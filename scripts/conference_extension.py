"""Opt-in, separately locked component audit using existing full-study caches.

This script does not change the original runner or its scientific identities.
Its own content identity is recorded in an additional immutable review lock.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.conference_diagnostics import (
    audit_sift,
    availability,
    coverage_contrasts,
    render_preflight,
    support_warnings,
)
from warpaudit.cache.hashing import file_digest, short_hash
from warpaudit.cache.store import ShardedTable
from warpaudit.cli import (
    _code_identity,
    _current_registration_rows,
    _feature_code_identity,
    _feature_matrix,
    _pairs_manifest,
    _project_root,
    _utc_now,
)
from warpaudit.config import load_config
from warpaudit.evaluation.full_study import _population
from warpaudit.evaluation.metrics import (
    auroc,
    average_precision,
    brier_score,
    calibration_in_the_large,
)
from warpaudit.evaluation.riskcoverage import risk_coverage_curve
from warpaudit.evaluation.weights import inverse_group_size_weights
from warpaudit.predictors.model import fit_logistic_detector
from warpaudit.predictors.policy import FrozenPolicy
from warpaudit.protocols.full_freeze import require_full_freeze
from warpaudit.protocols.splits import make_outer_folds


def clean_json(value):
    """Undefined scientific quantities are JSON null, never zero."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, payload, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(clean_json(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")


def select_arms(names, spec):
    if (
        spec.get("feature_contract", "available_signals_descriptive")
        != "available_signals_descriptive"
    ):
        raise ValueError(
            "unsupported feature contract; inverse-equivalent measurements are not implemented"
        )
    arms = {}
    for arm, selectors in spec["arms"].items():
        columns = []
        for selector in selectors:
            matches = [
                n
                for n in names
                if (n.startswith(selector) if selector.endswith((":", "_")) else n == selector)
            ]
            if not matches:
                raise ValueError(f"{arm}: selector {selector!r} matches no features")
            columns.extend(matches)
        arms[arm] = tuple(dict.fromkeys(columns))
    for left, right in spec["contrasts"]:
        if left not in arms or right not in arms:
            raise ValueError("contrast refers to an unknown arm")
    if any(not 0 < c <= 1 for c in spec["requested_coverages"]):
        raise ValueError("requested coverages must be in (0, 1]")
    if spec["analysis_status"] not in ("prospective_descriptive", "exploratory"):
        raise ValueError("extension supports descriptive or exploratory analysis only")
    if not spec.get("transfer_modes", ["dataset"]) or any(
        mode not in ("dataset", "dataset_and_pipeline")
        for mode in spec.get("transfer_modes", ["dataset"])
    ):
        raise ValueError("unknown or empty transfer modes")
    return arms


def assemble(cfg, root, datasets):
    """Restrict registrations to declared datasets BEFORE joining outcomes."""
    pairs, _ = _pairs_manifest(cfg, root)
    paths = cfg.paths.resolve(root)
    registrations = ShardedTable(paths["cache_root"], "registrations").load()
    if registrations.empty:
        raise ValueError("full-study registration cache absent; run on the original workstation")
    registrations = _current_registration_rows(registrations, pairs, cfg, root)
    registrations = registrations[
        registrations.dataset_id.astype(str).isin(datasets)
        & (registrations.direction.astype(str) == "canonical")
        & (registrations.condition.astype(str) == "clean")
    ].drop_duplicates("job_id", keep="last")
    labels = ShardedTable(paths["cache_root"], "labels").load()
    if labels.empty:
        raise ValueError("full-study label cache absent")
    labels = labels[labels.job_id.isin(registrations.job_id)].drop_duplicates("job_id", keep="last")
    needed = ["job_id", "operational_failure", "eligible_for_acceptance", "bounded_loss"]
    cases = registrations.merge(labels[needed], on="job_id", validate="one_to_one")
    if len(cases) != len(registrations) or cases.empty:
        raise ValueError("selected current registrations have missing labels or no cases")
    expected = pairs[pairs.dataset_id.astype(str).isin(datasets)]
    for dataset in datasets:
        for pipeline in cfg.common_block:
            want = set(
                expected.loc[expected.dataset_id.astype(str) == dataset, "pair_id"].astype(str)
            )
            got = set(
                cases.loc[
                    (cases.dataset_id.astype(str) == dataset)
                    & (cases.pipeline_id.astype(str) == pipeline),
                    "pair_id",
                ].astype(str)
            )
            if not want or got != want:
                raise ValueError(f"{dataset}/{pipeline}: incomplete canonical case coverage")
    features = ShardedTable(paths["cache_root"], "features", key_column="feature_id").load()
    if features.empty:
        raise ValueError("full-study feature cache absent")
    features = features[
        (features.config_hash.astype(str) == cfg.hash)
        & (features.code_hash.astype(str) == _feature_code_identity(root))
        & features.job_id.isin(cases.job_id)
    ]
    if set(cases.job_id) - set(features.job_id):
        raise ValueError("selected cases lack current feature bundles")
    matrix = _feature_matrix(features, cfg.full_study.augmented_families, cases.job_id.tolist())
    cases = pd.concat([cases.reset_index(drop=True), matrix.reset_index(drop=True)], axis=1)
    cases["explicit_failure"] = ~cases.eligible_for_acceptance.astype(bool)
    cases = cases.sort_values(["dataset_id", "pair_id", "pipeline_id"], kind="stable").reset_index(
        drop=True
    )
    return cases, tuple(matrix.columns)


def partition(dev, seed):
    groups = sorted(set(dev.group_id.astype(str)))
    if len(groups) < 5:
        raise ValueError("fewer than five development groups")
    folds = make_outer_folds(groups, n_folds=5, seed=seed)
    calibration = dev.group_id.astype(str).map(folds).to_numpy() == 0
    return dev.loc[~calibration], dev.loc[calibration]


def support(frame):
    y = frame.operational_failure.to_numpy(float)
    return {
        "cases": len(frame),
        "groups": frame.group_id.nunique(),
        "failures": int(np.sum(y == 1)),
        "successes": int(np.sum(y == 0)),
        "failure_groups": frame.loc[y == 1, "group_id"].nunique(),
        "success_groups": frame.loc[y == 0, "group_id"].nunique(),
        "valid_binary_labels": bool(np.isin(y, [0, 1]).all()),
    }


def fit_arm(train, calibration, columns, cfg, name):
    started = time.perf_counter()
    model = fit_logistic_detector(
        train.loc[:, columns],
        train.operational_failure.to_numpy(float),
        train.group_id.astype(str).to_numpy(),
        feature_names=columns,
        C_values=cfg.learners.logistic_C,
        seed=cfg.splits.seed,
    )
    model.calibrate(
        calibration.loc[:, columns],
        calibration.operational_failure.to_numpy(float),
        method=cfg.policy.calibration_method,
        sample_weight=inverse_group_size_weights(calibration.group_id.astype(str)),
    )
    probability = model.predict_probability(calibration.loc[:, columns])
    if not np.isfinite(probability).all():
        raise ValueError(f"{name}: nonfinite calibration predictions")
    policy = FrozenPolicy.from_calibration(
        f"extension-{name}",
        _population(calibration, probability),
        nominal_acceptance=cfg.policy.nominal_acceptance,
        detector_hash=model.fingerprint,
        preprocessing_hash=model.preprocessor.fingerprint,
    )
    return model, policy, time.perf_counter() - started


def training_sets(cases, cfg, modes):
    development = cases[cases.dataset_id.astype(str).isin(cfg.full_study.development_datasets)]
    for mode in modes:
        for pipeline in cfg.common_block:
            own = development.pipeline_id.astype(str) == pipeline
            dev = development.loc[own if mode == "dataset" else ~own]
            if mode == "dataset_and_pipeline" and len(cfg.common_block) < 2:
                raise ValueError("held-out-pipeline transfer requires at least two pipelines")
            yield mode, pipeline, dev


def preflight(cases, arms, cfg, modes=("dataset",)):
    report = {"all_passed": True, "pipelines": {}}
    for mode, pipeline, dev in training_sets(cases, cfg, modes):
        row = {
            "datasets": {str(d): support(f) for d, f in dev.groupby("dataset_id")},
            "arms": {},
            "issues": [],
            "warnings": [],
            "target_feature_contract": {},
        }
        row["transfer_mode"] = mode
        row["training_pipelines"] = sorted(set(dev.pipeline_id.astype(str)))
        report["pipelines"][pipeline if mode == "dataset" else f"held_out/{pipeline}"] = row
        target_dev = cases[
            cases.dataset_id.astype(str).isin(cfg.full_study.development_datasets)
            & (cases.pipeline_id.astype(str) == pipeline)
        ]
        row["target_feature_contract"] = {
            arm: availability(target_dev, columns) for arm, columns in arms.items()
        }
        if not np.isfinite(dev.bounded_loss.to_numpy(float)).all():
            row["issues"].append("development contains undefined bounded loss; audit ground truth")
        try:
            train, calibration = partition(dev, cfg.splits.seed)
            row["train"], row["calibration"] = support(train), support(calibration)
            for label, subset in (("train", train), ("calibration", calibration)):
                if (
                    not support(subset)["valid_binary_labels"]
                    or subset.operational_failure.nunique() != 2
                ):
                    row["issues"].append(f"{label} lacks two valid outcome classes")
            inner = make_outer_folds(
                sorted(set(train.group_id.astype(str))),
                n_folds=min(5, train.group_id.nunique()),
                seed=cfg.splits.seed,
            )
            row["inner_folds"] = []
            for fold in sorted(set(inner.values())):
                valid = train.group_id.astype(str).map(inner).to_numpy() == fold
                item = {
                    "fold": fold,
                    "train": support(train.loc[~valid]),
                    "validation": support(train.loc[valid]),
                }
                item["usable"] = all(
                    item[k]["failures"] and item[k]["successes"] for k in ("train", "validation")
                )
                row["inner_folds"].append(item)
            if not all(item["usable"] for item in row["inner_folds"]):
                row["issues"].append("one or more fixed inner folds lack class support")
            for arm, columns in arms.items():
                finite = np.isfinite(dev.loc[:, columns].to_numpy(float))
                result = {
                    "columns": columns,
                    "availability": availability(dev, columns),
                    "train_availability": availability(train, columns),
                    "calibration_availability": availability(calibration, columns),
                    "finite_fraction": float(finite.mean()),
                    "all_missing_columns": [
                        c for c, ok in zip(columns, finite.any(axis=0), strict=True) if not ok
                    ],
                }
                row["arms"][arm] = result
                try:
                    model, _, elapsed = fit_arm(train, calibration, columns, cfg, arm)
                    result.update(
                        fit_passed=True,
                        model_hash=model.fingerprint,
                        fit_seconds=elapsed,
                        selected_C=model.selected_C,
                    )
                except ValueError as exc:
                    result.update(fit_passed=False, reason=str(exc))
                    row["issues"].append(f"{arm}: {exc}")
        except ValueError as exc:
            row["issues"].append(str(exc))
        row["warnings"] = support_warnings(row)
        missing = sorted(
            {
                c
                for a in row["target_feature_contract"].values()
                for c in a["all_missing_measurements"]
            }
        )
        if missing:
            row["warnings"].append(
                {
                    "code": "unavailable_target_measurements",
                    "columns": missing,
                    "message": "Target pipeline lacks entire measurements on development; "
                    "arm comparisons use available signals, and pipeline transfer "
                    "also changes feature availability.",
                }
            )
        if row["issues"]:
            report["all_passed"] = False
    return report


def metrics(frame, probability, policy, coverages, cutoff):
    population = _population(frame, probability)
    outcome = policy.apply(population)
    curve = risk_coverage_curve(population)
    low = probability <= cutoff
    y, w = population.failure, population.weight
    requested = []
    for coverage in coverages:
        indices = np.flatnonzero(curve.coverage <= coverage + 1e-12)
        attainable = coverage <= curve.max_attainable_coverage + 1e-12 and len(indices) > 0
        requested.append(
            {
                "requested_coverage": coverage,
                "attainable": bool(attainable),
                "achieved_coverage": float(curve.coverage[indices[-1]]) if attainable else None,
                "risk": float(curve.risk[indices[-1]]) if attainable else None,
            }
        )
    return {
        **support(frame),
        "brier": brier_score(probability, y, w),
        "auroc": auroc(probability, y, w),
        "average_precision": average_precision(probability, y, w),
        "calibration_in_the_large": calibration_in_the_large(probability, y, w),
        "coverage": outcome.coverage,
        "accepted_risk": outcome.risk,
        "high_confidence_cases": int(low.sum()),
        "high_confidence_failures": int(np.sum(low & (y == 1))),
        "high_confidence_failure_risk": float(w[low & (y == 1)].sum() / w[low].sum())
        if w[low].sum()
        else None,
        "retrospective_risk_coverage": requested,
        "max_attainable_coverage": curve.max_attainable_coverage,
    }


def evaluate(cases, arms, spec, cfg):
    rows = []
    for mode, pipeline, dev in training_sets(cases, cfg, spec.get("transfer_modes", ["dataset"])):
        train, calibration = partition(dev, cfg.splits.seed)
        models = {
            arm: fit_arm(train, calibration, columns, cfg, arm) for arm, columns in arms.items()
        }
        for dataset in (
            *cfg.full_study.external_confirmatory_datasets,
            *cfg.full_study.external_descriptive_datasets,
        ):
            cell = cases[
                (cases.dataset_id.astype(str) == dataset)
                & (cases.pipeline_id.astype(str) == pipeline)
            ]
            row = {
                "dataset": dataset,
                "pipeline": pipeline,
                "transfer_mode": mode,
                "training_pipelines": sorted(set(dev.pipeline_id.astype(str))),
                "analysis_status": spec["analysis_status"],
                "population_intervals": None,
                "development_support": {
                    "train": support(train),
                    "calibration": support(calibration),
                },
                "review_warnings": support_warnings(
                    {
                        "datasets": {str(d): support(f) for d, f in dev.groupby("dataset_id")},
                        "train": support(train),
                        "calibration": support(calibration),
                    }
                ),
                "arms": {},
                "contrasts": [],
            }
            for arm, (model, policy, elapsed) in models.items():
                started = time.perf_counter()
                probability = model.predict_probability(cell.loc[:, arms[arm]])
                prediction_seconds = time.perf_counter() - started
                row["arms"][arm] = {
                    **metrics(
                        cell,
                        probability,
                        policy,
                        spec["requested_coverages"],
                        cfg.full_study.high_confidence_cutoff,
                    ),
                    "model_hash": model.fingerprint,
                    "fit_seconds": elapsed,
                    "prediction_seconds": prediction_seconds,
                    "feature_acquisition_seconds": None,
                    "training_feature_availability": availability(train, arms[arm]),
                    "target_feature_availability": availability(cell, arms[arm]),
                    "interpretation": "Available-signal bundle; missing features are not recovered by imputation.",
                }
            for left, right in spec["contrasts"]:
                a, b = row["arms"][left], row["arms"][right]
                row["contrasts"].append(
                    {
                        "reference": left,
                        "candidate": right,
                        "brier_improvement": a["brier"] - b["brier"],
                        "auroc_improvement": b["auroc"] - a["auroc"],
                        "accepted_risk_improvement": a["accepted_risk"] - b["accepted_risk"],
                        "coverage_change": b["coverage"] - a["coverage"],
                        "retrospective_matched_coverage": coverage_contrasts(
                            a["retrospective_risk_coverage"], b["retrospective_risk_coverage"]
                        ),
                    }
                )
            rows.append(row)
    return rows


def identity(cfg, root, spec):
    return {
        "config_hash": cfg.hash,
        "base_code_identity": _code_identity(root),
        "extension_code_hash": file_digest(Path(__file__)),
        "diagnostics_code_hash": file_digest(Path(__file__).with_name("conference_diagnostics.py")),
        "spec_hash": short_hash(spec),
    }


def read_lock(path, current):
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = lock.pop("lock_hash")
    if short_hash(lock) != digest or lock["identity"] != current:
        raise ValueError(
            "extension lock was edited or code/config/spec changed; reviewed replacement required"
        )
    return lock, digest


def validate_development_review(out, current, development_hash):
    review_path = out / "development_review.json"
    if not review_path.exists():
        raise ValueError(
            "run the development-only review command before locking; SIFT review evidence absent"
        )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    digest = review.pop("review_hash")
    if (
        short_hash(review) != digest
        or review["identity"] != current
        or review["development_hash"] != development_hash
    ):
        raise ValueError("development review is stale or edited; rerun review before locking")
    if not review["sift"]["passed"]:
        raise ValueError("SIFT label audit failed; resolve mismatches before locking")
    for record in review["sift"]["gallery"]:
        if record["file"] and file_digest(out / record["file"]) != record["sha256"]:
            raise ValueError("SIFT review image changed; rerun review")
    return digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["preflight", "review", "lock", "verify-lock", "evaluate"]
    )
    parser.add_argument("--config", default="configs/full_study.yaml")
    parser.add_argument("--spec", default="configs/conference_extension.json")
    parser.add_argument("--output", default="conference_outputs/review_v2")
    parser.add_argument("--signed-off-by")
    parser.add_argument("--review-note")
    args = parser.parse_args(argv)
    cfg, root = load_config(args.config), _project_root(args.config)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    out = Path(args.output)
    lock_path = out / "extension_lock.json"
    current = identity(cfg, root, spec)
    if args.command == "verify-lock":
        lock, _ = read_lock(lock_path, current)
        dev, names = assemble(cfg, root, cfg.full_study.development_datasets)
        arms = select_arms(names, spec)
        if clean_json(arms) != lock["arms"] or case_identity(dev, arms) != lock["development_hash"]:
            raise ValueError("development evidence differs from reviewed extension lock")
        return 0
    if args.command == "evaluate":
        lock, digest = read_lock(lock_path, current)
        base = require_full_freeze(cfg.paths.resolve(root)["manifests"], config_hash=cfg.hash)
        if base.code_identity != current["base_code_identity"]:
            raise ValueError("base code differs from G2 freeze")
        plan = json.loads(
            (cfg.paths.resolve(root)["manifests"] / "full_study_information_plan.json").read_text()
        )
        if short_hash(plan) != base.information_plan_hash:
            raise ValueError("base information plan differs from G2 freeze")
        datasets = (
            *cfg.full_study.development_datasets,
            *cfg.full_study.external_confirmatory_datasets,
            *cfg.full_study.external_descriptive_datasets,
        )
        cases, names = assemble(cfg, root, datasets)
        arms = select_arms(names, spec)
        if clean_json(arms) != lock["arms"]:
            raise ValueError("available feature columns differ from reviewed lock")
        dev = cases[cases.dataset_id.astype(str).isin(cfg.full_study.development_datasets)]
        if case_identity(dev, arms) != lock["development_hash"]:
            raise ValueError("development evidence differs from reviewed extension lock")
        write_json(
            out / "extension_results.json",
            {
                "lock_hash": digest,
                "g2_hash": base.hash,
                "identity": current,
                "cells": evaluate(cases, arms, spec, cfg),
            },
            exclusive=True,
        )
        return 0
    cases, names = assemble(cfg, root, cfg.full_study.development_datasets)
    arms = select_arms(names, spec)
    report = preflight(cases, arms, cfg, spec.get("transfer_modes", ["dataset"]))
    write_json(out / "development_preflight.json", {"identity": current, **report})
    (out / "development_preflight.md").write_text(render_preflight(report), encoding="utf-8")
    print(
        f"Preflight fits passed: {report['all_passed']}; "
        f"review warnings: {sum(len(r['warnings']) for r in report['pipelines'].values())}. "
        f"Read {out / 'development_preflight.md'}"
    )
    if args.command == "review":
        diagnostics = (
            audit_sift(cases, cfg, root, out)
            if "sift_h" in cfg.common_block
            else {"passed": True, "audited_cases": 0, "gallery": [], "note": "SIFT not configured."}
        )
        review = clean_json(
            {
                "identity": current,
                "development_hash": case_identity(cases, arms),
                "sift": diagnostics,
                "feature_contract": "available_signals_descriptive; TPS has no inverse sampler",
            }
        )
        write_json(out / "development_review.json", {**review, "review_hash": short_hash(review)})
        print(
            f"Development review: {out / 'development_review.json'}; "
            f"SIFT gallery: {out / 'sift_review' / 'contact_sheet.png'}"
        )
        return 0 if report["all_passed"] and diagnostics["passed"] else 3
    if args.command == "preflight":
        return 0 if report["all_passed"] else 3
    if not report["all_passed"]:
        raise ValueError("development preflight failed; inspect development_preflight.json")
    if (
        not args.signed_off_by
        or not args.signed_off_by.strip()
        or not args.review_note
        or not args.review_note.strip()
    ):
        raise ValueError("actual reviewer identity and review findings are required")
    if spec["analysis_status"] == "prospective_descriptive":
        cache = cfg.paths.resolve(root)["cache_root"]
        registrations = ShardedTable(cache, "registrations").load()
        external_ids = set(
            registrations.loc[
                ~registrations.dataset_id.astype(str).isin(cfg.full_study.development_datasets),
                "job_id",
            ]
        )
        labels = ShardedTable(cache, "labels").load()
        if set(labels.job_id) & external_ids:
            raise ValueError(
                "external labels already cached; cannot certify prospective lock; use exploratory spec"
            )
    review_digest = None
    if "sift_h" in cfg.common_block:
        review_digest = validate_development_review(out, current, case_identity(cases, arms))
    lock = {
        "generated_at": _utc_now(),
        "development_review_hash": review_digest,
        "identity": current,
        "spec": spec,
        "arms": arms,
        "development_hash": case_identity(cases, arms),
        "signed_off_by": args.signed_off_by,
        "review_note": args.review_note,
        "attestation": "Reviewer must disclose any external outcomes inspected outside this cache.",
    }
    lock = clean_json(lock)
    write_json(lock_path, {**lock, "lock_hash": short_hash(lock)}, exclusive=True)
    return 0


def case_identity(cases, arms):
    columns = [
        "job_id",
        "dataset_id",
        "group_id",
        "pipeline_id",
        "operational_failure",
        "explicit_failure",
        "bounded_loss",
        *dict.fromkeys(c for cols in arms.values() for c in cols),
    ]
    return short_hash(
        clean_json(cases.sort_values("job_id")[columns].to_dict(orient="records")), length=64
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"conference extension: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc
