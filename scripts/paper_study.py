"""Workstation paper-evidence study: development, external lock, external evaluation.

Default development stage adds repeated calibration budgets, one fixed robust
calibration sensitivity and feature-level margin forensics to cached evidence.
No expensive registration runs are launched. External stages require separately
prepared data/features and a reviewed cohort record; they never fabricate access.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

from scripts import development_decision as dd
from scripts import publication_audit as audit
from scripts.conference_extension import (
    assemble,
    case_identity,
    fit_arm,
    identity,
    partition,
    select_arms,
)
from warpaudit.cache.hashing import file_digest, short_hash
from warpaudit.cache.store import ShardedTable, atomic_write_bytes, atomic_write_text
from warpaudit.cli import (
    _compute_label_task,
    _current_registration_rows,
    _feature_code_identity,
    _feature_matrix,
    _pairs_manifest,
)
from warpaudit.config import load_config
from warpaudit.evaluation.full_study import _population
from warpaudit.evaluation.metrics import auroc
from warpaudit.evaluation.weights import inverse_group_size_weights
from warpaudit.predictors.policy import FrozenPolicy


def validate_design(design, cfg):
    """Reject ambiguous designs before fitting or opening external outcomes."""
    if design["outer_folds"] != 5:
        raise ValueError("The inherited development partitions require five folds")
    for key, minimum in [
        ("budget_repeats", 1),
        ("conditional_bootstrap_resamples", 100),
        ("external_min_test_groups", 2),
    ]:
        if type(design[key]) is not int or design[key] < minimum:
            raise ValueError(f"Invalid {key}")
    for key in ["primary_pipelines", "arms", "modes", "target_group_budgets"]:
        if not design[key] or len(set(design[key])) != len(design[key]):
            raise ValueError(f"Empty or duplicate {key}")
    if not set(design["modes"]) <= {"within_pipeline", "held_out_pipeline"}:
        raise ValueError("Unknown source mode")
    if not set(design["primary_pipelines"]) <= set(cfg.common_block):
        raise ValueError("Missing primary pipeline")
    if any(b != "all" and (type(b) is not int or b < 1) for b in design["target_group_budgets"]):
        raise ValueError("Budgets must be positive group counts or all")
    if not 0 < design["external_calibration_fraction"] < 1:
        raise ValueError("Invalid external calibration fraction")
    q = design["sensitivity"]["training_margin_quantiles"]
    penalty = design["sensitivity"]["slope_l2_penalty"]
    if len(q) != 2 or not 0 <= q[0] < q[1] <= 1 or not np.isfinite(penalty) or penalty < 0:
        raise ValueError("Invalid sensitivity bounds or penalty")
    if cfg.policy.calibration_method != "platt":
        raise ValueError(
            "This audit labels its unconstrained probability map Platt; use platt config"
        )


def planned_variants(design):
    yield "source", "source", 0
    yield "source_monotone_clipped", "source", 0
    for budget in design["target_group_budgets"]:
        for repeat in range(1 if budget == "all" else design["budget_repeats"]):
            for variant in ["threshold_only", "target_platt", "target_monotone_clipped"]:
                yield variant, str(budget), repeat


@dataclass
class MonotoneMap:
    """Frozen clipped-margin sensitivity; all scale bounds come from source train."""

    base: object
    low: float
    high: float
    slope: float
    intercept: float
    optimizer_message: str

    def predict_probability(self, frame):
        raw = self.base.decision_function(frame)
        x = (np.clip(raw, self.low, self.high) - self.low) / (self.high - self.low)
        return expit(self.intercept + self.slope * x)

    @property
    def fingerprint(self):
        return short_hash(
            {
                "base": self.base.fingerprint,
                "bounds": [self.low, self.high],
                "slope": self.slope,
                "intercept": self.intercept,
            }
        )

    @property
    def preprocessor(self):
        return self.base.preprocessor


def fit_monotone(base, train, calibration, columns, design):
    bounds = np.quantile(
        base.decision_function(train.loc[:, columns]),
        design["sensitivity"]["training_margin_quantiles"],
    )
    low, high = map(float, bounds)
    if not np.isfinite(bounds).all() or high <= low:
        raise ValueError("constant/nonfinite training-margin range")
    raw = base.decision_function(calibration.loc[:, columns])
    x = (np.clip(raw, low, high) - low) / (high - low)
    y = calibration.operational_failure.to_numpy(float)
    if len(set(y)) != 2:
        raise ValueError("both calibration classes required")
    w = inverse_group_size_weights(calibration.group_id.astype(str))
    penalty = design["sensitivity"]["slope_l2_penalty"]

    def objective(theta):
        z = theta[0] + theta[1] * x
        loss = np.sum(w * (np.logaddexp(0, z) - y * z)) + 0.5 * penalty * theta[1] ** 2
        error = w * (expit(z) - y)
        return loss, np.array([error.sum(), np.sum(error * x) + penalty * theta[1]])

    mean = float(np.average(y, weights=w))
    fit = minimize(
        objective,
        [np.log(mean / (1 - mean)), 1.0],
        jac=True,
        method="L-BFGS-B",
        bounds=[(None, None), (0.0, None)],
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"Monotone sensitivity fit failed: {fit.message}")
    return MonotoneMap(base, low, high, float(fit.x[1]), float(fit.x[0]), str(fit.message))


def target_sample(pool, budget, repeat, seed):
    groups = sorted(
        set(pool.group_id.astype(str)), key=lambda g: short_hash([seed, repeat, g], length=64)
    )
    if budget != "all" and budget > len(groups):
        return pool.iloc[:0]
    selected = groups if budget == "all" else groups[:budget]
    return pool[pool.group_id.astype(str).isin(selected)].copy()


def operating_policy(model, calibration, columns, cfg):
    return FrozenPolicy.from_calibration(
        "paper-study",
        _population(calibration, model.predict_probability(calibration.loc[:, columns])),
        nominal_acceptance=cfg.policy.nominal_acceptance,
        detector_hash=model.fingerprint,
        preprocessing_hash=model.preprocessor.fingerprint,
    )


def forensics(model, train, calibration, columns):
    """Explain largest absolute calibration margins without excluding any case."""
    matrix = model.preprocessor.transform(calibration.loc[:, columns])
    training_matrix = model.preprocessor.transform(train.loc[:, columns])
    terms = matrix * model.model.coef_[0]
    raw = model.decision_function(calibration.loc[:, columns])
    train_raw = model.decision_function(train.loc[:, columns])
    rows = []
    for pos in np.argsort(-np.abs(raw), kind="stable")[:10]:
        case = calibration.iloc[pos]
        top = np.argsort(-np.abs(terms[pos]), kind="stable")[:5]
        rows.append(
            {
                "job_id": case.job_id,
                "group_id": case.group_id,
                "dataset": case.dataset_id,
                "failure": float(case.operational_failure),
                "margin": float(raw[pos]),
                "train_margin_min": float(train_raw.min()),
                "train_margin_max": float(train_raw.max()),
                "intercept": float(model.model.intercept_[0]),
                "all_contributions_sum": float(terms[pos].sum()),
                "top_terms": [
                    {
                        "feature": model.preprocessor.output_names[j],
                        "scaled_value": float(matrix[pos, j]),
                        "training_scaled_min": float(training_matrix[:, j].min()),
                        "training_scaled_max": float(training_matrix[:, j].max()),
                        "coefficient": float(model.model.coef_[0, j]),
                        "contribution": float(terms[pos, j]),
                        "unscaled_value": float(case[columns[j]]) if j < len(columns) else None,
                    }
                    for j in top
                ],
            }
        )
    return dd.clean_json(rows)


def run_cell(train, source_cal, target_pool, test, arms, cfg, design, cell):
    result = {
        "cell": cell,
        "status": [],
        "metrics": [],
        "contrasts": [],
        "forensics": [],
        "predictions": [],
    }
    # Pipeline repeats may share subjects only within one partition.
    sets = [set(f.group_id.astype(str)) for f in (train, source_cal, test)]
    if (
        sets[0] & sets[1]
        or sets[0] & sets[2]
        or sets[1] & sets[2]
        or set(target_pool.group_id.astype(str)) & sets[2]
        or set(target_pool.group_id.astype(str)) & sets[0]
    ):
        raise ValueError("Training/calibration/test group leakage")
    reasons, inner = dd.eligibility(train, source_cal, cfg.splits.seed)
    result["support"] = {
        n: audit.support(f)
        for n, f in [
            ("train", train),
            ("source_calibration", source_cal),
            ("target_pool", target_pool),
            ("test", test),
        ]
    }
    result["partition_groups"] = {
        n: sorted(set(f.group_id.astype(str)))
        for n, f in [
            ("train", train),
            ("source_calibration", source_cal),
            ("target_pool", target_pool),
            ("test", test),
        ]
    }
    result["inner_support"] = inner
    if reasons or test.empty:
        result["status"].append({"status": "not_estimable", "reasons": reasons or ["empty test"]})
        return result
    for arm, columns in arms.items():
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            source, policy, fit_seconds = fit_arm(train, source_cal, columns, cfg, arm)
        result["forensics"].append(
            {"arm": arm, "cases": forensics(source, train, source_cal, columns)}
        )
        result["status"].append(
            {
                "arm": arm,
                "variant": "source_fit",
                "status": "ok",
                "warnings": [str(w.message) for w in captured],
                "fit_seconds": fit_seconds,
                "diagnostics": {
                    n: audit.model_diagnostics(source, f, columns)
                    for n, f in [("train", train), ("calibration", source_cal), ("test", test)]
                },
            }
        )
        baseline = None

        def measure(
            variant,
            budget,
            repeat,
            fitted,
            calibration,
            policy=None,
            *,
            columns=columns,
            source=source,
            arm=arm,
        ):
            nonlocal baseline
            if policy is None:
                policy = operating_policy(fitted, calibration, columns, cfg)
            prob = fitted.predict_probability(test.loc[:, columns])
            if not np.isfinite(prob).all():
                raise ValueError("Nonfinite test probability")
            frame = test[
                [
                    "job_id",
                    "group_id",
                    "pair_id",
                    "dataset_id",
                    "pipeline_id",
                    "operational_failure",
                    "explicit_failure",
                    "bounded_loss",
                ]
            ].copy()
            frame["raw_score"] = source.decision_function(test.loc[:, columns])
            frame["probability"] = prob
            frame["accepted"] = ~frame.explicit_failure & (prob <= policy.threshold)
            frame["fold"] = cell["fold"]
            key = {**cell, "arm": arm, "variant": variant, "budget": str(budget), "repeat": repeat}
            if variant == "source":
                baseline = frame.copy()
            result["status"].append(
                {
                    **key,
                    "status": "ok",
                    "calibration_groups": calibration.group_id.nunique(),
                    "calibration_cases": len(calibration),
                    "threshold": policy.threshold,
                    "threshold_negative_infinity": policy.threshold == -np.inf,
                    "model_hash": fitted.fingerprint,
                    "monotone_parameters": {
                        "low": fitted.low,
                        "high": fitted.high,
                        "slope": fitted.slope,
                        "intercept": fitted.intercept,
                    }
                    if isinstance(fitted, MonotoneMap)
                    else None,
                }
            )
            for dataset, sub in [("ALL", frame), *[(d, f) for d, f in frame.groupby("dataset_id")]]:
                m = dd.summarize(sub, cfg.policy.requested_coverages)
                result["metrics"].append({**key, "dataset": str(dataset), **m})
                if baseline is not None and variant != "source":
                    ref = baseline if dataset == "ALL" else baseline[baseline.dataset_id == dataset]
                    bm = dd.summarize(ref, cfg.policy.requested_coverages)
                    result["contrasts"].append(
                        {
                            **key,
                            "dataset": str(dataset),
                            "brier_improvement": bm["brier"] - m["brier"],
                            "source_coverage": bm["coverage"],
                            "coverage": m["coverage"],
                            "source_risk": bm["accepted_risk"],
                            "risk": m["accepted_risk"],
                            "matched_coverage": dd.safe_coverage_contrasts(
                                bm["risk_coverage"], m["risk_coverage"]
                            ),
                        }
                    )
            result["predictions"].extend({**r, **key} for r in frame.to_dict("records"))

        measure("source", "source", 0, source, source_cal, policy)
        try:
            robust = fit_monotone(source, train, source_cal, columns, design)
        except ValueError as exc:
            result["status"].append(
                {
                    "arm": arm,
                    "variant": "source_monotone_clipped",
                    "status": "not_estimable",
                    "reason": str(exc),
                }
            )
        else:
            measure("source_monotone_clipped", "source", 0, robust, source_cal)
        for budget in design["target_group_budgets"]:
            for repeat in range(1 if budget == "all" else design["budget_repeats"]):
                target = target_sample(target_pool, budget, repeat, cfg.splits.seed)
                key = {"arm": arm, "budget": str(budget), "repeat": repeat}
                if target.empty:
                    result["status"].append(
                        {**key, "status": "not_estimable", "reason": "insufficient target groups"}
                    )
                    continue
                measure("threshold_only", budget, repeat, source, target)
                if target.operational_failure.nunique() != 2:
                    result["status"].append(
                        {
                            **key,
                            "variant": "target_recalibration",
                            "status": "not_estimable",
                            "reason": "target calibration one class",
                        }
                    )
                    continue
                recal = copy.deepcopy(source)
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    recal.calibrate(
                        target.loc[:, columns],
                        target.operational_failure.to_numpy(float),
                        method=cfg.policy.calibration_method,
                        sample_weight=inverse_group_size_weights(target.group_id.astype(str)),
                    )
                result["status"].append(
                    {
                        **key,
                        "variant": "target_platt_fit",
                        "status": "ok",
                        "warnings": [str(w.message) for w in captured],
                        "platt_slope": float(recal.calibrator.coef_[0, 0])
                        if hasattr(recal.calibrator, "coef_")
                        else None,
                    }
                )
                measure("target_platt", budget, repeat, recal, target)
                try:
                    robust = fit_monotone(source, train, target, columns, design)
                except ValueError as exc:
                    result["status"].append(
                        {
                            **key,
                            "variant": "target_monotone_clipped",
                            "status": "not_estimable",
                            "reason": str(exc),
                        }
                    )
                else:
                    measure("target_monotone_clipped", budget, repeat, robust, target)
    return dd.clean_json(result)


def worker(task):
    train, cal, pool, test, arms, cfg, design, cell, path, run_id = task
    with threadpool_limits(limits=1):
        result = run_cell(train, cal, pool, test, arms, cfg, design, cell)
    dd.save(path, {"run_id": run_id, "hash": short_hash(result, length=64), "result": result})
    return str(path)


def execute(tasks, paths, workers, run_id):
    todo = []
    for task, path in zip(tasks, paths, strict=True):
        if path.exists():
            data = json.loads(path.read_text())
            if data["run_id"] != run_id or data["hash"] != short_hash(data["result"], length=64):
                raise ValueError(f"Checkpoint mismatch: {path}")
        else:
            todo.append(task)
    print(f"{len(paths)-len(todo)}/{len(paths)} cells reusable", flush=True)
    if workers == 1:
        for task in todo:
            print("Completed " + worker(task), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            for future in as_completed([pool.submit(worker, t) for t in todo]):
                print("Completed " + future.result(), flush=True)
    return [json.loads(p.read_text())["result"] for p in paths]


def external_inputs(cfg, dataset):
    """Read only current ground-truth-free registrations/features and index metadata."""
    pairs, _ = _pairs_manifest(cfg, ROOT)
    indexed = pairs[pairs.dataset_id.astype(str) == dataset]
    declared_external = set(cfg.full_study.external_confirmatory_datasets) | set(
        cfg.full_study.external_descriptive_datasets
    )
    if (
        indexed.empty
        or dataset in cfg.full_study.development_datasets
        or dataset not in declared_external
    ):
        raise ValueError("External cohort must be indexed separately from development")
    cache = cfg.paths.resolve(ROOT)["cache_root"]
    regs = _current_registration_rows(ShardedTable(cache, "registrations").load(), pairs, cfg, ROOT)
    regs = regs[
        (regs.dataset_id.astype(str) == dataset)
        & (regs.condition == "clean")
        & (regs.direction == "canonical")
    ].drop_duplicates("job_id", keep="last")
    for pipeline in cfg.common_block:
        if set(regs.loc[regs.pipeline_id == pipeline, "pair_id"]) != set(indexed.pair_id):
            raise ValueError(
                f"Incomplete external registrations for {pipeline}; prepare ground-truth-free caches first"
            )
    feat = ShardedTable(cache, "features", key_column="feature_id").load()
    feat = feat[
        (feat.config_hash == cfg.hash)
        & (feat.code_hash == _feature_code_identity(ROOT))
        & feat.job_id.isin(regs.job_id)
    ]
    if set(regs.job_id) - set(feat.job_id):
        raise ValueError("External current features incomplete")
    matrix = _feature_matrix(feat, cfg.full_study.augmented_families, regs.job_id.tolist())
    frame = pd.concat([regs.reset_index(drop=True), matrix.reset_index(drop=True)], axis=1)
    digest = short_hash(
        {
            "pairs": dd.clean_json(indexed.to_dict("records")),
            "registrations": dd.clean_json(regs.to_dict("records")),
            "features": dd.clean_json(feat.to_dict("records")),
        },
        length=64,
    )
    return frame, indexed, tuple(matrix.columns), digest


def review_cohort(path, cfg, design):
    record = json.loads(Path(path).read_text())
    for key in ("dataset", "reviewer", "review_note", "registration_appropriateness"):
        if not str(record.get(key, "")).strip():
            raise ValueError(f"Cohort review needs {key}")
    if (
        record.get("annotation_kind") != "natural_landmarks"
        or record.get("outcomes_previously_inspected") is not False
    ):
        raise ValueError(
            "Prospective external cohort requires natural landmarks and no prior outcome inspection"
        )
    if record.get("independent_grouping_reviewed") is not True:
        raise ValueError("Independent grouping review required")
    hashes = {}
    for key in ("landmark_review", "grouping_review", "permission_record"):
        file = Path(record.get("evidence_files", {}).get(key, ""))
        if not file.is_file():
            raise ValueError(f"Missing actual evidence file: {key}")
        hashes[key] = file_digest(file)
    frame, indexed, names, digest = external_inputs(cfg, record["dataset"])
    if indexed.group_id.isna().any() or indexed.group_id.astype(str).str.strip().eq("").any():
        raise ValueError("External group identities missing")
    if not indexed.annotation_kind.eq("landmarks").all():
        raise ValueError("External index must contain landmark annotations")
    if (
        indexed.annotation_provenance.fillna("")
        .str.contains("synthetic|published.transform|generated.control", case=False, regex=True)
        .any()
    ):
        raise ValueError(
            "Synthetic/transform-derived annotations cannot serve as natural validation"
        )
    groups = sorted(
        set(indexed.group_id.astype(str)),
        key=lambda g: short_hash([cfg.splits.seed, "external", g], length=64),
    )
    ncal = max(1, int(np.ceil(len(groups) * design["external_calibration_fraction"])))
    if len(groups) - ncal < design["external_min_test_groups"] or ncal < max(
        [b for b in design["target_group_budgets"] if type(b) is int], default=1
    ):
        raise ValueError(
            "Insufficient external calibration/test groups; do not manufacture independence"
        )
    return record, hashes, frame, indexed, names, digest, groups[:ncal], groups[ncal:]


def render(results, out, design, stage, cfg):
    metrics = [m for r in results for m in r["metrics"]]
    if not metrics:
        dd.save(out / "not_estimable.json", results)
        raise ValueError("All cells not estimable; inspect not_estimable.json")
    predictions = pd.DataFrame([p for r in results for p in r["predictions"]])
    atomic_write_text(out / "predictions.csv", predictions.to_csv(index=False))
    flat = [{k: v for k, v in m.items() if k != "risk_coverage"} for m in metrics]
    atomic_write_text(out / "cell_metrics.csv", pd.DataFrame(flat).to_csv(index=False))
    dd.save(
        out / "cell_contrasts.json",
        [{"cell": r["cell"], "contrasts": r["contrasts"]} for r in results],
    )
    dd.save(
        out / "fit_diagnostics.json",
        [{"cell": r["cell"], "support": r["support"], "status": r["status"]} for r in results],
    )
    dd.save(
        out / "forensic_cases.json",
        [{"cell": r["cell"], "records": r["forensics"]} for r in results],
    )
    # Export terms and scores without job/group identifiers, keeping local linkage separately.
    terms = []
    for r in results:
        for a in r["forensics"]:
            for rank, c in enumerate(a["cases"]):
                for term in c["top_terms"]:
                    terms.append(
                        {
                            **r["cell"],
                            "arm": a["arm"],
                            "absolute_margin_rank": rank + 1,
                            **{
                                k: v
                                for k, v in c.items()
                                if k not in ("job_id", "group_id", "top_terms")
                            },
                            **term,
                        }
                    )
    atomic_write_text(out / "margin_terms.csv", pd.DataFrame(terms).to_csv(index=False))
    summary, pooled, reliability, curves = [], [], [], []
    expected = (
        predictions[predictions.variant == "source"]
        .groupby(["mode", "pipeline", "arm"])
        .job_id.agg(set)
        .to_dict()
    )
    for key, f in predictions.groupby(["mode", "pipeline", "arm", "variant", "budget", "repeat"]):
        mode, pipe, arm, variant, budget, repeat = key
        complete = (
            len(f) == len(expected[(mode, pipe, arm)])
            and set(f.job_id) == expected[(mode, pipe, arm)]
        )
        record = dict(
            zip(["mode", "pipeline", "arm", "variant", "budget", "repeat"], key, strict=True)
        )
        # Require every planned cell in this mode/pipeline, not just successful source folds.
        planned = [
            r for r in results if r["cell"]["mode"] == mode and r["cell"]["pipeline"] == pipe
        ]
        complete = complete and set(f.fold) == {r["cell"]["fold"] for r in planned}
        for dataset, sub in [("ALL", f), *[(d, g) for d, g in f.groupby("dataset_id")]]:
            entry = {
                **record,
                "dataset": str(dataset),
                "status": "complete" if complete else "partial_not_estimable",
            }
            if complete:
                m = dd.summarize(sub, cfg.policy.requested_coverages)
                entry.update({k: v for k, v in m.items() if k != "risk_coverage"})
                weights = inverse_group_size_weights(sub.group_id.astype(str))
                entry["raw_auroc"] = auroc(
                    sub.raw_score.to_numpy(float), sub.operational_failure.to_numpy(float), weights
                )
                curves.extend(
                    {**record, "dataset": str(dataset), **point} for point in m["risk_coverage"]
                )
                # Fixed equal-width bins; not selected to make calibration look good.
                probs = sub.probability.to_numpy(float)
                labels = sub.operational_failure.to_numpy(float)
                bins = np.minimum((probs * 10).astype(int), 9)
                for b in range(10):
                    selected = bins == b
                    if selected.any():
                        reliability.append(
                            {
                                **record,
                                "dataset": str(dataset),
                                "bin_low": b / 10,
                                "bin_high": (b + 1) / 10,
                                "cases": int(selected.sum()),
                                "groups": sub.loc[selected, "group_id"].nunique(),
                                "weight_mass": float(weights[selected].sum()),
                                "mean_probability": float(
                                    np.average(probs[selected], weights=weights[selected])
                                ),
                                "failure_rate": float(
                                    np.average(labels[selected], weights=weights[selected])
                                ),
                            }
                        )
                ref = predictions[
                    (predictions["mode"] == mode)
                    & (predictions.pipeline_id == pipe)
                    & (predictions.arm == arm)
                    & (predictions.variant == "source")
                ]
                if dataset != "ALL":
                    ref = ref[ref.dataset_id == dataset]
                if variant != "source":
                    contrast = dd.paired_contrast(
                        ref, sub, cfg.splits.seed, design["conditional_bootstrap_resamples"]
                    )
                    pooled.append({**entry, **contrast})
            summary.append(entry)
    # Include wholly absent variants, including single-class budget repeats.
    present = {
        (r["mode"], r["pipeline"], r["arm"], r["variant"], r["budget"], r["repeat"], r["dataset"])
        for r in summary
    }
    for mode in design["modes"]:
        for pipe in design["primary_pipelines"]:
            datasets = [
                "ALL",
                *sorted(set(predictions.dataset_id)),
            ]
            for arm in design["arms"]:
                for variant, budget, repeat in planned_variants(design):
                    for dataset in datasets:
                        key = (mode, pipe, arm, variant, budget, repeat, dataset)
                        if key not in present:
                            summary.append(
                                dict(
                                    zip(
                                        [
                                            "mode",
                                            "pipeline",
                                            "arm",
                                            "variant",
                                            "budget",
                                            "repeat",
                                            "dataset",
                                        ],
                                        key,
                                        strict=True,
                                    ),
                                    status="not_estimable",
                                )
                            )
    table = pd.DataFrame(summary)
    atomic_write_text(out / "performance.csv", table.to_csv(index=False))
    dd.save(out / "paired_brier.json", pooled)
    atomic_write_text(out / "reliability_bins.csv", pd.DataFrame(reliability).to_csv(index=False))
    atomic_write_text(out / "risk_coverage.csv", pd.DataFrame(curves).to_csv(index=False))
    repeated = table[(table.budget != "all") & (table.budget != "source")]
    variability = []
    for key, f in repeated.groupby(["mode", "pipeline", "arm", "variant", "budget", "dataset"]):
        good = f[f.status == "complete"]
        item = dict(
            zip(["mode", "pipeline", "arm", "variant", "budget", "dataset"], key, strict=True)
        )
        item.update(
            planned_repeats=design["budget_repeats"],
            complete_repeats=len(good),
            interpretation="Repeated calibration subsets share test groups. Ranges describe subset sensitivity, not independent replicates or population confidence intervals.",
        )
        for metric in ["brier", "coverage", "accepted_risk"]:
            v = good[metric].dropna() if metric in good else pd.Series(dtype=float)
            item.update(
                {
                    metric + "_median": float(v.median()) if len(v) else None,
                    metric + "_min": float(v.min()) if len(v) else None,
                    metric + "_max": float(v.max()) if len(v) else None,
                }
            )
        variability.append(item)
    atomic_write_text(out / "budget_variability.csv", pd.DataFrame(variability).to_csv(index=False))
    main = table[
        (table.dataset == "ALL")
        & (table.arm == "non_stability")
        & (table.budget.isin(["source", "all"]))
    ]
    atomic_write_text(
        out / "report.html",
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Paper evidence</title><h1>Registration confidence audit</h1><p>'
        + stage
        + "; conditional estimates, no automatic conference-readiness claim.</p>"
        + main.to_html(index=False, escape=True)
        + "</html>",
    )
    lines = [
        "# Paper evidence package",
        "",
        f"Stage: {stage}. The original decision and publication audit are retained unchanged.",
        "",
        "Read performance.csv, paired_brier.json, budget_variability.csv, margin_terms.csv and fit_diagnostics.json.",
        "",
        "The monotone clipped calibration map is an exploratory sensitivity chosen after observing extreme margins. Its bounds come from source-training margin quantiles. It constrains slope >=0 and can create ties. It does not remove cases, flip scores using test labels, or establish a novel method.",
        "",
        "Primary question: do ranking, probabilities and selective thresholds fail differently under shift? Do not claim recalibration caused an acceptance benefit also achieved by threshold-only adaptation.",
        "",
        "Repeated subsets are dependent measurements. Fixed-prediction bootstrap intervals omit training and calibration-fit uncertainty. No power or familywise confirmatory claim is made.",
        "",
        "Independent natural validation remains pending unless a separately reviewed and locked external cohort has actually completed. A development completion marker is not a full conference-ready study.",
    ]
    atomic_write_text(out / "EVIDENCE.md", "\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["development", "lock-external", "external"], default="development"
    )
    parser.add_argument("--config", default="configs/full_study.yaml")
    parser.add_argument("--design", default="configs/paper_study.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default="conference_outputs/paper_study")
    parser.add_argument("--handoff", default="reports/paper_study_latest")
    parser.add_argument("--cohort-review")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run focused workstation regression checks before the study",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16:
        parser.error("workers must be 1..16")
    os.chdir(ROOT)
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[name] = "1"
    if args.check:
        import subprocess

        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_paper_study.py",
                "tests/test_publication_audit.py",
                "tests/test_development_decision.py",
            ],
            check=True,
        )
    cfg = load_config(args.config)
    design = json.loads(Path(args.design).read_text())
    validate_design(design, cfg)
    spec = json.loads((ROOT / "configs/conference_extension.json").read_text())
    cases, names = assemble(cfg, ROOT, cfg.full_study.development_datasets)
    dd.validate_cases(cases, cfg)
    all_arms = select_arms(names, spec)
    arms = {k: all_arms[k] for k in design["arms"]}
    keep = [
        "job_id",
        "group_id",
        "pair_id",
        "dataset_id",
        "pipeline_id",
        "operational_failure",
        "explicit_failure",
        "bounded_loss",
        *(c for cols in arms.values() for c in cols),
    ]
    cases = cases[list(dict.fromkeys(keep))].copy()
    out = Path(args.output)
    handoff = Path(args.handoff)
    if out.resolve() == handoff.resolve():
        raise ValueError("Keep handoff separate from detailed outputs")
    out.mkdir(parents=True, exist_ok=True)
    ident = {
        **identity(cfg, ROOT, spec),
        "paper_code": file_digest(Path(__file__)),
        "audit_code": file_digest(ROOT / "scripts/publication_audit.py"),
        "decision_code": file_digest(ROOT / "scripts/development_decision.py"),
        "design": design,
        "development": case_identity(cases, arms),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn"]},
        },
    }
    digest = short_hash(ident, length=64)
    manifest = out / "identity.json"
    if manifest.exists() and json.loads(manifest.read_text())["run_id"] != digest:
        raise ValueError("Study identity changed; use fresh --output")
    dd.save(manifest, {"run_id": digest, "identity": ident})
    tasks = []
    paths = []
    started = time.perf_counter()
    if args.stage == "development":
        for mode in design["modes"]:
            for pipe in design["primary_pipelines"]:
                for fold in range(5):
                    train, cal, test = dd.split_cell(cases, cfg, design, mode, pipe, fold)
                    _, pool, _ = dd.split_cell(cases, cfg, design, "within_pipeline", pipe, fold)
                    cell = {"mode": mode, "pipeline": pipe, "fold": fold}
                    path = out / "development" / f"{mode}-{pipe}-{fold}.json"
                    paths.append(path)
                    tasks.append((train, cal, pool, test, arms, cfg, design, cell, path, digest))
    else:
        if not args.cohort_review:
            raise ValueError("External stage requires --cohort-review with actual evidence files")
        review, evidence, external, indexed, extnames, extdigest, calgroups, testgroups = (
            review_cohort(args.cohort_review, cfg, design)
        )
        if any(c not in extnames for cols in arms.values() for c in cols):
            raise ValueError("External feature contract differs")
        if set(external.group_id.astype(str)) & set(cases.group_id.astype(str)):
            raise ValueError("External/development groups overlap")
        lockpath = out / "external_lock.json"
        lock = {
            "study_id": digest,
            "review": review,
            "evidence_hashes": evidence,
            "external_identity": extdigest,
            "calibration_groups": calgroups,
            "test_groups": testgroups,
            "scope": "prospective descriptive natural-cohort evaluation; no population power claim",
        }
        lockhash = short_hash(lock, length=64)
        if args.stage == "lock-external":
            if lockpath.exists():
                raise ValueError("External lock already exists; preserve it")
            existing = ShardedTable(cfg.paths.resolve(ROOT)["cache_root"], "labels").load()
            if not existing.empty and set(existing.job_id) & set(external.job_id):
                raise ValueError(
                    "External outcomes already cached; cannot create outcome-blind lock"
                )
            # Exclusive write prevents replacement of an earlier lock.
            with lockpath.open("x", encoding="utf-8") as handle:
                json.dump({"hash": lockhash, "lock": lock}, handle, indent=2)
            print(
                f"Locked {review['dataset']}; calibration groups={len(calgroups)}, test groups={len(testgroups)}. Run --stage external with the same arguments."
            )
            return 0
        if not lockpath.exists() or json.loads(lockpath.read_text()) != {
            "hash": lockhash,
            "lock": lock,
        }:
            raise ValueError("External lock absent or changed")
        # Landmark outcomes are computed only after verifying the separate paper-study lock.
        labelpath = out / "external_labels.json"
        if labelpath.exists():
            stored = json.loads(labelpath.read_text())
            if stored["lock_hash"] != lockhash or stored["hash"] != short_hash(
                stored["rows"], length=64
            ):
                raise ValueError("External label checkpoint mismatch")
            labels = pd.DataFrame(stored["rows"])
        else:
            by_pair = indexed.set_index("pair_id", drop=False)
            rows = [
                _compute_label_task((row, by_pair.loc[row.pair_id], cfg, ROOT))
                for _, row in external.iterrows()
            ]
            clean = dd.clean_json(rows)
            dd.save(
                labelpath,
                {"lock_hash": lockhash, "hash": short_hash(clean, length=64), "rows": clean},
            )
            labels = pd.DataFrame(rows)
        external = external.merge(
            labels[["job_id", "operational_failure", "eligible_for_acceptance", "bounded_loss"]],
            on="job_id",
            validate="one_to_one",
        )
        external["explicit_failure"] = ~external.eligible_for_acceptance.astype(bool)
        if (
            not np.isfinite(external.bounded_loss).all()
            or not np.isin(external.operational_failure, [0, 1]).all()
        ):
            raise ValueError(
                "Undefined external endpoints; investigate without excluding outcome-selected cases"
            )
        external = external[list(dict.fromkeys(keep))]
        for mode in design["modes"]:
            for pipe in design["primary_pipelines"]:
                source = (
                    cases[cases.pipeline_id == pipe]
                    if mode == "within_pipeline"
                    else cases[cases.pipeline_id != pipe]
                )
                train, cal = partition(source, cfg.splits.seed)
                target = external[external.pipeline_id == pipe]
                pool = target[target.group_id.astype(str).isin(calgroups)]
                test = target[target.group_id.astype(str).isin(testgroups)]
                cell = {"mode": mode, "pipeline": pipe, "fold": 0}
                path = out / "external" / f"{mode}-{pipe}.json"
                paths.append(path)
                tasks.append((train, cal, pool, test, arms, cfg, design, cell, path, lockhash))
    results = execute(tasks, paths, args.workers, tasks[0][-1])
    resultout = out / (args.stage + "_report")
    resultout.mkdir(parents=True, exist_ok=True)
    print(
        "All cells collected; aggregating fixed-prediction bootstrap and manuscript tables...",
        flush=True,
    )
    render(results, resultout, design, args.stage, cfg)
    dd.write_costs(resultout, cases, arms, cfg)
    audit.provenance(resultout)
    scope = {
        "stage": args.stage,
        "acquisition_cost_scope": "cached development registration and feature acquisition only; excludes external and repeated analysis fitting",
        "planned_cells": len(results),
        "successful_source_fits": sum(
            s.get("variant") == "source_fit" and s.get("status") == "ok"
            for r in results
            for s in r["status"]
        ),
        "planned_source_fits": len(results) * len(arms),
    }
    if args.stage == "external":
        scope.update(
            dataset=review["dataset"],
            calibration_groups=len(calgroups),
            test_groups=len(testgroups),
            evidence_hashes=evidence,
            lock_hash=lockhash,
            interpretation=lock["scope"],
        )
    dd.save(resultout / "study_scope.json", scope)
    prior = {}
    for folder in ["development_decision_latest", "publication_audit_latest"]:
        directory = ROOT / "reports" / folder
        prior[folder] = {p.name: file_digest(p) for p in sorted(directory.glob("*")) if p.is_file()}
    dd.save(
        resultout / "prior_evidence.json",
        {
            "reports": prior,
            "interpretation": "Hashes identify retained historical artifacts; no new independent evidence is implied.",
        },
    )
    handoff = handoff / args.stage
    handoff.mkdir(parents=True, exist_ok=True)
    (handoff / "completion.json").unlink(missing_ok=True)
    hashes = {}
    for name in [
        "prior_evidence.json",
        "reliability_bins.csv",
        "risk_coverage.csv",
        "study_scope.json",
        "EVIDENCE.md",
        "performance.csv",
        "paired_brier.json",
        "budget_variability.csv",
        "cell_metrics.csv",
        "cell_contrasts.json",
        "fit_diagnostics.json",
        "margin_terms.csv",
        "acquisition_cost.json",
        "provenance.json",
        "report.html",
    ]:
        atomic_write_bytes(handoff / name, (resultout / name).read_bytes())
        hashes[name] = file_digest(handoff / name)
    atomic_write_bytes(handoff / "identity.json", manifest.read_bytes())
    hashes["identity.json"] = file_digest(handoff / "identity.json")
    dd.save(
        handoff / "completion.json",
        {
            "status": "completed_" + args.stage,
            "run_id": digest,
            "external_lock_hash": tasks[0][-1] if args.stage == "external" else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "artifact_sha256": hashes,
            "publication_readiness": "requires scientific review, independent validation, novelty comparison and manuscript",
        },
    )
    print(
        f"DONE: {handoff}. Commit this aggregate folder; detailed cases remain in {out}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"Paper study stopped: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc
