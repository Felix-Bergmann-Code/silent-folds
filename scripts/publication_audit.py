"""One-command exploratory publication audit over corrected development caches.

No external labels are evaluated. Recalibration uses target calibration groups
in the existing calibration fold; outer test groups never participate.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts import development_decision as dd
from scripts.conference_extension import (
    assemble,
    case_identity,
    fit_arm,
    identity,
    select_arms,
    support,
)
from warpaudit.cache.hashing import file_digest, short_hash
from warpaudit.cache.store import ShardedTable, atomic_write_bytes, atomic_write_text
from warpaudit.cli import _pairs_manifest
from warpaudit.config import load_config
from warpaudit.evaluation.full_study import _population
from warpaudit.evaluation.metrics import auroc
from warpaudit.evaluation.weights import inverse_group_size_weights
from warpaudit.predictors.policy import FrozenPolicy

DESIGN = {
    "revision": "1-exploratory-publication-audit",
    "outer_folds": 5,
    "primary_pipelines": ["xfeat_h", "sp_lg_h"],
    "arms": [
        "non_stability",
        "non_stability_resampling",
        "non_stability_seed",
        "non_stability_e1",
        "non_stability_e2",
        "non_stability_e1_e2",
    ],
    "modes": ["within_pipeline", "held_out_pipeline"],
    "target_group_budgets": [5, 10, "all"],
    "bootstrap_resamples": 2000,
    "stable_quantile": 0.25,
    "stable_feature": "E1:seed_only_spread_mean",
    "interpretation": "Development-informed exploratory analysis; fixed source ranking model; no test-directed score flips or retries",
}


def target_subset(cases, cfg, design, pipeline, fold, budget):
    _, target, _ = dd.split_cell(cases, cfg, design, "within_pipeline", pipeline, fold)
    # Stable, outcome-independent nested subsets; no class-balancing retries.
    groups = sorted(
        set(target.group_id.astype(str)),
        key=lambda g: short_hash([cfg.splits.seed, fold, g], length=64),
    )
    if budget != "all" and len(groups) < budget:
        return target.iloc[:0], "insufficient target calibration groups"
    selected = groups if budget == "all" else groups[:budget]
    return target[target.group_id.astype(str).isin(selected)].copy(), ""


def policy_for(model, frame, columns, cfg):
    return FrozenPolicy.from_calibration(
        "publication-audit",
        _population(frame, model.predict_probability(frame.loc[:, columns])),
        nominal_acceptance=cfg.policy.nominal_acceptance,
        detector_hash=model.fingerprint,
        preprocessing_hash=model.preprocessor.fingerprint,
    )


def model_diagnostics(model, frame, columns):
    raw = model.decision_function(frame.loc[:, columns])
    p = model.predict_probability(frame.loc[:, columns])
    if not np.isfinite(raw).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite raw margins or probabilities; inspect numerical fitting")
    weights = inverse_group_size_weights(frame.group_id.astype(str))
    slope = float(model.calibrator.coef_[0, 0]) if hasattr(model.calibrator, "coef_") else None
    return {
        **support(frame),
        "raw_auroc": auroc(raw, frame.operational_failure, weights),
        "calibrated_auroc": auroc(p, frame.operational_failure, weights),
        "platt_slope": slope,
        "platt_intercept": float(model.calibrator.intercept_[0]) if slope is not None else None,
        "ranking_reversed_by_calibration": slope is not None and slope < 0,
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "probability_min": float(p.min()),
        "probability_max": float(p.max()),
        "probability_std": float(p.std()),
        "selected_C": model.selected_C,
        "model_iterations": model.model.n_iter_.tolist(),
        "calibrator_iterations": model.calibrator.n_iter_.tolist()
        if hasattr(model.calibrator, "n_iter_")
        else None,
        "model_hash": model.fingerprint,
        "preprocessor_hash": model.preprocessor.fingerprint,
        "missing_feature_fraction": float(
            (~np.isfinite(frame.loc[:, columns].to_numpy(float))).mean()
        ),
        "all_missing_training_features": model.preprocessor.all_missing_features,
    }


def fold_audit(cases, arms, cfg, design, mode, pipeline, fold):
    train, source_cal, test = dd.split_cell(cases, cfg, design, mode, pipeline, fold)
    reasons, inner = dd.eligibility(train, source_cal, cfg.splits.seed)
    result = {
        "mode": mode,
        "pipeline": pipeline,
        "fold": fold,
        "fits": [],
        "predictions": [],
        "partition_groups": {
            name: sorted(set(f.group_id.astype(str)))
            for name, f in [("train", train), ("source_calibration", source_cal), ("test", test)]
        },
        "inner_support": inner,
    }
    for arm, columns in arms.items():
        if reasons or test.empty:
            result["fits"].append(
                {
                    "arm": arm,
                    "variant": "source",
                    "status": "not_estimable",
                    "reasons": reasons or ["empty test"],
                }
            )
            continue
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always")
            model, source_policy, seconds = fit_arm(train, source_cal, columns, cfg, arm)
        variants = [
            (
                "source",
                "source",
                model,
                source_policy,
                source_cal,
                [str(w.message) for w in messages],
                seconds,
            )
        ]
        for budget in design["target_group_budgets"]:
            target, reason = target_subset(cases, cfg, design, pipeline, fold, budget)
            if reason:
                for variant in ("threshold_only", "recalibrated"):
                    result["fits"].append(
                        {
                            "arm": arm,
                            "variant": variant,
                            "budget": budget,
                            "status": "not_estimable",
                            "reasons": [reason],
                        }
                    )
                continue
            variants.append(
                (
                    "threshold_only",
                    budget,
                    model,
                    policy_for(model, target, columns, cfg),
                    target,
                    [],
                    None,
                )
            )
            if target.operational_failure.nunique() != 2:
                result["fits"].append(
                    {
                        "arm": arm,
                        "variant": "recalibrated",
                        "budget": budget,
                        "status": "not_estimable",
                        "reasons": ["target calibration lacks both classes"],
                        "support": support(target),
                    }
                )
                continue
            recal = copy.deepcopy(model)
            with warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter("always")
                recal.calibrate(
                    target.loc[:, columns],
                    target.operational_failure.to_numpy(float),
                    method=cfg.policy.calibration_method,
                    sample_weight=inverse_group_size_weights(target.group_id.astype(str)),
                )
            variants.append(
                (
                    "recalibrated",
                    budget,
                    recal,
                    policy_for(recal, target, columns, cfg),
                    target,
                    [str(w.message) for w in messages],
                    None,
                )
            )
        for variant, budget, fitted, policy, calibration, messages, seconds in variants:
            cutoff_values = (
                calibration[design["stable_feature"]].replace([np.inf, -np.inf], np.nan).dropna()
            )
            # The mechanism definition stays fixed to source calibration across remedies.
            source_values = (
                source_cal[design["stable_feature"]].replace([np.inf, -np.inf], np.nan).dropna()
            )
            cutoff = (
                float(source_values.quantile(design["stable_quantile"]))
                if len(source_values)
                else None
            )
            info = {
                "arm": arm,
                "variant": variant,
                "budget": budget,
                "status": "ok",
                "warnings": messages,
                "source_fit_seconds": seconds,
                "calibration_groups": sorted(set(calibration.group_id.astype(str))),
                "threshold": policy.threshold,
                "threshold_negative_infinity": policy.threshold == -np.inf,
                "calibration_coverage": policy.selection.realised_calibration_coverage,
                "stable_cutoff": cutoff,
                "target_stability_measured": len(cutoff_values),
                "diagnostics": {
                    name: model_diagnostics(fitted, frame, columns)
                    for name, frame in [
                        ("train", train),
                        ("calibration", calibration),
                        ("test", test),
                    ]
                },
            }
            result["fits"].append(info)
            raw = fitted.decision_function(test.loc[:, columns])
            prob = fitted.predict_probability(test.loc[:, columns])
            for (_, row), margin, p in zip(test.iterrows(), raw, prob, strict=True):
                measured = cutoff is not None and np.isfinite(row[design["stable_feature"]])
                rec = {
                    k: row[k]
                    for k in (
                        "job_id",
                        "group_id",
                        "pair_id",
                        "dataset_id",
                        "pipeline_id",
                        "operational_failure",
                        "explicit_failure",
                        "bounded_loss",
                    )
                }
                rec.update(
                    mode=mode,
                    fold=fold,
                    arm=arm,
                    variant=variant,
                    budget=str(budget),
                    raw_score=margin,
                    probability=p,
                    accepted=bool(not row.explicit_failure and p <= policy.threshold),
                    stable_measured=bool(measured),
                    stable=bool(measured and row[design["stable_feature"]] <= cutoff),
                )
                for k in (
                    "tre_px",
                    "tre_norm",
                    design["stable_feature"],
                    "E1:resample_fixed_seed_spread_mean",
                    "E2:perturbation_spread_mean",
                ):
                    rec[k] = row.get(k)
                result["predictions"].append(rec)
    return dd.clean_json(result)


def worker(task):
    cases, arms, cfg, design, mode, pipeline, fold, path, run_id = task
    with threadpool_limits(limits=1):
        result = fold_audit(cases, arms, cfg, design, mode, pipeline, fold)
    dd.save(path, {"run_id": run_id, "hash": short_hash(result, length=64), "result": result})
    return str(path)


def provenance(out):
    previous = ROOT / "reports/development_decision_latest/run_identity.json"
    old = json.loads(previous.read_text())["identity"] if previous.exists() else {}
    rows = []
    for filename, key in [
        ("scripts/conference_diagnostics.py", "diagnostics_code_hash"),
        ("scripts/conference_extension.py", "extension_code_hash"),
        ("scripts/development_decision.py", "decision_code"),
    ]:
        data = (ROOT / filename).read_bytes()
        git = subprocess.run(
            ["git", "show", f"HEAD:{filename}"], cwd=ROOT, capture_output=True, check=True
        ).stdout
        normalized = data.replace(b"\r\n", b"\n")
        variants = {"exact": data, "LF": normalized, "CRLF": normalized.replace(b"\n", b"\r\n")}
        match = [k for k, v in variants.items() if hashlib.sha256(v).hexdigest() == old.get(key)]
        rows.append(
            {
                "file": filename,
                "old_recorded_hash": old.get(key),
                "current_hash": hashlib.sha256(data).hexdigest(),
                "historical_match": match,
                "matches_git_normalized": normalized == git.replace(b"\r\n", b"\n"),
                "normalized_diff": "".join(
                    difflib.unified_diff(
                        git.decode().splitlines(True),
                        normalized.decode().splitlines(True),
                        fromfile="git",
                        tofile="workstation",
                    )
                ),
            }
        )
        atomic_write_bytes(out / "source_snapshot" / Path(filename).name, data)
    dd.save(
        out / "provenance.json",
        {
            "files": rows,
            "limitation": "An unmatched historical hash cannot be reconstructed from current source; preserve historical source if available.",
        },
    )


def report(results, cases, cfg, design, out):
    records = pd.DataFrame([p for r in results for p in r["predictions"]])
    if records.empty:
        raise ValueError("No estimable cells; inspect checkpoints")
    atomic_write_text(out / "predictions.csv", records.to_csv(index=False))
    perf, contrasts, stable, diagnostic = [], [], [], []
    for r in results:
        for fit in r["fits"]:
            for partition, values in fit.get("diagnostics", {"none": {}}).items():
                diagnostic.append(
                    {
                        "mode": r["mode"],
                        "pipeline": r["pipeline"],
                        "fold": r["fold"],
                        "arm": fit["arm"],
                        "variant": fit["variant"],
                        "budget": fit.get("budget", "source"),
                        "status": fit["status"],
                        "threshold": fit.get("threshold"),
                        "threshold_negative_infinity": fit.get("threshold_negative_infinity"),
                        "calibration_coverage": fit.get("calibration_coverage"),
                        "partition": partition,
                        "warnings": " | ".join(fit.get("warnings", [])),
                        "reasons": " | ".join(fit.get("reasons", [])),
                        **values,
                    }
                )
    for mode in design["modes"]:
        for pipe in design["primary_pipelines"]:
            base = cases[cases.pipeline_id == pipe]
            for dataset in ["ALL", *sorted(set(base.dataset_id))]:
                expected = base if dataset == "ALL" else base[base.dataset_id == dataset]
                f = records[(records["mode"] == mode) & (records.pipeline_id == pipe)]
                if dataset != "ALL":
                    f = f[f.dataset_id == dataset]
                for arm in design["arms"]:
                    a = f[(f.arm == arm) & (f.variant == "source")]
                    for variant, budget in [
                        ("source", "source"),
                        *[
                            (v, str(b))
                            for b in design["target_group_budgets"]
                            for v in ["threshold_only", "recalibrated"]
                        ],
                    ]:
                        b = f[(f.arm == arm) & (f.variant == variant) & (f.budget == budget)]
                        key = dict(
                            mode=mode,
                            pipeline=pipe,
                            dataset=dataset,
                            arm=arm,
                            variant=variant,
                            budget=budget,
                        )
                        complete = len(b) == len(expected) and set(b.job_id) == set(expected.job_id)
                        perf.append(
                            {
                                **key,
                                "status": "complete" if complete else "not_estimable",
                                "expected_cases": len(expected),
                                "predicted_cases": len(b),
                                **(
                                    {
                                        k: v
                                        for k, v in dd.summarize(
                                            b, cfg.policy.requested_coverages
                                        ).items()
                                        if k != "risk_coverage"
                                    }
                                    if complete
                                    else {}
                                ),
                            }
                        )
                        if complete and len(a) == len(expected) and variant != "source":
                            sa, sb = (
                                dd.summarize(a, cfg.policy.requested_coverages),
                                dd.summarize(b, cfg.policy.requested_coverages),
                            )
                            contrasts.append(
                                {
                                    **key,
                                    **dd.paired_contrast(
                                        a, b, cfg.splits.seed, design["bootstrap_resamples"]
                                    ),
                                    "source_coverage": sa["coverage"],
                                    "candidate_coverage": sb["coverage"],
                                    "source_risk": sa["accepted_risk"],
                                    "candidate_risk": sb["accepted_risk"],
                                    "matched_coverage": dd.safe_coverage_contrasts(
                                        sa["risk_coverage"], sb["risk_coverage"]
                                    ),
                                }
                            )
                        if len(b):
                            wrong = b.stable & ~b.explicit_failure & (b.operational_failure == 1)
                            stable.append(
                                {
                                    **key,
                                    "status": "complete" if complete else "partial",
                                    "cases": len(b),
                                    "groups": b.group_id.nunique(),
                                    "measured_cases": int(b.stable_measured.sum()),
                                    "stable_cases": int(b.stable.sum()),
                                    "stable_wrong_cases": int(wrong.sum()),
                                    "stable_wrong_groups": b.loc[wrong, "group_id"].nunique(),
                                    "stable_wrong_accepted": int((wrong & b.accepted).sum()),
                                }
                            )
    fold_rows = []
    for keys, frame in records.groupby(["mode", "pipeline_id", "fold", "arm", "variant", "budget"]):
        fold_rows.append(
            {
                **dict(
                    zip(["mode", "pipeline", "fold", "arm", "variant", "budget"], keys, strict=True)
                ),
                **{
                    k: v
                    for k, v in dd.summarize(frame, cfg.policy.requested_coverages).items()
                    if k != "risk_coverage"
                },
            }
        )
    atomic_write_text(out / "fold_performance.csv", pd.DataFrame(fold_rows).to_csv(index=False))
    for name, rows in [
        ("performance", perf),
        ("diagnostics", diagnostic),
        ("stable_wrong_summary", stable),
    ]:
        atomic_write_text(out / f"{name}.csv", pd.DataFrame(rows).to_csv(index=False))
    dd.save(out / "contrasts.json", contrasts)
    dd.save(
        out / "fit_details.json",
        [{k: v for k, v in r.items() if k != "predictions"} for r in results],
    )
    primary = pd.DataFrame(perf)
    primary = primary[(primary.dataset == "ALL") & (primary.arm == "non_stability")]
    lines = [
        "# Publication audit",
        "",
        "Exploratory development analysis. Lower Brier is better. Risk must be interpreted jointly with actual coverage. Conditional bootstrap holds predictions fixed and omits training/calibration uncertainty.",
        "",
        "| Mode | Pipeline | Variant | Target groups | Status | Brier | Coverage | Risk |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for _, r in primary.iterrows():
        lines.append(
            f'| {r["mode"]} | {r.pipeline} | {r.variant} | {r.budget} | {r.status} | {r.get("brier", np.nan):.4f} | {r.get("coverage", np.nan):.4f} | {r.get("accepted_risk", np.nan):.4f} |'
        )
    reversals = [
        r
        for r in diagnostic
        if r.get("ranking_reversed_by_calibration") and r["partition"] == "test"
    ]
    lines += [
        "",
        f"Detected {len(reversals)} arm/variant/fold test records with a negative Platt slope. See diagnostics.csv for raw versus calibrated AUROC; repeated variants are not independent failures.",
        "",
        "Threshold-only adapts the operating point using target scores without target outcome fitting. Recalibrated replaces the probability map using target labels and then chooses a target threshold. Both share the identical frozen source ranking model. Compare both to avoid attributing threshold adaptation to calibration.",
        "",
        "Read dataset strata, all component arms, stable_wrong_summary.csv, contrasts.json, provenance.json and acquisition_cost.json before choosing a claim. No automatic publication success is inferred.",
        "",
        "Independent natural validation is NOT VERIFIED. See dataset_readiness.json. No new external labels were evaluated.",
    ]
    atomic_write_text(out / "AUDIT.md", "\n".join(lines) + "\n")
    atomic_write_text(
        out / "report.html",
        '<!doctype html><meta charset="utf-8"><title>Publication audit</title><h1>Exploratory recalibration audit</h1><p>Actual coverage and risk must be interpreted together. No independent external validation.</p>'
        + primary.to_html(index=False, escape=True),
    )
    atomic_write_text(
        out / "MANUSCRIPT_OUTLINE.md",
        "# Working manuscript outline\n\nWorking question: When do registration failure probabilities and selective policies transport between pipelines?\n\n1. Introduction: separate ranking, probability calibration, and selective acceptance.\n2. Related work: uncertainty/error association, registration QA, calibration under shift. Verify exact novelty against cited papers.\n3. Methods: fixed grouped development folds, frozen source ranking models, nested target calibration budgets, threshold-only control, explicit failed fits.\n4. Exploratory results: cite performance.csv and diagnostics.csv; retain the original decision run and its fold anomaly.\n5. Independent validation: pending accessible natural landmarks, independent grouping, appropriate transforms, and a frozen protocol.\n6. Discussion: measured computational costs, conditional inference limits, dataset/pipeline confounding.\n\nDo not write a confirmed generalization claim or final abstract until independent validation is complete.\n",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/full_study.yaml")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default="conference_outputs/publication_audit")
    parser.add_argument("--handoff", default="reports/publication_audit_latest")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16:
        parser.error("workers must be 1..16")
    os.chdir(ROOT)
    for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[k] = "1"
    started = time.perf_counter()
    cfg = load_config(args.config)
    spec = json.loads((ROOT / "configs/conference_extension.json").read_text())
    design = copy.deepcopy(DESIGN)
    if not set(design["primary_pipelines"]) <= set(cfg.common_block):
        raise ValueError("Primary pipelines absent from study configuration")
    out, handoff = Path(args.output), Path(args.handoff)
    if out.resolve() == handoff.resolve():
        raise ValueError("Output and aggregate handoff must differ")
    out.mkdir(parents=True, exist_ok=True)
    print(
        "Loading corrected development caches; no new registration or external evaluation",
        flush=True,
    )
    cases, names = assemble(cfg, ROOT, cfg.full_study.development_datasets)
    dd.validate_cases(cases, cfg)
    if design["stable_feature"] not in cases:
        raise ValueError("Required stability diagnostic measurement absent")
    all_arms = select_arms(names, spec)
    arms = {k: all_arms[k] for k in design["arms"]}
    metadata = [
        "job_id",
        "group_id",
        "pair_id",
        "dataset_id",
        "pipeline_id",
        "operational_failure",
        "explicit_failure",
        "bounded_loss",
    ]
    cases = cases[list(dict.fromkeys([*metadata, *names]))].copy()
    labels = ShardedTable(cfg.paths.resolve(ROOT)["cache_root"], "labels").load()
    labels = labels[labels.job_id.isin(cases.job_id)].drop_duplicates("job_id", keep="last")
    extra = [k for k in ["tre_px", "tre_norm"] if k in labels]
    cases = cases.merge(labels[["job_id", *extra]], on="job_id", validate="one_to_one")
    ident = {
        **identity(cfg, ROOT, spec),
        "design": design,
        "audit_code": file_digest(Path(__file__)),
        "decision_code": file_digest(ROOT / "scripts/development_decision.py"),
        "evidence": case_identity(cases, arms),
        "diagnostic_labels": short_hash(
            dd.clean_json(cases[["job_id", *extra]].to_dict("records")), length=64
        ),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn"]},
    }
    run_id = short_hash(ident, length=64)
    manifest = out / "identity.json"
    if manifest.exists() and json.loads(manifest.read_text())["run_id"] != run_id:
        raise ValueError("Output identity changed; use a fresh --output directory")
    dd.save(manifest, {"run_id": run_id, "identity": ident})
    provenance(out)
    tasks, paths = [], []
    for mode in design["modes"]:
        for pipe in design["primary_pipelines"]:
            for fold in range(5):
                path = out / "folds" / f"{mode}-{pipe}-{fold}.json"
                paths.append(path)
                if path.exists():
                    saved = json.loads(path.read_text())
                    if saved["run_id"] != run_id or saved["hash"] != short_hash(
                        saved["result"], length=64
                    ):
                        raise ValueError(f"Invalid checkpoint: {path}")
                else:
                    tasks.append((cases, arms, cfg, design, mode, pipe, fold, path, run_id))
    print(f"{len(paths)-len(tasks)}/{len(paths)} checkpoints reusable", flush=True)
    if args.workers == 1:
        for task in tasks:
            print(worker(task), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            for future in as_completed([pool.submit(worker, t) for t in tasks]):
                print("Completed " + future.result(), flush=True)
    results = [json.loads(p.read_text())["result"] for p in paths]
    report(results, cases, cfg, design, out)
    dd.write_costs(out, cases, arms, cfg)
    pairs, _ = _pairs_manifest(cfg, ROOT)
    dd.save(
        out / "dataset_readiness.json",
        {
            "status": "independent natural validation not verified",
            "external_outcomes_evaluated": False,
            "label_access_scope": "Only declared development jobs are joined and analyzed; the shared loader reads the label table before filtering",
            "inventory": [
                {"dataset": str(d), "pairs": len(f), "manifest_groups": f.group_id.nunique()}
                for d, f in pairs.groupby("dataset_id")
            ],
            "required_before_freeze": [
                "verified natural landmark files",
                "independent subject/specimen mapping",
                "documented use permission",
                "domain-appropriate registration",
                "group-level precision plan",
            ],
            "decision": "Existing MultiReg synthetic-transform cases cannot substitute for independent natural anatomical validation. No new dataset access was obtained by this script.",
        },
    )
    handoff.mkdir(parents=True, exist_ok=True)
    (handoff / "completion.json").unlink(missing_ok=True)
    hashes = {}
    for name in [
        "AUDIT.md",
        "performance.csv",
        "diagnostics.csv",
        "fold_performance.csv",
        "stable_wrong_summary.csv",
        "contrasts.json",
        "provenance.json",
        "acquisition_cost.json",
        "dataset_readiness.json",
        "identity.json",
        "MANUSCRIPT_OUTLINE.md",
        "report.html",
    ]:
        atomic_write_bytes(handoff / name, (out / name).read_bytes())
        hashes[name] = file_digest(handoff / name)
    dd.save(
        handoff / "completion.json",
        {
            "run_id": run_id,
            "status": "completed_exploratory",
            "folds": len(results),
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "artifact_sha256": hashes,
        },
    )
    print(
        f"DONE. Read {handoff / 'AUDIT.md'}. Commit and push {handoff} to share aggregate evidence.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"Publication audit stopped: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc
