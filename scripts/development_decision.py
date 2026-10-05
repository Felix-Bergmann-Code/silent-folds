"""Exploratory, cache-only nested development evaluation; never evaluates external data.

Run from any directory with the workstation evaluation Python. Fold checkpoints
are atomic and bound to code, config, evidence, and analysis identities.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import html
import json
import os
import platform
import sys
import time
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

from scripts.conference_diagnostics import availability, coverage_contrasts
from scripts.conference_extension import (
    assemble,
    case_identity,
    clean_json,
    fit_arm,
    identity,
    select_arms,
    support,
)
from warpaudit.cache.hashing import file_digest, short_hash
from warpaudit.cache.store import ShardedTable, atomic_write_bytes, atomic_write_text
from warpaudit.cli import _current_registration_rows, _pairs_manifest
from warpaudit.config import load_config
from warpaudit.evaluation.full_study import _population
from warpaudit.evaluation.metrics import auroc, average_precision, brier_score
from warpaudit.evaluation.riskcoverage import risk_coverage_curve
from warpaudit.evaluation.weights import inverse_group_size_weights
from warpaudit.predictors.policy import FrozenPolicy
from warpaudit.protocols.splits import make_outer_folds


def save(path, value):
    atomic_write_text(path, json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")


def validate_cases(cases, cfg):
    if cases.empty or not set(cases.dataset_id) <= set(cfg.full_study.development_datasets):
        raise ValueError("Only nonempty declared development data are permitted")
    if cases.job_id.duplicated().any() or cases.group_id.isna().any():
        raise ValueError("Duplicate jobs or missing independent group identifiers")
    if not np.isin(cases.operational_failure, [0, 1]).all():
        raise ValueError("Nonbinary development labels")
    if not np.isfinite(cases.bounded_loss).all():
        raise ValueError("Undefined development loss: run the coordinate-label repair first")
    if (cases.explicit_failure & (cases.operational_failure != 1)).any():
        raise ValueError("Explicit failures have inconsistent labels")


def split_cell(cases, cfg, design, mode, pipeline, fold):
    assignment = make_outer_folds(
        sorted(set(cases.group_id.astype(str))),
        n_folds=design["outer_folds"],
        seed=cfg.splits.seed,
    )
    f = cases.group_id.astype(str).map(assignment)
    own = cases.pipeline_id == pipeline
    source = own if mode == "within_pipeline" else ~own
    cal_fold = (fold + 1) % design["outer_folds"]
    train = cases[source & ~f.isin([fold, cal_fold])].copy()
    calibration = cases[source & (f == cal_fold)].copy()
    test = cases[own & (f == fold)].copy()
    return train, calibration, test


def eligibility(train, calibration, seed):
    reasons = []
    for name, frame in (("train", train), ("calibration", calibration)):
        if frame.operational_failure.nunique() != 2:
            reasons.append(f"{name} lacks both classes")
    groups = sorted(set(train.group_id.astype(str)))
    inner = []
    if len(groups) < 2:
        reasons.append("fewer than two training groups")
    else:
        assignment = make_outer_folds(groups, n_folds=min(5, len(groups)), seed=seed)
        for fold in sorted(set(assignment.values())):
            mask = train.group_id.astype(str).map(assignment) == fold
            item = {
                "fold": fold,
                "train": support(train[~mask]),
                "validation": support(train[mask]),
            }
            inner.append(item)
            if any(
                item[k]["failures"] == 0 or item[k]["successes"] == 0
                for k in ("train", "validation")
            ):
                reasons.append(f"inner fold {fold} lacks both classes")
    return reasons, inner


def evaluate_fold(cases, arms, cfg, design, mode, pipeline, fold):
    train, calibration, test = split_cell(cases, cfg, design, mode, pipeline, fold)
    reasons, inner = eligibility(train, calibration, cfg.splits.seed)
    row = {
        "mode": mode,
        "pipeline": pipeline,
        "fold": fold,
        "role": "primary" if pipeline in design["primary_pipelines"] else "secondary",
        "support": {
            k: support(v)
            for k, v in (("train", train), ("calibration", calibration), ("test", test))
        },
        "groups": {
            k: sorted(set(v.group_id.astype(str)))
            for k, v in (("train", train), ("calibration", calibration), ("test", test))
        },
        "inner_folds": inner,
        "arms": {},
        "predictions": [],
    }
    feature = design["stable_feature"]
    stable_cutoff = None
    if feature in calibration:
        vals = calibration.loc[np.isfinite(calibration[feature]), feature]
        if len(vals):
            stable_cutoff = float(vals.quantile(design["stable_quantile"]))
    for arm, columns in {"training_prevalence": (), **arms}.items():
        info = {"status": "not_estimable", "reasons": reasons, "fit_seconds": None}
        row["arms"][arm] = info
        if test.empty or train.empty or calibration.empty:
            info["reasons"] = ["empty training, calibration, or test partition"]
            continue
        if arm != "training_prevalence" and reasons:
            continue
        started = time.perf_counter()
        if arm == "training_prevalence":
            w = inverse_group_size_weights(train.group_id.astype(str))
            value = float(np.average(train.operational_failure, weights=w))
            policy = FrozenPolicy.from_calibration(
                "development-prevalence",
                _population(calibration, np.full(len(calibration), value)),
                nominal_acceptance=cfg.policy.nominal_acceptance,
            )
            model_hash = short_hash(
                {"training_prevalence": value, "groups": row["groups"]["train"]}
            )
            selected_C = None
            fit_seconds = time.perf_counter() - started
            predicted_at = time.perf_counter()
            probability = np.full(len(test), value)
            prediction_seconds = time.perf_counter() - predicted_at
        else:
            # Unexpected fit/data errors propagate. They are not hidden as class ineligibility.
            model, policy, fit_seconds = fit_arm(train, calibration, columns, cfg, arm)
            predicted_at = time.perf_counter()
            probability = model.predict_probability(test.loc[:, columns])
            prediction_seconds = time.perf_counter() - predicted_at
            model_hash, selected_C = model.fingerprint, model.selected_C
            info["availability"] = availability(train, columns)
            info["test_availability"] = availability(test, columns)
        if not np.isfinite(probability).all():
            raise ValueError("Nonfinite held-out probability")
        info.update(
            status="ok",
            reasons=[],
            fit_seconds=fit_seconds,
            prediction_seconds=prediction_seconds,
            model_hash=model_hash,
            selected_C=selected_C,
            threshold=policy.threshold,
            threshold_is_negative_infinity=bool(policy.threshold == -np.inf),
            calibration_coverage=policy.selection.realised_calibration_coverage
            if hasattr(policy, "selection")
            else None,
        )
        for (_, case), p in zip(test.iterrows(), probability, strict=True):
            stable_measured = (
                stable_cutoff is not None
                and feature in case
                and pd.notna(case[feature])
                and np.isfinite(case[feature])
            )
            stable = (
                stable_cutoff is not None
                and feature in case
                and pd.notna(case[feature])
                and np.isfinite(case[feature])
                and case[feature] <= stable_cutoff
            )
            record = {
                k: case[k]
                for k in (
                    "job_id",
                    "dataset_id",
                    "pair_id",
                    "group_id",
                    "pipeline_id",
                    "operational_failure",
                    "explicit_failure",
                    "bounded_loss",
                )
            }
            record.update(
                mode=mode,
                fold=fold,
                arm=arm,
                probability=float(p),
                accepted=bool(not case.explicit_failure and p <= policy.threshold),
                stable=bool(stable),
                stable_measured=bool(stable_measured),
                stable_cutoff=stable_cutoff,
                stable_value=case.get(feature),
                model_hash=model_hash,
            )
            for key in (
                "tre_px",
                "tre_norm",
                "tau",
                "E1:resample_fixed_seed_spread_mean",
                "E2:perturbation_spread_mean",
            ):
                record[key] = case.get(key)
            row["predictions"].append(record)
    return clean_json(row)


def summarize(frame, coverages):
    p, y = frame.probability.to_numpy(float), frame.operational_failure.to_numpy(float)
    w = inverse_group_size_weights(frame.group_id.astype(str))
    accepted = frame.accepted.to_numpy(bool)
    curve = risk_coverage_curve(_population(frame, p))
    rc = []
    for c in coverages:
        i = np.flatnonzero(curve.coverage <= c + 1e-12)[-1]
        attainable = c <= curve.max_attainable_coverage + 1e-12
        rc.append(
            {
                "requested_coverage": c,
                "attainable": bool(attainable),
                "achieved_coverage": float(curve.coverage[i]) if attainable else None,
                "risk": float(curve.risk[i]) if attainable else None,
            }
        )
    return clean_json(
        {
            **support(frame),
            "brier": brier_score(p, y, w),
            "auroc": auroc(p, y, w),
            "average_precision": average_precision(p, y, w),
            "calibration_bias": float(np.average(p - y, weights=w)),
            "coverage": float(w[accepted].sum() / w.sum()),
            "accepted_risk": float(np.average(y[accepted], weights=w[accepted]))
            if accepted.any()
            else None,
            "accepted_groups": frame.loc[accepted, "group_id"].nunique(),
            "risk_coverage": rc,
        }
    )


def safe_coverage_contrasts(reference, candidate):
    # Restore numerical undefined values for the shared comparator, then write null.
    def numeric(curve):
        return [{**c, "risk": np.nan if c["risk"] is None else c["risk"]} for c in curve]

    return clean_json(coverage_contrasts(numeric(reference), numeric(candidate)))


def paired_contrast(a, b, seed, resamples):
    keys = ["job_id", "group_id", "fold"]
    joined = a[keys + ["probability", "operational_failure"]].merge(
        b[keys + ["probability", "operational_failure"]],
        on=keys,
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    if len(joined) != len(a) or len(joined) != len(b):
        raise ValueError("Paired contrast requires identical evaluation jobs")
    if not np.array_equal(joined.operational_failure_a, joined.operational_failure_b):
        raise ValueError("Paired contrast labels differ")
    y = joined.operational_failure_a
    joined["delta"] = (joined.probability_a - y) ** 2 - (joined.probability_b - y) ** 2
    group = joined.groupby("group_id").delta.mean().to_numpy()
    rng = np.random.default_rng(seed)
    draws = [float(rng.choice(group, len(group), replace=True).mean()) for _ in range(resamples)]
    return {
        "brier_improvement": float(group.mean()),
        "groups": len(group),
        "conditional_group_bootstrap_low": float(np.quantile(draws, 0.025))
        if len(group) > 1
        else None,
        "conditional_group_bootstrap_high": float(np.quantile(draws, 0.975))
        if len(group) > 1
        else None,
        "interval_scope": "exploratory fixed OOF predictions; excludes training/selection variability; not confirmatory",
        "fold_differences": {
            str(k): float(v.groupby("group_id").delta.mean().mean())
            for k, v in joined.groupby("fold")
        },
    }


def build_tables(results, cases, arms, spec, design, seed):
    predictions = pd.DataFrame([p for result in results for p in result["predictions"]])
    rows, contrasts, curves, diagnostics = [], [], [], []
    for (mode, pipeline), cell in predictions.groupby(["mode", "pipeline_id"]):
        expected = cases[cases.pipeline_id == pipeline]
        strata = [("ALL", cell, expected)] + [
            (d, cell[cell.dataset_id == d], expected[expected.dataset_id == d])
            for d in sorted(set(expected.dataset_id))
        ]
        for dataset, frame, target in strata:
            summaries = {}
            for arm in ["training_prevalence", *arms]:
                a = frame[frame.arm == arm]
                complete = set(a.job_id) == set(target.job_id) and len(a) == len(target)
                key = {
                    "mode": mode,
                    "pipeline": pipeline,
                    "dataset": dataset,
                    "arm": arm,
                    "role": "primary" if pipeline in design["primary_pipelines"] else "secondary",
                    "status": "complete" if complete else "not_estimable",
                    "expected_cases": len(target),
                    "predicted_cases": len(a),
                }
                if complete and len(a):
                    m = summarize(a, spec["requested_coverages"])
                    rows.append({**key, **{k: v for k, v in m.items() if k != "risk_coverage"}})
                    curves.extend({**key, **c} for c in m["risk_coverage"])
                    summaries[arm] = (a, m)
                else:
                    rows.append(key)
            for reference, candidate in [
                ["training_prevalence", "match_count"],
                *spec["contrasts"],
            ]:
                key = {
                    "mode": mode,
                    "pipeline": pipeline,
                    "dataset": dataset,
                    "reference": reference,
                    "candidate": candidate,
                }
                if reference not in summaries or candidate not in summaries:
                    contrasts.append({**key, "status": "not_estimable"})
                    continue
                a, ma = summaries[reference]
                b, mb = summaries[candidate]
                contrast = paired_contrast(a, b, seed, design["bootstrap_resamples"])
                contrasts.append(
                    {
                        **key,
                        "status": "complete",
                        **contrast,
                        "coverage_change": mb["coverage"] - ma["coverage"],
                        "matched_coverage": safe_coverage_contrasts(
                            ma["risk_coverage"], mb["risk_coverage"]
                        ),
                    }
                )
        # Each job is listed once per mode/arm; group summaries retain independent denominators.
        for (arm, group), f in cell.groupby(["arm", "group_id"]):
            wrong = f.stable & ~f.explicit_failure & (f.operational_failure == 1)
            diagnostics.append(
                {
                    "mode": mode,
                    "pipeline": pipeline,
                    "arm": arm,
                    "group_id": group,
                    "cases": len(f),
                    "stable_measured_cases": int(f.stable_measured.sum()),
                    "stable_cases": int(f.stable.sum()),
                    "stable_wrong_cases": int(wrong.sum()),
                    "stable_wrong_accepted": int((wrong & f.accepted).sum()),
                }
            )
    fold_metrics = []
    for (mode, pipeline, fold, arm), f in predictions.groupby(
        ["mode", "pipeline_id", "fold", "arm"]
    ):
        fold_metrics.append(
            {
                "mode": mode,
                "pipeline": pipeline,
                "fold": fold,
                "arm": arm,
                **{
                    k: v
                    for k, v in summarize(f, spec["requested_coverages"]).items()
                    if k != "risk_coverage"
                },
            }
        )
    return predictions, rows, contrasts, curves, diagnostics, fold_metrics


def write_report(out, tables, results, design):
    predictions, rows, contrasts, curves, diagnostics, fold_metrics = tables
    for name, frame in (
        ("predictions", predictions),
        ("performance", rows),
        ("risk_coverage", curves),
        ("stable_wrong_groups", diagnostics),
        ("fold_performance", fold_metrics),
    ):
        atomic_write_text(out / f"{name}.csv", pd.DataFrame(frame).to_csv(index=False))
    flat = [
        {k: v for k, v in c.items() if k not in ("fold_differences", "matched_coverage")}
        for c in contrasts
    ]
    atomic_write_text(out / "contrasts.csv", pd.DataFrame(flat).to_csv(index=False))
    save(out / "contrasts.json", contrasts)
    stable = predictions[
        predictions.stable & ~predictions.explicit_failure & (predictions.operational_failure == 1)
    ]
    atomic_write_text(out / "stable_wrong_cases.csv", stable.to_csv(index=False))
    save(
        out / "fit_status.json",
        [{k: v for k, v in r.items() if k != "predictions"} for r in results],
    )
    fit_rows = []
    for result in results:
        for arm, info in result["arms"].items():
            fit_rows.append(
                {
                    "mode": result["mode"],
                    "pipeline": result["pipeline"],
                    "fold": result["fold"],
                    "arm": arm,
                    "status": info["status"],
                    "reasons": "; ".join(info["reasons"]),
                    "fit_seconds": info.get("fit_seconds"),
                    "prediction_seconds": info.get("prediction_seconds"),
                    **{
                        f"{part}_{k}": v
                        for part, counts in result["support"].items()
                        for k, v in counts.items()
                    },
                }
            )
    atomic_write_text(out / "fit_status.csv", pd.DataFrame(fit_rows).to_csv(index=False))
    primary = [r for r in rows if r["role"] == "primary" and r["dataset"] == "ALL"]
    table = pd.DataFrame(primary).to_html(index=False, na_rep="not estimable", escape=True)
    # Standalone SVG avoids requiring matplotlib on the evaluation workstation.
    bars = [r for r in primary if r.get("brier") is not None]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="{50+26*len(bars)}" role="img" aria-label="Held-out development Brier scores; lower is better">'
    ]
    svg.append(
        '<text x="10" y="20" font-size="16">Exploratory held-out development Brier score (lower is better)</text>'
    )
    for i, r in enumerate(bars):
        y = 48 + i * 26
        label = html.escape(f'{r["mode"]} / {r["pipeline"]} / {r["arm"]}')
        svg.extend(
            [
                f'<text x="10" y="{y}" font-size="12">{label}</text>',
                f'<rect x="580" y="{y-12}" width="{r["brier"]*350}" height="15" fill="#236b8e"/>',
                f'<text x="950" y="{y}" font-size="12">{r["brier"]:.4f}</text>',
            ]
        )
    svg.append("</svg>")
    atomic_write_text(out / "brier.svg", "\n".join(svg))
    note = "Exploratory development evidence. Conditional bootstrap intervals hold OOF predictions fixed and omit training/selection variability. No external confirmation or automatic publication verdict."
    atomic_write_text(
        out / "report.html",
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>WarpAudit development decision</title><style>body{font:15px system-ui;margin:32px}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ddd}img{max-width:100%}</style><h1>Development decision package</h1><p>'
        + note
        + '</p><img src="brier.svg" alt="Brier scores">'
        + table
        + "</html>",
    )
    lines = [
        "# Development decision package",
        "",
        note,
        "",
        "Primary scope: " + ", ".join(design["primary_pipelines"]),
        "",
        "| Mode | Pipeline | Arm | Status | Brier | AUROC | Coverage | Accepted risk |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]

    def fmt(v):
        return "NA" if v is None else f"{v:.4f}"

    for r in primary:
        lines.append(
            f'| {r["mode"]} | {r["pipeline"]} | {r["arm"]} | {r["status"]} | '
            + " | ".join(fmt(r.get(k)) for k in ("brier", "auroc", "coverage", "accepted_risk"))
            + " |"
        )
    lines += [
        "",
        "## Primary paired comparisons",
        "",
        "Positive Brier differences favor the candidate. Interval: conditional group bootstrap, exploratory only.",
        "",
        "| Mode | Pipeline | Reference → candidate | Difference | 95% conditional interval | Positive folds |",
        "|---|---|---|---:|---|---:|",
    ]
    for c in contrasts:
        if c["pipeline"] in design["primary_pipelines"] and c["dataset"] == "ALL":
            bounds = f'{fmt(c.get("conditional_group_bootstrap_low"))} to {fmt(c.get("conditional_group_bootstrap_high"))}'
            folds = c.get("fold_differences", {})
            signs = f"{sum(v > 0 for v in folds.values())}/{len(folds)}" if folds else "NA"
            lines.append(
                f'| {c["mode"]} | {c["pipeline"]} | {c["reference"]} → {c["candidate"]} | {fmt(c.get("brier_improvement"))} | {bounds} | {signs} |'
            )
    lines += [
        "",
        "Read contrasts.csv/json for paired Brier differences, conditional intervals and fold differences. Risk comparisons are only labelled matched when achieved coverage is equal. Undefined risk stays NA.",
        "",
        "SIFT/TPS remain in secondary tables; failed fits are explicit in fit_status.json. Incomplete arms are not compared on a favorable subset.",
        "",
        "Stable-wrong cases use a calibration-only 25th-percentile seed-spread threshold, all ties included. This is an operational low-spread definition, not proof of correctness or a universal stability threshold. Cases and group denominators are separate CSVs.",
        "",
        "External natural cohort access remains NOT VERIFIED. This run cannot obtain permissions, certify clinical annotations, or establish independent external replication.",
        "",
        "Decision: examine whether the prespecified 0.01 Brier improvement is plausible and consistent across folds/datasets, or whether a consequential natural blind spot is replicated. Otherwise stop broad expansion. Do not use conditional intervals as a confirmatory pass/fail test.",
    ]
    atomic_write_text(out / "DECISION.md", "\n".join(lines) + "\n")


def feature_costs(features, cases, arms):
    """Each family bundle repeats its time on every scalar row; count it once."""
    keys = ["job_id", "family"]
    required = {*keys, "incremental_wall_s"}
    if not required <= set(features):
        return {"status": "unavailable", "reason": "incremental_wall_s absent"}
    selected = features[features.job_id.isin(cases.job_id)].copy()
    selected["incremental_wall_s"] = pd.to_numeric(selected.incremental_wall_s, errors="coerce")
    valid = (
        selected.incremental_wall_s.notna()
        & np.isfinite(selected.incremental_wall_s)
        & (selected.incremental_wall_s >= 0)
    )
    selected.loc[~valid, "incremental_wall_s"] = np.nan
    # Inconsistent repeated values cannot be turned into a precise cost estimate.
    variation = selected.groupby(keys).incremental_wall_s.nunique()
    conflicts = variation[variation > 1]
    bundles = selected.groupby(keys, as_index=False).incremental_wall_s.first()
    bad_keys = set(conflicts.index)
    for i, row in bundles.iterrows():
        if (row.job_id, row.family) in bad_keys:
            bundles.loc[i, "incremental_wall_s"] = np.nan
    rows = []
    for pipeline, frame in cases.groupby("pipeline_id"):
        costs = bundles[bundles.job_id.isin(frame.job_id)]
        for arm, columns in arms.items():
            families = sorted({c.split(":", 1)[0] for c in columns})
            subset = costs[costs.family.isin(families)]
            expected = len(frame) * len(families)
            complete = len(subset) == expected and subset.incremental_wall_s.notna().all()
            rows.append(
                {
                    "pipeline": pipeline,
                    "arm": arm,
                    "families": families,
                    "expected_job_families": expected,
                    "timed_job_families": int(subset.incremental_wall_s.notna().sum()),
                    "all_families_measured": bool(complete),
                    "sum_family_seconds": float(subset.incremental_wall_s.sum())
                    if complete
                    else None,
                    "mean_family_seconds_per_case": float(
                        subset.incremental_wall_s.sum() / len(frame)
                    )
                    if complete
                    else None,
                }
            )
    return {
        "status": "measured_family_bundles",
        "arms": rows,
        "conflicting_job_families": len(conflicts),
        "interpretation": "Measured full-family extraction cost. E1 component arms share the full E1 bundle cost; subcomponent costs are not separately measured. Excludes prerequisite registrations/E2 draws. Job sums are not parallel elapsed time.",
    }


def write_costs(out, cases, arms, cfg):
    from warpaudit.cli import _feature_code_identity

    cache = cfg.paths.resolve(ROOT)["cache_root"]
    features = ShardedTable(cache, "features", key_column="feature_id").load()
    features = features[
        (features.config_hash == cfg.hash) & (features.code_hash == _feature_code_identity(ROOT))
    ]
    costs = feature_costs(features, cases, arms)
    regs = ShardedTable(cache, "registrations").load()
    pairs, _ = _pairs_manifest(cfg, ROOT)
    regs = _current_registration_rows(regs, pairs, cfg, ROOT)
    regs = regs[regs.dataset_id.isin(cfg.full_study.development_datasets)].drop_duplicates(
        "job_id", keep="last"
    )
    timing = []
    for key, frame in regs.groupby(["pipeline_id", "condition", "direction"]):
        values = pd.to_numeric(frame.runtime_s, errors="coerce")
        good = values[np.isfinite(values) & (values >= 0)]
        timing.append(
            dict(zip(["pipeline", "condition", "direction"], key, strict=True))
            | {
                "jobs": len(frame),
                "timed_jobs": len(good),
                "sum_job_seconds": float(good.sum()) if len(good) else None,
                "median_job_seconds": float(good.median()) if len(good) else None,
            }
        )
    save(
        out / "acquisition_cost.json",
        {
            "feature_families": costs,
            "registration_jobs": timing,
            "interpretation": "E2 perturbation extraction depends on auxiliary registration jobs listed separately. Reverse jobs can be shared by multiple features. Do not equate family timing with total end-to-end inference cost or count shared prerequisites twice.",
        },
    )


def worker(task):
    cases, arms, cfg, design, mode, pipeline, fold, path, run_id = task
    with threadpool_limits(limits=1):
        result = evaluate_fold(cases, arms, cfg, design, mode, pipeline, fold)
    save(path, {"run_id": run_id, "result_hash": short_hash(result, length=64), "result": result})
    return str(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/full_study.yaml")
    parser.add_argument("--spec", default="configs/conference_extension.json")
    parser.add_argument("--design", default="configs/development_decision.json")
    parser.add_argument("--output", default="conference_outputs/development_decision")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--handoff",
        default="reports/development_decision_latest",
        help="Aggregate-only Git handoff directory (no patient rows or images)",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be 1..16")
    os.chdir(ROOT)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    cfg = load_config(args.config)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    design = json.loads(Path(args.design).read_text(encoding="utf-8"))
    if design["outer_folds"] != 5 or design["bootstrap_resamples"] < 100:
        raise ValueError("Decision design requires five fixed outer folds and >=100 resamples")
    if not 0 < design["stable_quantile"] < 1:
        raise ValueError("stable_quantile must be inside (0,1)")
    if not design["transfer_modes"] or not set(design["transfer_modes"]) <= {
        "within_pipeline",
        "held_out_pipeline",
    }:
        raise ValueError("Unknown transfer mode")
    if not set(design["primary_pipelines"]) <= set(cfg.common_block):
        raise ValueError("Primary pipelines missing from config")
    print(
        "Loading corrected development caches only; no registration jobs or external evaluation.",
        flush=True,
    )
    cases, names = assemble(cfg, ROOT, cfg.full_study.development_datasets)
    validate_cases(cases, cfg)
    arms = select_arms(names, spec)
    if design["stable_feature"] not in cases:
        raise ValueError("Configured stability diagnostic feature is absent")
    # Do not serialize large correspondence/transform arrays to every CPU worker.
    metadata = [
        "job_id",
        "dataset_id",
        "pair_id",
        "group_id",
        "pipeline_id",
        "operational_failure",
        "explicit_failure",
        "bounded_loss",
    ]
    cases = cases.loc[:, list(dict.fromkeys([*metadata, *names]))].copy()
    print(
        f"Assembled {len(cases)} cases, {cases.group_id.nunique()} groups, {len(names)} scalar features.",
        flush=True,
    )
    # Supplement case diagnostics without changing the base cache or its identities.
    labels = ShardedTable(cfg.paths.resolve(ROOT)["cache_root"], "labels").load()
    labels = labels[labels.job_id.isin(cases.job_id)].drop_duplicates("job_id", keep="last")
    extra = [k for k in ("tre_px", "tre_norm", "tau") if k in labels and k not in cases]
    cases = cases.merge(labels[["job_id", *extra]], on="job_id", validate="one_to_one")
    run_identity = {
        **identity(cfg, ROOT, spec),
        "design": design,
        "decision_code": file_digest(Path(__file__)),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn")},
        },
        "development": case_identity(cases, arms),
        "diagnostic_labels": short_hash(
            clean_json(cases[["job_id", *extra]].to_dict("records")), length=64
        ),
    }
    run_id = short_hash(run_identity, length=64)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "run_identity.json"
    if manifest.exists() and json.loads(manifest.read_text())["run_id"] != run_id:
        raise ValueError("Output contains a different run identity; use a new --output directory")
    save(manifest, {"run_id": run_id, "identity": run_identity})
    tasks, paths = [], []
    for mode in design["transfer_modes"]:
        for pipeline in cfg.common_block:
            for fold in range(design["outer_folds"]):
                path = out / "folds" / f"{mode}-{pipeline}-{fold}.json"
                paths.append(path)
                if path.exists():
                    stored = json.loads(path.read_text())
                    if stored["run_id"] != run_id or stored["result_hash"] != short_hash(
                        stored["result"], length=64
                    ):
                        raise ValueError(f"Checkpoint identity/content mismatch: {path}")
                    continue
                tasks.append((cases, arms, cfg, design, mode, pipeline, fold, path, run_id))
    print(
        f"{len(paths)-len(tasks)}/{len(paths)} fold checkpoints reusable; workers={args.workers}",
        flush=True,
    )
    started = time.perf_counter()
    if args.workers == 1:
        for task in tasks:
            print("Completed " + worker(task), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            futures = [pool.submit(worker, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), 1):
                print(
                    f"[{i}/{len(tasks)}] {future.result()} ({(time.perf_counter()-started)/60:.1f} min)",
                    flush=True,
                )
    results = [json.loads(path.read_text())["result"] for path in paths]
    tables = build_tables(results, cases, arms, spec, design, cfg.splits.seed)
    write_report(out, tables, results, design)
    write_costs(out, cases, arms, cfg)
    pairs, _ = _pairs_manifest(cfg, ROOT)
    external = pairs[~pairs.dataset_id.isin(cfg.full_study.development_datasets)]
    save(
        out / "external_inventory.json",
        {
            "label_access": "none; manifest metadata only",
            "status": "natural independent validation NOT VERIFIED",
            "datasets": [
                {
                    "dataset": str(d),
                    "pairs": len(f),
                    "manifest_groups": f.group_id.nunique(),
                    "natural_landmarks_verified": False,
                    "permissions_verified": False,
                }
                for d, f in external.groupby("dataset_id")
            ],
            "required": [
                "accessible paired images and natural landmarks",
                "independent patient/specimen mapping",
                "documented permissions",
                "appropriate transform family",
                "group-level precision plan",
            ],
        },
    )
    save(
        out / "completion.json",
        {
            "run_id": run_id,
            "status": "completed_exploratory",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "folds": len(results),
            "not_estimable_arm_folds": sum(
                a["status"] != "ok" for r in results for a in r["arms"].values()
            ),
            "external_evidence": "not_verified",
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
        },
    )
    handoff = Path(args.handoff)
    handoff.mkdir(parents=True, exist_ok=True)
    if handoff.resolve() == out.resolve():
        raise ValueError("Handoff must be separate from the detailed output directory")
    (handoff / "completion.json").unlink(missing_ok=True)
    exported = {}
    for name in (
        "DECISION.md",
        "performance.csv",
        "contrasts.json",
        "contrasts.csv",
        "fold_performance.csv",
        "risk_coverage.csv",
        "fit_status.csv",
        "acquisition_cost.json",
        "external_inventory.json",
        "run_identity.json",
        "brier.svg",
        "report.html",
    ):
        atomic_write_bytes(handoff / name, (out / name).read_bytes())
        exported[name] = file_digest(handoff / name)
    save(
        handoff / "completion.json",
        {**json.loads((out / "completion.json").read_text()), "artifact_sha256": exported},
    )
    print(
        f"Aggregate handoff ready: {handoff}. Commit this directory to share the result.",
        flush=True,
    )
    print(
        f"DONE. Read {out / 'DECISION.md'} or {out / 'report.html'}. Share performance.csv, contrasts.json, fit_status.json and completion.json.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"Decision run stopped: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc
