#!/usr/bin/env python3
"""SuperRetina projective-pole audit (evaluation-environment side).

``export-jobs`` writes the pair list consumed by
``scripts/superretina/run_superretina.py`` (SuperRetina environment).
``evaluate`` joins its homographies to the study's landmark annotations and
reports: the official FIRE S/P/A AUC reproduction (sanity check against the
released 0.950/0.554/0.783), failures at this paper's 0.005-diagonal
endpoint, exact rectangle/FOV pole crossings, normalized clearance, the
training-free clearance AUROC, and a re-fit of SuperRetina's stage-1
correspondences with the study RANSAC with and without the support
constraint.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts import isbi_submission_experiments as experiments
from scripts.isbi_submission_experiments import (
    _parallel_map,
    circle_crossing,
    clearance_score,
    fit_homography_constrained,
    perspective_score,
    rank_average,
    rectangle_crossing,
)

TAU = 0.005
OFFICIAL_EXCLUDED = "FIRE/P37"  # official test_on_FIRE.py skips control_points_P37_1_2


def export_jobs(output: Path, seed: int) -> None:
    from warpaudit.cli import _pairs_manifest
    from warpaudit.config import load_config

    cfg = load_config(ROOT / "configs/full_study.yaml")
    pairs, _ = _pairs_manifest(cfg, ROOT)
    pairs = pairs[pairs.dataset_id.astype(str).isin(cfg.full_study.development_datasets)]
    jobs = []
    for row in pairs.sort_values(["dataset_id", "pair_id"]).to_dict("records"):
        jobs.append({
            "job_key": f"{row['pair_id']}|canonical", "pair_id": row["pair_id"],
            "direction": "canonical", "seed": seed,
            "query_path": row["moving_path"], "refer_path": row["fixed_path"],
        })
        if row["dataset_id"] == "FIRE":
            # Official orientation: query = second-named image, refer = first.
            jobs.append({
                "job_key": f"{row['pair_id']}|official", "pair_id": row["pair_id"],
                "direction": "reverse", "seed": seed,
                "query_path": row["fixed_path"], "refer_path": row["moving_path"],
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"seed": seed, "jobs": jobs}, indent=1) + "\n")
    print(f"{len(jobs)} SuperRetina jobs -> {output}")


def official_fire_auc(errors: dict[str, list[float]]) -> dict[str, float]:
    """``common/eval_util.compute_auc`` from the SuperRetina release, verbatim logic."""

    out = {}
    for category, values in errors.items():
        e = np.asarray(values, dtype=float)
        acc = sum(np.sum(e < i) * 100 / len(e) for i in range(1, 26))
        out[category] = acc / (25 * 100)
    out["mAUC"] = float(np.mean([out[c] for c in ("S", "P", "A")]))
    return out


def map_points(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    mapped = np.c_[points, np.ones(len(points))] @ np.asarray(h, float).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return mapped[:, :2] / mapped[:, 2:]


def _evaluate_one(result):
    from warpaudit.cli import _annotation_points
    from warpaudit.geometry.projective import projective_pole_diagnostics
    from warpaudit.registration.fitting import FittingPolicy

    # _parallel_map's initializer sets the state on the experiments module.
    state = experiments._STATE
    cfg, pairs, jobs = state["cfg"], state["pairs"], state["jobs"]
    base_policy = FittingPolicy()
    job = jobs[result["job_key"]]
    pair = pairs.loc[job["pair_id"]]
    query_pts, refer_pts = _annotation_points(pair, ROOT, direction=job["direction"])
    canonical = job["direction"] == "canonical"
    moving_hw = (int(pair.moving_h), int(pair.moving_w))
    fixed_hw = (int(pair.fixed_h), int(pair.fixed_w))
    query_hw, refer_hw = (moving_hw, fixed_hw) if canonical else (fixed_hw, moving_hw)
    diagonal = float(np.hypot(*refer_hw))
    h1 = None if result.get("h1") is None else np.asarray(result["h1"], float)
    h2 = None if result.get("h2") is None else np.asarray(result["h2"], float)
    h = None if h1 is None else (h1 if h2 is None else h2 @ h1)
    official_failed = result.get("status") != "ok" or (
        result.get("official_inlier_rate") or 0.0) < 1e-6
    row = {
        "job_key": result["job_key"], "pair_id": job["pair_id"],
        "direction": job["direction"], "dataset_id": pair.dataset_id,
        "group_id": pair.group_id, "category": pair.dataset_category,
        "status": result.get("status"), "returned": h is not None,
        "official_failed": bool(official_failed),
        "inlier_rate": float(result.get("official_inlier_rate") or 0.0),
    }
    height, width = query_hw
    if h is not None:
        mapped = map_points(h, query_pts)
        finite = np.isfinite(mapped).all()
        tre = float(np.linalg.norm(mapped - refer_pts, axis=1).mean()) if finite else np.inf
        diag = projective_pole_diagnostics(h, width=width, height=height)
        row.update(
            tre_px=tre, tre_norm=tre / diagonal,
            failure=bool(not np.isfinite(tre) or tre / diagonal > TAU),
            rectangle_crossing=rectangle_crossing(h, width, height),
            fov_crossing=circle_crossing(h, width, height),
            clearance=diag.pole_clearance_diagonal_fraction,
            stage1_rectangle_crossing=rectangle_crossing(h1, width, height),
            perspective=perspective_score(h, width, height),
        )
    else:
        row.update(tre_px=np.nan, tre_norm=np.nan, failure=True,
                   rectangle_crossing=False, fov_crossing=False, clearance=np.nan,
                   stage1_rectangle_crossing=False, perspective=np.nan)

    # Study RANSAC on stage-1 matches, threshold scaled to original pixels.
    src = np.asarray(result.get("stage1_query_points") or [], float).reshape(-1, 2)
    dst = np.asarray(result.get("stage1_refer_points") or [], float).reshape(-1, 2)
    scale = max(query_hw) / cfg.geometry.working_long_edge
    policy = FittingPolicy(threshold_px=base_policy.threshold_px * scale)
    refits = []
    for variant in ("none", "rectangle", "fov"):
        fit = fit_homography_constrained(
            src, dst, policy, seed=int(job["seed"]), constraint=variant,
            width=width, height=height,
        )
        ref = {"job_key": result["job_key"], "dataset_id": pair.dataset_id,
               "direction": job["direction"], "variant": variant, "status": fit["status"],
               "refit_fallback": fit["refit_fallback"]}
        if fit["matrix"] is not None:
            m = np.asarray(fit["matrix"])
            mapped = map_points(m, query_pts)
            tre = (float(np.linalg.norm(mapped - refer_pts, axis=1).mean())
                   if np.isfinite(mapped).all() else np.inf)
            ref.update(tre_px=tre, failure=bool(not np.isfinite(tre) or tre / diagonal > TAU),
                       rectangle_crossing=rectangle_crossing(m, width, height))
        else:
            ref.update(tre_px=np.nan, failure=True, rectangle_crossing=False)
        refits.append(ref)
    return row, refits


def evaluate(jobs_path: Path, results_dir: Path, output: Path, workers: int) -> None:
    from warpaudit.cli import _pairs_manifest
    from warpaudit.config import load_config
    from warpaudit.evaluation.metrics import auroc
    from warpaudit.evaluation.weights import inverse_group_size_weights

    cfg = load_config(ROOT / "configs/full_study.yaml")
    pairs, _ = _pairs_manifest(cfg, ROOT)
    pairs = pairs.set_index("pair_id", drop=False)
    jobs = {j["job_key"]: j for j in json.loads(jobs_path.read_text())["jobs"]}
    results = []
    shards = sorted(results_dir.glob("homographies*.jsonl"))
    if not shards:
        raise SystemExit(f"no homographies*.jsonl shards in {results_dir}")
    for shard in shards:
        results += [json.loads(x) for x in shard.read_text().splitlines() if x.strip()]
    missing = set(jobs) - {r["job_key"] for r in results}
    if missing:
        raise SystemExit(f"{len(missing)} SuperRetina jobs have no result; rerun stage 1")
    state = {"cfg": cfg, "pairs": pairs, "jobs": jobs}
    rows, refits = [], []
    for row, refit_rows in _parallel_map(_evaluate_one, results, state, workers):
        rows.append(row)
        refits.extend(refit_rows)

    frame = pd.DataFrame(rows)
    refit = pd.DataFrame(refits)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "superretina_cases.csv", index=False)
    refit.to_csv(output / "superretina_refit_cases.csv", index=False)

    official = frame[frame.direction.eq("reverse") & frame.pair_id.ne(OFFICIAL_EXCLUDED)]
    errors = {c: [] for c in ("S", "P", "A")}
    for r in official.itertuples():
        errors[r.category].append(1e6 if r.official_failed or not np.isfinite(r.tre_px)
                                  else r.tre_px)
    fire_auc = official_fire_auc(errors) if all(errors.values()) else {}

    summary = []
    canonical = frame[frame.direction.eq("canonical")]
    for dataset in ("ALL", "FIRE", "COph100"):
        part = canonical if dataset == "ALL" else canonical[canonical.dataset_id.eq(dataset)]
        if part.empty:
            continue
        y = part.failure.to_numpy(bool)
        w = inverse_group_size_weights(part.group_id.astype(str))
        returned = part.returned.to_numpy(bool)
        score = clearance_score(part.clearance, ~returned)
        # Same pre-specified conventions as the cached pipelines.
        inlier_score = np.where(returned, -part.inlier_rate.to_numpy(float), 1.0)
        perspective = np.where(
            returned, np.log10(np.clip(part.perspective.to_numpy(float), 1e-6, 1e6)), 6.0)
        near = part.returned & (part.clearance <= 1.0)
        summary.append({
            "dataset": dataset, "pairs": len(part), "returned": int(part.returned.sum()),
            "failures": int(y.sum()),
            "rectangle_poles": int(part.rectangle_crossing.sum()),
            "fov_poles": int(part.fov_crossing.sum()),
            "failed_poles": int((part.rectangle_crossing & part.failure).sum()),
            "pole_groups": int(part.group_id[part.rectangle_crossing].nunique()),
            "stage1_rectangle_poles": int(part.stage1_rectangle_crossing.sum()),
            "rho_le_1": int(near.sum()), "rho_le_1_failures": int((near & part.failure).sum()),
            "clearance_auroc": auroc(score, y, w) if 0 < y.sum() < len(y) else np.nan,
            "inlier_ratio_auroc": (auroc(inlier_score, y, w)
                                   if 0 < y.sum() < len(y) else np.nan),
            "perspective_auroc": (auroc(perspective, y, w)
                                  if 0 < y.sum() < len(y) else np.nan),
            "clearance_plus_inlier_ratio_auroc": (
                auroc(rank_average(score, inlier_score), y, w)
                if 0 < y.sum() < len(y) else np.nan),
        })
    pd.DataFrame(summary).to_csv(output / "superretina_summary.csv", index=False)
    refit_summary = (
        refit[refit.direction.eq("canonical")]
        .groupby(["dataset_id", "variant"], as_index=False)
        .agg(cases=("job_key", "size"), returned=("status", lambda s: int((s == "ok").sum())),
             failures=("failure", "sum"),
             rectangle_poles=("rectangle_crossing", "sum"),
             refit_fallbacks=("refit_fallback", "sum"))
    )
    refit_summary.to_csv(output / "superretina_refit_summary.csv", index=False)
    (output / "fire_official_reproduction.json").write_text(json.dumps({
        "reproduced": fire_auc,
        "released": {"S": 0.950, "P": 0.554, "A": 0.783, "mAUC": 0.762},
        "pairs": len(official),
    }, indent=2) + "\n")
    print(pd.DataFrame(summary).to_string(index=False))
    print("official FIRE AUC reproduction:", fire_auc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export-jobs")
    e.add_argument("--output", type=Path,
                   default=ROOT / "reports/isbi_submission_latest/superretina/jobs.json")
    e.add_argument("--seed", type=int, default=20260907)
    v = sub.add_parser("evaluate")
    v.add_argument("--jobs", type=Path,
                   default=ROOT / "reports/isbi_submission_latest/superretina/jobs.json")
    v.add_argument("--results-dir", type=Path,
                   default=ROOT / "reports/isbi_submission_latest/superretina")
    v.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    v.add_argument("--output", type=Path,
                   default=ROOT / "reports/isbi_submission_latest/superretina")
    args = parser.parse_args()
    if args.command == "export-jobs":
        export_jobs(args.output, args.seed)
    else:
        evaluate(args.jobs, args.results_dir, args.output, args.workers)


if __name__ == "__main__":
    main()
