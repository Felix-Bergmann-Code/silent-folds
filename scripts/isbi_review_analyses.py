#!/usr/bin/env python3
"""Review analyses for the ISBI paper, from committed result CSVs only.

No registration, matcher, or detector is run. Inputs are the workstation
outputs in ``reports/isbi_submission_latest`` and the saved detector
predictions in ``reports/pole_guard_ablation_latest``. Outputs go to
``reports/isbi_review_analyses``; ``scripts/build_isbi_numbers.py`` turns
them into paper macros.

1. ``auroc_ci.csv``: group-weighted AUROC with a 95% group-bootstrap interval
   (patients / FIRE components resampled) for every training-free score and
   the detector, on all attempted pairs and on returned transforms only
   (SuperRetina also on estimates with at least one LMedS inlier).
2. ``auroc_differences.csv``: paired bootstrap intervals of AUROC differences
   (fused - detector, fused - inlier ratio, clearance - perspective,
   clearance - detector).
3. ``threshold_sensitivity.csv``: the same AUROCs and the rho<=1 precision
   when the failure threshold is 0.0025, 0.005, 0.01, or 0.02 of the
   fixed-image diagonal, with the crossing counts.
4. ``calibration_quality.csv``: group-weighted Brier score and NLL of the
   calibrated probabilities per treatment, over the 21 assignments.
5. ``crossings_by_group.csv``: image crossings and their failures per patient
   group, to show how concentrated the pole events are.

The detector enters with its original-assignment pooled raw score (Base arm,
pole guard), so its point estimate differs slightly from the 21-assignment
median in Table 1.

The fused score is cross-fitted so that it can score a single new case: each
component is mapped through its empirical percentile transform (mid-ranks)
estimated on the other four group folds of the original assignment only, and
the two percentiles are averaged. No label enters the transform.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.isbi_submission_experiments import (  # noqa: E402
    clearance_score,
    group_bootstrap_auroc,
)
from warpaudit.evaluation.metrics import auroc  # noqa: E402
from warpaudit.evaluation.weights import inverse_group_size_weights  # noqa: E402

RESULTS = ROOT / "reports/isbi_submission_latest"
ABLATION = ROOT / "reports/pole_guard_ablation_latest/test_predictions.csv"
OUTPUT = ROOT / "reports/isbi_review_analyses"
ORIGINAL_SEED = 20260907
FAILURE_THRESHOLD = 0.005
# 0.0025 * 2**(k/3), k = 0..9: contains the study endpoint 0.005 and 0.01, 0.02.
THRESHOLDS = tuple(float(f"{0.0025 * 2 ** (k / 3):.6g}") for k in range(10))
RESAMPLES = 2000
# Working resolution of the generic pipelines (1,024-pixel long edge).
WORKING_WH = {"FIRE": (1024, 1024), "COph100": (1024, 768)}
SUPERRETINA_H = sorted((ROOT / "reports/isbi_submission_latest/superretina").glob(
    "homographies-*.jsonl"))
SCORES = ("clearance", "inlier_ratio", "inlier_count", "perspective", "affine", "condition",
          "scale", "fused", "detector")
# Projective-magnitude baselines that ask only "how projective is H".
PROJECTIVE = ("perspective", "affine", "condition", "scale")
DIFFERENCES = (("fused", "detector"), ("fused", "inlier_ratio"),
               ("clearance", "perspective"), ("clearance", "affine"),
               ("clearance", "condition"), ("clearance", "scale"),
               ("clearance", "detector"), ("fused", "clearance"))
CAP = 1e6


def corners(width: float, height: float) -> np.ndarray:
    return np.array([[0.0, 0.0], [width - 1, 0.0], [width - 1, height - 1], [0.0, height - 1]])


def affine_deviation(h: np.ndarray, width: float, height: float) -> float:
    """Mean corner displacement between H and its least-squares affine fit on
    the four image corners, in image diagonals; inf if the pole crosses."""
    c = corners(width, height)
    mapped = np.c_[c, np.ones(4)] @ np.asarray(h, float).T
    if not (np.all(mapped[:, 2] > 0) or np.all(mapped[:, 2] < 0)):
        return np.inf
    q = mapped[:, :2] / mapped[:, 2:]
    design = np.c_[c, np.ones(4)]
    fit, *_ = np.linalg.lstsq(design, q, rcond=None)
    residual = np.linalg.norm(design @ fit - q, axis=1).mean()
    return float(residual / np.hypot(width - 1, height - 1))


def condition_number(h: np.ndarray, width: float, height: float) -> float:
    """log10 condition number of H in centred coordinates scaled by the half
    diagonal (the same normalization on both images, which share a size)."""
    half = np.hypot(width - 1, height - 1) / 2
    n = np.array([[1 / half, 0, -(width - 1) / 2 / half],
                  [0, 1 / half, -(height - 1) / 2 / half], [0, 0, 1]])
    singular = np.linalg.svd(n @ np.asarray(h, float) @ np.linalg.inv(n), compute_uv=False)
    if singular[-1] <= 0:
        return np.log10(CAP)
    return float(np.log10(min(singular[0] / singular[-1], CAP)))


def projective_baselines(h: np.ndarray | None, width: float, height: float) -> dict:
    """Most-failing values for a missing transform, as for every training-free score."""
    if h is None:
        return {"perspective": 6.0, "affine": CAP, "condition": np.log10(CAP), "scale": CAP}
    return {"perspective": perspective(h, width, height),
            "affine": min(affine_deviation(h, width, height), CAP),
            "condition": condition_number(h, width, height),
            "scale": min(scale_deviation(h, width, height), CAP)}


def group_folds() -> dict[str, int]:
    """Group-to-fold map of the original detector assignment (one per group)."""
    predictions = pd.read_csv(ABLATION, usecols=["seed", "fold", "group_id"])
    original = predictions[predictions.seed.eq(ORIGINAL_SEED)].drop_duplicates("group_id")
    return dict(zip(original.group_id.astype(str), original.fold.astype(int)))


def crossfit_percentile(values, folds) -> np.ndarray:
    """Mid-rank empirical percentile of each case relative to the cases of the
    other folds only, so a case never enters its own reference."""
    values, folds = np.asarray(values, float), np.asarray(folds)
    out = np.empty_like(values)
    for fold in np.unique(folds):
        test = folds == fold
        reference = np.sort(values[~test])
        below = np.searchsorted(reference, values[test], side="left")
        at_or_below = np.searchsorted(reference, values[test], side="right")
        out[test] = (below + at_or_below) / (2 * len(reference))
    return out


def perspective(h: np.ndarray, width: float, height: float) -> float:
    """log10 of the pole's inverse distance from the image center (diagonals)."""
    a = h[2, :2]
    center = np.array([(width - 1) / 2, (height - 1) / 2])
    norm, denominator = float(np.hypot(*a)), abs(float(a @ center + h[2, 2]))
    if norm == 0.0:
        return -6.0
    value = np.inf if denominator == 0.0 else np.hypot(width - 1, height - 1) * norm / denominator
    return float(np.log10(np.clip(value, 1e-6, 1e6)))


def scale_deviation(h: np.ndarray, width: float, height: float) -> float:
    """|log| of the area ratio of the warped image rectangle; inf if it folds."""
    c = corners(width, height)
    mapped = np.c_[c, np.ones(4)] @ h.T
    if not (np.all(mapped[:, 2] > 0) or np.all(mapped[:, 2] < 0)):
        return np.inf
    q = mapped[:, :2] / mapped[:, 2:]
    x, y = q[:, 0], q[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if area <= 0 or not np.isfinite(area):
        return np.inf
    return float(abs(np.log(area / ((width - 1) * (height - 1)))))


def generic_cases() -> pd.DataFrame:
    """XFeat-H and SP/LG-H: one row per attempted pair with every score."""
    cases = pd.read_csv(RESULTS / "clearance/clearance_cases.csv")
    replay = pd.read_csv(RESULTS / "constrained_ransac/constrained_ransac_cases.csv")
    replay = replay[replay.variant.eq("none")]
    frame = cases.merge(
        replay[["job_id", "pipeline_id", "tre_norm", "n_matches", "n_inliers", "matrix_json"]],
        on=["job_id", "pipeline_id"], validate="one_to_one")
    returned = frame.status.eq("ok")
    ratio = frame.n_inliers / frame.n_matches.replace(0, np.nan)
    frame["inlier_ratio"] = np.where(returned, -ratio.fillna(0), 1.0)
    frame["inlier_count"] = np.where(returned, -frame.n_inliers.fillna(0), 1.0)
    baselines = pd.DataFrame([
        projective_baselines(
            np.asarray(json.loads(row.matrix_json), dtype=float) if row.status == "ok" else None,
            *WORKING_WH[row.dataset_id])
        for row in frame.itertuples()], index=frame.index)
    frame[list(PROJECTIVE)] = baselines[list(PROJECTIVE)]
    frame["clearance"] = frame.clearance_score
    frame["crossing"] = returned & frame.pole_crosses_image.astype(bool)
    predictions = pd.read_csv(ABLATION)
    predictions = predictions[
        predictions.seed.eq(ORIGINAL_SEED) & predictions.treatment.eq("pole_guard")
        & predictions.feature_arm.eq("non_stability")]
    frame = frame.merge(
        predictions[["job_id", "pipeline", "raw_score"]].rename(
            columns={"pipeline": "pipeline_id", "raw_score": "detector"}),
        on=["job_id", "pipeline_id"], how="left", validate="one_to_one")
    frame["pipeline"] = frame.pipeline_id
    frame["inliers"] = frame.n_inliers
    return frame[frame.pipeline_id.isin(["xfeat_h", "sp_lg_h"])]


def superretina_homographies() -> dict[str, tuple[np.ndarray, tuple[int, int]]]:
    """Final released homography (stage 2 composed with stage 1, as in the
    audit) and query image size, keyed by job."""
    matrices = {}
    for path in SUPERRETINA_H:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get("h1") is None:
                continue
            h = np.asarray(record["h1"], float)
            if record.get("h2") is not None:
                h = np.asarray(record["h2"], float) @ h
            matrices[record["job_key"]] = (h, tuple(record["query_hw"]))
    return matrices


def superretina_cases() -> pd.DataFrame:
    cases = pd.read_csv(RESULTS / "superretina/superretina_cases.csv")
    frame = cases[cases.direction.eq("canonical")].copy()
    returned = frame.returned.astype(bool)
    frame["status"] = np.where(returned, "ok", "no_output")
    frame["clearance"] = clearance_score(
        frame.clearance.astype(float).where(returned), explicit_failure=~returned)
    frame["inlier_ratio"] = np.where(returned, -frame.inlier_rate.astype(float), 1.0)
    frame["inliers"] = np.where(frame.inlier_rate.astype(float) > 0, 1, 0)
    matrices = superretina_homographies()
    rows = []
    for row in frame.itertuples():
        h, (height, width) = matrices.get(row.job_key, (None, (0, 0)))
        rows.append(projective_baselines(h if row.returned else None, width, height))
    baselines = pd.DataFrame(rows, index=frame.index)
    # The audit's own perspective column must agree with the recomputed one.
    audit = np.where(returned, np.log10(frame.perspective.astype(float).clip(1e-6, 1e6)), 6.0)
    if not np.allclose(audit, baselines.perspective, atol=1e-6):
        raise SystemExit("superretina: recomputed perspective does not match the audit")
    frame[list(PROJECTIVE)] = baselines[list(PROJECTIVE)]
    frame["crossing"] = returned & frame.rectangle_crossing.astype(bool)
    frame["pipeline"] = "superretina"
    frame["operational_failure"] = frame.failure.astype(bool)
    return frame


def relabel(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    frame = frame.copy()
    frame["operational_failure"] = (
        frame.status.ne("ok") | (frame.tre_norm.astype(float) > threshold)).to_numpy()
    return frame


def available(frame: pd.DataFrame) -> list[str]:
    return [s for s in SCORES
            if s in frame.columns and s != "fused" and frame[s].notna().all()] + ["fused"]


def with_fused(frame: pd.DataFrame, folds: dict[str, int]) -> pd.DataFrame:
    """Cross-fitted fused score on all attempted pairs. It uses no labels, so
    it is computed once and shared by every population and endpoint."""
    frame = frame.copy()
    fold = frame.group_id.astype(str).map(folds)
    if fold.isna().any():
        raise SystemExit("a group has no fold in the original assignment")
    frame["fused"] = (crossfit_percentile(frame.clearance, fold)
                      + crossfit_percentile(frame.inlier_ratio, fold)) / 2
    return frame


def point(frame: pd.DataFrame, score: str) -> float:
    y = frame.operational_failure.to_numpy(bool)
    w = inverse_group_size_weights(frame.group_id.astype(str))
    return float(auroc(frame[score].to_numpy(float), y, w))


def populations(pipeline: str, frame: pd.DataFrame):
    yield "all_attempted", frame
    yield "returned_only", frame[frame.status.eq("ok")]
    # rho > 1 (s < 0): no crossing and not near one, so the score must work
    # beyond the obvious pole cases.
    yield "rho_gt_1", frame[frame.status.eq("ok") & frame.clearance.lt(0.0)]
    if pipeline == "superretina":
        yield "with_inliers", frame[frame.status.eq("ok") & frame.inliers.gt(0)]


def analyse(frames: dict[str, pd.DataFrame], resamples: int = RESAMPLES):
    ci_rows, diff_rows = [], []
    for pipeline, full in frames.items():
        for population, part in populations(pipeline, full):
            part = part.reset_index(drop=True)
            scores = available(part)
            draws = group_bootstrap_auroc(part, scores, resamples=resamples, seed=ORIGINAL_SEED)
            for score in scores:
                ci_rows.append({
                    "pipeline": pipeline, "population": population, "score": score,
                    "cases": len(part), "failures": int(part.operational_failure.sum()),
                    "auroc": point(part, score),
                    "ci_low": np.quantile(draws[score], 0.025),
                    "ci_high": np.quantile(draws[score], 0.975)})
            for left, right in DIFFERENCES:
                if left not in scores or right not in scores:
                    continue
                diff = draws[left] - draws[right]
                diff_rows.append({
                    "pipeline": pipeline, "population": population,
                    "comparison": f"{left}-{right}",
                    "difference": point(part, left) - point(part, right),
                    "ci_low": np.quantile(diff, 0.025), "ci_high": np.quantile(diff, 0.975),
                    "fraction_positive": float(np.mean(diff > 0))})
    return pd.DataFrame(ci_rows), pd.DataFrame(diff_rows)


def sensitivity(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for pipeline, full in frames.items():
        for threshold in THRESHOLDS:
            part = relabel(full, threshold).reset_index(drop=True)
            returned = part.status.eq("ok")
            near = returned & (part.clearance >= 0.0)  # s >= 0  <=>  rho <= 1
            crossing = returned & part.clearance.ge(3.0)  # rho = 0 is floored at 1e-3
            rows.append({
                "pipeline": pipeline, "threshold": threshold,
                "failure_rate": float(part.operational_failure.mean()),
                "clearance_auroc": point(part, "clearance"),
                "inlier_ratio_auroc": point(part, "inlier_ratio"),
                "fused_auroc": point(part, "fused"),
                "rho_le_1": int(near.sum()),
                "rho_le_1_failed": int((near & part.operational_failure).sum()),
                "crossings": int(crossing.sum()),
                "crossings_failed": int((crossing & part.operational_failure).sum())})
    return pd.DataFrame(rows)


def crossings_by_group(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for pipeline, frame in frames.items():
        for (dataset, group), part in frame.groupby(["dataset_id", "group_id"]):
            rows.append({"pipeline": pipeline, "dataset": dataset, "group_id": group,
                         "pairs": len(part), "crossings": int(part.crossing.sum()),
                         "crossing_failures": int(
                             (part.crossing & part.operational_failure).sum())})
    return pd.DataFrame(rows)


def calibration_quality() -> pd.DataFrame:
    """Group-weighted Brier score and NLL of the calibrated probabilities per
    assignment, summarized over the 21 assignments (median, IQR, worst)."""
    ablation = pd.read_csv(ABLATION).rename(columns={"treatment": "variant"})
    conditional = pd.read_csv(RESULTS / "clearance/conditional_value_predictions.csv")
    conditional = conditional[conditional.variant.isin(["log_curvature", "quantile_curvature"])]
    predictions = pd.concat([ablation, conditional], ignore_index=True)
    per_assignment = []
    for (seed, pipeline, arm, variant), part in predictions.groupby(
            ["seed", "pipeline", "feature_arm", "variant"]):
        y = part.operational_failure.astype(float).to_numpy()
        p = np.clip(part.probability.to_numpy(float), 1e-12, 1 - 1e-12)
        w = np.asarray(inverse_group_size_weights(part.group_id.astype(str)), dtype=float)
        per_assignment.append({
            "seed": seed, "pipeline": pipeline, "feature_arm": arm, "variant": variant,
            "brier": np.average((p - y) ** 2, weights=w),
            "nll": np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=w)})
    table = pd.DataFrame(per_assignment)
    return (table.groupby(["pipeline", "feature_arm", "variant"])
            .agg(assignments=("seed", "nunique"),
                 brier_median=("brier", "median"),
                 brier_q25=("brier", lambda v: v.quantile(0.25)),
                 brier_q75=("brier", lambda v: v.quantile(0.75)),
                 brier_worst=("brier", "max"),
                 nll_median=("nll", "median"),
                 nll_worst=("nll", "max"))
            .reset_index())


def main(resamples: int = RESAMPLES) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    generic = generic_cases()
    frames = {p: generic[generic.pipeline.eq(p)] for p in ("xfeat_h", "sp_lg_h")}
    frames["superretina"] = superretina_cases()
    folds = group_folds()
    for name, frame in frames.items():  # the relabelling must reproduce the study labels
        again = relabel(frame, FAILURE_THRESHOLD).operational_failure
        if not np.array_equal(again.to_numpy(bool), frame.operational_failure.to_numpy(bool)):
            raise SystemExit(f"{name}: failure labels do not reproduce")
        frames[name] = with_fused(frame, folds)
    ci, differences = analyse(frames, resamples)
    ci.to_csv(OUTPUT / "auroc_ci.csv", index=False)
    differences.to_csv(OUTPUT / "auroc_differences.csv", index=False)
    sensitivity(frames).to_csv(OUTPUT / "threshold_sensitivity.csv", index=False)
    crossings_by_group(frames).to_csv(OUTPUT / "crossings_by_group.csv", index=False)
    calibration_quality().to_csv(OUTPUT / "calibration_quality.csv", index=False)
    (OUTPUT / "run_metadata.json").write_text(json.dumps({
        "resamples": resamples, "seed": ORIGINAL_SEED, "detector": "original assignment, "
        "Base arm, pole_guard, pooled raw score", "thresholds": THRESHOLDS,
        "fused": "mean of mid-rank percentiles of -log10(rho) and the negated inlier "
        "ratio, each transform estimated on the other four group folds of the "
        "original assignment"}, indent=2))
    with pd.option_context("display.width", 200):
        print(ci.round(3).to_string(index=False))
        print(differences.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
