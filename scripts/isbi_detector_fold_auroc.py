#!/usr/bin/env python3
"""Per-fold AUROC of the guarded detector versus RANSAC inlier ratio.

Reads saved outputs only (no model is refitted): the pole-guard ablation's
test predictions and the unconstrained RANSAC replay, which reproduces the
cached transforms and carries their match and inlier counts. Pooled raw
scores from five separately fitted fold models are not on one scale, so this
checks the pooled comparison within folds (unweighted AUROC per fold, mean
over folds, median over the 21 assignments).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "reports/pole_guard_ablation_latest/test_predictions.csv"
REPLAY = ROOT / "reports/isbi_submission_latest/constrained_ransac/constrained_ransac_cases.csv"


def auroc(score, failed) -> float:
    failed = np.asarray(failed, dtype=bool)
    positives, negatives = failed.sum(), (~failed).sum()
    if not positives or not negatives:
        return float("nan")
    ranks = rankdata(np.asarray(score, dtype=float))
    return float((ranks[failed].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def main() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS)
    predictions = predictions[predictions.treatment.eq("pole_guard")]
    replay = pd.read_csv(REPLAY)
    replay = replay[replay.variant.eq("none")].copy()
    # Same orientation as isbi_submission_experiments: lower ratio => failure,
    # no-output cases get the most-failing score.
    ratio = replay.n_inliers / replay.n_matches.replace(0, np.nan)
    replay["inlier_score"] = np.where(replay.status.eq("ok"), -ratio.fillna(0), 1.0)
    joined = predictions.merge(
        replay[["job_id", "pipeline_id", "inlier_score"]],
        left_on=["job_id", "pipeline"], right_on=["job_id", "pipeline_id"], how="left")
    if joined.inlier_score.isna().any():
        raise SystemExit("predictions without a replayed transform")
    rows = []
    for (pipeline, arm), part in joined.groupby(["pipeline", "feature_arm"]):
        folds = part.groupby(["seed", "fold"]).apply(lambda d: pd.Series({
            "detector": auroc(d.raw_score, d.operational_failure),
            "inlier_ratio": auroc(d.inlier_score, d.operational_failure)}))
        per_seed = folds.groupby("seed").mean().median()
        rows.append({"pipeline": pipeline, "feature_arm": arm,
                     "fold_mean_detector_auroc": per_seed.detector,
                     "fold_mean_inlier_ratio_auroc": per_seed.inlier_ratio,
                     "folds_detector_better": float((folds.detector > folds.inlier_ratio).mean())})
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))
    return table


if __name__ == "__main__":
    main()
