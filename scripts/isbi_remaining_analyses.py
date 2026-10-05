"""Produce the compact evidence tables used by the ISBI 2027 manuscript.

This script only summarizes existing registration/ablation outputs.  It never
reruns a matcher or a detector.  The optional fundus-mask audit operates on the
locally available RIDIRP/COph100 moving images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from warpaudit.evaluation.metrics import auroc
from warpaudit.evaluation.weights import inverse_group_size_weights

ROOT = Path(__file__).resolve().parents[1]
ABLATION = ROOT / "reports/pole_guard_ablation_latest"
HPATCHES = ROOT / "benchmark_pole_outputs_hpatches_v2/pole_cases.csv"
MEGADEPTH = ROOT / "benchmark_pole_outputs/pole_cases.csv"
OUTPUT = ROOT / "reports/isbi_remaining_work_latest"

ARMS = {
    "non_stability": "Base",
    "non_stability_e1_e2": "+Stability",
}
PIPELINES = {"xfeat_h": "XFeat-H", "sp_lg_h": "SP/LG-H"}
TREATMENTS = {
    "original": "Original",
    "original_isotonic": "Original + isotonic",
    "curvature_clipped": "Clip",
    "curvature_removed": "Remove",
    "fov_curvature": "FOV-only curvature",
    "pole_guard": "Rectangle guard",
    "fov_guard_on_rectangle": "Rectangle curvature + FOV guard",
    "rectangle_guard_on_fov": "FOV curvature + rectangle guard",
    "fov_guard_on_fov": "FOV curvature + FOV guard",
}


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUTPUT / name, index=False, lineterminator="\n")


def treatment_and_brier_tables() -> None:
    metrics = pd.read_csv(ABLATION / "assignment_metrics.csv")
    diagnostics = pd.read_csv(ABLATION / "split_diagnostics.csv")
    if "pooled_calibrated_auroc" not in metrics.columns:
        predictions = pd.read_csv(ABLATION / "test_predictions.csv")
        calibrated_rows = []
        for keys, part in predictions.groupby(
            ["seed", "pipeline", "feature_arm", "treatment"], sort=True
        ):
            weights = inverse_group_size_weights(part.group_id.astype(str))
            calibrated_rows.append(
                {
                    "seed": keys[0],
                    "pipeline": keys[1],
                    "feature_arm": keys[2],
                    "treatment": keys[3],
                    "pooled_calibrated_auroc": auroc(
                        part.probability, part.operational_failure, weights
                    ),
                }
            )
        metrics = metrics.merge(
            pd.DataFrame(calibrated_rows),
            on=["seed", "pipeline", "feature_arm", "treatment"],
            validate="one_to_one",
        )
    original = metrics[metrics.treatment.eq("original")][
        ["seed", "pipeline", "feature_arm", "pooled_brier"]
    ].rename(columns={"pooled_brier": "original_brier"})
    paired = metrics.merge(original, on=["seed", "pipeline", "feature_arm"])
    paired["paired_brier_difference"] = (
        paired.pooled_brier - paired.original_brier
    )

    rows = []
    for (pipeline, arm, treatment), group in paired.groupby(
        ["pipeline", "feature_arm", "treatment"], sort=False
    ):
        folds = diagnostics[
            diagnostics.pipeline.eq(pipeline)
            & diagnostics.feature_arm.eq(arm)
            & diagnostics.treatment.eq(treatment)
        ]
        rows.append(
            {
                "pipeline": pipeline,
                "pipeline_label": PIPELINES[pipeline],
                "feature_arm": arm,
                "feature_arm_label": ARMS[arm],
                "treatment": treatment,
                "treatment_label": TREATMENTS[treatment],
                "assignments": int(group.seed.nunique()),
                "reversals": int(folds.ranking_reversed_by_calibration.sum()),
                "median_brier": float(group.pooled_brier.median()),
                "median_raw_auroc": float(group.pooled_raw_auroc.median()),
                "median_calibrated_auroc": float(
                    group.pooled_calibrated_auroc.median()
                ),
                "worst_raw_auroc": float(group.pooled_raw_auroc.min()),
                "brier_no_worse_than_original": int(
                    (group.paired_brier_difference <= 0).sum()
                ),
                "median_paired_brier_difference": float(
                    group.paired_brier_difference.median()
                ),
                "min_paired_brier_difference": float(
                    group.paired_brier_difference.min()
                ),
                "max_paired_brier_difference": float(
                    group.paired_brier_difference.max()
                ),
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["treatment", "pipeline", "feature_arm"]
    )
    write_csv(summary, "treatment_summary.csv")
    factorial = summary[
        summary.treatment.isin(
            {
                "pole_guard",
                "fov_guard_on_rectangle",
                "rectangle_guard_on_fov",
                "fov_guard_on_fov",
            }
        )
    ].copy()
    factorial["sampling_domain"] = factorial.treatment.map(
        {
            "pole_guard": "rectangle",
            "fov_guard_on_rectangle": "rectangle",
            "rectangle_guard_on_fov": "FOV",
            "fov_guard_on_fov": "FOV",
        }
    )
    factorial["guard_domain"] = factorial.treatment.map(
        {
            "pole_guard": "rectangle",
            "fov_guard_on_rectangle": "FOV",
            "rectangle_guard_on_fov": "rectangle",
            "fov_guard_on_fov": "FOV",
        }
    )
    write_csv(
        factorial[
            [
                "sampling_domain",
                "guard_domain",
                "pipeline_label",
                "feature_arm_label",
                "assignments",
                "reversals",
                "median_brier",
                "median_raw_auroc",
                "median_calibrated_auroc",
                "worst_raw_auroc",
                "brier_no_worse_than_original",
                "median_paired_brier_difference",
                "min_paired_brier_difference",
                "max_paired_brier_difference",
            ]
        ].sort_values(
            ["sampling_domain", "guard_domain", "pipeline_label", "feature_arm_label"]
        ),
        "support_guard_2x2.csv",
    )
    write_csv(
        summary[summary.treatment.eq("pole_guard")][
            [
                "pipeline_label",
                "feature_arm_label",
                "assignments",
                "brier_no_worse_than_original",
                "median_paired_brier_difference",
                "min_paired_brier_difference",
                "max_paired_brier_difference",
            ]
        ],
        "paired_guard_brier.csv",
    )


def clearance_table() -> None:
    """Summarize observed noncrossing clearance without selecting a cutoff."""

    frame = pd.read_csv(ABLATION / "pole_clearance.csv")
    rows = []
    for pipeline, group in frame[frame.pipeline_id.isin(PIPELINES)].groupby(
        "pipeline_id", sort=True
    ):
        clearance = pd.to_numeric(
            group.pole_clearance_diagonal_fraction, errors="coerce"
        )
        noncrossing = group[
            ~group.pole_crosses_image.astype(bool)
            & np.isfinite(clearance)
            & clearance.gt(0)
        ].copy()
        noncrossing["clearance"] = pd.to_numeric(
            noncrossing.pole_clearance_diagonal_fraction, errors="coerce"
        )
        record: dict[str, object] = {
            "pipeline": pipeline,
            "pipeline_label": PIPELINES[pipeline],
            "returned_homographies": int(len(group)),
            "crossing_poles": int(group.pole_crosses_image.astype(bool).sum()),
            "noncrossing_homographies": int(len(noncrossing)),
            "nearest_failed_clearance": float(
                noncrossing.loc[
                    noncrossing.operational_failure.astype(bool), "clearance"
                ].min()
            ),
            "nearest_successful_clearance": float(
                noncrossing.loc[
                    ~noncrossing.operational_failure.astype(bool), "clearance"
                ].min()
            ),
        }
        for threshold in (0.05, 0.1, 0.25, 0.5, 1.0):
            label = str(threshold).replace(".", "p")
            subset = noncrossing[noncrossing.clearance.le(threshold)]
            record[f"clearance_le_{label}_cases"] = int(len(subset))
            record[f"clearance_le_{label}_failures"] = int(
                subset.operational_failure.astype(bool).sum()
            )
        rows.append(record)
    write_csv(pd.DataFrame(rows), "empirical_clearance_summary.csv")


def splg_mechanism_table() -> None:
    diagnostics = pd.read_csv(ABLATION / "split_diagnostics.csv")
    pole_folds = diagnostics[
        diagnostics.pipeline.eq("sp_lg_h")
        & diagnostics.treatment.eq("original")
        & diagnostics.calibration_poles.gt(0)
    ].copy()
    pole_folds["negative_curvature_coefficient"] = (
        pole_folds.curvature_coefficient < 0
    )
    pole_folds["out_of_training_range_pole"] = (
        pole_folds.calibration_poles_outside_training_range > 0
    )
    pole_folds["reversal"] = pole_folds.ranking_reversed_by_calibration.astype(bool)
    counts = (
        pole_folds.groupby(
            [
                "feature_arm",
                "negative_curvature_coefficient",
                "out_of_training_range_pole",
                "reversal",
            ],
            dropna=False,
        )
        .size()
        .rename("folds")
        .reset_index()
    )
    counts.insert(1, "feature_arm_label", counts.feature_arm.map(ARMS))
    write_csv(counts, "splg_mechanism_crosstab.csv")


def saturation_and_warning_tables() -> None:
    diagnostics = pd.read_csv(ABLATION / "split_diagnostics.csv")
    original = diagnostics[
        diagnostics.seed.eq(20260907)
        & diagnostics.pipeline.eq("xfeat_h")
        & diagnostics["fold"].eq(3)
        & diagnostics.feature_arm.eq("non_stability_e1_e2")
        & diagnostics.treatment.eq("original")
    ].iloc[0]
    deletions = pd.read_csv(ABLATION / "calibration_case_deletion.csv")
    deletions = deletions[deletions.feature_arm.eq("non_stability_e1_e2")]
    fits = [
        ("both poles present", original.platt_slope, original.platt_intercept),
    ]
    for case_id, label in [
        ("a3b2e2934c364e2a", "corner-only pole deleted"),
        ("11bcaf81af6206b4", "FOV-crossing pole deleted"),
    ]:
        row = deletions[deletions.deleted_case.eq(case_id)].iloc[0]
        fits.append((label, row.platt_slope, row.platt_intercept))
    scores = [
        ("a3b2e2934c364e2a", "corner-only pole", -779618.030474),
        ("11bcaf81af6206b4", "FOV-crossing pole", -94113.780187),
    ]
    rows = []
    for fit, slope, intercept in fits:
        for case_id, case, score in scores:
            logit = slope * score + intercept
            rows.append(
                {
                    "fit": fit,
                    "case_id": case_id,
                    "case": case,
                    "raw_margin": score,
                    "platt_slope": slope,
                    "platt_intercept": intercept,
                    "platt_logit": logit,
                    "probability": 1 / (1 + np.exp(-logit)),
                }
            )
    write_csv(pd.DataFrame(rows), "saturation_case_predictions.csv")

    warning = diagnostics[
        diagnostics.seed.eq(20260915)
        & diagnostics.pipeline.eq("xfeat_h")
        & diagnostics["fold"].eq(3)
        & diagnostics.feature_arm.eq("non_stability")
        & diagnostics.treatment.eq("original")
    ][
        [
            "seed",
            "pipeline",
            "fold",
            "feature_arm",
            "ranking_reversed_by_calibration",
            "platt_slope",
            "calibrated_auroc",
        ]
    ].copy()
    warning["warning_source"] = "reports/ROBUSTNESS_RESULTS_REVIEW.md"
    warning["warning"] = "L-BFGS line-search warning retained"
    write_csv(warning, "warning_bearing_fit.csv")


def modern_retinal_poles() -> pd.DataFrame:
    poles = pd.read_csv(ABLATION / "pole_cases.csv")
    poles = poles[
        poles.pipeline_id.isin(PIPELINES)
        & poles.pole_crosses_image.astype(bool)
    ].copy()
    poles["pipeline_label"] = poles.pipeline_id.map(PIPELINES)
    poles["patient_id"] = poles.group_id.str.rsplit("/", n=1).str[-1]
    poles["dataset_category"] = np.where(
        poles.dataset_id.eq("FIRE"),
        poles.pair_id.str.extract(r"FIRE/([SPA])", expand=False),
        "not applicable",
    )
    columns = [
        "job_id",
        "pipeline_id",
        "pipeline_label",
        "dataset_id",
        "dataset_category",
        "patient_id",
        "group_id",
        "pair_id",
        "operational_failure",
        "pole_crosses_inscribed_circle",
        "moving_image_id",
    ]
    write_csv(poles[columns].sort_values(["pipeline_id", "patient_id", "pair_id"]), "retinal_pole_locations.csv")
    return poles


def hpatches_table() -> None:
    cases = pd.read_csv(HPATCHES)
    modern = cases[cases.pipeline_id.isin(PIPELINES)].copy()
    reference = cases[cases.pipeline_id.eq("ground_truth")][
        ["pair_id", "denominator_crosses_image"]
    ].rename(columns={"denominator_crosses_image": "reference_crosses_image"})
    modern = modern.merge(reference, on="pair_id", how="left", validate="many_to_one")
    modern["reference_crosses_image"] = modern.reference_crosses_image.fillna(False).astype(bool)
    valid = modern[~modern.reference_crosses_image]
    rows = []
    for pipeline, group in valid.groupby("pipeline_id"):
        returned = group[group.returned_homography.astype(bool)]
        poles = returned[returned.denominator_crosses_image.astype(bool)]
        high_error = returned[returned.mean_corner_error_px.gt(10)]
        rows.append(
            {
                "pipeline": pipeline,
                "pipeline_label": PIPELINES[pipeline],
                "valid_reference_pairs": int(group.pair_id.nunique()),
                "returned_homographies": int(len(returned)),
                "poles": int(len(poles)),
                "poles_over_10_px": int(poles.mean_corner_error_px.gt(10).sum()),
                "high_error_cases": int(len(high_error)),
                "high_error_cases_flagged": int(
                    high_error.denominator_crosses_image.astype(bool).sum()
                ),
            }
        )
    write_csv(pd.DataFrame(rows), "hpatches_valid_reference_summary.csv")
    excluded = modern[modern.reference_crosses_image]
    write_csv(
        excluded[
            [
                "pipeline_id",
                "pair_id",
                "denominator_crosses_image",
                "mean_corner_error_px",
            ]
        ],
        "hpatches_reference_pole_exclusion.csv",
    )


def megadepth_pairing() -> None:
    cases = pd.read_csv(MEGADEPTH)
    frame = cases[
        cases.dataset_id.eq("MegaDepth-1500") & cases.pipeline_id.isin(PIPELINES)
    ].copy()
    returned = frame[frame.returned_homography.astype(bool)]
    flags = returned.pivot_table(
        index="pair_id",
        columns="pipeline_id",
        values="denominator_crosses_image",
        aggfunc="first",
    ).dropna()
    for column in PIPELINES:
        flags[column] = flags[column].astype(bool)
    x_only = int((flags.xfeat_h & ~flags.sp_lg_h).sum())
    sp_only = int((~flags.xfeat_h & flags.sp_lg_h).sum())
    write_csv(
        pd.DataFrame(
            [
                {
                    "pairs_returned_by_both": len(flags),
                    "xfeat_only_poles": x_only,
                    "splg_only_poles": sp_only,
                }
            ]
        ),
        "megadepth_paired_poles.csv",
    )


def fundus_mask_audit(poles: pd.DataFrame) -> None:
    records = []
    structure = np.ones((7, 7), dtype=bool)
    for row in poles.itertuples(index=False):
        image_path = ROOT / "data" / f"{row.moving_image_id}.jpg"
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        mask = rgb.max(axis=2) > 10
        mask = ndimage.binary_closing(mask, structure=structure)
        mask = ndimage.binary_fill_holes(mask)
        labels, count = ndimage.label(mask)
        if count:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            mask = labels == sizes.argmax()
        matrix = np.asarray(json.loads(row.original_homography_json), dtype=float)
        yy, xx = np.nonzero(mask)
        denominator = matrix[2, 0] * xx + matrix[2, 1] * yy + matrix[2, 2]
        mask_crosses = bool(denominator.min() <= 0 <= denominator.max())
        records.append(
            {
                "job_id": row.job_id,
                "pipeline_id": row.pipeline_id,
                "pair_id": row.pair_id,
                "threshold": 10,
                "mask_fraction": float(mask.mean()),
                "pole_crosses_estimated_fundus_mask": mask_crosses,
                "pole_crosses_inscribed_circle": bool(
                    row.pole_crosses_inscribed_circle
                ),
                "mask_circle_agree": mask_crosses
                == bool(row.pole_crosses_inscribed_circle),
            }
        )
    write_csv(pd.DataFrame(records), "fundus_mask_pole_audit.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablation-only",
        action="store_true",
        help="summarize the pole-guard run without requiring image/benchmark assets",
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    treatment_and_brier_tables()
    clearance_table()
    if args.ablation_only:
        return
    splg_mechanism_table()
    saturation_and_warning_tables()
    poles = modern_retinal_poles()
    hpatches_table()
    megadepth_pairing()
    fundus_mask_audit(poles)
    (OUTPUT / "README.md").write_text(
        "# ISBI remaining-work analyses\n\n"
        "Generated by `scripts/isbi_remaining_analyses.py` from existing cached "
        "aggregate results. This includes treatment/calibration summaries and "
        "the empirical continuous-clearance audit; only the mask audit uses "
        "local de-identified COph100/RIDIRP images. No registration or detector "
        "fit is rerun.\n"
    )


if __name__ == "__main__":
    main()
