"""Versioned cache and release schemas (specification §12.4).

The release must let a third party fit *new predictors over the cached
features* without rerunning registration. That is only true if every table
below is complete and versioned, so the column lists here are the contract,
not documentation.

Known limitation, stated in the README and in any abstract that claims the
cache as a contribution: new *image* features, perturbation estimators, or
native uncertainty methods still require source images and additional compute.
"""

from __future__ import annotations

SCHEMA_VERSION = "1.1.0"

#: One row per (dataset_version, pair_id, pipeline_version, direction,
#: condition, severity, seed).
REGISTRATION_COLUMNS: tuple[str, ...] = (
    "job_id",
    "dataset_id",
    "dataset_version",
    "pair_id",
    "moving_image_id",
    "fixed_image_id",
    "group_id",
    "group_basis",
    "fold",
    "is_development",
    "sampling_probability",
    "pipeline_id",
    "pipeline_version",
    "direction",
    "condition",
    "severity",
    "seed",
    # coordinates
    "original_moving_hw",
    "original_fixed_hw",
    "working_moving_hw",
    "working_fixed_hw",
    "A_moving",
    "A_fixed",
    "valid_support_convention",
    "transform_family",
    "transform_params",
    "transform_checksum",
    # exact cached matcher output for B/E1 and future predictor features
    "matches_moving",
    "matches_fixed",
    "inlier_mask",
    "match_scores",
    # outcome
    "status",
    "diagnostic_code",
    "diagnostics",
    "explicit_failure",
    "n_matches",
    "n_inliers",
    "overlap_fraction",
    "runtime_s",
    "cpu_time_s",
    "peak_vram_bytes",
    # provenance
    "code_hash",
    "env_hash",
    "checkpoint_hash",
    "config_hash",
    "hardware",
    "timestamp",
)

#: Feature values live in their own table so a new feature version can be
#: added without rewriting registration rows.
FEATURE_COLUMNS: tuple[str, ...] = (
    "feature_id",
    "job_id",
    "family",
    "feature_name",
    "value",
    "available",
    "reason",
    "unit",
    "definition_version",
    "incremental_wall_s",
    "incremental_cpu_s",
    "feature_hash",
    "code_hash",
    "config_hash",
)

#: Separate labels table. Joined to predictions only in the evaluation
#: process, on immutable IDs, after predictions are written and hashed.
LABEL_COLUMNS: tuple[str, ...] = (
    "job_id",
    "dataset_id",
    "pair_id",
    "annotation_kind",
    "annotation_provenance",
    "n_landmarks",
    "n_evaluated",
    "tre_px",
    "tre_norm",
    "tre_median_px",
    "tre_max_px",
    "tre_p95_px",
    "tre_defined",
    "tre_reason",
    "tau",
    "silent_failure",
    "operational_failure",
    "eligible_for_acceptance",
    "bounded_loss",
    "landmark_success_fraction_12p5px",
)

#: Separate prediction table (§12.4).
PREDICTION_COLUMNS: tuple[str, ...] = (
    #: Primary key: one row per (job, arm, fold, experiment).
    "prediction_id",
    "job_id",
    "experiment_id",
    "protocol",  # P1..P6
    "arm",  # transferred | target_reference | single_score | control
    "fold",
    "training_groups_hash",
    "annotation_budget_groups",
    "annotation_budget_pairs",
    "learner_config_hash",
    "preprocessing_hash",
    "calibration_hash",
    "score",
    "probability",
    "threshold",
    "accepted",
    "policy_id",
)

TABLES: dict[str, tuple[str, ...]] = {
    "registrations": REGISTRATION_COLUMNS,
    "features": FEATURE_COLUMNS,
    "labels": LABEL_COLUMNS,
    "predictions": PREDICTION_COLUMNS,
}


def validate_columns(table: str, columns) -> list[str]:
    """Return required columns missing from ``columns``."""
    required = TABLES[table]
    present = set(columns)
    return [c for c in required if c not in present]
