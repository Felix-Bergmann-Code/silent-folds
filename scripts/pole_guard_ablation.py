#!/usr/bin/env python3
"""Pole-guard, sign-condition, and outside-training-range robustness audit.

This runner uses frozen registration/feature/label caches and never launches a
matcher.  It preserves case identifiers, reports pole labels and calibration
membership for the original plus 20 alternative assignments, repeats group and
single-pole deletions in the anomalous fold, and compares both prespecified
non-stability feature arms under the original curvature, training-range
clipping, curvature removal, an exact pole guard, and an FOV-restricted
curvature calculation. A 2x2 support study crosses rectangle- versus
FOV-sampled curvature with rectangle- versus FOV-domain guards. Each guard
marks bending energy missing whenever its exact crossing test fires. The
preprocessor's existing missingness flag records that event; the separately
exported pole flags are diagnostic columns and are not duplicated in the model
design.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts import development_decision as dd
from scripts import paper_study as ps
from scripts import publication_audit as audit
from scripts.conference_extension import assemble, select_arms
from warpaudit.config import load_config
from warpaudit.evaluation.metrics import auroc, brier_score
from warpaudit.evaluation.weights import inverse_group_size_weights
from warpaudit.geometry.projective import (
    projective_pole_crosses_circle,
    projective_pole_diagnostics,
)


def add_pole_guard_features(
    cases: pd.DataFrame, feature_names: tuple[str, ...]
) -> tuple[pd.DataFrame, str, str]:
    """Add guarded bending energy and an exact-pole indicator."""

    bending = [name for name in feature_names if name.endswith(":bending_energy")]
    if len(bending) != 1:
        raise ValueError(f"expected one bending-energy feature, found {bending}")
    source_name = bending[0]
    guarded_name = source_name + "_pole_guarded"
    indicator_name = "D:projective_pole_in_image"
    fov_indicator_name = "D:projective_pole_in_fov"
    fov_name = source_name + "_fov_restricted"
    frame = cases.copy()
    rectangle_flags: list[bool] = []
    circle_flags: list[bool] = []
    pole_clearances: list[float] = []
    fov_curvature: list[float] = []
    original_homographies: list[str | None] = []
    for record in frame.to_dict("records"):
        is_homography = str(record.get("transform_family")) == "homography"
        status_ok = str(record.get("status")) == "ok"
        if not (is_homography and status_ok):
            rectangle_flags.append(False)
            circle_flags.append(False)
            pole_clearances.append(np.nan)
            fov_curvature.append(np.nan)
            original_homographies.append(None)
            continue
        params = record.get("transform_params")
        payload = json.loads(params) if isinstance(params, str) else params
        matrix = np.asarray(payload["matrix"], dtype=float)
        height, width = map(int, record["working_moving_hw"])
        fixed_height, fixed_width = map(int, record["working_fixed_hw"])
        pole_diagnostics = projective_pole_diagnostics(
            matrix, width=width, height=height
        )
        rectangle_flags.append(pole_diagnostics.denominator_crosses_image)
        pole_clearances.append(
            pole_diagnostics.pole_clearance_diagonal_fraction
        )
        circle_flags.append(
            projective_pole_crosses_circle(
                matrix,
                center_x=(width - 1.0) / 2.0,
                center_y=(height - 1.0) / 2.0,
                radius=min(width - 1.0, height - 1.0) / 2.0,
            )
        )
        fov_curvature.append(
            fov_restricted_bending_energy(
                matrix,
                moving_width=width,
                moving_height=height,
                fixed_width=fixed_width,
                fixed_height=fixed_height,
            )
        )
        a_moving = np.stack(record["A_moving"]).astype(float)
        a_fixed = np.stack(record["A_fixed"]).astype(float)
        original = np.linalg.inv(a_fixed) @ matrix @ a_moving
        original_homographies.append(json.dumps(original.tolist(), separators=(",", ":")))
    frame["pole_crosses_image"] = rectangle_flags
    frame["pole_crosses_inscribed_circle"] = circle_flags
    frame["pole_clearance_diagonal_fraction"] = pole_clearances
    frame[guarded_name] = frame[source_name].mask(frame.pole_crosses_image)
    frame[indicator_name] = frame.pole_crosses_image.astype(float)
    frame[fov_name] = fov_curvature
    frame[fov_indicator_name] = frame.pole_crosses_inscribed_circle.astype(float)
    frame[source_name + "_fov_pole_guarded"] = frame[source_name].mask(
        frame.pole_crosses_inscribed_circle
    )
    frame[fov_name + "_rectangle_guarded"] = frame[fov_name].mask(
        frame.pole_crosses_image
    )
    frame[fov_name + "_fov_guarded"] = frame[fov_name].mask(
        frame.pole_crosses_inscribed_circle
    )
    frame["original_homography_json"] = original_homographies
    return frame, guarded_name, indicator_name


def fov_restricted_bending_energy(
    matrix: np.ndarray,
    *,
    moving_width: int,
    moving_height: int,
    fixed_width: int,
    fixed_height: int,
    size: int = 32,
) -> float:
    """Compute curvature where every second-difference stencil stays in the FOV."""

    xs = (np.arange(size, dtype=float) + 0.5) * (moving_width / size) - 0.5
    ys = (np.arange(size, dtype=float) + 0.5) * (moving_height / size) - 0.5
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.c_[xx.ravel(), yy.ravel(), np.ones(size * size)]
    homogeneous = points @ np.asarray(matrix, dtype=float).T
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        mapped = homogeneous[:, :2] / homogeneous[:, 2, None]
    moving_diagonal = float(np.hypot(moving_width, moving_height))
    fixed_diagonal = float(np.hypot(fixed_width, fixed_height))
    source_n = (points[:, :2] / moving_diagonal).reshape(size, size, 2)
    mapped_n = (mapped / fixed_diagonal).reshape(size, size, 2)
    d_col = float(source_n[0, 1, 0] - source_n[0, 0, 0])
    d_row = float(source_n[1, 0, 1] - source_n[0, 0, 1])
    dv_drow = np.gradient(mapped_n, d_row, axis=0)
    dv_dcol = np.gradient(mapped_n, d_col, axis=1)
    d2_row = np.gradient(dv_drow, d_row, axis=0)
    d2_col = np.gradient(dv_dcol, d_col, axis=1)
    d2_cross = np.gradient(dv_drow, d_col, axis=1)
    energy = d2_row**2 + 2.0 * d2_cross**2 + d2_col**2
    center_x, center_y = (moving_width - 1.0) / 2.0, (moving_height - 1.0) / 2.0
    radius = min(moving_width - 1.0, moving_height - 1.0) / 2.0
    # Two applications of ``gradient`` can reach two lattice steps away.  The
    # eroded support prevents values outside the retinal proxy from entering an
    # otherwise in-FOV curvature estimate through a finite-difference stencil.
    step = max(moving_width / size, moving_height / size)
    stencil_radius = max(0.0, radius - 2.0 * step)
    inside = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= stencil_radius**2
    values = energy[inside[..., None] & np.isfinite(energy)]
    return float(values.mean()) if values.size else np.nan


def guarded_columns(
    columns: tuple[str, ...], source_name: str, guarded_name: str, indicator_name: str
) -> tuple[str, ...]:
    """Replace curvature by its guarded value without a redundant indicator.

    ``SourcePreprocessor`` already appends one missingness flag for every input
    feature.  A finite curvature value becomes missing exactly when the guard
    fires, so adding ``indicator_name`` to the input columns would duplicate
    that flag in the design matrix.  The named pole indicator remains on the
    exported case table for diagnostics and traceability.
    """

    del indicator_name
    if source_name not in columns:
        raise ValueError(f"arm does not contain {source_name}")
    return tuple(guarded_name if name == source_name else name for name in columns)


def replace_column(
    columns: tuple[str, ...], source_name: str, replacement_name: str
) -> tuple[str, ...]:
    if source_name not in columns:
        raise ValueError(f"arm does not contain {source_name}")
    return tuple(replacement_name if name == source_name else name for name in columns)


def treatment_frames(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    *,
    treatment: str,
    columns: tuple[str, ...],
    source_name: str,
    guarded_name: str,
    indicator_name: str,
    train_min: float,
    train_max: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Return split-local feature treatment without mutating the case table."""

    if treatment == "original":
        return train, calibration, test, columns
    if treatment == "curvature_removed":
        reduced = tuple(name for name in columns if name != source_name)
        return train, calibration, test, reduced
    if treatment == "pole_guard":
        guarded = guarded_columns(columns, source_name, guarded_name, indicator_name)
        return train, calibration, test, guarded
    if treatment == "fov_curvature":
        fov_name = source_name + "_fov_restricted"
        return train, calibration, test, replace_column(columns, source_name, fov_name)
    if treatment == "fov_guard_on_rectangle":
        guarded = guarded_columns(
            columns,
            source_name,
            source_name + "_fov_pole_guarded",
            "D:projective_pole_in_fov",
        )
        return train, calibration, test, guarded
    if treatment == "rectangle_guard_on_fov":
        fov_name = source_name + "_fov_restricted"
        guarded = guarded_columns(
            columns,
            source_name,
            fov_name + "_rectangle_guarded",
            indicator_name,
        )
        return train, calibration, test, guarded
    if treatment == "fov_guard_on_fov":
        fov_name = source_name + "_fov_restricted"
        guarded = guarded_columns(
            columns,
            source_name,
            fov_name + "_fov_guarded",
            "D:projective_pole_in_fov",
        )
        return train, calibration, test, guarded
    if treatment == "curvature_clipped":
        clipped_name = source_name + "_training_range_clipped"
        frames = []
        for frame in (train, calibration, test):
            copy_frame = frame.copy()
            copy_frame[clipped_name] = pd.to_numeric(
                copy_frame[source_name], errors="coerce"
            ).clip(train_min, train_max)
            frames.append(copy_frame)
        return (*frames, replace_column(columns, source_name, clipped_name))
    raise ValueError(f"unknown treatment {treatment!r}")


def curvature_coefficient(model, columns: tuple[str, ...], feature: str) -> float:
    if feature not in columns:
        return np.nan
    index = model.preprocessor.feature_names.index(feature)
    return float(model.model.coef_[0, index])


_POLE_WORKER_STATE: tuple | None = None


def _initialize_pole_worker(
    cases: pd.DataFrame,
    feature_arms: dict[str, tuple[str, ...]],
    design: dict,
    cfg,
    source_name: str,
    guarded_name: str,
    indicator_name: str,
) -> None:
    global _POLE_WORKER_STATE
    _POLE_WORKER_STATE = (
        cases,
        feature_arms,
        design,
        cfg,
        source_name,
        guarded_name,
        indicator_name,
    )


def _run_assignment(seed: int) -> tuple[list[dict[str, object]], ...]:
    if _POLE_WORKER_STATE is None:
        raise RuntimeError("pole worker was not initialized")
    (
        cases,
        feature_arms,
        design,
        cfg,
        source_name,
        guarded_name,
        indicator_name,
    ) = _POLE_WORKER_STATE
    rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    pole_split_rows: list[dict[str, object]] = []
    deletion_rows: list[dict[str, object]] = []
    case_deletion_rows: list[dict[str, object]] = []
    original_seed = cfg.splits.seed
    local = copy.deepcopy(cfg)
    local.splits.seed = seed
    for pipeline in design["primary_pipelines"]:
        for fold in range(5):
            train, calibration, test = dd.split_cell(
                cases, local, design, "within_pipeline", pipeline, fold
            )
            train_values = pd.to_numeric(train[source_name], errors="coerce")
            train_values = train_values[np.isfinite(train_values)]
            train_min = float(train_values.min())
            train_max = float(train_values.max())
            cal_poles = calibration[calibration.pole_crosses_image]
            cal_values = pd.to_numeric(cal_poles[source_name], errors="coerce")
            outside = (cal_values < train_min) | (cal_values > train_max)
            for partition_name, partition in (
                ("train", train),
                ("calibration", calibration),
                ("test", test),
            ):
                partition_poles = partition[partition.pole_crosses_image]
                for pole in partition_poles.to_dict("records"):
                    value = float(pole[source_name])
                    pole_split_rows.append(
                        {
                            "seed": seed,
                            "pipeline": pipeline,
                            "fold": fold,
                            "partition": partition_name,
                            "job_id": pole["job_id"],
                            "pair_id": pole["pair_id"],
                            "group_id": pole["group_id"],
                            "dataset_id": pole["dataset_id"],
                            "operational_failure": bool(pole["operational_failure"]),
                            "pole_crosses_inscribed_circle": bool(
                                pole["pole_crosses_inscribed_circle"]
                            ),
                            "bending_energy": value,
                            "training_bending_min": train_min,
                            "training_bending_max": train_max,
                            "outside_training_range": bool(
                                value < train_min or value > train_max
                            ),
                        }
                    )
            common = {
                "seed": seed,
                "pipeline": pipeline,
                "fold": fold,
                "calibration_poles": len(cal_poles),
                "calibration_pole_failures": int(
                    cal_poles.operational_failure.sum()
                ),
                "calibration_poles_outside_training_range": int(outside.sum()),
                "training_bending_min": train_min,
                "training_bending_max": train_max,
            }
            for feature_arm, base_columns in feature_arms.items():
                for treatment in (
                    "original",
                    "curvature_clipped",
                    "curvature_removed",
                    "pole_guard",
                    "fov_curvature",
                    "fov_guard_on_rectangle",
                    "rectangle_guard_on_fov",
                    "fov_guard_on_fov",
                ):
                    tr, ca, te, columns = treatment_frames(
                        train,
                        calibration,
                        test,
                        treatment=treatment,
                        columns=base_columns,
                        source_name=source_name,
                        guarded_name=guarded_name,
                        indicator_name=indicator_name,
                        train_min=train_min,
                        train_max=train_max,
                    )
                    with threadpool_limits(limits=1):
                        model, _, _ = ps.fit_arm(
                            tr, ca, columns, local, f"{feature_arm}:{treatment}"
                        )
                    coefficient_feature = (
                        source_name + "_training_range_clipped"
                        if treatment == "curvature_clipped"
                        else source_name
                    )
                    coefficient = curvature_coefficient(
                        model, columns, coefficient_feature
                    )
                    model_variants = [(treatment, model)]
                    if treatment == "original":
                        isotonic = copy.deepcopy(model)
                        with threadpool_limits(limits=1):
                            isotonic.calibrate(
                                ca.loc[:, columns],
                                ca.operational_failure.to_numpy(float),
                                method="isotonic",
                                sample_weight=inverse_group_size_weights(
                                    ca.group_id.astype(str)
                                ),
                            )
                        model_variants.append(("original_isotonic", isotonic))
                    for variant_name, variant_model in model_variants:
                        diagnostics = audit.model_diagnostics(
                            variant_model, te, columns
                        )
                        rows.append(
                            {
                                **common,
                                "feature_arm": feature_arm,
                                "treatment": variant_name,
                                "training_pole_cases": int(
                                    train.pole_crosses_image.sum()
                                ),
                                "curvature_coefficient": coefficient,
                                "curvature_coefficient_sign": (
                                    "negative"
                                    if coefficient < 0
                                    else "positive"
                                    if coefficient > 0
                                    else "zero_or_absent"
                                ),
                                "pole_indicator_varies_in_training": bool(
                                    train.pole_crosses_image.nunique() > 1
                                ),
                                **diagnostics,
                            }
                        )
                        raw = variant_model.decision_function(te.loc[:, columns])
                        probability = variant_model.predict_probability(
                            te.loc[:, columns]
                        )
                        for record, raw_value, probability_value in zip(
                            te.to_dict("records"), raw, probability, strict=True
                        ):
                            prediction_rows.append(
                                {
                                    "seed": seed,
                                    "pipeline": pipeline,
                                    "fold": fold,
                                    "feature_arm": feature_arm,
                                    "treatment": variant_name,
                                    "job_id": record["job_id"],
                                    "group_id": record["group_id"],
                                    "operational_failure": bool(
                                        record["operational_failure"]
                                    ),
                                    "raw_score": float(raw_value),
                                    "probability": float(probability_value),
                                }
                            )
                    if not (
                        seed == original_seed
                        and pipeline == "xfeat_h"
                        and fold == 3
                        and treatment == "original"
                    ):
                        continue
                    for deletion_index, group_id in enumerate(
                        sorted(calibration.group_id.astype(str).unique())
                    ):
                        subset = calibration[
                            calibration.group_id.astype(str) != group_id
                        ]
                        record = {
                            "seed": seed,
                            "pipeline": pipeline,
                            "fold": fold,
                            "feature_arm": feature_arm,
                            "deleted_group_index": deletion_index,
                            "deleted_group_id": group_id,
                            "deleted_group_contains_pole": bool(
                                cal_poles.group_id.astype(str).eq(group_id).any()
                            ),
                            "deleted_group_pole_failures": int(
                                cal_poles.loc[
                                    cal_poles.group_id.astype(str).eq(group_id),
                                    "operational_failure",
                                ].sum()
                            ),
                            "remaining_groups": subset.group_id.nunique(),
                        }
                        deletion_rows.append(
                            recalibration_deletion(
                                model, subset, test, base_columns, record
                            )
                        )
                    pole_ids = cal_poles.job_id.astype(str).tolist()
                    deletion_sets = [(job_id, [job_id]) for job_id in pole_ids]
                    deletion_sets.append(("both_poles", pole_ids))
                    for label, deleted_ids in deletion_sets:
                        subset = calibration[
                            ~calibration.job_id.astype(str).isin(deleted_ids)
                        ]
                        case_deletion_rows.append(
                            recalibration_deletion(
                                model,
                                subset,
                                test,
                                base_columns,
                                {
                                    "seed": seed,
                                    "pipeline": pipeline,
                                    "fold": fold,
                                    "feature_arm": feature_arm,
                                    "deleted_case": label,
                                    "deleted_case_count": len(deleted_ids),
                                    "deleted_case_crosses_fov": bool(
                                        cal_poles.loc[
                                            cal_poles.job_id.astype(str).isin(
                                                deleted_ids
                                            ),
                                            "pole_crosses_inscribed_circle",
                                        ].any()
                                    ),
                                },
                            )
                        )
    return (
        rows,
        prediction_rows,
        pole_split_rows,
        deletion_rows,
        case_deletion_rows,
    )


def run(output: Path, workers: int = 1) -> None:
    cfg = load_config(ROOT / "configs/full_study.yaml")
    design = json.loads((ROOT / "configs/paper_study.json").read_text())
    spec = json.loads((ROOT / "configs/conference_extension.json").read_text())
    cases, names = assemble(cfg, ROOT, cfg.full_study.development_datasets)
    selected = select_arms(names, spec)
    cases, guarded_name, indicator_name = add_pole_guard_features(cases, names)
    source_name = guarded_name.removesuffix("_pole_guarded")
    feature_arms = {
        name: selected[name] for name in ("non_stability", "non_stability_e1_e2")
    }

    output.mkdir(parents=True, exist_ok=True)
    pole_columns = [
        "job_id",
        "pair_id",
        "group_id",
        "dataset_id",
        "pipeline_id",
        "operational_failure",
        "pole_crosses_image",
        "pole_clearance_diagonal_fraction",
        "pole_crosses_inscribed_circle",
        source_name,
        "moving_image_id",
        "original_homography_json",
    ]
    dd.atomic_write_text(
        output / "pole_cases.csv",
        cases.loc[cases.pole_crosses_image, pole_columns].to_csv(index=False),
    )
    clearance_columns = [
        "job_id",
        "pair_id",
        "group_id",
        "dataset_id",
        "pipeline_id",
        "operational_failure",
        "pole_crosses_image",
        "pole_clearance_diagonal_fraction",
        source_name,
    ]
    dd.atomic_write_text(
        output / "pole_clearance.csv",
        cases.loc[cases.status.astype(str).eq("ok"), clearance_columns].to_csv(
            index=False
        ),
    )

    rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    pole_split_rows: list[dict[str, object]] = []
    deletion_rows: list[dict[str, object]] = []
    case_deletion_rows: list[dict[str, object]] = []
    original_seed = cfg.splits.seed
    seeds = range(original_seed, original_seed + 21)
    initializer_args = (
        cases,
        feature_arms,
        design,
        cfg,
        source_name,
        guarded_name,
        indicator_name,
    )
    with threadpool_limits(limits=1):
        if workers == 1:
            _initialize_pole_worker(*initializer_args)
            results = map(_run_assignment, seeds)
        else:
            pool = ThreadPoolExecutor(
                max_workers=workers,
                initializer=_initialize_pole_worker,
                initargs=initializer_args,
            )
            results = pool.map(_run_assignment, seeds)
        try:
            for index, result in enumerate(results, start=1):
                for destination, values in zip(
                    (
                        rows,
                        prediction_rows,
                        pole_split_rows,
                        deletion_rows,
                        case_deletion_rows,
                    ),
                    result,
                    strict=True,
                ):
                    destination.extend(values)
                print(f"[{index}/21] completed assignment", flush=True)
        finally:
            if workers != 1:
                pool.shutdown()
    frame = pd.DataFrame(rows)
    dd.atomic_write_text(output / "split_diagnostics.csv", frame.to_csv(index=False))
    dd.atomic_write_text(
        output / "pole_split_membership.csv",
        pd.DataFrame(pole_split_rows).to_csv(index=False),
    )
    dd.atomic_write_text(
        output / "calibration_group_deletion.csv",
        pd.DataFrame(deletion_rows).to_csv(index=False),
    )
    dd.atomic_write_text(
        output / "calibration_case_deletion.csv",
        pd.DataFrame(case_deletion_rows).to_csv(index=False),
    )
    predictions = pd.DataFrame(prediction_rows)
    dd.atomic_write_text(
        output / "test_predictions.csv", predictions.to_csv(index=False)
    )
    assignment_rows = []
    for keys, part in predictions.groupby(
        ["seed", "pipeline", "feature_arm", "treatment"], sort=True
    ):
        weights = inverse_group_size_weights(part.group_id.astype(str))
        assignment_rows.append(
            {
                "seed": keys[0],
                "pipeline": keys[1],
                "feature_arm": keys[2],
                "treatment": keys[3],
                "pooled_raw_auroc": auroc(
                    part.raw_score, part.operational_failure, weights
                ),
                "pooled_calibrated_auroc": auroc(
                    part.probability, part.operational_failure, weights
                ),
                "pooled_brier": brier_score(
                    part.probability, part.operational_failure, weights
                ),
            }
        )
    assignment_metrics = pd.DataFrame(assignment_rows)
    dd.atomic_write_text(
        output / "assignment_metrics.csv", assignment_metrics.to_csv(index=False)
    )
    summary = (
        frame.groupby(["pipeline", "feature_arm", "treatment"], as_index=False)
        .agg(
            assignments=("seed", "nunique"),
            folds=("fold", "size"),
            negative_slope_folds=("ranking_reversed_by_calibration", "sum"),
            calibration_pole_folds=("calibration_poles", lambda x: int((x > 0).sum())),
            outside_range_pole_folds=(
                "calibration_poles_outside_training_range",
                lambda x: int((x > 0).sum()),
            ),
            negative_curvature_coefficient_folds=(
                "curvature_coefficient", lambda x: int((x < 0).sum())
            ),
        )
    )
    sign_rows = []
    original_rows = frame[
        (frame.treatment == "original") & (frame.calibration_poles > 0)
    ]
    for keys, part in original_rows.groupby(
        ["pipeline", "feature_arm"], sort=True
    ):
        negative = part.curvature_coefficient < 0
        reversed_map = part.ranking_reversed_by_calibration.astype(bool)
        sign_rows.append(
            {
                "pipeline": keys[0],
                "feature_arm": keys[1],
                "negative_curvature_coefficient_pole_folds": int(negative.sum()),
                "sign_matches_reversal_pole_folds": int(
                    (negative == reversed_map).sum()
                ),
                "pole_folds_for_sign_check": len(part),
            }
        )
    summary = summary.merge(
        pd.DataFrame(sign_rows),
        on=["pipeline", "feature_arm"],
        how="left",
        validate="many_to_one",
    )
    dd.atomic_write_text(output / "summary.csv", summary.to_csv(index=False))


def warning_audit(output: Path) -> dict[str, object]:
    """Refit the previously warning-bearing fold and record current diagnostics."""

    cfg = load_config(ROOT / "configs/full_study.yaml")
    design = json.loads((ROOT / "configs/paper_study.json").read_text())
    spec = json.loads((ROOT / "configs/conference_extension.json").read_text())
    cases, names = assemble(cfg, ROOT, cfg.full_study.development_datasets)
    columns = select_arms(names, spec)["non_stability"]
    local = copy.deepcopy(cfg)
    local.splits.seed = 20260915
    train, calibration, test = dd.split_cell(
        cases, local, design, "within_pipeline", "xfeat_h", 3
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with threadpool_limits(limits=1):
            model, _, _ = ps.fit_arm(
                train, calibration, columns, local, "optimizer-warning-audit"
            )
    warning_rows = [
        {
            "category": item.category.__name__,
            "message": str(item.message),
        }
        for item in caught
    ]
    diagnostics = audit.model_diagnostics(model, test, columns)
    payload: dict[str, object] = {
        "seed": 20260915,
        "pipeline": "xfeat_h",
        "fold": 3,
        "feature_arm": "non_stability",
        "fit_implementation": "current LogisticRegression max_iter=2000",
        "warning_free": not warning_rows,
        "warnings": warning_rows,
        "model_iterations": model.model.n_iter_.tolist(),
        "calibrator_iterations": model.calibrator.n_iter_.tolist(),
        "selected_C": model.selected_C,
        "diagnostics": diagnostics,
    }
    output.mkdir(parents=True, exist_ok=True)
    dd.atomic_write_text(
        output / "optimizer_warning_audit.json",
        json.dumps(dd.clean_json(payload), indent=2, sort_keys=True) + "\n",
    )
    return payload


def recalibration_deletion(
    model,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    columns: tuple[str, ...],
    record: dict[str, object],
) -> dict[str, object]:
    if calibration.operational_failure.nunique() != 2:
        return {**record, "status": "not_estimable", "reason": "single class"}
    candidate = copy.deepcopy(model)
    with threadpool_limits(limits=1):
        candidate.calibrate(
            calibration.loc[:, columns],
            calibration.operational_failure.to_numpy(float),
            method="platt",
            sample_weight=inverse_group_size_weights(
                calibration.group_id.astype(str)
            ),
        )
    return {
        **record,
        "status": "ok",
        **audit.model_diagnostics(candidate, test, columns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/pole_guard_ablation_latest")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--warning-audit-only",
        action="store_true",
        help="refit only seed 20260915, XFeat-H Base, fold 3 and capture warnings",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be 1..16")
    if args.warning_audit_only:
        warning_audit(ROOT / args.output)
    else:
        run(ROOT / args.output, workers=args.workers)


if __name__ == "__main__":
    main()
