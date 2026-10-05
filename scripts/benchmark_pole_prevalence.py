#!/usr/bin/env python3
"""Run label-free projective-pole prevalence experiments on image benchmarks.

The runner deliberately does not download data and never evaluates matching
accuracy.  It consumes image pairs, runs the already configured matcher and
shared RANSAC/DLT adapters, and applies the exact four-corner pole test.  An
optional HPatches ground-truth homography is audited as a separate ``GT``
control without exposing it to a matcher.

Outputs are one atomic JSON shard per pair/pipeline (safe to resume), a flat
``pole_cases.csv``, ``pole_summary.csv``, run metadata, and the two-panel figure
used by the paper.  Source images and external labels are never copied.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image

from warpaudit.cache.hashing import file_digest, short_hash
from warpaudit.cache.store import atomic_write_text
from warpaudit.config import Config, PipelineConfig, load_config
from warpaudit.geometry.coordinates import homography_original_to_working, make_frame
from warpaudit.geometry.projective import (
    HOMOGRAPHY_NORMALIZATION,
    image_corners,
    projective_pole_crosses_circle,
    projective_pole_diagnostics,
)
from warpaudit.registration.adapters import load_registrar
from warpaudit.types import (
    GROUP_BASIS_PRIORITY,
    CoordinateMetadata,
    PairInput,
    RegistrationStatus,
)

ANALYSIS_REVISION = "exact-corner-v2-fov-and-error"
DEFAULT_PIPELINES = ("xfeat_h", "sp_lg_h", "sift_h")
REQUIRED_COLUMNS = (
    "dataset_id",
    "pair_id",
    "group_id",
    "group_basis",
    "moving_path",
    "fixed_path",
    "moving_h",
    "moving_w",
    "fixed_h",
    "fixed_w",
    "moving_sha256",
    "fixed_sha256",
)
IMPLEMENTATION_HASH = short_hash(
    {
        "runner": file_digest(Path(__file__)),
        "geometry": file_digest(ROOT / "warpaudit/geometry/projective.py"),
    },
    length=32,
)


def _image_hw(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return int(height), int(width)


def _read_homography(path: Path) -> np.ndarray:
    try:
        matrix = np.loadtxt(path, dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read homography {path}: {exc}") from exc
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{path}: expected a finite 3x3 homography, got {matrix.shape}")
    if abs(float(np.linalg.det(matrix))) < 1e-15:
        raise ValueError(f"{path}: ground-truth homography is singular")
    return matrix


def _write_manifest(rows: list[dict[str, Any]], output: Path) -> Path:
    if not rows:
        raise ValueError("benchmark index would be empty")
    frame = pd.DataFrame(rows).sort_values(["dataset_id", "pair_id"], kind="stable")
    duplicate = frame.duplicated(["dataset_id", "pair_id"], keep=False)
    if duplicate.any():
        keys = frame.loc[duplicate, ["dataset_id", "pair_id"]].to_dict("records")
        raise ValueError(f"duplicate benchmark pair identifiers: {keys[:5]}")
    atomic_write_text(output, frame.to_csv(index=False))
    return output


def index_hpatches(root: Path, output: Path, dataset_id: str = "HPatches") -> Path:
    """Index the standard 116-sequence, image-1-to-image-i HPatches layout."""

    root = root.resolve()
    # Accept either the extracted archive root or its sequences-release child.
    candidates = [root, root / "hpatches-sequences-release"]
    base = next((path for path in candidates if any(path.glob("*/1.ppm"))), None)
    if base is None:
        raise FileNotFoundError(
            f"HPatches layout not found below {root}; expected '<sequence>/1.ppm' and H_1_i"
        )
    digest_cache: dict[Path, str] = {}

    def digest(path: Path) -> str:
        if path not in digest_cache:
            digest_cache[path] = file_digest(path)
        return digest_cache[path]

    rows: list[dict[str, Any]] = []
    for sequence in sorted(path for path in base.iterdir() if path.is_dir()):
        reference = sequence / "1.ppm"
        if not reference.is_file():
            continue
        for index in range(2, 7):
            fixed = sequence / f"{index}.ppm"
            homography = sequence / f"H_1_{index}"
            if not fixed.is_file() or not homography.is_file():
                raise FileNotFoundError(
                    f"incomplete HPatches sequence {sequence.name}: expected {fixed.name} and "
                    f"{homography.name}"
                )
            _read_homography(homography)
            moving_hw, fixed_hw = _image_hw(reference), _image_hw(fixed)
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "pair_id": f"{sequence.name}/1-{index}",
                    "group_id": sequence.name,
                    "group_basis": "sequence",
                    "moving_path": str(reference.resolve()),
                    "fixed_path": str(fixed.resolve()),
                    "moving_h": moving_hw[0],
                    "moving_w": moving_hw[1],
                    "fixed_h": fixed_hw[0],
                    "fixed_w": fixed_hw[1],
                    "moving_sha256": digest(reference),
                    "fixed_sha256": digest(fixed),
                    "ground_truth_homography_path": str(homography.resolve()),
                    "ground_truth_homography_sha256": digest(homography),
                }
            )
    return _write_manifest(rows, output)


def index_pairs(root: Path, pairs: Path, output: Path, dataset_id: str) -> Path:
    """Canonicalize a CSV pair list for MegaDepth/ScanNet/another benchmark.

    Required input columns are ``moving_path`` and ``fixed_path``.  Optional
    columns are ``pair_id``, ``group_id``, ``group_basis`` and
    ``ground_truth_homography_path``.  Relative paths are resolved below
    ``root``.  No failure or correctness labels are accepted or needed.
    """

    source = pd.read_csv(pairs)
    missing = {"moving_path", "fixed_path"} - set(source.columns)
    if missing:
        raise ValueError(f"{pairs}: missing columns {sorted(missing)}")
    root = root.resolve()
    digest_cache: dict[Path, str] = {}

    def resolve(value: Any) -> Path:
        path = Path(str(value))
        return (path if path.is_absolute() else root / path).resolve()

    def digest(path: Path) -> str:
        if path not in digest_cache:
            digest_cache[path] = file_digest(path)
        return digest_cache[path]

    rows: list[dict[str, Any]] = []
    for number, record in source.iterrows():
        moving, fixed = resolve(record.moving_path), resolve(record.fixed_path)
        if not moving.is_file() or not fixed.is_file():
            raise FileNotFoundError(f"pair row {number}: absent image {moving} or {fixed}")
        moving_hw, fixed_hw = _image_hw(moving), _image_hw(fixed)
        pair_value = record.get("pair_id")
        pair_id = (
            str(pair_value).strip()
            if pd.notna(pair_value) and str(pair_value).strip()
            else f"pair-{number:06d}"
        )
        group_value = record.get("group_id")
        group_id = (
            str(group_value).strip()
            if pd.notna(group_value) and str(group_value).strip()
            else pair_id
        )
        basis_value = record.get("group_basis")
        group_basis = (
            str(basis_value).strip()
            if pd.notna(basis_value) and str(basis_value).strip()
            else "image_component"
        )
        row: dict[str, Any] = {
            "dataset_id": dataset_id,
            "pair_id": pair_id,
            "group_id": group_id,
            "group_basis": group_basis,
            "moving_path": str(moving),
            "fixed_path": str(fixed),
            "moving_h": moving_hw[0],
            "moving_w": moving_hw[1],
            "fixed_h": fixed_hw[0],
            "fixed_w": fixed_hw[1],
            "moving_sha256": digest(moving),
            "fixed_sha256": digest(fixed),
            "ground_truth_homography_path": "",
            "ground_truth_homography_sha256": "",
        }
        gt_value = record.get("ground_truth_homography_path")
        if pd.notna(gt_value) and str(gt_value).strip():
            gt = resolve(gt_value)
            _read_homography(gt)
            row["ground_truth_homography_path"] = str(gt)
            row["ground_truth_homography_sha256"] = digest(gt)
        rows.append(row)
    return _write_manifest(rows, output)


def load_manifests(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path).fillna("")
        missing = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing manifest columns {sorted(missing)}")
        frame["manifest_path"] = str(path.resolve())
        frames.append(frame)
    if not frames:
        raise ValueError("at least one manifest is required")
    result = pd.concat(frames, ignore_index=True)
    duplicate = result.duplicated(["dataset_id", "pair_id"], keep=False)
    if duplicate.any():
        raise ValueError("dataset_id/pair_id must be unique across all supplied manifests")
    invalid_bases = sorted(set(result["group_basis"].astype(str)) - set(GROUP_BASIS_PRIORITY))
    if invalid_bases:
        raise ValueError(
            f"unsupported group_basis value(s) {invalid_bases}; expected one of "
            f"{list(GROUP_BASIS_PRIORITY)}"
        )
    return result.sort_values(["dataset_id", "pair_id"], kind="stable").reset_index(drop=True)


def verify_manifest_inputs(frame: pd.DataFrame) -> None:
    expected: dict[Path, str] = {}
    for record in frame.to_dict("records"):
        for path_key, hash_key in (
            ("moving_path", "moving_sha256"),
            ("fixed_path", "fixed_sha256"),
            ("ground_truth_homography_path", "ground_truth_homography_sha256"),
        ):
            value = str(record.get(path_key) or "")
            if not value:
                continue
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(path)
            declared = str(record.get(hash_key) or "")
            if not declared:
                raise ValueError(f"{path_key} has no {hash_key}; regenerate the benchmark index")
            if path in expected and expected[path] != declared:
                raise ValueError(f"conflicting hashes declared for {path}")
            expected[path] = declared
    for path, declared in expected.items():
        actual = file_digest(path)
        if actual != declared:
            raise ValueError(
                f"input changed since indexing: {path}; expected {declared}, got {actual}"
            )


def pair_from_record(record: dict[str, Any], cfg: Config) -> PairInput:
    moving_hw = (int(record["moving_h"]), int(record["moving_w"]))
    fixed_hw = (int(record["fixed_h"]), int(record["fixed_w"]))
    moving_id = f"{record['dataset_id']}/{record['pair_id']}/moving"
    fixed_id = f"{record['dataset_id']}/{record['pair_id']}/fixed"
    moving = make_frame(
        moving_id,
        moving_hw,
        long_edge=cfg.geometry.working_long_edge,
        pad_to_square=cfg.geometry.pad_to_square,
    )
    fixed = make_frame(
        fixed_id,
        fixed_hw,
        long_edge=cfg.geometry.working_long_edge,
        pad_to_square=cfg.geometry.pad_to_square,
    )
    return PairInput(
        dataset_id=str(record["dataset_id"]),
        pair_id=str(record["pair_id"]),
        group_id=str(record.get("group_id") or record["pair_id"]),
        group_basis=str(record.get("group_basis") or "image_component"),  # type: ignore[arg-type]
        moving_image_id=moving_id,
        fixed_image_id=fixed_id,
        moving_path=Path(str(record["moving_path"])),
        fixed_path=Path(str(record["fixed_path"])),
        coordinates=CoordinateMetadata(
            moving=moving,
            fixed=fixed,
            pixel_center_convention=cfg.geometry.pixel_center_convention,
            interpolation=cfg.geometry.interpolation,
            padding_mode=cfg.geometry.padding_mode,
        ),
    )


def _job_id(
    record: dict[str, Any], pipeline: PipelineConfig | None, cfg: Config, seed: int, grid: int
) -> str:
    identity = {
        "analysis_revision": ANALYSIS_REVISION,
        "implementation_hash": IMPLEMENTATION_HASH,
        "dataset_id": record["dataset_id"],
        "pair_id": record["pair_id"],
        "moving_sha256": record["moving_sha256"],
        "fixed_sha256": record["fixed_sha256"],
        "gt_sha256": record.get("ground_truth_homography_sha256", ""),
        "pipeline": "ground_truth" if pipeline is None else asdict(pipeline),
        "geometry": asdict(cfg.geometry),
        "seed": int(seed),
        "magnitude_grid_size": int(grid),
    }
    return short_hash(identity, length=32)


def _analyse_matrix(
    matrix: np.ndarray, pair: PairInput, *, magnitude_grid_size: int
) -> dict[str, Any]:
    moving = pair.coordinates.moving
    pole = projective_pole_diagnostics(
        matrix,
        width=moving.working_width,
        height=moving.working_height,
        magnitude_grid_size=magnitude_grid_size,
    )
    center_x = (moving.working_width - 1.0) / 2.0
    center_y = (moving.working_height - 1.0) / 2.0
    radius = min(moving.working_width - 1.0, moving.working_height - 1.0) / 2.0
    return {
        "returned_homography": True,
        "denominator_crosses_image": pole.denominator_crosses_image,
        "denominator_crosses_inscribed_circle": projective_pole_crosses_circle(
            matrix,
            center_x=center_x,
            center_y=center_y,
            radius=radius,
        ),
        "inscribed_circle_center": [center_x, center_y],
        "inscribed_circle_radius": radius,
        "denominator_corner_values": list(pole.corner_denominators),
        "denominator_corner_min": pole.denominator_min,
        "denominator_corner_max": pole.denominator_max,
        "min_abs_sampled_denominator": pole.min_abs_sampled_denominator,
        "magnitude_grid_size": pole.sampled_grid_size,
        "homography_normalization": pole.homography_normalization,
        "denominator_coordinate_frame": "working_pixel_centres",
        "moving_working_width": moving.working_width,
        "moving_working_height": moving.working_height,
    }


def _mean_corner_error_px(
    estimated: np.ndarray, reference: np.ndarray, pair: PairInput
) -> float:
    """Mean fixed-frame error over the four working moving-image corners."""

    moving = pair.coordinates.moving
    points = image_corners(moving.working_width, moving.working_height)

    def project(matrix: np.ndarray) -> np.ndarray:
        homogeneous = np.c_[points, np.ones(len(points))] @ np.asarray(matrix, float).T
        with np.errstate(divide="ignore", invalid="ignore"):
            return homogeneous[:, :2] / homogeneous[:, 2, None]

    left, right = project(estimated), project(reference)
    errors = np.linalg.norm(left - right, axis=1)
    return float(np.mean(errors)) if np.isfinite(errors).all() else float("inf")


def _base_row(record: dict[str, Any], pipeline_id: str, job_id: str, seed: int) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "dataset_id": str(record["dataset_id"]),
        "pair_id": str(record["pair_id"]),
        "group_id": str(record.get("group_id") or record["pair_id"]),
        "pipeline_id": pipeline_id,
        "seed": int(seed),
        "analysis_revision": ANALYSIS_REVISION,
    }


def run_task(task: tuple[dict[str, Any], str, Config, int, int]) -> dict[str, Any]:
    record, pipeline_id, cfg, seed, grid = task
    pipeline = None if pipeline_id == "ground_truth" else cfg.pipeline(pipeline_id)
    job_id = _job_id(record, pipeline, cfg, seed, grid)
    row = _base_row(record, pipeline_id, job_id, seed)
    pair = pair_from_record(record, cfg)
    if pipeline is None:
        gt_path = Path(str(record["ground_truth_homography_path"]))
        original = _read_homography(gt_path)
        matrix = homography_original_to_working(original, pair.coordinates)
        row.update(
            status="ok",
            estimator_kind="ground_truth_control",
            n_matches=None,
            n_inliers=None,
            runtime_s=0.0,
        )
        row.update(_analyse_matrix(matrix, pair, magnitude_grid_size=grid))
        row["mean_corner_error_px"] = 0.0
        return row

    row["estimator_kind"] = "estimated"
    try:
        registrar = load_registrar(pipeline, project_root=ROOT)
        result = registrar.register(pair, seed=seed)
        row.update(
            status=result.status.value,
            n_matches=result.n_matches,
            n_inliers=result.n_inliers,
            runtime_s=result.runtime_s,
            peak_vram_bytes=result.peak_vram_bytes,
            diagnostics=result.diagnostics,
        )
        if result.status is RegistrationStatus.OK:
            matrix = getattr(result.forward_moving_to_fixed, "matrix", None)
            if matrix is None or np.asarray(matrix).shape != (3, 3):
                raise ValueError("configured homography pipeline returned a non-matrix transform")
            row.update(_analyse_matrix(matrix, pair, magnitude_grid_size=grid))
            gt_value = str(record.get("ground_truth_homography_path") or "")
            if gt_value:
                reference = homography_original_to_working(
                    _read_homography(Path(gt_value)), pair.coordinates
                )
                row["mean_corner_error_px"] = _mean_corner_error_px(
                    np.asarray(matrix), reference, pair
                )
        else:
            row.update(
                returned_homography=False,
                denominator_crosses_image=None,
                denominator_crosses_inscribed_circle=None,
                mean_corner_error_px=None,
            )
    except Exception as exc:  # preserve the pair and make infrastructure failures visible
        row.update(
            status="infrastructure_error",
            returned_homography=False,
            denominator_crosses_image=None,
            error_type=type(exc).__name__,
            error=str(exc),
        )
    return row


def _json_row(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _clean_json(value: Any) -> Any:
    """Replace undefined numeric diagnostics by JSON null, recursively."""

    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating | float):
        return float(value) if np.isfinite(value) else None
    return value


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def import_retinal_geometry(path: Path, *, magnitude_grid_size: int) -> list[dict[str, Any]]:
    """Convert a deidentified exact-corner geometry audit into prevalence rows.

    No outcome or failure-label column is copied.  Transform rows are selected
    at the same magnitude lattice as the benchmark run; no-transform attempts
    are retained once so the attempted count remains visible.
    """

    source = pd.read_csv(path)
    required = {
        "pipeline",
        "status",
        "grid_size",
        "denominator_crosses_image",
        "min_abs_lattice_denominator",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"{path}: missing retinal geometry columns {sorted(missing)}")
    transform_rows = source[
        (source.status.astype(str) == "ok")
        & (pd.to_numeric(source.get("grid_size"), errors="coerce") == magnitude_grid_size)
    ]
    no_transform_rows = source[source.status.astype(str) == "no_transform"]
    selected = pd.concat([transform_rows, no_transform_rows], ignore_index=True)
    selected = selected[selected.pipeline.astype(str).isin(DEFAULT_PIPELINES)].copy()
    if selected.empty:
        raise ValueError(
            f"{path}: no retinal rows at grid {magnitude_grid_size} for {DEFAULT_PIPELINES}"
        )

    returned = selected.status.astype(str) == "ok"
    if returned.any():
        for column in ("homography_normalization", "denominator_coordinate_frame"):
            if column not in selected:
                raise ValueError(
                    f"{path}: {column} absent; rerun scripts/evidence_followup.py with "
                    "the exact-corner implementation"
                )
        normalizations = set(selected.loc[returned, "homography_normalization"].astype(str))
        frames = set(selected.loc[returned, "denominator_coordinate_frame"].astype(str))
        if normalizations != {HOMOGRAPHY_NORMALIZATION} or frames != {"working_pixel_centres"}:
            raise ValueError(
                f"{path}: incompatible denominator convention: normalization={normalizations}, "
                f"coordinate_frame={frames}"
            )

    rows: list[dict[str, Any]] = []
    for number, record in selected.reset_index(drop=True).iterrows():
        is_returned = str(record.status) == "ok"
        crossing_value = record.get("denominator_crosses_image")
        if is_returned and pd.notna(crossing_value):
            if isinstance(crossing_value, str):
                normalized = crossing_value.strip().lower()
                if normalized not in {"true", "false"}:
                    raise ValueError(
                        f"{path}: invalid denominator_crosses_image value {crossing_value!r}"
                    )
                crossing = normalized == "true"
            else:
                crossing = bool(crossing_value)
        else:
            crossing = None
        magnitude = record.get("min_abs_lattice_denominator")
        rows.append(
            {
                "job_id": f"retinal-import-{number:06d}",
                "dataset_id": "Retinal",
                "pair_id": f"deidentified-{number:06d}",
                "group_id": "",
                "pipeline_id": str(record.pipeline),
                "seed": None,
                "analysis_revision": ANALYSIS_REVISION,
                "estimator_kind": "estimated",
                "status": str(record.status),
                "returned_homography": is_returned,
                "denominator_crosses_image": crossing,
                "min_abs_sampled_denominator": (
                    float(magnitude) if is_returned and pd.notna(magnitude) else None
                ),
                "magnitude_grid_size": magnitude_grid_size if is_returned else None,
                "homography_normalization": (HOMOGRAPHY_NORMALIZATION if is_returned else None),
                "denominator_coordinate_frame": ("working_pixel_centres" if is_returned else None),
                "source_kind": "deidentified_retinal_geometry_import",
            }
        )
    return rows


def summarize(
    output: Path,
    metadata: dict[str, Any] | None = None,
    *,
    figure: bool = True,
    imported_cases: list[dict[str, Any]] | None = None,
) -> None:
    shard_dir = output / "cases"
    rows = [_json_row(path) for path in sorted(shard_dir.glob("*.json"))]
    rows.extend(imported_cases or [])
    if not rows:
        raise ValueError(f"no completed case shards below {shard_dir}")
    frame = pd.DataFrame(rows)
    flat = frame.copy()
    for column in flat.columns:
        if flat[column].map(lambda value: isinstance(value, dict | list)).any():
            flat[column] = flat[column].map(
                lambda value: json.dumps(value, sort_keys=True)
                if isinstance(value, dict | list)
                else value
            )
    atomic_write_text(output / "pole_cases.csv", flat.to_csv(index=False))

    summary_rows: list[dict[str, Any]] = []
    for (dataset, pipeline), part in frame.groupby(["dataset_id", "pipeline_id"], sort=True):
        returned = part[part.returned_homography.fillna(False).astype(bool)]
        crossing = returned[returned.denominator_crosses_image.fillna(False).astype(bool)]
        count, denominator = len(crossing), len(returned)
        low, high = _wilson(count, denominator)
        magnitude_values = (
            crossing["min_abs_sampled_denominator"]
            if "min_abs_sampled_denominator" in crossing
            else pd.Series(dtype=float)
        )
        magnitudes = pd.to_numeric(magnitude_values, errors="coerce")
        magnitudes = magnitudes[np.isfinite(magnitudes)]
        summary_rows.append(
            {
                "dataset_id": dataset,
                "pipeline_id": pipeline,
                "attempted_pairs": len(part),
                "returned_homographies": denominator,
                "no_returned_homography": len(part) - denominator,
                "in_frame_denominator_crossings": count,
                "crossing_fraction": count / denominator if denominator else float("nan"),
                "crossing_wilson95_low": low,
                "crossing_wilson95_high": high,
                "median_min_abs_sampled_denominator_among_crossings": (
                    float(magnitudes.median()) if len(magnitudes) else float("nan")
                ),
                "zero_sampled_denominators_among_crossings": int((magnitudes == 0).sum()),
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    atomic_write_text(output / "pole_summary.csv", summary_frame.to_csv(index=False))
    if metadata is not None:
        atomic_write_text(
            output / "run_metadata.json",
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
    if figure:
        render_figure(frame, summary_frame, output / "pole_prevalence")


def render_figure(cases: pd.DataFrame, summary: pd.DataFrame, destination: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - reporting extra is optional
        raise RuntimeError(
            "install the report extra to render figures: pip install -e '.[report]'"
        ) from exc

    pipelines = [p for p in (*DEFAULT_PIPELINES, "ground_truth") if p in set(summary.pipeline_id)]
    datasets = sorted(summary.dataset_id.unique())
    colors = {
        "xfeat_h": "#087e8b",
        "sp_lg_h": "#364c75",
        "sift_h": "#b55b31",
        "ground_truth": "#777777",
    }
    names = {
        "xfeat_h": "XFeat-H",
        "sp_lg_h": "SP/LightGlue-H",
        "sift_h": "SIFT-H",
        "ground_truth": "GT homography",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.25), layout="constrained")
    x = np.arange(len(datasets), dtype=float)
    width = 0.82 / max(1, len(pipelines))
    for offset, pipeline in enumerate(pipelines):
        values, errors_low, errors_high = [], [], []
        for dataset in datasets:
            part = summary[(summary.dataset_id == dataset) & (summary.pipeline_id == pipeline)]
            if part.empty:
                values.append(np.nan)
                errors_low.append(np.nan)
                errors_high.append(np.nan)
                continue
            row = part.iloc[0]
            rate = 100 * float(row.crossing_fraction)
            values.append(rate)
            errors_low.append(rate - 100 * float(row.crossing_wilson95_low))
            errors_high.append(100 * float(row.crossing_wilson95_high) - rate)
        positions = x - 0.41 + width / 2 + offset * width
        axes[0].bar(
            positions,
            values,
            width=width,
            color=colors[pipeline],
            label=names[pipeline],
            yerr=np.asarray([errors_low, errors_high]),
            capsize=2,
        )
    axes[0].set_xticks(x, datasets, rotation=15, ha="right")
    axes[0].set_ylabel("Returned homographies with\nin-frame pole (%)")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(axis="y", alpha=0.18)
    axes[0].legend(fontsize=7, frameon=False)

    distributions, positions, labels, box_colors = [], [], [], []
    position = 0
    any_zero = False
    for dataset in datasets:
        for pipeline in pipelines:
            selected = cases[
                (cases.dataset_id == dataset)
                & (cases.pipeline_id == pipeline)
                & cases.denominator_crosses_image.fillna(False).astype(bool)
            ]
            raw_values = (
                selected["min_abs_sampled_denominator"]
                if "min_abs_sampled_denominator" in selected
                else pd.Series(dtype=float)
            )
            values = pd.to_numeric(raw_values, errors="coerce")
            values = values[np.isfinite(values)].to_numpy(float)
            if not len(values):
                continue
            any_zero |= bool(np.any(values == 0))
            distributions.append(values)
            positions.append(position)
            labels.append(f"{dataset}\n{names[pipeline]}")
            box_colors.append(colors[pipeline])
            position += 1
        position += 0.5
    if distributions:
        boxes = axes[1].boxplot(
            distributions, positions=positions, widths=0.65, showfliers=False, patch_artist=True
        )
        for patch, color in zip(boxes["boxes"], box_colors, strict=True):
            patch.set_facecolor(color)
        axes[1].set_xticks(positions, labels, rotation=25, ha="right", fontsize=6)
        if any_zero:
            axes[1].set_yscale("symlog", linthresh=1e-12)
        else:
            axes[1].set_yscale("log")
    else:
        axes[1].text(0.5, 0.5, "No crossing homographies", ha="center", va="center")
        axes[1].set_xticks([])
    axes[1].set_ylabel(
        r"Min sampled $|h_{31}x+h_{32}y+h_{33}|$" "\n(unit-Frobenius H; working pixels)"
    )
    axes[1].grid(axis="y", alpha=0.18)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(destination.with_suffix(".png"), dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> int:
    if not args.no_figure:
        try:
            import matplotlib  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "figure output requires the report extra; install with pip install -e '.[report]' "
                "or pass --no-figure"
            ) from exc
    cfg = load_config(args.config)
    manifests = [Path(path) for path in args.manifests]
    pairs = load_manifests(manifests)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        pairs = pairs.head(args.limit)
    print(f"Verifying hashes for {len(pairs)} indexed pairs...", flush=True)
    verify_manifest_inputs(pairs)
    pipelines = tuple(args.pipelines)
    for pipeline_id in pipelines:
        pipeline = cfg.pipeline(pipeline_id)
        if pipeline.transform_family != "homography":
            raise ValueError(f"{pipeline_id}: pole prevalence requires a homography pipeline")

    output = Path(args.output)
    shard_dir = output / "cases"
    shard_dir.mkdir(parents=True, exist_ok=True)
    retinal_geometry = Path(args.retinal_geometry) if args.retinal_geometry else None
    contract = {
        "analysis_revision": ANALYSIS_REVISION,
        "implementation_hash": IMPLEMENTATION_HASH,
        "config_hash": cfg.hash,
        "manifests": [
            {"path": str(path.resolve()), "sha256": file_digest(path)} for path in manifests
        ],
        "selected_pairs": [
            {
                key: record.get(key, "")
                for key in (
                    "dataset_id",
                    "pair_id",
                    "moving_sha256",
                    "fixed_sha256",
                    "ground_truth_homography_sha256",
                )
            }
            for record in pairs.to_dict("records")
        ],
        "pipelines": list(pipelines),
        "seed": int(args.seed),
        "magnitude_grid_size": int(args.magnitude_grid_size),
        "retinal_geometry": (
            None
            if retinal_geometry is None
            else {"path": str(retinal_geometry.resolve()), "sha256": file_digest(retinal_geometry)}
        ),
    }
    contract_path = output / "run_contract.json"
    if contract_path.is_file():
        if _json_row(contract_path) != _clean_json(contract):
            raise ValueError("run contract changed; use a fresh --output directory")
    elif any(shard_dir.glob("*.json")):
        raise ValueError("case shards exist without a run contract; use a fresh --output directory")
    else:
        atomic_write_text(
            contract_path,
            json.dumps(_clean_json(contract), indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
    tasks: list[tuple[dict[str, Any], str, Config, int, int]] = []
    for record in pairs.to_dict("records"):
        pipeline_ids = list(pipelines)
        if str(record.get("ground_truth_homography_path") or ""):
            pipeline_ids.append("ground_truth")
        for pipeline_id in pipeline_ids:
            pipeline = None if pipeline_id == "ground_truth" else cfg.pipeline(pipeline_id)
            job_id = _job_id(record, pipeline, cfg, args.seed, args.magnitude_grid_size)
            path = shard_dir / f"{job_id}.json"
            if path.is_file():
                existing = _json_row(path)
                if not (args.retry_errors and existing.get("status") == "infrastructure_error"):
                    continue
            tasks.append((record, pipeline_id, cfg, args.seed, args.magnitude_grid_size))

    total = len(tasks)
    completed = 0
    if tasks:
        executor = ThreadPoolExecutor if args.workers == 1 else ProcessPoolExecutor
        executor_options = (
            {"max_workers": 1}
            if args.workers == 1
            else {"max_workers": args.workers, "mp_context": get_context("spawn")}
        )
        with executor(**executor_options) as pool:
            futures = {pool.submit(run_task, task): task for task in tasks}
            for future in as_completed(futures):
                row = _clean_json(future.result())
                atomic_write_text(
                    shard_dir / f"{row['job_id']}.json",
                    json.dumps(row, sort_keys=True, allow_nan=False) + "\n",
                )
                completed += 1
                print(
                    f"[{completed}/{total}] {row['dataset_id']} {row['pair_id']} "
                    f"{row['pipeline_id']}: {row['status']}",
                    flush=True,
                )

    imported_cases = (
        []
        if retinal_geometry is None
        else import_retinal_geometry(retinal_geometry, magnitude_grid_size=args.magnitude_grid_size)
    )
    metadata = {
        "analysis_revision": ANALYSIS_REVISION,
        "implementation_hash": IMPLEMENTATION_HASH,
        "config": str(Path(args.config).resolve()),
        "config_hash": cfg.hash,
        "input_manifests": [str(path.resolve()) for path in manifests],
        "datasets": sorted(pairs.dataset_id.unique()),
        "pipelines": list(pipelines),
        "seed": int(args.seed),
        "pairs": len(pairs),
        "new_tasks_completed": completed,
        "crossing_test": "exact sign bracketing at four closed image-rectangle corners",
        "magnitude_statistic": f"minimum absolute denominator on a {args.magnitude_grid_size}x{args.magnitude_grid_size} lattice including corners",
        "homography_normalization": HOMOGRAPHY_NORMALIZATION,
        "denominator_coordinate_frame": "working integer-centred pixels",
        "working_long_edge": cfg.geometry.working_long_edge,
        "ground_truth_used_by_matchers": False,
        "failure_labels_used": False,
        "retinal_geometry_import": (
            None if retinal_geometry is None else str(retinal_geometry.resolve())
        ),
        "retinal_imported_attempts": len(imported_cases),
    }
    summarize(
        output,
        metadata,
        figure=not args.no_figure,
        imported_cases=imported_cases,
    )
    print(f"Wrote {output / 'pole_summary.csv'}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    hp = commands.add_parser("index-hpatches", help="index an extracted HPatches archive")
    hp.add_argument("--root", required=True)
    hp.add_argument("--output", required=True)
    hp.add_argument("--dataset-id", default="HPatches")

    generic = commands.add_parser("index-pairs", help="index a CSV pair list")
    generic.add_argument("--dataset", required=True)
    generic.add_argument("--root", required=True)
    generic.add_argument("--pairs", required=True)
    generic.add_argument("--output", required=True)

    execute = commands.add_parser("run", help="run/resume matchers and build prevalence outputs")
    execute.add_argument("--manifests", nargs="+", required=True)
    execute.add_argument("--config", default=str(ROOT / "configs/full_study.yaml"))
    execute.add_argument("--output", default=str(ROOT / "benchmark_pole_outputs"))
    execute.add_argument("--pipelines", nargs="+", default=list(DEFAULT_PIPELINES))
    execute.add_argument("--seed", type=int, default=20260907)
    execute.add_argument("--workers", type=int, default=1)
    execute.add_argument("--magnitude-grid-size", type=int, default=32)
    execute.add_argument("--limit", type=int)
    execute.add_argument("--retry-errors", action="store_true")
    execute.add_argument(
        "--retinal-geometry",
        help="exact-corner geometry.csv to merge as the Retinal dataset in the final outputs",
    )
    execute.add_argument("--no-figure", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "index-hpatches":
        path = index_hpatches(Path(args.root), Path(args.output), args.dataset_id)
        print(f"Wrote {path}")
        return 0
    if args.command == "index-pairs":
        path = index_pairs(Path(args.root), Path(args.pairs), Path(args.output), args.dataset)
        print(f"Wrote {path}")
        return 0
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
