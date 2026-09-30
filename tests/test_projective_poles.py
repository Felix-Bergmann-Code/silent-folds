from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts import benchmark_pole_prevalence as benchmark
from warpaudit.config import load_config
from warpaudit.geometry.projective import (
    normalise_homography,
    projective_pole_crosses_circle,
    projective_pole_diagnostics,
)
from warpaudit.geometry.transforms import HomographyTransform
from warpaudit.types import RegistrationResult, RegistrationStatus


def test_corner_test_is_exact_and_scale_invariant_between_lattice_samples():
    # The zero at x=49.123 is not required to coincide with a lattice sample.
    h = np.array([[1.0, 0, 0], [0, 1.0, 0], [1.0, 0, -49.123]])
    first = projective_pole_diagnostics(h, width=100, height=80, magnitude_grid_size=4)
    second = projective_pole_diagnostics(-17.5 * h, width=100, height=80, magnitude_grid_size=64)
    assert first.denominator_crosses_image
    assert second.denominator_crosses_image
    assert first.pole_clearance_diagonal_fraction == 0.0
    assert second.pole_clearance_diagonal_fraction == 0.0
    # Frobenius normalization also makes a fixed-grid magnitude scale invariant.
    same_grid = projective_pole_diagnostics(1e9 * h, width=100, height=80, magnitude_grid_size=4)
    assert same_grid.min_abs_sampled_denominator == pytest.approx(first.min_abs_sampled_denominator)


def test_corner_touch_counts_as_crossing_and_non_crossing_is_rejected():
    touching = np.array([[1.0, 0, 0], [0, 1.0, 0], [1.0, 0, 0]])
    outside = np.array([[1.0, 0, 0], [0, 1.0, 0], [0.001, 0, 1.0]])
    assert projective_pole_diagnostics(touching, width=20, height=10).denominator_crosses_image
    outside_result = projective_pole_diagnostics(outside, width=20, height=10)
    scaled_result = projective_pole_diagnostics(1000 * outside, width=20, height=10)
    assert not outside_result.denominator_crosses_image
    assert outside_result.pole_clearance_diagonal_fraction > 0
    assert scaled_result.pole_clearance_diagonal_fraction == pytest.approx(
        outside_result.pole_clearance_diagonal_fraction
    )
    coordinate_scaled = outside.copy()
    coordinate_scaled[2, :2] /= 2.0
    coordinate_scaled_result = projective_pole_diagnostics(
        coordinate_scaled, width=39, height=19
    )
    assert coordinate_scaled_result.pole_clearance_diagonal_fraction == pytest.approx(
        outside_result.pole_clearance_diagonal_fraction
    )
    affine_result = projective_pole_diagnostics(np.eye(3), width=20, height=10)
    assert affine_result.pole_clearance_diagonal_fraction == float("inf")
    with pytest.raises(ValueError, match="Frobenius norm"):
        normalise_homography(np.zeros((3, 3)))


def test_circle_guard_rejects_black_corner_crossing_but_keeps_fov_crossing():
    corner_only = np.array([[1.0, 0, 0], [0, 1.0, 0], [1.0, 1.0, -5.0]])
    through_fov = np.array([[1.0, 0, 0], [0, 1.0, 0], [1.0, 0.0, -50.0]])
    assert projective_pole_diagnostics(corner_only, width=100, height=80).denominator_crosses_image
    assert not projective_pole_crosses_circle(
        corner_only, center_x=49.5, center_y=39.5, radius=39.5
    )
    assert projective_pole_crosses_circle(
        through_fov, center_x=49.5, center_y=39.5, radius=39.5
    )


def _hpatches_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "hpatches-sequences-release" / "v_fixture"
    root.mkdir(parents=True)
    for index in range(1, 7):
        Image.new("RGB", (24, 16), (index, index, index)).save(root / f"{index}.ppm")
        if index > 1:
            np.savetxt(root / f"H_1_{index}", np.eye(3))
    return tmp_path


def test_hpatches_index_has_five_pairs_and_ground_truth_hashes(tmp_path):
    root = _hpatches_fixture(tmp_path)
    path = benchmark.index_hpatches(root, tmp_path / "hpatches.csv")
    frame = pd.read_csv(path)
    assert len(frame) == 5
    assert set(frame.group_id) == {"v_fixture"}
    assert frame.ground_truth_homography_sha256.str.len().eq(64).all()
    benchmark.verify_manifest_inputs(frame)


def test_pair_index_and_ground_truth_task_need_no_labels(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (30, 20)).save(images / "a.png")
    Image.new("RGB", (30, 20)).save(images / "b.png")
    np.savetxt(images / "H.txt", np.eye(3))
    source = tmp_path / "pairs.csv"
    source.write_text(
        "pair_id,moving_path,fixed_path,ground_truth_homography_path\n" "one,a.png,b.png,H.txt\n",
        encoding="utf-8",
    )
    manifest = benchmark.index_pairs(images, source, tmp_path / "indexed.csv", "PLANAR")
    record = pd.read_csv(manifest).fillna("").iloc[0].to_dict()
    cfg = load_config(benchmark.ROOT / "configs/full_study.yaml")
    row = benchmark.run_task((record, "ground_truth", cfg, 7, 32))
    assert row["status"] == "ok"
    assert row["estimator_kind"] == "ground_truth_control"
    assert not row["denominator_crosses_image"]
    assert not row["denominator_crosses_inscribed_circle"]
    assert row["mean_corner_error_px"] == 0.0


def test_estimated_task_and_summary_keep_returned_denominator(monkeypatch, tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("a", "b"):
        Image.new("RGB", (100, 80)).save(images / f"{name}.png")
    np.savetxt(images / "H.txt", np.eye(3))
    source = tmp_path / "pairs.csv"
    source.write_text(
        "moving_path,fixed_path,ground_truth_homography_path\n"
        "a.png,b.png,H.txt\n",
        encoding="utf-8",
    )
    manifest = benchmark.index_pairs(images, source, tmp_path / "indexed.csv", "PLANAR")
    record = pd.read_csv(manifest).fillna("").iloc[0].to_dict()
    cfg = load_config(benchmark.ROOT / "configs/full_study.yaml")
    crossing = HomographyTransform(np.array([[1.0, 0, 0], [0, 1.0, 0], [1 / 100, 0, -0.5]]))
    result = RegistrationResult(
        pipeline_id="xfeat_h",
        status=RegistrationStatus.OK,
        forward_moving_to_fixed=crossing,
        matches_moving=np.zeros((4, 2)),
        matches_fixed=np.zeros((4, 2)),
        inlier_mask=np.ones(4, dtype=bool),
        runtime_s=1.0,
    )
    monkeypatch.setattr(
        benchmark,
        "load_registrar",
        lambda *args, **kwargs: type(
            "Registrar", (), {"register": lambda self, pair, seed: result}
        )(),
    )
    row = benchmark._clean_json(benchmark.run_task((record, "xfeat_h", cfg, 7, 8)))
    assert row["denominator_crosses_image"]
    assert not row["denominator_crosses_inscribed_circle"]
    assert row["mean_corner_error_px"] > 0
    case_dir = tmp_path / "out" / "cases"
    case_dir.mkdir(parents=True)
    (case_dir / "row.json").write_text(json.dumps(row), encoding="utf-8")
    benchmark.summarize(tmp_path / "out", figure=False)
    summary = pd.read_csv(tmp_path / "out" / "pole_summary.csv").iloc[0]
    assert summary.returned_homographies == 1
    assert summary.in_frame_denominator_crossings == 1
    assert summary.crossing_fraction == 1.0


def test_retinal_import_selects_one_grid_and_keeps_no_transform_attempts(tmp_path):
    source = pd.DataFrame(
        [
            {
                "pipeline": "xfeat_h",
                "status": "ok",
                "grid_size": 16,
                "denominator_crosses_image": "False",
                "min_abs_lattice_denominator": 0.2,
                "homography_normalization": "frobenius_unit_norm",
                "denominator_coordinate_frame": "working_pixel_centres",
                "operational_failure": True,
            },
            {
                "pipeline": "xfeat_h",
                "status": "ok",
                "grid_size": 32,
                "denominator_crosses_image": "False",
                "min_abs_lattice_denominator": 0.1,
                "homography_normalization": "frobenius_unit_norm",
                "denominator_coordinate_frame": "working_pixel_centres",
                "operational_failure": True,
            },
            {
                "pipeline": "xfeat_h",
                "status": "no_transform",
                "grid_size": None,
                "denominator_crosses_image": None,
                "min_abs_lattice_denominator": None,
            },
        ]
    )
    path = tmp_path / "geometry.csv"
    source.to_csv(path, index=False)
    rows = benchmark.import_retinal_geometry(path, magnitude_grid_size=32)
    assert len(rows) == 2
    assert sum(row["returned_homography"] for row in rows) == 1
    assert rows[0]["denominator_crosses_image"] is False
    assert all("operational_failure" not in row for row in rows)


def test_retinal_import_rejects_old_unnormalized_geometry(tmp_path):
    path = tmp_path / "old.csv"
    pd.DataFrame(
        {
            "pipeline": ["xfeat_h"],
            "status": ["ok"],
            "grid_size": [32],
            "denominator_crosses_image": [True],
            "min_abs_lattice_denominator": [1e-6],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="rerun scripts/evidence_followup"):
        benchmark.import_retinal_geometry(path, magnitude_grid_size=32)
