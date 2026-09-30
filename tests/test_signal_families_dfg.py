"""Families D, F, and G: warp plausibility, cross-pipeline disagreement, structure."""

from __future__ import annotations

import numpy as np
import pytest

from warpaudit.geometry.grids import prespecified_grid
from warpaudit.geometry.transforms import HomographyTransform
from warpaudit.signals import compute_families
from warpaudit.signals.family_f_disagreement import PIPELINE_PREFIX
from warpaudit.types import RegistrationResult, RegistrationStatus, SignalConfig, SignalContext


def _result(matrix: np.ndarray | None, pipeline_id: str = "fixture") -> RegistrationResult:
    return RegistrationResult(
        pipeline_id=pipeline_id,
        status=RegistrationStatus.OK if matrix is not None else RegistrationStatus.INVALID_TRANSFORM,
        forward_moving_to_fixed=None if matrix is None else HomographyTransform(matrix),
        matches_moving=None,
        matches_fixed=None,
        inlier_mask=None,
        match_scores=None,
    )


def _context(pair_input, matrix, *, auxiliaries=None, images=None, size=16) -> SignalContext:
    return SignalContext(
        pair=pair_input,
        result=_result(matrix),
        reverse_estimate=None,
        auxiliary_results=auxiliaries or {},
        grid=prespecified_grid(pair_input.coordinates.fixed, size=size),
        config=SignalConfig(bootstrap_B=4, seed=3, grid_size=size),
        images=images or {},
    )


def test_identity_warp_is_rigid_unfolded_and_unbent(pair_input) -> None:
    values = compute_families(_context(pair_input, np.eye(3)), ["D"])["D"].values
    assert values["folding_fraction"].value == 0.0
    # Identity between identically sized frames: unit area, unit shape, no bending.
    assert values["log_jacobian_determinant_median"].value == pytest.approx(0.0, abs=1e-9)
    assert values["anisotropy_median"].value == pytest.approx(1.0, abs=1e-9)
    assert values["scale_median"].value == pytest.approx(1.0, abs=1e-9)
    assert values["bending_energy"].value == pytest.approx(0.0, abs=1e-9)
    assert values["displacement_mean"].value == pytest.approx(0.0, abs=1e-12)


def test_reflection_is_reported_as_folding_everywhere(pair_input) -> None:
    """A mirrored map is orientation-reversing at every point, not merely odd."""
    mirror = np.array([[-1.0, 0.0, 119.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    values = compute_families(_context(pair_input, mirror), ["D"])["D"].values
    assert values["folding_fraction"].value == 1.0
    # Magnitudes are unchanged by the flip: area and shape stay unit.
    assert values["anisotropy_median"].value == pytest.approx(1.0, abs=1e-9)


def test_uniform_scale_is_reported_on_a_symmetric_log_area_scale(pair_input) -> None:
    half = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]])
    double = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    shrink = compute_families(_context(pair_input, half), ["D"])["D"].values
    grow = compute_families(_context(pair_input, double), ["D"])["D"].values
    assert shrink["log_jacobian_determinant_median"].value == pytest.approx(
        -grow["log_jacobian_determinant_median"].value, abs=1e-9
    )
    assert shrink["scale_median"].value == pytest.approx(0.5, abs=1e-9)


def test_projective_warp_bends_where_an_affine_one_does_not(pair_input) -> None:
    affine = np.array([[1.0, 0.2, 5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    projective = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1e-3, 0.0, 1.0]])
    flat = compute_families(_context(pair_input, affine), ["D"])["D"].values
    bent = compute_families(_context(pair_input, projective), ["D"])["D"].values
    assert flat["bending_energy"].value == pytest.approx(0.0, abs=1e-9)
    assert bent["bending_energy"].value > flat["bending_energy"].value
    # Shear shows up in shape, not in area.
    assert flat["anisotropy_median"].value > 1.0
    assert flat["log_jacobian_determinant_median"].value == pytest.approx(0.0, abs=1e-9)


def test_disagreement_is_zero_for_identical_maps_and_scales_with_translation(
    pair_input,
) -> None:
    same = {f"{PIPELINE_PREFIX}other": _result(np.eye(3), "other")}
    shifted = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    apart = {f"{PIPELINE_PREFIX}other": _result(shifted, "other")}

    agree = compute_families(_context(pair_input, np.eye(3), auxiliaries=same), ["F"])["F"].values
    differ = compute_families(_context(pair_input, np.eye(3), auxiliaries=apart), ["F"])["F"].values
    assert agree["disagreement_mean"].value == pytest.approx(0.0, abs=1e-12)
    diagonal = pair_input.coordinates.fixed.working_diagonal
    assert differ["disagreement_mean"].value == pytest.approx(10.0 / diagonal, rel=1e-9)
    assert differ["comparison_pipelines"].value == 1.0


def test_a_comparison_pipeline_without_a_transform_is_counted_not_dropped(
    pair_input,
) -> None:
    missing = {f"{PIPELINE_PREFIX}other": _result(None, "other")}
    values = compute_families(_context(pair_input, np.eye(3), auxiliaries=missing), ["F"])[
        "F"
    ].values
    assert values["comparison_pipelines"].value == 1.0
    assert values["comparison_without_transform"].value == 1.0
    assert not values["disagreement_mean"].available
    assert "returned no transform" in values["disagreement_mean"].reason


def _textured(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = np.indices(shape)
    return (
        120.0
        + 60.0 * np.sin(rows / 6.0)
        + 40.0 * np.cos(cols / 4.0)
        + 25.0 * np.sin((rows + cols) / 9.0)
    )


def test_structure_agreement_is_high_when_aligned_and_falls_when_shifted(
    pair_input,
) -> None:
    image = _textured((100, 120))
    images = {"moving": image, "fixed": image}
    aligned = compute_families(_context(pair_input, np.eye(3), images=images), ["G"])["G"].values
    shifted_matrix = np.array([[1.0, 0.0, 9.0], [0.0, 1.0, 7.0], [0.0, 0.0, 1.0]])
    shifted = compute_families(
        _context(pair_input, shifted_matrix, images=images), ["G"]
    )["G"].values

    assert aligned["gradient_magnitude_ncc"].value == pytest.approx(1.0, abs=1e-9)
    assert aligned["gradient_orientation_agreement"].value == pytest.approx(1.0, abs=1e-9)
    assert aligned["edge_dice_q90"].value == pytest.approx(1.0, abs=1e-9)
    assert shifted["edge_dice_q90"].value < aligned["edge_dice_q90"].value
    assert shifted["gradient_magnitude_ncc"].value < aligned["gradient_magnitude_ncc"].value


def test_edge_sets_use_a_within_support_quantile_so_brightness_cannot_move_them(
    pair_input,
) -> None:
    """The definition must not become a contrast detector across domains."""
    image = _textured((100, 120))
    plain = {"moving": image, "fixed": image}
    # A different intensity range in one "domain" must not change the edge set.
    rescaled = {"moving": image * 0.25 + 500.0, "fixed": image}
    base = compute_families(_context(pair_input, np.eye(3), images=plain), ["G"])["G"].values
    scaled = compute_families(_context(pair_input, np.eye(3), images=rescaled), ["G"])["G"].values
    assert scaled["edge_dice_q80"].value == pytest.approx(base["edge_dice_q80"].value, abs=1e-9)
    assert scaled["edge_fraction_warped_q80"].value == pytest.approx(
        base["edge_fraction_warped_q80"].value, abs=1e-9
    )
