from __future__ import annotations

import pytest

from scripts.build_isbi_near_pole_figure import (
    sampled_bending_energy,
    synthetic_homography,
)
from warpaudit.geometry.projective import projective_pole_diagnostics


@pytest.mark.parametrize("clearance", [1e-3, 1e-2, 0.1, 1.0])
def test_synthetic_homography_has_requested_clearance(clearance: float) -> None:
    matrix = synthetic_homography(clearance)
    result = projective_pole_diagnostics(matrix, width=640, height=480)
    assert not result.denominator_crosses_image
    assert result.pole_clearance_diagonal_fraction == pytest.approx(clearance)


def test_bending_energy_explodes_as_pole_approaches_support() -> None:
    far = sampled_bending_energy(synthetic_homography(1.0), size=32)
    near = sampled_bending_energy(synthetic_homography(1e-3), size=32)
    assert near / far > 1e6
