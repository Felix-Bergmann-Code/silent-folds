from __future__ import annotations

import numpy as np

from warpaudit.geometry.transforms import HomographyTransform
from warpaudit.signals.diagnostics import classify_cycle, classify_e1
from warpaudit.signals.family_e2_perturbation import correct_perturbed_transform


def test_cycle_degeneracy_rule_uses_median_and_iqr() -> None:
    result = classify_cycle(np.array([1e-6, 2e-6, 3e-6, 4e-6]), tolerance=1e-4)
    assert result.classification == "degenerate"


def test_e1_gate_requires_repeatability_seed_control_and_error_association() -> None:
    spread = np.arange(1, 11, dtype=float) / 1000
    error = np.arange(1, 11, dtype=float)
    result = classify_e1(
        spread,
        spread * 0.01,
        error,
        repeatability_tolerance=1e-3,
        seed_dominance_max_ratio=0.5,
        min_error_spearman=0.15,
    )
    assert result.classification == "informative"


def test_e2_coordinate_correction_recovers_baseline_transform() -> None:
    baseline = np.array([[1, 0, 4], [0, 1, -2], [0, 0, 1]], float)
    moving_perturbation = np.array([[1, 0, 3], [0, 1, 1], [0, 0, 1]], float)
    fixed_perturbation = np.array([[1, 0, -2], [0, 1, 5], [0, 0, 1]], float)
    perturbed = fixed_perturbation @ baseline @ np.linalg.inv(moving_perturbation)
    corrected = correct_perturbed_transform(
        HomographyTransform(perturbed), moving_perturbation, fixed_perturbation
    )
    np.testing.assert_allclose(corrected.matrix, baseline)
