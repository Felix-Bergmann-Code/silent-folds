from __future__ import annotations

import numpy as np
import pytest

from scripts.isbi_submission_experiments import (
    circle_crossing,
    clearance_score,
    fit_homography_constrained,
    nested_threshold,
    rectangle_crossing,
    sample_orientation_consistent,
    sampled_folding,
)
from warpaudit.registration.fitting import FittingPolicy, fit_homography


def _pole_at_x(x_pole: float) -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0 / x_pole, 0.0, 1.0]])


def test_crossing_tests_agree_with_pole_position():
    assert rectangle_crossing(_pole_at_x(300.0), 640, 480)
    assert not rectangle_crossing(_pole_at_x(700.0), 640, 480)
    # Pole at x=600 crosses the rectangle but misses the inscribed circle (x<=559).
    assert rectangle_crossing(_pole_at_x(600.0), 640, 480)
    assert not circle_crossing(_pole_at_x(600.0), 640, 480)
    assert circle_crossing(_pole_at_x(400.0), 640, 480)


def test_sampled_folding_matches_exact_test_away_from_boundary():
    assert sampled_folding(_pole_at_x(300.0), 640, 480)
    assert not sampled_folding(_pole_at_x(700.0), 640, 480)
    # A pole clipping the last half lattice cell is missed by the lattice only.
    assert rectangle_crossing(_pole_at_x(635.0), 640, 480)
    assert not sampled_folding(_pole_at_x(635.0), 640, 480)


def test_sample_orientation():
    h = _pole_at_x(300.0)
    same_side = np.array([[0, 0], [100, 0], [0, 100], [100, 100]], dtype=float)
    split = np.array([[0, 0], [400, 0], [0, 100], [100, 100]], dtype=float)
    assert sample_orientation_consistent(h, same_side)
    assert not sample_orientation_consistent(h, split)


def test_clearance_score_orders_and_flags_explicit_failures():
    score = clearance_score(np.array([0.0, 0.5, 10.0, np.inf, np.nan]),
                            np.array([False, False, False, False, True]))
    assert score[0] == pytest.approx(3.0)
    assert score[0] > score[1] > score[2] > score[3]
    assert score[4] == pytest.approx(3.0)


def test_nested_threshold_is_largest_cutoff_meeting_precision():
    rho = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 2.0, 3.0, 4.0])
    failure = np.array([1, 1, 1, 1, 1, 1, 0, 1, 0], dtype=bool)
    assert nested_threshold(rho, failure, target_precision=1.0, min_flagged=5) == 0.5
    assert nested_threshold(rho, failure, target_precision=0.85, min_flagged=5) == 3.0
    assert nested_threshold(rho, failure, target_precision=1.0, min_flagged=50) == 0.0


def _synthetic_correspondences(seed: int = 3):
    rng = np.random.default_rng(seed)
    src = rng.uniform(0, 600, size=(120, 2))
    h = np.array([[1.02, 0.01, 5.0], [-0.02, 0.99, -3.0], [1e-5, -2e-5, 1.0]])
    mapped = np.c_[src, np.ones(len(src))] @ h.T
    dst = mapped[:, :2] / mapped[:, 2:]
    dst += rng.normal(0, 0.3, size=dst.shape)
    dst[:40] = rng.uniform(0, 600, size=(40, 2))  # outliers
    return src, dst


def test_unconstrained_refit_reproduces_production_fitter_bitwise():
    src, dst = _synthetic_correspondences()
    policy = FittingPolicy()
    production = fit_homography(src, dst, policy, seed=11)
    replay = fit_homography_constrained(src, dst, policy, seed=11, constraint="none",
                                        width=640, height=480)
    assert replay["status"] == "ok"
    np.testing.assert_array_equal(np.asarray(replay["matrix"]), production.transform.matrix)
    assert replay["n_iterations"] == production.n_iterations


@pytest.mark.parametrize("constraint", ["sample_orientation", "rectangle", "fov"])
def test_constraints_leave_a_valid_fit_unchanged(constraint):
    src, dst = _synthetic_correspondences()
    policy = FittingPolicy()
    base = fit_homography_constrained(src, dst, policy, seed=11, constraint="none",
                                      width=640, height=480)
    constrained = fit_homography_constrained(src, dst, policy, seed=11,
                                             constraint=constraint, width=640, height=480)
    assert not rectangle_crossing(np.asarray(base["matrix"]), 640, 480)
    assert constrained["status"] == "ok"
    assert not rectangle_crossing(np.asarray(constrained["matrix"]), 640, 480)


def test_rectangle_constraint_never_returns_a_crossing():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 640, size=(30, 2))
    dst = rng.uniform(0, 640, size=(30, 2))  # pure outliers invite wild hypotheses
    policy = FittingPolicy(max_iters=500)
    fit = fit_homography_constrained(src, dst, policy, seed=1, constraint="rectangle",
                                     width=640, height=480)
    if fit["status"] == "ok":
        assert not rectangle_crossing(np.asarray(fit["matrix"]), 640, 480)
    assert fit["rejected_hypotheses"] > 0


def test_perspective_and_orientation_checks():
    from scripts.isbi_submission_experiments import (
        orientation_reversing,
        perspective_score,
        projected_quad_consistent,
    )

    assert perspective_score(np.eye(3), 640, 480) == 0.0
    near = perspective_score(_pole_at_x(700.0), 640, 480)
    far = perspective_score(_pole_at_x(7000.0), 640, 480)
    assert near > far > 0
    mirror = np.diag([-1.0, 1.0, 1.0])
    assert orientation_reversing(mirror, 640, 480)
    assert not orientation_reversing(np.eye(3), 640, 480)
    assert not orientation_reversing(_pole_at_x(300.0), 640, 480)  # crossing, not global
    assert projected_quad_consistent(np.eye(3), 640, 480)
    assert not projected_quad_consistent(mirror, 640, 480)


def test_training_quantile_maps_out_of_range_to_bounds():
    import pandas as pd

    from scripts.isbi_submission_experiments import training_quantile

    out = training_quantile(np.array([1.0, 2.0, 3.0, 4.0]),
                            pd.Series([0.0, 2.5, 1e10, np.nan]))
    assert out[0] == 0.0 and out[1] == 0.5 and out[2] == 1.0 and np.isnan(out[3])


def test_rank_average_is_monotone_fusion():
    from scripts.isbi_submission_experiments import rank_average

    fused = rank_average(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert np.all(np.diff(fused) > 0)
