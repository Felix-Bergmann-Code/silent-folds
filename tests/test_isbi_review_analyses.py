"""Pure helpers of the review analyses (no result files needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import isbi_review_analyses as review


def test_scale_deviation_is_zero_for_identity_and_log_area_for_zoom():
    assert review.scale_deviation(np.eye(3), 101, 51) == 0.0
    zoom = np.diag([2.0, 2.0, 1.0])
    assert np.isclose(review.scale_deviation(zoom, 101, 51), np.log(4.0))


def test_scale_deviation_is_infinite_when_the_pole_crosses_the_image():
    h = np.eye(3)
    h[2] = [0.02, 0.0, -1.0]  # pole at x = 50, inside a 101-wide image
    assert np.isinf(review.scale_deviation(h, 101, 51))


def test_perspective_grows_as_the_pole_approaches_the_center():
    far, near = np.eye(3), np.eye(3)
    far[2] = [1e-4, 0.0, 1.0]
    near[2] = [1e-2, 0.0, 1.0]
    assert review.perspective(near, 101, 51) > review.perspective(far, 101, 51)
    assert review.perspective(np.eye(3), 101, 51) == -6.0


def test_relabel_reproduces_threshold_endpoint():
    frame = pd.DataFrame({"status": ["ok", "ok", "no_output"],
                          "tre_norm": [0.004, 0.006, np.nan]})
    labels = review.relabel(frame, 0.005).operational_failure.tolist()
    assert labels == [False, True, True]


def test_crossfit_percentile_never_uses_the_case_or_its_fold():
    values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    folds = np.array([0, 0, 0, 1, 1, 1])
    # Fold 0 is ranked only against fold 1 and vice versa.
    assert review.crossfit_percentile(values, folds).tolist() == [0, 0, 0, 1, 1, 1]
    # Mid-ranks for ties with the reference.
    tied = review.crossfit_percentile(np.array([5.0, 5.0]), np.array([0, 1]))
    assert tied.tolist() == [0.5, 0.5]


def test_affine_deviation_is_zero_for_affine_and_infinite_for_a_crossing():
    affine = np.array([[1.1, 0.1, 5.0], [-0.2, 0.9, 3.0], [0.0, 0.0, 1.0]])
    assert np.isclose(review.affine_deviation(affine, 101, 51), 0.0)
    keystone = np.eye(3)
    keystone[2] = [1e-3, 0.0, 1.0]
    assert review.affine_deviation(keystone, 101, 51) > 0
    crossing = np.eye(3)
    crossing[2] = [0.02, 0.0, -1.0]
    assert np.isinf(review.affine_deviation(crossing, 101, 51))


def test_condition_number_is_scale_free_and_zero_for_identity():
    assert np.isclose(review.condition_number(np.eye(3), 101, 51), 0.0)
    h = np.array([[1.0, 0.2, 3.0], [0.1, 1.2, -2.0], [1e-4, 2e-4, 1.0]])
    assert np.isclose(review.condition_number(h, 101, 51),
                      review.condition_number(7.0 * h, 101, 51))
