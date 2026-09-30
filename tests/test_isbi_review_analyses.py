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
