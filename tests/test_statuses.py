from __future__ import annotations

import numpy as np

from warpaudit.geometry.coordinates import identity_frame
from warpaudit.geometry.support import intersect_support
from warpaudit.geometry.transforms import HomographyTransform
from warpaudit.labels.errors import point_errors
from warpaudit.labels.targets import case_outcome
from warpaudit.types import RegistrationStatus


def test_no_output_has_missing_tre_and_is_auto_rejected() -> None:
    frame = identity_frame("f", (20, 20))
    errors = point_errors(None, np.array([[1, 1]]), np.array([[1, 1]]), frame)
    outcome = case_outcome(RegistrationStatus.NO_MATCHES, errors)
    assert not errors.defined and np.isnan(errors.tre_px)
    assert outcome.operational_failure and not outcome.eligible_for_acceptance
    assert outcome.bounded_loss == 10.0


def test_out_of_frame_finite_mapping_is_scored() -> None:
    frame = identity_frame("f", (10, 10))
    t = HomographyTransform(np.array([[1, 0, 100], [0, 1, 0], [0, 0, 1]], float))
    errors = point_errors(t, np.array([[1, 1]]), np.array([[1, 1]]), frame)
    assert errors.defined and errors.tre_px == 100.0


def test_nonfinite_landmark_mapping_invalidates_tre() -> None:
    frame = identity_frame("f", (10, 10))
    t = HomographyTransform(np.array([[1, 0, 0], [0, 1, 0], [1, 0, -1]], float))
    errors = point_errors(t, np.array([[0, 0], [1, 0]]), np.array([[0, 0], [1, 0]]), frame)
    assert not errors.defined and np.isnan(errors.tre_px)


def test_empty_support_is_visible() -> None:
    frame = identity_frame("f", (8, 8))
    t = HomographyTransform(np.array([[1, 0, 100], [0, 1, 100], [0, 0, 1]], float))
    support = intersect_support(t, frame, frame)
    assert support.count == 0 and support.fraction == 0.0
