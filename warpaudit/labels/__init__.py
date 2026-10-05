"""Label construction. Imported only by the evaluation process (spec §7.3)."""

from .errors import R8_LANDMARK_SUCCESS_PX, PointErrors, landmark_success_fraction, point_errors
from .targets import (
    BOUNDED_LOSS_CAP,
    BOUNDED_LOSS_SENSITIVITY_CAPS,
    PRIMARY_TAU,
    SECONDARY_TAUS,
    CaseOutcome,
    bounded_loss,
    case_outcome,
    pixel_equivalent,
)

__all__ = [
    "BOUNDED_LOSS_CAP",
    "BOUNDED_LOSS_SENSITIVITY_CAPS",
    "CaseOutcome",
    "PRIMARY_TAU",
    "PointErrors",
    "R8_LANDMARK_SUCCESS_PX",
    "SECONDARY_TAUS",
    "bounded_loss",
    "case_outcome",
    "landmark_success_fraction",
    "pixel_equivalent",
    "point_errors",
]
