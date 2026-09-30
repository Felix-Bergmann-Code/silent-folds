"""Geometric endpoints (specification §6.1).

For moving landmarks ``x_i`` and fixed landmarks ``y_i`` in ORIGINAL
coordinates::

    e_i      = ||T_original(x_i) - y_i||_2
    tre_px   = mean_i(e_i)
    tre_norm = tre_px / sqrt(H_fixed_original^2 + W_fixed_original^2)

Rules that this module enforces rather than documents:

* An out-of-image but **finite** forward mapping is a geometric error, not a
  reason to omit a landmark.
* A non-finite map, or no output at all, yields a **missing** TRE. Never zero,
  never a fabricated large value.
* ``p95`` over 6-10 landmarks is a sparse descriptive statistic. It is
  computed and stored, and callers are expected not to read it as dense
  clinical tail risk.

This module belongs to the evaluation process. It is the only place where
annotation arrays and transforms meet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry.transforms import Transform, as_points
from ..types import ImageFrame

__all__ = [
    "PointErrors",
    "R8_LANDMARK_SUCCESS_PX",
    "landmark_success_fraction",
    "point_errors",
]

#: Landmark-level success radius reported by the retinal comparison study
#: [R8], printed page 8. Reproduced as its own endpoint and never equated with
#: the pair-mean target (spec §6.2).
R8_LANDMARK_SUCCESS_PX = 12.5


@dataclass(frozen=True)
class PointErrors:
    """Per-pair landmark error summary in original fixed-image coordinates."""

    n_landmarks: int
    n_evaluated: int
    errors_px: np.ndarray  # (n_evaluated,) finite per-landmark distances
    tre_px: float
    tre_norm: float
    median_px: float
    max_px: float
    p95_px: float
    fixed_diagonal_px: float
    n_nonfinite: int
    defined: bool
    reason: str = ""
    fraction_exceeding: dict[float, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, float | int | bool | str]:
        row: dict[str, float | int | bool | str] = {
            "n_landmarks": self.n_landmarks,
            "n_evaluated": self.n_evaluated,
            "n_nonfinite_mapped": self.n_nonfinite,
            "tre_px": self.tre_px,
            "tre_norm": self.tre_norm,
            "tre_median_px": self.median_px,
            "tre_max_px": self.max_px,
            "tre_p95_px": self.p95_px,
            "fixed_diagonal_px": self.fixed_diagonal_px,
            "tre_defined": self.defined,
            "tre_reason": self.reason,
        }
        for thr, frac in self.fraction_exceeding.items():
            row[f"frac_landmarks_gt_{thr:g}norm"] = frac
        return row


_UNDEFINED_THRESHOLDS = (0.0025, 0.005, 0.01)


def _undefined(n_landmarks: int, diagonal: float, reason: str) -> PointErrors:
    nan = float("nan")
    return PointErrors(
        n_landmarks=n_landmarks,
        n_evaluated=0,
        errors_px=np.empty(0, dtype=np.float64),
        tre_px=nan,
        tre_norm=nan,
        median_px=nan,
        max_px=nan,
        p95_px=nan,
        fixed_diagonal_px=diagonal,
        n_nonfinite=n_landmarks,
        defined=False,
        reason=reason,
        fraction_exceeding={t: nan for t in _UNDEFINED_THRESHOLDS},
    )


def point_errors(
    transform_original: Transform | None,
    points_moving: np.ndarray,
    points_fixed: np.ndarray,
    fixed_frame: ImageFrame,
    *,
    normalised_thresholds: tuple[float, ...] = _UNDEFINED_THRESHOLDS,
) -> PointErrors:
    """Evaluate a forward map against landmark correspondences.

    ``transform_original`` must already be in the ORIGINAL frame -- convert
    with :func:`warpaudit.geometry.coordinates.to_original_frame` first.
    Passing ``None`` (an explicit registration failure) returns an undefined
    result whose ``reason`` records why, so the case survives in every
    operational denominator.
    """
    pm = as_points(points_moving)
    pf = as_points(points_fixed)
    if pm.shape != pf.shape:
        raise ValueError("moving and fixed landmark arrays must have equal shape")
    diagonal = fixed_frame.original_diagonal
    n = int(pm.shape[0])

    if transform_original is None:
        return _undefined(n, diagonal, "no returned transform")
    if not transform_original.is_valid():
        return _undefined(n, diagonal, "non-finite or degenerate transform")
    if n == 0:
        return _undefined(0, diagonal, "no annotated landmarks")

    mapped = transform_original.apply(pm)
    finite = np.isfinite(mapped).all(axis=1)
    n_nonfinite = int((~finite).sum())
    if not finite.all():
        return _undefined(
            n,
            diagonal,
            f"{n_nonfinite} landmark(s) map to a non-finite location; transform is not evaluable",
        )

    # Out-of-image but finite mappings are retained deliberately (§6.2).
    err = np.linalg.norm(mapped - pf, axis=1)
    tre_px = float(err.mean())
    tre_norm = tre_px / diagonal

    fraction_exceeding = {
        float(t): float(np.mean(err / diagonal > t)) for t in normalised_thresholds
    }

    return PointErrors(
        n_landmarks=n,
        n_evaluated=n,
        errors_px=err,
        tre_px=tre_px,
        tre_norm=tre_norm,
        median_px=float(np.median(err)),
        max_px=float(err.max()),
        p95_px=float(np.percentile(err, 95)),
        fixed_diagonal_px=diagonal,
        n_nonfinite=n_nonfinite,
        defined=True,
        reason="",
        fraction_exceeding=fraction_exceeding,
    )


def landmark_success_fraction(
    errors: PointErrors, radius_px: float = R8_LANDMARK_SUCCESS_PX
) -> float:
    """Fraction of landmarks within ``radius_px`` -- the [R8] landmark-level rule.

    Reported separately from the pair-mean binary target; the two are not
    interchangeable (spec §6.2).
    """
    if not errors.defined or errors.n_evaluated == 0:
        return float("nan")
    return float(np.mean(errors.errors_px <= radius_px))
