"""Exact projective-pole diagnostics for homographies.

A homography maps ``(x, y)`` with projective denominator

``d(x, y) = h31*x + h32*y + h33``.

Because ``d`` is affine, its minimum and maximum over an axis-aligned image
rectangle occur at corners.  A zero therefore lies in the closed rectangle if
and only if the four corner values bracket zero.  The crossing decision below
is exact with respect to the supplied floating-point matrix and does not use a
sampling lattice.

Denominator magnitudes are not intrinsic to a homography: multiplying the
matrix by a non-zero scalar leaves the mapping unchanged.  Every magnitude
returned here is consequently computed after Frobenius normalization.  Its
coordinate frame is still part of the definition and must be reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "HOMOGRAPHY_NORMALIZATION",
    "PoleDiagnostics",
    "image_corners",
    "normalise_homography",
    "projective_pole_diagnostics",
    "projective_pole_crosses_circle",
]

HOMOGRAPHY_NORMALIZATION = "frobenius_unit_norm"


def normalise_homography(matrix: np.ndarray) -> np.ndarray:
    """Return a finite 3x3 homography with unit Frobenius norm.

    This normalization is defined even when ``h33`` is zero, unlike the common
    ``h33 = 1`` convention.  It fixes only homogeneous matrix scale; it does
    not make denominator magnitudes invariant to a change of coordinate frame.
    """

    h = np.asarray(matrix, dtype=np.float64)
    if h.shape != (3, 3):
        raise ValueError(f"homography must have shape (3, 3), got {h.shape}")
    if not np.isfinite(h).all():
        raise ValueError("homography must be finite")
    norm = float(np.linalg.norm(h, ord="fro"))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("homography Frobenius norm must be positive and finite")
    return h / norm


def image_corners(width: int, height: int) -> np.ndarray:
    """Four closed-domain corners under the integer pixel-centre convention."""

    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive, got {(width, height)}")
    return np.asarray(
        ((0.0, 0.0), (width - 1.0, 0.0), (0.0, height - 1.0), (width - 1.0, height - 1.0)),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class PoleDiagnostics:
    """Scale-fixed diagnostics in one declared pixel coordinate frame."""

    denominator_crosses_image: bool
    corner_denominators: tuple[float, float, float, float]
    denominator_min: float
    denominator_max: float
    pole_clearance_diagonal_fraction: float
    min_abs_sampled_denominator: float
    sampled_grid_size: int
    homography_normalization: str = HOMOGRAPHY_NORMALIZATION


def projective_pole_diagnostics(
    matrix: np.ndarray,
    *,
    width: int,
    height: int,
    magnitude_grid_size: int = 32,
) -> PoleDiagnostics:
    """Compute an exact crossing flag and a descriptive sampled magnitude.

    ``denominator_crosses_image`` uses only the four corner extrema and is
    resolution-independent.  ``min_abs_sampled_denominator`` deliberately
    remains a descriptive ``N x N`` lattice statistic for continuity with the
    existing Figure 3 right panel; it is never used to decide whether a pole
    crosses the image.  Samples include the four image corners.

    Parameters are interpreted in integer-centred pixel coordinates over the
    closed rectangle ``[0, width-1] x [0, height-1]``.
    """

    grid_size = int(magnitude_grid_size)
    if grid_size < 2:
        raise ValueError("magnitude_grid_size must be at least 2")
    h = normalise_homography(matrix)
    corners = image_corners(width, height)
    corner_den = np.c_[corners, np.ones(4)] @ h[2]
    minimum = float(np.min(corner_den))
    maximum = float(np.max(corner_den))

    # This is the exact affine-on-a-rectangle test.  Equality includes a pole
    # that touches the image boundary or a corner.
    crossing = minimum <= 0.0 <= maximum
    line_norm = float(np.hypot(h[2, 0], h[2, 1]))
    image_diagonal = float(np.hypot(width - 1.0, height - 1.0))
    if crossing:
        clearance = 0.0
    elif line_norm == 0.0:
        # A nonzero constant denominator is an affine map whose pole line is
        # at infinity.  It is not merely a large finite clearance.
        clearance = float("inf")
    elif image_diagonal == 0.0:
        raise ValueError("pole clearance is undefined for a one-pixel image")
    else:
        # With no crossing, d has one sign over the rectangle.  Its minimum
        # absolute value is therefore attained at a corner.  Dividing by the
        # line-normal norm gives Euclidean distance to the pole line; dividing
        # again by the image diagonal makes the quantity scale-free.
        clearance = float(np.min(np.abs(corner_den)) / line_norm / image_diagonal)

    xs = np.linspace(0.0, float(width - 1), grid_size)
    ys = np.linspace(0.0, float(height - 1), grid_size)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    sampled_den = h[2, 0] * xx + h[2, 1] * yy + h[2, 2]
    magnitude = float(np.min(np.abs(sampled_den)))
    return PoleDiagnostics(
        denominator_crosses_image=bool(crossing),
        corner_denominators=tuple(float(value) for value in corner_den),  # type: ignore[arg-type]
        denominator_min=minimum,
        denominator_max=maximum,
        pole_clearance_diagonal_fraction=clearance,
        min_abs_sampled_denominator=magnitude,
        sampled_grid_size=grid_size,
    )


def projective_pole_crosses_circle(
    matrix: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    radius: float,
) -> bool:
    """Return whether the projective denominator vanishes in a circular FOV.

    The zero set of the denominator is the line ``h31*x+h32*y+h33=0``.
    It intersects a closed circle exactly when the centre-to-line distance is
    no larger than the radius.  Homogeneous matrix scale cancels from the
    comparison, but the circle and homography must use the same coordinates.
    """

    h = normalise_homography(matrix)
    cx, cy, r = float(center_x), float(center_y), float(radius)
    if not np.isfinite([cx, cy, r]).all() or r <= 0:
        raise ValueError("circle centre must be finite and radius must be positive")
    a, b, c = h[2]
    line_norm = float(np.hypot(a, b))
    if line_norm == 0.0:
        return bool(c == 0.0)
    return bool(abs(a * cx + b * cy + c) <= r * line_norm)
