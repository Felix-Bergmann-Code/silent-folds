"""Valid support: where a pairwise score may legitimately be computed (§5.3).

Appearance metrics are computed over the documented intersection of the fixed
image's valid support and the warped moving image's valid support, excluding
padding and invalid pixels. Two numbers always travel with such a score: the
overlap fraction and whether the score was computable at all. Cropped or
out-of-bounds support must not disappear from operational evaluation just
because its similarity score is inconvenient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import ImageFrame
from .transforms import Transform

__all__ = [
    "SupportMask",
    "frame_content_mask",
    "intersect_support",
    "warp_support_mask",
]


@dataclass(frozen=True)
class SupportMask:
    """A boolean mask in the fixed working frame plus its provenance."""

    mask: np.ndarray  # (H, W) bool in the FIXED working frame
    definition: str
    fixed_height: int
    fixed_width: int

    def __post_init__(self) -> None:
        m = np.asarray(self.mask, dtype=bool)
        if m.shape != (self.fixed_height, self.fixed_width):
            raise ValueError("mask shape does not match declared fixed frame")
        object.__setattr__(self, "mask", m)

    @property
    def count(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def fraction(self) -> float:
        total = self.fixed_height * self.fixed_width
        return float(self.count) / float(total) if total else float("nan")


def frame_content_mask(frame: ImageFrame) -> np.ndarray:
    """True inside real image content, False inside recorded padding."""
    mask = np.zeros((frame.working_height, frame.working_width), dtype=bool)
    y0, y1 = frame.pad_top, frame.working_height - frame.pad_bottom
    x0, x1 = frame.pad_left, frame.working_width - frame.pad_right
    if y1 > y0 and x1 > x0:
        mask[y0:y1, x0:x1] = True
    return mask


def warp_support_mask(
    transform: Transform,
    moving: ImageFrame,
    fixed: ImageFrame,
    *,
    moving_content: np.ndarray | None = None,
) -> np.ndarray:
    """Support of the warped moving image, expressed in the fixed working frame.

    Implemented by *backward* sampling: for each fixed pixel centre, find its
    pre-image under the forward map and test whether it lands inside moving
    content. This is the same direction used for raster warping and is kept
    distinct from the forward point map (spec §5.2).

    A non-invertible forward transform yields an all-False mask; the caller
    reports empty support as an explicit status rather than a zero score.
    """
    if moving_content is None:
        moving_content = frame_content_mask(moving)

    ys, xs = np.mgrid[0 : fixed.working_height, 0 : fixed.working_width]
    fixed_pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)

    inverse = _inverse_of(transform)
    if inverse is None:
        return np.zeros((fixed.working_height, fixed.working_width), dtype=bool)

    src = inverse.apply(fixed_pts)
    ok = np.isfinite(src).all(axis=1)
    cols = np.rint(src[:, 0]).astype(np.int64)
    rows = np.rint(src[:, 1]).astype(np.int64)
    inside = (
        ok
        & (cols >= 0)
        & (cols < moving.working_width)
        & (rows >= 0)
        & (rows < moving.working_height)
    )
    out = np.zeros(fixed_pts.shape[0], dtype=bool)
    idx = np.nonzero(inside)[0]
    out[idx] = moving_content[rows[idx], cols[idx]]
    return out.reshape(fixed.working_height, fixed.working_width)


def _inverse_of(transform: Transform):
    """Analytic inverse where the family provides one, else None.

    Note the naming discipline of spec §5.2: this is the *inverse map* of an
    estimate, used for raster sampling. It is never stored as, or compared
    with, an independently estimated reverse registration.
    """
    inv = getattr(transform, "inverse", None)
    if inv is None:
        return None
    try:
        return inv()
    except (ValueError, np.linalg.LinAlgError):
        return None


def intersect_support(
    transform: Transform,
    moving: ImageFrame,
    fixed: ImageFrame,
    *,
    fixed_valid: np.ndarray | None = None,
) -> SupportMask:
    """Documented intersection used by every Family A appearance metric."""
    fixed_content = frame_content_mask(fixed)
    if fixed_valid is not None:
        fixed_content = fixed_content & np.asarray(fixed_valid, dtype=bool)
    warped = warp_support_mask(transform, moving, fixed)
    return SupportMask(
        mask=fixed_content & warped,
        definition="fixed content AND warped moving content, padding excluded",
        fixed_height=fixed.working_height,
        fixed_width=fixed.working_width,
    )
