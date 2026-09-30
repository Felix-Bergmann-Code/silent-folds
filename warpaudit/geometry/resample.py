"""The shared evaluation resampler (specification §5.2).

A common resampler is good engineering, not a research contribution. Its only
job is to make appearance scores comparable: one documented pixel-centre
convention, one interpolation rule, one padding rule, one valid-support mask.

Point evaluation never passes through this module. Direct landmark transforms
are independent of raster resampling and are verified as such in
``tests/test_geometry.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import ImageFrame
from .support import _inverse_of, frame_content_mask
from .transforms import Transform

__all__ = ["ResampleResult", "resample_to_frame", "warp_moving_to_fixed"]


@dataclass(frozen=True)
class ResampleResult:
    image: np.ndarray  # (H, W) or (H, W, C) float64 in the fixed working frame
    valid: np.ndarray  # (H, W) bool -- pixels with real sampled content
    convention: str


def _sample_bilinear(image: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear sample at ``(N, 2)`` pixel-centre coordinates.

    Returns ``(values, inside)``. Samples outside the image return 0 and
    ``inside=False``; the caller never treats an outside sample as a dark
    pixel.
    """
    if image.ndim == 2:
        image = image[:, :, None]
    H, W, C = image.shape

    x, y = xy[:, 0], xy[:, 1]
    finite = np.isfinite(x) & np.isfinite(y)
    inside = finite & (x >= -0.5) & (x <= W - 0.5) & (y >= -0.5) & (y <= H - 0.5)

    xc = np.clip(np.where(finite, x, 0.0), 0.0, W - 1.0)
    yc = np.clip(np.where(finite, y, 0.0), 0.0, H - 1.0)
    x0 = np.floor(xc).astype(np.int64)
    y0 = np.floor(yc).astype(np.int64)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    wx = xc - x0
    wy = yc - y0

    out = (
        image[y0, x0] * ((1 - wx) * (1 - wy))[:, None]
        + image[y0, x1] * (wx * (1 - wy))[:, None]
        + image[y1, x0] * ((1 - wx) * wy)[:, None]
        + image[y1, x1] * (wx * wy)[:, None]
    )
    out[~inside] = 0.0
    return (out[:, 0] if C == 1 else out), inside


def resample_to_frame(image: np.ndarray, frame: ImageFrame) -> np.ndarray:
    """Bring an ORIGINAL-resolution image into its declared working frame.

    Uses the frame's own ``to_working`` affine, so resize and padding cannot
    drift apart from the coordinate metadata.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.shape[:2] != (frame.original_height, frame.original_width):
        raise ValueError(
            f"image {img.shape[:2]} does not match frame original "
            f"{(frame.original_height, frame.original_width)}"
        )
    inv = np.linalg.inv(frame.to_working)
    ys, xs = np.mgrid[0 : frame.working_height, 0 : frame.working_width]
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], axis=1)
    src = pts @ inv.T
    values, inside = _sample_bilinear(img, src[:, :2])
    shape = (frame.working_height, frame.working_width)
    if values.ndim == 2:
        shape = shape + (values.shape[1],)
    out = values.reshape(shape)
    out[~inside.reshape(frame.working_height, frame.working_width)] = 0.0
    return out


def warp_moving_to_fixed(
    moving_working: np.ndarray,
    transform: Transform,
    moving: ImageFrame,
    fixed: ImageFrame,
) -> ResampleResult:
    """Warp a working-frame moving image into the fixed working frame.

    Backward sampling with fixed-to-moving coordinates, as required by §5.2.
    An unusable transform produces an all-invalid result rather than an
    exception, so the caller can record ``invalid_transform`` and keep the
    attempted case in every denominator.
    """
    inverse = _inverse_of(transform)
    convention = "backward sampling, bilinear, zero padding, pixel centre (0,0) at (0.0, 0.0)"
    if inverse is None:
        empty_shape = (fixed.working_height, fixed.working_width)
        if moving_working.ndim == 3:
            empty_shape = empty_shape + (moving_working.shape[2],)
        return ResampleResult(
            image=np.zeros(empty_shape, dtype=np.float64),
            valid=np.zeros((fixed.working_height, fixed.working_width), dtype=bool),
            convention=convention + "; forward transform not invertible",
        )

    ys, xs = np.mgrid[0 : fixed.working_height, 0 : fixed.working_width]
    fixed_pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    src = inverse.apply(fixed_pts)
    values, inside = _sample_bilinear(np.asarray(moving_working, dtype=np.float64), src)

    content = frame_content_mask(moving)
    cols = np.clip(np.rint(src[:, 0]).astype(np.int64), 0, moving.working_width - 1)
    rows = np.clip(np.rint(src[:, 1]).astype(np.int64), 0, moving.working_height - 1)
    valid = inside & content[rows, cols]

    shape = (fixed.working_height, fixed.working_width)
    if values.ndim == 2:
        shape = shape + (values.shape[1],)
    return ResampleResult(
        image=values.reshape(shape),
        valid=valid.reshape(fixed.working_height, fixed.working_width),
        convention=convention,
    )
