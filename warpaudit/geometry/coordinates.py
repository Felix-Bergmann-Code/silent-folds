"""Original <-> working coordinate maps and canonical-frame conversion (§5.2).

The one equation that must hold everywhere in this project::

    T_original = inverse(A_f) . T_working . A_m

``A_m`` and ``A_f`` are the explicit homogeneous affine maps from ORIGINAL
pixel-centre coordinates to the WORKING coordinates a pipeline actually saw.
Every published pixel endpoint is evaluated in original fixed-image
coordinates. Aspect ratio is preserved by construction and padding is recorded
explicitly, so no code path ever has to infer a stretch from a size ratio.

Pixel-centre convention: the centre of pixel ``(row 0, col 0)`` is ``(x, y) =
(0.0, 0.0)``. Under a resize by factor ``s`` this gives

    x_working = s * (x_original + 0.5) - 0.5

which is the only mapping that keeps image corners at ``-0.5`` in both frames.
Using ``x_working = s * x_original`` instead introduces a half-pixel bias that
grows with ``s`` and would silently corrupt landmark errors.
"""

from __future__ import annotations

import numpy as np

from ..types import CoordinateMetadata, ImageFrame
from .transforms import (
    AffineTransform,
    ComposedTransform,
    Transform,
    as_points,
)

__all__ = [
    "affine_from_matrix",
    "frame_to_transform",
    "identity_frame",
    "make_frame",
    "pixel_centre_scale_matrix",
    "to_original_frame",
    "to_working_frame",
    "working_scale_of",
]


def pixel_centre_scale_matrix(
    scale_x: float,
    scale_y: float,
    pad_left: float = 0.0,
    pad_top: float = 0.0,
) -> np.ndarray:
    """Homogeneous 3x3 map original -> working under the pixel-centre rule."""
    A = np.eye(3, dtype=np.float64)
    A[0, 0] = scale_x
    A[1, 1] = scale_y
    A[0, 2] = 0.5 * scale_x - 0.5 + pad_left
    A[1, 2] = 0.5 * scale_y - 0.5 + pad_top
    return A


def identity_frame(image_id: str, original_hw: tuple[int, int]) -> ImageFrame:
    """A frame whose working coordinates equal its original coordinates."""
    H, W = int(original_hw[0]), int(original_hw[1])
    return ImageFrame(
        image_id=image_id,
        original_height=H,
        original_width=W,
        working_height=H,
        working_width=W,
        to_working=np.eye(3),
        resample_note="native resolution; no resize or padding",
    )


def make_frame(
    image_id: str,
    original_hw: tuple[int, int],
    *,
    long_edge: int | None = None,
    pad_to_square: bool = False,
    pad_centre: bool = True,
) -> ImageFrame:
    """Build a working frame by aspect-preserving resize and explicit padding.

    Parameters
    ----------
    long_edge:
        Target size of the longer side. ``None`` keeps native resolution.
    pad_to_square:
        Pad the resized image to ``long_edge x long_edge``. Padding is
        recorded in the frame and folded into ``A``; it is never inferred from
        the size ratio downstream.

    Anisotropic scaling is deliberately not offered. A pipeline that requires a
    square input gets padding, not a stretch, because a stretch cannot be
    undone by dividing by a diagonal (spec §5.2).
    """
    H, W = int(original_hw[0]), int(original_hw[1])
    if H <= 0 or W <= 0:
        raise ValueError(f"non-positive image size {(H, W)}")

    if long_edge is None:
        scale = 1.0
        work_h, work_w = H, W
    else:
        scale = float(long_edge) / float(max(H, W))
        work_h = int(round(H * scale))
        work_w = int(round(W * scale))

    pad_left = pad_top = pad_right = pad_bottom = 0
    if pad_to_square:
        side = long_edge if long_edge is not None else max(work_h, work_w)
        if pad_centre:
            pad_left = (side - work_w) // 2
            pad_top = (side - work_h) // 2
        pad_right = side - work_w - pad_left
        pad_bottom = side - work_h - pad_top
        if pad_right < 0 or pad_bottom < 0:
            raise ValueError("padding target smaller than resized image")
        work_h, work_w = side, side

    return ImageFrame(
        image_id=image_id,
        original_height=H,
        original_width=W,
        working_height=work_h,
        working_width=work_w,
        to_working=pixel_centre_scale_matrix(scale, scale, pad_left, pad_top),
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        resample_note=(
            f"aspect-preserving resize by {scale:.6f}"
            + (f"; centred pad to {work_w}x{work_h}" if pad_to_square else "")
        ),
    )


def affine_from_matrix(A: np.ndarray) -> AffineTransform:
    """Wrap a 3x3 affine matrix as a :class:`AffineTransform`."""
    return AffineTransform(np.asarray(A, dtype=np.float64)[:2])


def frame_to_transform(frame: ImageFrame) -> AffineTransform:
    return affine_from_matrix(frame.to_working)


def working_scale_of(frame: ImageFrame) -> float:
    """Isotropic working-per-original scale factor of a frame."""
    return float(frame.to_working[0, 0])


def to_original_frame(t_working: Transform, coords: CoordinateMetadata) -> Transform:
    """``inverse(A_f) . T_working . A_m`` (spec §5.2).

    The result maps ORIGINAL moving pixel centres to ORIGINAL fixed pixel
    centres and is the only map used for landmark scoring.
    """
    A_m = affine_from_matrix(coords.A_m)
    A_f_inv = affine_from_matrix(np.linalg.inv(coords.A_f))
    # Preserve the fit's validity in its working frame. A raw homography's
    # singular-value ratio changes under pixel-coordinate scaling; collapsing
    # this chain and revalidating it can reject a finite, accepted working fit
    # solely because the original image is larger. Each constituent is still
    # validated, and point_errors separately rejects non-finite mapped points.
    return ComposedTransform((A_m, t_working, A_f_inv))


def to_working_frame(t_original: Transform, coords: CoordinateMetadata) -> Transform:
    """Inverse of :func:`to_original_frame`: ``A_f . T_original . inverse(A_m)``."""
    A_m_inv = affine_from_matrix(np.linalg.inv(coords.A_m))
    A_f = affine_from_matrix(coords.A_f)
    return ComposedTransform((A_m_inv, t_original, A_f)).collapse()


def map_points_original_to_working(points: np.ndarray, frame: ImageFrame) -> np.ndarray:
    return frame_to_transform(frame).apply(as_points(points))


def map_points_working_to_original(points: np.ndarray, frame: ImageFrame) -> np.ndarray:
    return affine_from_matrix(np.linalg.inv(frame.to_working)).apply(as_points(points))


def homography_original_to_working(
    H_original: np.ndarray, coords: CoordinateMetadata
) -> np.ndarray:
    """Convert an original-frame ground-truth homography to working coordinates.

    Used only inside the evaluation process (HPatches annotations arrive in
    original coordinates).
    """
    return coords.A_f @ np.asarray(H_original, dtype=np.float64) @ np.linalg.inv(coords.A_m)


def homography_working_to_original(H_working: np.ndarray, coords: CoordinateMetadata) -> np.ndarray:
    return np.linalg.inv(coords.A_f) @ np.asarray(H_working, dtype=np.float64) @ coords.A_m
