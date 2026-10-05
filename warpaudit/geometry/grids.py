"""Evaluation grids (specification §5.3, §6.1).

Two distinct grid kinds live here and must not be confused:

``prespecified_grid``
    A ground-truth-free grid handed to *signals*. Summaries are reported both
    over the full fixed grid and restricted to valid support, so a warp cannot
    improve its score by shrinking its own evaluation region.

``hpatches_grid``
    A deterministic 20x20 grid in the ORIGINAL reference image used as
    *ground-truth supervision* for HPatches. It requires the true homography
    and therefore lives in the evaluation process only.
"""

from __future__ import annotations

import numpy as np

from ..types import EvaluationGridWithoutGroundTruth, ImageFrame
from .support import SupportMask
from .transforms import HomographyTransform

__all__ = [
    "HPATCHES_GRID_SIDE",
    "HPATCHES_MIN_VALID_POINTS",
    "hpatches_grid",
    "prespecified_grid",
]

HPATCHES_GRID_SIDE = 20
#: Fewer than 20 evaluable points is a predeclared annotation-support
#: exclusion, listed *before* model scoring (spec §6.1).
HPATCHES_MIN_VALID_POINTS = 20


def _uniform_centres(n: int, extent: int) -> np.ndarray:
    """``n`` uniformly spaced pixel-centre coordinates spanning ``extent``."""
    return (np.arange(n, dtype=np.float64) + 0.5) * (extent / n) - 0.5


def prespecified_grid(
    fixed: ImageFrame,
    *,
    size: int = 32,
    support: SupportMask | None = None,
) -> EvaluationGridWithoutGroundTruth:
    """Deterministic ``size x size`` grid over the fixed working frame."""
    xs = _uniform_centres(size, fixed.working_width)
    ys = _uniform_centres(size, fixed.working_height)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

    if support is None:
        valid = np.ones(pts.shape[0], dtype=bool)
        note = "full fixed working frame"
    else:
        cols = np.clip(np.rint(pts[:, 0]).astype(int), 0, support.fixed_width - 1)
        rows = np.clip(np.rint(pts[:, 1]).astype(int), 0, support.fixed_height - 1)
        valid = support.mask[rows, cols]
        note = support.definition

    return EvaluationGridWithoutGroundTruth(
        points_fixed=pts,
        valid_mask=valid,
        frame="working",
        spec=f"{size}x{size} uniform pixel-centre grid; valid = {note}",
    )


def hpatches_grid(
    reference_hw: tuple[int, int],
    other_hw: tuple[int, int],
    homography_ref_to_other: np.ndarray,
    *,
    side: int = HPATCHES_GRID_SIDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic HPatches evaluation points in ORIGINAL coordinates.

    Returns ``(points_reference, points_other_true, retained_mask)``. Points
    are retained only when the ground-truth homography maps them inside the
    other image. The count of retained points is recorded by the caller; a
    pair with fewer than :data:`HPATCHES_MIN_VALID_POINTS` retained points is
    excluded on annotation-support grounds before any model scoring.
    """
    Hr, Wr = int(reference_hw[0]), int(reference_hw[1])
    Ho, Wo = int(other_hw[0]), int(other_hw[1])

    xs = _uniform_centres(side, Wr)
    ys = _uniform_centres(side, Hr)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    ref = np.stack([gx.ravel(), gy.ravel()], axis=1)

    mapped = HomographyTransform(homography_ref_to_other).apply(ref)
    retained = (
        np.isfinite(mapped).all(axis=1)
        & (mapped[:, 0] >= -0.5)
        & (mapped[:, 0] <= Wo - 0.5)
        & (mapped[:, 1] >= -0.5)
        & (mapped[:, 1] <= Ho - 0.5)
    )
    return ref, mapped, retained
