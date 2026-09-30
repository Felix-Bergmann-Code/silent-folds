"""Canonical geometry: transforms, coordinate conversion, support, resampling."""

from .coordinates import (
    identity_frame,
    make_frame,
    map_points_original_to_working,
    map_points_working_to_original,
    to_original_frame,
    to_working_frame,
)
from .grids import hpatches_grid, prespecified_grid
from .resample import resample_to_frame, warp_moving_to_fixed
from .support import SupportMask, frame_content_mask, intersect_support, warp_support_mask
from .transforms import (
    AffineTransform,
    ComposedTransform,
    HomographyTransform,
    ThinPlateSplineTransform,
    Transform,
    identity,
)

__all__ = [
    "AffineTransform",
    "ComposedTransform",
    "HomographyTransform",
    "SupportMask",
    "ThinPlateSplineTransform",
    "Transform",
    "frame_content_mask",
    "hpatches_grid",
    "identity",
    "identity_frame",
    "intersect_support",
    "make_frame",
    "map_points_original_to_working",
    "map_points_working_to_original",
    "prespecified_grid",
    "resample_to_frame",
    "to_original_frame",
    "to_working_frame",
    "warp_moving_to_fixed",
    "warp_support_mask",
]
