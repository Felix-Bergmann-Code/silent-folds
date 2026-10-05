"""Local-only registration overlays for the development geometry review.

Panels contain source-image pixels and therefore belong under an ignored local
directory. The review report may record pair IDs and observations, but this
module never treats rendered images as releasable artifacts.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw

from ..geometry.resample import resample_to_frame, warp_moving_to_fixed
from ..geometry.transforms import Transform
from ..types import CoordinateMetadata

__all__ = ["contact_sheet", "render_registration_panel"]


def _rgb8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"expected grayscale/RGB/RGBA image, got {array.shape}")
    return np.clip(array[:, :, :3], 0.0, 255.0).round().astype(np.uint8)


def render_registration_panel(
    moving_original: np.ndarray,
    fixed_original: np.ndarray,
    transform_working: Transform,
    coordinates: CoordinateMetadata,
    *,
    title: str = "",
) -> tuple[np.ndarray, float]:
    """Render fixed, warped-moving, and red/green agreement views.

    Fixed luminance is placed in red and warped-moving luminance in green, so
    aligned structures appear yellow while displacement produces colored
    edges. Invalid warped support is darkened in the agreement view and its
    fraction is returned for the review ledger.
    """
    moving = resample_to_frame(moving_original, coordinates.moving)
    fixed = resample_to_frame(fixed_original, coordinates.fixed)
    warped = warp_moving_to_fixed(
        moving, transform_working, coordinates.moving, coordinates.fixed
    )
    fixed_rgb = _rgb8(fixed)
    warped_rgb = _rgb8(warped.image)
    fixed_luma = np.mean(fixed_rgb.astype(np.float64), axis=2)
    warped_luma = np.mean(warped_rgb.astype(np.float64), axis=2)
    agreement = np.zeros_like(fixed_rgb)
    agreement[:, :, 0] = fixed_luma.round().astype(np.uint8)
    agreement[:, :, 1] = warped_luma.round().astype(np.uint8)
    agreement[~warped.valid] = (fixed_rgb[~warped.valid] * 0.25).astype(np.uint8)

    header, footer = 32, 24
    height, width = fixed_rgb.shape[:2]
    canvas = Image.new("RGB", (3 * width, height + header + footer), color=(20, 20, 20))
    for index, panel in enumerate((fixed_rgb, warped_rgb, agreement)):
        canvas.paste(Image.fromarray(panel), (index * width, header))
    draw = ImageDraw.Draw(canvas)
    labels = ("fixed", "warped moving", "agreement: fixed=red, moving=green")
    for index, label in enumerate(labels):
        draw.text((index * width + 8, 8), label, fill=(245, 245, 245))
    if title:
        draw.text((8, height + header + 4), title[:160], fill=(245, 245, 245))
    return np.asarray(canvas), float(np.mean(warped.valid))


def contact_sheet(
    panels: Sequence[np.ndarray], *, columns: int = 2, thumbnail_width: int = 720
) -> np.ndarray:
    """Build a deterministic thumbnail sheet for efficient visual inspection."""
    if not panels:
        raise ValueError("at least one panel is required")
    if columns <= 0 or thumbnail_width <= 0:
        raise ValueError("columns and thumbnail_width must be positive")
    thumbnails: list[Image.Image] = []
    for panel in panels:
        image = Image.fromarray(_rgb8(panel))
        height = max(1, round(image.height * thumbnail_width / image.width))
        thumbnails.append(image.resize((thumbnail_width, height), Image.Resampling.LANCZOS))
    cell_height = max(image.height for image in thumbnails)
    rows = (len(thumbnails) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumbnail_width, rows * cell_height), (16, 16, 16))
    for index, image in enumerate(thumbnails):
        x = (index % columns) * thumbnail_width
        y = (index // columns) * cell_height
        sheet.paste(image, (x, y))
    return np.asarray(sheet)
