from __future__ import annotations

import numpy as np

from warpaudit.evaluation.overlays import contact_sheet, render_registration_panel
from warpaudit.geometry.coordinates import make_frame
from warpaudit.geometry.transforms import identity
from warpaudit.types import CoordinateMetadata


def test_identity_overlay_has_full_support_and_deterministic_shape() -> None:
    image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    frame = make_frame("fixture", (12, 16), long_edge=16)
    panel, overlap = render_registration_panel(
        image,
        image,
        identity(),
        CoordinateMetadata(moving=frame, fixed=frame),
        title="fixture",
    )
    assert panel.shape == (12 + 32 + 24, 16 * 3, 3)
    assert overlap == 1.0
    sheet = contact_sheet([panel, panel], columns=2, thumbnail_width=96)
    assert sheet.shape[1] == 192
    assert sheet.dtype == np.uint8


def test_contact_sheet_rejects_empty_input() -> None:
    try:
        contact_sheet([])
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:  # pragma: no cover - documents required failure
        raise AssertionError("empty contact sheet should fail")
