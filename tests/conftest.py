from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from warpaudit.geometry.coordinates import identity_frame
from warpaudit.types import CoordinateMetadata, PairInput


@pytest.fixture
def pair_input(tmp_path: Path) -> PairInput:
    moving = identity_frame("moving", (100, 120))
    fixed = identity_frame("fixed", (100, 120))
    return PairInput(
        dataset_id="fixture",
        pair_id="fixture/pair-1",
        group_id="fixture/patient/1",
        group_basis="patient",
        moving_image_id="moving",
        fixed_image_id="fixed",
        moving_path=tmp_path / "moving.png",
        fixed_path=tmp_path / "fixed.png",
        coordinates=CoordinateMetadata(moving=moving, fixed=fixed),
        acquisition_meta={"modality": "fundus", "patient_id": None}
        if False
        else {"modality": "fundus"},
    )


@pytest.fixture
def correspondences() -> tuple[np.ndarray, np.ndarray]:
    src = np.array(
        [[5, 5], [60, 5], [110, 5], [5, 50], [60, 50], [110, 50], [5, 90], [110, 90]],
        dtype=float,
    )
    dst = src + np.array([3.0, -2.0])
    return src, dst
