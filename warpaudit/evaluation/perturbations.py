"""Deterministic development-only input perturbations for the E2 smoke test.

This is an attributed 2-D sparse-registration adaptation of the transformation
equivariance baseline by Tian, Hu, and Iglesias. The smoke distribution uses
moving-image translations only. It verifies full matcher reruns and coordinate
correction without prematurely freezing the focused-study perturbation family.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..geometry.coordinates import identity_frame
from ..geometry.resample import warp_moving_to_fixed
from ..geometry.transforms import HomographyTransform
from ..signals.family_e2_perturbation import (
    FIXED_CORRECTION_KEY,
    MOVING_CORRECTION_KEY,
)
from ..types import PairInput, Registrar, RegistrationResult

__all__ = [
    "CoordinateCorrectingRegistrar",
    "translation_perturbations",
    "warp_by_input_map",
]


def translation_perturbations(
    height: int,
    width: int,
    *,
    count: int,
    seed: int,
    max_fraction: float,
) -> tuple[np.ndarray, ...]:
    """Draw reproducible baseline-to-perturbed translation matrices."""
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if count < 2:
        raise ValueError("at least two perturbations are required")
    if not 0.0 < max_fraction < 0.5:
        raise ValueError("max_fraction must lie in (0, 0.5)")
    radius = max_fraction * min(height, width)
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(-radius, radius, size=(count, 2))
    return tuple(
        np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
        for dx, dy in offsets
    )


def warp_by_input_map(image: np.ndarray, baseline_to_perturbed: np.ndarray) -> np.ndarray:
    """Resample one working image into the perturbed frame with zero padding."""
    array = np.asarray(image, dtype=np.float64)
    frame = identity_frame("e2-working", array.shape[:2])
    warped = warp_moving_to_fixed(
        array,
        HomographyTransform(np.asarray(baseline_to_perturbed, dtype=np.float64)),
        frame,
        frame,
    )
    return np.clip(warped.image, 0.0, 255.0).round().astype(np.uint8)


class CoordinateCorrectingRegistrar:
    """Attach immutable input-coordinate maps to full perturbed reruns."""

    def __init__(
        self,
        base: Registrar,
        corrections: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    ) -> None:
        self.base = base
        self.pipeline_id = base.pipeline_id
        self.corrections = dict(corrections)

    def register(self, pair: PairInput, seed: int) -> RegistrationResult:
        try:
            moving_map, fixed_map = self.corrections[(pair.pair_id, pair.condition)]
        except KeyError as exc:
            raise ValueError(f"no E2 correction maps for condition {pair.condition!r}") from exc
        result = self.base.register(pair, seed)
        result.diagnostics = {
            **result.diagnostics,
            MOVING_CORRECTION_KEY: np.asarray(moving_map, dtype=np.float64).tolist(),
            FIXED_CORRECTION_KEY: np.asarray(fixed_map, dtype=np.float64).tolist(),
            "e2_smoke_distribution": "moving translation, uniform square",
        }
        return result
