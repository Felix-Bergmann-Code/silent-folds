from __future__ import annotations

import numpy as np

from warpaudit.evaluation.perturbations import translation_perturbations, warp_by_input_map


def test_translation_perturbations_are_reproducible_and_bounded() -> None:
    first = translation_perturbations(100, 200, count=8, seed=17, max_fraction=0.02)
    second = translation_perturbations(100, 200, count=8, seed=17, max_fraction=0.02)
    assert len(first) == 8
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
        assert np.max(np.abs(left[:2, 2])) <= 2.0


def test_warp_by_input_map_obeys_forward_translation() -> None:
    image = np.zeros((9, 11), dtype=np.uint8)
    image[3, 4] = 255
    translation = np.array([[1, 0, 2], [0, 1, -1], [0, 0, 1]], dtype=float)
    perturbed = warp_by_input_map(image, translation)
    assert perturbed[2, 6] == 255
