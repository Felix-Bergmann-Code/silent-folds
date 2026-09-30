from __future__ import annotations

import numpy as np

from warpaudit.geometry.coordinates import make_frame, to_original_frame, to_working_frame
from warpaudit.geometry.resample import warp_moving_to_fixed
from warpaudit.geometry.transforms import (
    AffineTransform,
    HomographyTransform,
    ThinPlateSplineTransform,
    identity,
)
from warpaudit.labels.errors import point_errors
from warpaudit.types import CoordinateMetadata


def test_original_frame_preserves_working_fit_validity():
    frame = make_frame("fire", (2912, 2912), long_edge=728)
    coords = CoordinateMetadata(frame, frame)
    working = HomographyTransform(
        np.array([[0.001, 0, 1000], [0, 1, 0], [0, 0, 1]], float)
    )
    assert working.is_valid()
    collapsed = HomographyTransform(np.linalg.inv(coords.A_f) @ working.matrix @ coords.A_m)
    assert not collapsed.is_valid()  # The old label path rejected this solely after scaling.
    original = to_original_frame(working, coords)
    assert original.is_valid()
    moving = np.array([[100, 200], [800, 900]], float)
    expected = collapsed.apply(moving)
    np.testing.assert_allclose(original.apply(moving), expected)
    errors = point_errors(original, moving, moving, frame)
    assert errors.defined
    np.testing.assert_allclose(errors.tre_px, np.linalg.norm(expected - moving, axis=1).mean())


def test_original_frame_still_rejects_invalid_working_fit():
    frame = make_frame("fire", (2912, 2912), long_edge=728)
    coords = CoordinateMetadata(frame, frame)
    working = HomographyTransform(np.diag([1e-12, 1, 1]))
    original = to_original_frame(working, coords)
    assert not original.is_valid()
    assert not point_errors(original, np.array([[1, 2]]), np.array([[1, 2]]), frame).defined


def test_identity_translation_and_homography() -> None:
    points = np.array([[0.0, 0.0], [3.0, 4.0], [10.0, 7.0]])
    np.testing.assert_allclose(identity().apply(points), points)
    translation = HomographyTransform(np.array([[1, 0, 4], [0, 1, -3], [0, 0, 1]], float))
    np.testing.assert_allclose(translation.apply(points), points + [4, -3])
    projective = HomographyTransform(np.array([[2, 0, 1], [0, 3, 2], [0.01, 0, 1]], float))
    expected = np.column_stack(
        (
            (2 * points[:, 0] + 1) / (0.01 * points[:, 0] + 1),
            (3 * points[:, 1] + 2) / (0.01 * points[:, 0] + 1),
        )
    )
    np.testing.assert_allclose(projective.apply(points), expected)


def test_original_working_roundtrip_with_unequal_sizes_and_padding() -> None:
    moving = make_frame("m", (80, 160), long_edge=100, pad_to_square=True)
    fixed = make_frame("f", (200, 100), long_edge=100, pad_to_square=True)
    coords = CoordinateMetadata(moving=moving, fixed=fixed)
    original = HomographyTransform(np.array([[1.1, 0.05, 9], [-0.03, 0.9, 4], [0.0002, 0, 1]]))
    working = to_working_frame(original, coords)
    recovered = to_original_frame(working, coords)
    points = np.array([[0, 0], [50, 20], [159, 79]], float)
    np.testing.assert_allclose(recovered.apply(points), original.apply(points), atol=1e-10)
    assert moving.working_height == moving.working_width == 100
    assert fixed.working_height == fixed.working_width == 100
    assert moving.pad_top > 0 and fixed.pad_left > 0


def test_direct_point_scoring_is_independent_of_raster_interpolation() -> None:
    frame = make_frame("x", (8, 8))
    transform = HomographyTransform(np.array([[1, 0, 2], [0, 1, 0], [0, 0, 1]], float))
    moving_points = np.array([[1.0, 1.0], [3.0, 2.0]])
    fixed_points = moving_points + [2, 0]
    before = point_errors(transform, moving_points, fixed_points, frame)
    warped = warp_moving_to_fixed(np.arange(64).reshape(8, 8), transform, frame, frame)
    after = point_errors(transform, moving_points, fixed_points, frame)
    assert warped.valid.any()
    assert before.tre_px == after.tre_px == 0.0


def test_small_nonrigid_tps_fixture() -> None:
    norm = AffineTransform(np.array([[1, 0, 0], [0, 1, 0]], float))
    transform = ThinPlateSplineTransform(
        control_points=np.array([[0, 0], [1, 0], [0, 1]], float),
        weights=np.array([[0.05, 0], [-0.02, 0.01], [0, -0.03]], float),
        affine=np.array([[0, 0], [1, 0], [0, 1]], float),
        normalisation=norm,
        denormalisation=norm,
    )
    points = np.array([[0.25, 0.25], [0.5, 0.5]])
    mapped = transform.apply(points)
    assert transform.is_valid()
    assert np.isfinite(mapped).all()
    assert not np.allclose(mapped, points)
