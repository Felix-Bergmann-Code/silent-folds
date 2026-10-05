from __future__ import annotations

import numpy as np

from warpaudit.geometry.grids import prespecified_grid
from warpaudit.geometry.transforms import HomographyTransform
from warpaudit.signals import compute_families
from warpaudit.signals.family_e1_stability import positional_spread
from warpaudit.types import RegistrationResult, RegistrationStatus, SignalConfig, SignalContext


def test_equal_radius_opposite_predictions_have_positive_spread() -> None:
    predictions = np.array([[[1.0, 0.0]], [[-1.0, 0.0]]])
    np.testing.assert_allclose(positional_spread(predictions), [1.0])


def test_family_b_and_e1_compute_without_annotations(pair_input, correspondences) -> None:
    src, dst = correspondences
    result = RegistrationResult(
        pipeline_id="fixture",
        status=RegistrationStatus.OK,
        forward_moving_to_fixed=HomographyTransform(
            np.array([[1, 0, 3], [0, 1, -2], [0, 0, 1]], float)
        ),
        matches_moving=src,
        matches_fixed=dst,
        inlier_mask=np.ones(len(src), bool),
        match_scores=np.linspace(0.5, 1, len(src)),
        diagnostics={"retained_capacity": 16},
    )
    grid = prespecified_grid(pair_input.coordinates.fixed, size=4)
    ctx = SignalContext(
        pair=pair_input,
        result=result,
        reverse_estimate=None,
        auxiliary_results={},
        grid=grid,
        config=SignalConfig(bootstrap_B=8, seed=17),
        images={},
    )
    bundles = compute_families(ctx, ["B", "E1"])
    assert bundles["B"].values["inlier_ratio"].value == 1.0
    assert bundles["B"].values["fit_residual_norm_mean"].value < 1e-12
    assert bundles["E1"].values["bootstrap_invalid_fit_fraction"].available
