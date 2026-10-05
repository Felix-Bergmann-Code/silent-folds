"""Family C: independently estimated inverse consistency (specification §7.1a)."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..geometry.transforms import HomographyTransform
from ..types import FeatureBundle, RegistrationResult, SignalContext
from .registry import SignalSpec, register_signal

__all__ = ["compute_cycle_consistency"]


def _summary(bundle: FeatureBundle, prefix: str, values: np.ndarray, diagonal: float) -> None:
    finite = values[np.isfinite(values)] / diagonal
    reason = "no finite grid displacement"
    bundle.add(
        f"{prefix}_mean",
        float(np.mean(finite)) if len(finite) else None,
        unit="frame diagonal",
        reason=reason,
    )
    bundle.add(
        f"{prefix}_median",
        float(np.median(finite)) if len(finite) else None,
        unit="frame diagonal",
        reason=reason,
    )
    bundle.add(
        f"{prefix}_p95",
        float(np.percentile(finite, 95)) if len(finite) else None,
        unit="frame diagonal",
        reason=reason,
    )


def _reverse_overlap(
    forward: RegistrationResult, reverse: RegistrationResult, tolerance: float
) -> float:
    """Fraction of forward matches recovered (swapped) by the reverse matcher."""
    arrays = (
        forward.matches_moving,
        forward.matches_fixed,
        reverse.matches_moving,
        reverse.matches_fixed,
    )
    if any(value is None for value in arrays):
        return float("nan")
    fm, ff, rm, rf = (np.asarray(value) for value in arrays)
    if len(fm) == 0 or len(rm) == 0:
        return float("nan")
    forward_pairs = np.concatenate((fm, ff), axis=1)
    reverse_swapped = np.concatenate((rf, rm), axis=1)
    distance, _ = cKDTree(reverse_swapped).query(forward_pairs, k=1)
    return float(np.mean(distance <= tolerance))


def compute_cycle_consistency(ctx: SignalContext) -> FeatureBundle:
    bundle = FeatureBundle(family="C")
    forward = ctx.result.forward_moving_to_fixed
    reverse_result = ctx.reverse_estimate
    reverse = None if reverse_result is None else reverse_result.forward_moving_to_fixed
    if forward is None or reverse is None:
        bundle.add(
            "family_available", None, reason="independent forward or reverse estimate absent"
        )
        return bundle

    # The fixed grid is the input domain of the independent reverse estimate.
    fixed_grid = ctx.grid.points_fixed
    back_to_moving = reverse.apply(fixed_grid)
    cycled_fixed = forward.apply(back_to_moving)
    valid = np.isfinite(back_to_moving).all(axis=1) & np.isfinite(cycled_fixed).all(axis=1)
    displacement = np.full(len(fixed_grid), np.nan)
    displacement[valid] = np.linalg.norm(cycled_fixed[valid] - fixed_grid[valid], axis=1)
    _summary(bundle, "cycle", displacement, ctx.pair.coordinates.fixed.working_diagonal)
    bundle.add("cycle_invalid_fraction", float(1.0 - valid.mean()), unit="fraction")

    if isinstance(forward, HomographyTransform):
        try:
            analytic = forward.inverse().apply(fixed_grid)
            disagreement = np.linalg.norm(back_to_moving - analytic, axis=1)
            _summary(
                bundle,
                "reverse_vs_analytic_inverse",
                disagreement,
                ctx.pair.coordinates.moving.working_diagonal,
            )
        except ValueError:
            for name in (
                "reverse_vs_analytic_inverse_mean",
                "reverse_vs_analytic_inverse_median",
                "reverse_vs_analytic_inverse_p95",
            ):
                bundle.add(name, None, unit="frame diagonal", reason="forward map not invertible")
    else:
        for name in (
            "reverse_vs_analytic_inverse_mean",
            "reverse_vs_analytic_inverse_median",
            "reverse_vs_analytic_inverse_p95",
        ):
            bundle.add(name, None, unit="frame diagonal", reason="analytic inverse unavailable")

    tolerance = float(ctx.config.options.get("correspondence_overlap_tolerance_px", 1.0))
    bundle.add(
        "reverse_correspondence_overlap",
        _reverse_overlap(ctx.result, reverse_result, tolerance),
        unit="fraction",
        reason="forward/reverse correspondence arrays absent",
    )
    return bundle


register_signal(
    SignalSpec(
        family="C",
        description="independent forward/reverse cycle consistency",
        requires=("forward_transform", "reverse_estimate", "grid"),
        compute=compute_cycle_consistency,
        definition_version="1",
        cost_class="expensive",
        notes="The reverse estimate is never replaced by the analytic inverse.",
    )
)
