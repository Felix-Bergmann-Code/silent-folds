"""Family E2: coordinate-corrected input-perturbation stability (§7.1, §7.2a).

This is an attributed adaptation of Tian, Hu, and Iglesias [R4] for 2-D
sparse registration.  Each auxiliary registration must record the affine
maps applied to the original moving and fixed input coordinates.  The result
is corrected back before spread is measured; uncorrected maps are rejected.
"""

from __future__ import annotations

import numpy as np

from ..geometry.transforms import AffineTransform, ComposedTransform, HomographyTransform, Transform
from ..types import FeatureBundle, SignalContext
from .family_e1_stability import positional_spread
from .registry import SignalSpec, register_signal

__all__ = ["compute_perturbation_stability", "correct_perturbed_transform"]

MOVING_CORRECTION_KEY = "input_to_perturbed_moving"
FIXED_CORRECTION_KEY = "input_to_perturbed_fixed"


def correct_perturbed_transform(
    transform: Transform,
    input_to_perturbed_moving: np.ndarray,
    input_to_perturbed_fixed: np.ndarray,
) -> Transform:
    """Return ``inv(P_f) @ T_perturbed @ P_m`` in baseline coordinates."""
    Pm = np.asarray(input_to_perturbed_moving, dtype=np.float64)
    Pf = np.asarray(input_to_perturbed_fixed, dtype=np.float64)
    if Pm.shape != (3, 3) or Pf.shape != (3, 3):
        raise ValueError("input perturbation coordinate maps must be 3x3")
    if not np.isfinite(Pm).all() or not np.isfinite(Pf).all():
        raise ValueError("input perturbation coordinate maps must be finite")
    if isinstance(transform, HomographyTransform):
        return HomographyTransform(np.linalg.inv(Pf) @ transform.matrix @ Pm)
    return ComposedTransform(
        (
            AffineTransform(Pm[:2]),
            transform,
            AffineTransform(np.linalg.inv(Pf)[:2]),
        )
    )


def _moving_grid(ctx: SignalContext) -> np.ndarray:
    points = ctx.grid.points_fixed
    moving, fixed = ctx.pair.coordinates.moving, ctx.pair.coordinates.fixed
    return np.column_stack(
        (
            (points[:, 0] + 0.5) / fixed.working_width * moving.working_width - 0.5,
            (points[:, 1] + 0.5) / fixed.working_height * moving.working_height - 0.5,
        )
    )


def compute_perturbation_stability(ctx: SignalContext) -> FeatureBundle:
    bundle = FeatureBundle(family="E2")
    candidates = [
        result for key, result in sorted(ctx.auxiliary_results.items()) if key.startswith("e2:")
    ]
    attempted = len(candidates)
    if attempted == 0:
        bundle.add("family_available", None, reason="no e2:* auxiliary registrations")
        return bundle

    grid = _moving_grid(ctx)
    predictions: list[np.ndarray] = []
    invalid = 0
    for result in candidates:
        transform = result.forward_moving_to_fixed
        diagnostics = result.diagnostics
        if (
            transform is None
            or MOVING_CORRECTION_KEY not in diagnostics
            or FIXED_CORRECTION_KEY not in diagnostics
        ):
            invalid += 1
            continue
        try:
            corrected = correct_perturbed_transform(
                transform,
                diagnostics[MOVING_CORRECTION_KEY],
                diagnostics[FIXED_CORRECTION_KEY],
            )
            mapped = corrected.apply(grid)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            invalid += 1
            continue
        if not np.isfinite(mapped).all():
            invalid += 1
            continue
        predictions.append(mapped)

    bundle.add("perturbation_valid_fits", float(len(predictions)), unit="count")
    bundle.add("perturbation_invalid_fit_fraction", invalid / attempted, unit="fraction")
    if len(predictions) < 2:
        reason = f"only {len(predictions)} corrected perturbation fit(s); at least two required"
        bundle.add("perturbation_spread_mean", None, unit="fixed working diagonal", reason=reason)
        bundle.add("perturbation_spread_p95", None, unit="fixed working diagonal", reason=reason)
        bundle.add("perturbation_common_support_fraction", None, unit="fraction", reason=reason)
        return bundle

    stack = np.stack(predictions)
    spread = positional_spread(stack) / ctx.pair.coordinates.fixed.working_diagonal
    inside = (
        (stack[..., 0] >= -0.5)
        & (stack[..., 0] <= ctx.pair.coordinates.fixed.working_width - 0.5)
        & (stack[..., 1] >= -0.5)
        & (stack[..., 1] <= ctx.pair.coordinates.fixed.working_height - 0.5)
    )
    common = np.all(inside, axis=0) & ctx.grid.valid_mask
    bundle.add("perturbation_spread_mean", float(np.mean(spread)), unit="fixed working diagonal")
    bundle.add(
        "perturbation_spread_p95", float(np.percentile(spread, 95)), unit="fixed working diagonal"
    )
    bundle.add("perturbation_common_support_fraction", float(np.mean(common)), unit="fraction")
    bundle.add(
        "perturbation_spread_common_mean",
        float(np.mean(spread[common])) if common.any() else None,
        unit="fixed working diagonal",
        reason="corrected perturbations have empty common support",
    )

    # Subset sensitivity at the configured rerun counts (§7.2a). The
    # perturbation draws are prefix-stable -- offset k is the same whether 4 or
    # 16 were requested -- so the first k corrected fits *are* the B = k
    # estimate, and the check costs no extra registration.
    for count in sorted({int(b) for b in ctx.config.options.get("perturbation_B_sensitivity", ())}):
        name = f"perturbation_spread_mean_B{count}"
        if count < 2:
            bundle.add(name, None, unit="fixed working diagonal", reason="B must be at least two")
        elif count > len(predictions):
            bundle.add(
                name,
                None,
                unit="fixed working diagonal",
                reason=f"only {len(predictions)} valid corrected fit(s) cached; B={count} needs more",
            )
        else:
            subset = positional_spread(stack[:count]) / ctx.pair.coordinates.fixed.working_diagonal
            bundle.add(name, float(np.mean(subset)), unit="fixed working diagonal")
    return bundle


register_signal(
    SignalSpec(
        family="E2",
        description="coordinate-corrected input-perturbation stability",
        requires=("forward_transform", "auxiliary_results", "grid"),
        compute=compute_perturbation_stability,
        definition_version="2",
        cost_class="expensive",
        notes="Adaptation of Tian, Hu, and Iglesias [R4]; requires full registration reruns.",
    )
)
