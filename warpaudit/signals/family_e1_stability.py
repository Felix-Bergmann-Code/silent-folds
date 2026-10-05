"""Family E1: cached-correspondence bootstrap stability (specification §7.2).

E1 refits cached matches; it does not rerun the matcher and therefore cannot
measure matcher or image ambiguity.  The implementation exposes the two
development controls required by §7.2a: full correspondences with varying
RANSAC seeds, and resampled correspondences with the RANSAC seed held fixed.
"""

from __future__ import annotations

import numpy as np

from ..registration.fitting import FittingPolicy, fit_correspondences
from ..types import FeatureBundle, SignalContext
from .registry import SignalSpec, register_signal

__all__ = ["compute_cached_match_stability", "positional_spread"]


def positional_spread(predictions: np.ndarray) -> np.ndarray:
    """RMS vector displacement for each point, using denominator ``1/B``.

    This is intentionally not ``std(norm(displacement))``.  Predictions at
    equal radius on opposite sides of their mean have positive spread.
    """
    pred = np.asarray(predictions, dtype=np.float64)
    if pred.ndim != 3 or pred.shape[2] != 2:
        raise ValueError("predictions must have shape (B, N, 2)")
    if pred.shape[0] < 2:
        return np.full(pred.shape[1], np.nan)
    centre = np.mean(pred, axis=0)
    return np.sqrt(np.mean(np.sum((pred - centre) ** 2, axis=2), axis=0))


def _moving_grid(ctx: SignalContext) -> np.ndarray:
    """Map the prespecified fixed grid to matching relative moving positions.

    The map is based only on frame extents, not on annotations or the estimated
    warp.  It supplies a fixed, ground-truth-free set of transform-domain
    points even when moving and fixed working sizes differ.
    """
    pts = ctx.grid.points_fixed
    mf, ff = ctx.pair.coordinates.moving, ctx.pair.coordinates.fixed
    unit_x = (pts[:, 0] + 0.5) / ff.working_width
    unit_y = (pts[:, 1] + 0.5) / ff.working_height
    return np.column_stack((unit_x * mf.working_width - 0.5, unit_y * mf.working_height - 0.5))


def _policy(ctx: SignalContext) -> FittingPolicy:
    raw = dict(ctx.config.options.get("fitting_policy", {}))
    allowed = {
        "threshold_px",
        "max_iters",
        "confidence",
        "min_matches",
        "refine_on_inliers",
        "name",
    }
    return FittingPolicy(**{k: v for k, v in raw.items() if k in allowed})


def _fit(ctx: SignalContext, src: np.ndarray, dst: np.ndarray, seed: int):
    options = dict(ctx.config.options.get("fitting_policy", {}))
    return fit_correspondences(
        src,
        dst,
        _policy(ctx),
        seed=seed,
        transform_family=str(options.get("transform_family", "homography")),
        tps_regularisation=float(options.get("tps_regularisation", 1e-3)),
    )


def _summarise(
    bundle: FeatureBundle,
    prefix: str,
    predictions: list[np.ndarray],
    invalid: int,
    attempted: int,
    diagonal: float,
    valid_grid: np.ndarray,
) -> None:
    bundle.add(f"{prefix}_valid_fits", float(len(predictions)), unit="count")
    bundle.add(f"{prefix}_invalid_fit_fraction", invalid / attempted, unit="fraction")
    if len(predictions) < 2:
        reason = f"only {len(predictions)} valid fit(s); at least two required"
        for suffix in ("spread_mean", "spread_p95", "spread_valid_mean", "spread_valid_p95"):
            bundle.add(f"{prefix}_{suffix}", None, unit="fixed working diagonal", reason=reason)
        return

    spread = positional_spread(np.stack(predictions)) / diagonal
    bundle.add(f"{prefix}_spread_mean", float(np.mean(spread)), unit="fixed working diagonal")
    bundle.add(
        f"{prefix}_spread_p95", float(np.percentile(spread, 95)), unit="fixed working diagonal"
    )
    supported = spread[valid_grid]
    bundle.add(
        f"{prefix}_spread_valid_mean",
        float(np.mean(supported)) if len(supported) else None,
        unit="fixed working diagonal",
        reason="valid-support grid is empty",
    )
    bundle.add(
        f"{prefix}_spread_valid_p95",
        float(np.percentile(supported, 95)) if len(supported) else None,
        unit="fixed working diagonal",
        reason="valid-support grid is empty",
    )


def compute_cached_match_stability(ctx: SignalContext) -> FeatureBundle:
    bundle = FeatureBundle(family="E1")
    src = ctx.result.matches_moving
    dst = ctx.result.matches_fixed
    if src is None or dst is None:
        bundle.add("family_available", None, reason="cached correspondences absent")
        return bundle
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        bundle.add("family_available", None, reason="correspondences are not matching (N, 2)")
        return bundle

    B = int(ctx.config.bootstrap_B)
    n = len(src)
    if n < 4:
        bundle.add("family_available", None, reason=f"{n} correspondences; homography needs four")
        return bundle

    grid = _moving_grid(ctx)
    rng = np.random.default_rng(ctx.config.seed)
    draws = [rng.integers(0, n, size=n) for _ in range(B)]

    def run(indices: list[np.ndarray], seeds: list[int]) -> tuple[list[np.ndarray], int]:
        predictions: list[np.ndarray] = []
        invalid = 0
        for index, seed in zip(indices, seeds, strict=False):
            fit = _fit(ctx, src[index], dst[index], seed)
            if fit.ok and fit.transform is not None:
                mapped = fit.transform.apply(grid)
                if np.isfinite(mapped).all():
                    predictions.append(mapped)
                    continue
            invalid += 1
        return predictions, invalid

    base_seed = int(ctx.config.seed)
    boot, boot_invalid = run(draws, [base_seed + b for b in range(B)])
    fixed_seed, fixed_invalid = run(draws, [base_seed] * B)
    whole = [np.arange(n)] * B
    seed_only, seed_invalid = run(whole, [base_seed + b for b in range(B)])

    diagonal = ctx.pair.coordinates.fixed.working_diagonal
    valid_grid = np.asarray(ctx.grid.valid_mask, dtype=bool)
    _summarise(bundle, "bootstrap", boot, boot_invalid, B, diagonal, valid_grid)
    _summarise(bundle, "resample_fixed_seed", fixed_seed, fixed_invalid, B, diagonal, valid_grid)
    _summarise(bundle, "seed_only", seed_only, seed_invalid, B, diagonal, valid_grid)
    # Full-study names. ``bootstrap`` remains as a compatibility alias for the
    # pilot cache; it varied resampling and estimator seed simultaneously.
    _summarise(bundle, "combined", boot, boot_invalid, B, diagonal, valid_grid)
    if len(boot) >= 2 and len(fixed_seed) >= 2 and len(seed_only) >= 2:
        combined = positional_spread(np.stack(boot)) / diagonal
        resampling = positional_spread(np.stack(fixed_seed)) / diagonal
        estimator = positional_spread(np.stack(seed_only)) / diagonal
        # Descriptive non-additivity only; this is deliberately not labelled a
        # variance decomposition because RANSAC and resampling can interact.
        interaction = combined - resampling - estimator
        bundle.add("factor_interaction_mean", float(np.mean(interaction)),
                   unit="fixed working diagonal")
        bundle.add("factor_interaction_p95_abs", float(np.percentile(np.abs(interaction), 95)),
                   unit="fixed working diagonal")
    return bundle


register_signal(
    SignalSpec(
        family="E1",
        description="cached-match bootstrap/refit positional stability",
        requires=("forward_transform", "correspondences", "grid", "refit_capable"),
        compute=compute_cached_match_stability,
        definition_version="2",
        cost_class="moderate",
        notes="Conditional on cached matches; excludes matcher and image ambiguity.",
    )
)
