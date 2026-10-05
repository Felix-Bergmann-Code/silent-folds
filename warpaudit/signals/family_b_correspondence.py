"""Family B: correspondence and fitted-geometry diagnostics (specification §7.1).

These are measurements of the matcher and its in-sample fit.  They are not
surrogates for landmark error: no annotation is available through
``SignalContext``.  Quantities with pixel units are also stored normalised by
the fixed working-frame diagonal so datasets with different resolutions can
be compared without erasing the original units.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError

from ..registration.fitting import condition_number
from ..types import FeatureBundle, ImageFrame, SignalContext
from .registry import SignalSpec, register_signal

__all__ = ["compute_correspondence_geometry"]


def _normalised_entropy(points: np.ndarray, frame: ImageFrame, bins: int = 4) -> float:
    if len(points) < 2:
        return float("nan")
    x = np.clip((points[:, 0] + 0.5) / frame.working_width, 0.0, 1.0)
    y = np.clip((points[:, 1] + 0.5) / frame.working_height, 0.0, 1.0)
    hist, _, _ = np.histogram2d(y, x, bins=bins, range=((0, 1), (0, 1)))
    p = hist.ravel() / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / math.log(bins * bins))


def _hull_fraction(points: np.ndarray, frame: ImageFrame) -> float:
    if len(np.unique(points, axis=0)) < 3:
        return float("nan")
    try:
        area = float(ConvexHull(points).volume)  # ``volume`` is area in 2-D.
    except QhullError:
        return float("nan")
    image_area = float(frame.working_height * frame.working_width)
    return area / image_area if image_area > 0 else float("nan")


def _extrapolation_fraction(points: np.ndarray, query: np.ndarray) -> float:
    if len(np.unique(points, axis=0)) < 3 or len(query) == 0:
        return float("nan")
    try:
        inside = Delaunay(points).find_simplex(query) >= 0
    except QhullError:
        return float("nan")
    return float(1.0 - inside.mean())


def _add_distribution(bundle: FeatureBundle, prefix: str, values: np.ndarray, unit: str) -> None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    reason = "no finite values"
    bundle.add(
        f"{prefix}_mean", float(np.mean(finite)) if len(finite) else None, unit=unit, reason=reason
    )
    bundle.add(
        f"{prefix}_median",
        float(np.median(finite)) if len(finite) else None,
        unit=unit,
        reason=reason,
    )
    bundle.add(
        f"{prefix}_p95",
        float(np.percentile(finite, 95)) if len(finite) else None,
        unit=unit,
        reason=reason,
    )


def compute_correspondence_geometry(ctx: SignalContext) -> FeatureBundle:
    bundle = FeatureBundle(family="B")
    result = ctx.result
    moving = result.matches_moving
    fixed = result.matches_fixed
    if moving is None or fixed is None:
        bundle.add("family_available", None, reason="matching correspondence arrays absent")
        return bundle

    moving = np.asarray(moving, dtype=np.float64)
    fixed = np.asarray(fixed, dtype=np.float64)
    if moving.shape != fixed.shape or moving.ndim != 2 or moving.shape[1] != 2:
        bundle.add("family_available", None, reason="correspondence arrays are not matching (N, 2)")
        return bundle

    n = len(moving)
    inlier = (
        np.asarray(result.inlier_mask, dtype=bool)
        if result.inlier_mask is not None and len(result.inlier_mask) == n
        else np.ones(n, dtype=bool)
    )
    n_inlier = int(inlier.sum())
    bundle.add("match_count", float(n), unit="count")
    bundle.add("inlier_count", float(n_inlier), unit="count")
    bundle.add(
        "inlier_ratio", n_inlier / n if n else None, unit="fraction", reason="no correspondences"
    )

    capacity = result.diagnostics.get("retained_capacity")
    if isinstance(capacity, int | float) and capacity > 0:
        bundle.add("match_capacity_ratio", n / float(capacity), unit="fraction")
    else:
        bundle.add(
            "match_capacity_ratio",
            None,
            unit="fraction",
            reason="matcher did not report retained_capacity",
        )

    if result.match_scores is not None and len(result.match_scores) == n:
        _add_distribution(bundle, "native_score", np.asarray(result.match_scores), "native")
    else:
        for name in ("native_score_mean", "native_score_median", "native_score_p95"):
            bundle.add(name, None, unit="native", reason="native match scores absent")

    transform = result.forward_moving_to_fixed
    residual = np.linalg.norm(transform.apply(moving) - fixed, axis=1)
    _add_distribution(bundle, "fit_residual_px", residual[inlier], "working pixels")
    diagonal = ctx.pair.coordinates.fixed.working_diagonal
    _add_distribution(
        bundle, "fit_residual_norm", residual[inlier] / diagonal, "fixed working diagonal"
    )

    mf = ctx.pair.coordinates.moving
    ff = ctx.pair.coordinates.fixed
    bundle.add("match_spatial_entropy_moving", _normalised_entropy(moving, mf), unit="[0,1]")
    bundle.add("match_spatial_entropy_fixed", _normalised_entropy(fixed, ff), unit="[0,1]")
    bundle.add(
        "inlier_hull_fraction_moving",
        _hull_fraction(moving[inlier], mf),
        unit="fraction",
        reason="fewer than three non-collinear inliers",
    )
    bundle.add(
        "inlier_hull_fraction_fixed",
        _hull_fraction(fixed[inlier], ff),
        unit="fraction",
        reason="fewer than three non-collinear inliers",
    )

    grid = ctx.grid.points_fixed[ctx.grid.valid_mask]
    bundle.add(
        "extrapolation_fraction_valid_grid",
        _extrapolation_fraction(fixed[inlier], grid),
        unit="fraction",
        reason="inlier hull or valid grid unavailable",
    )
    bundle.add(
        "design_condition_moving",
        condition_number(moving[inlier]),
        unit="ratio",
        reason="insufficient inlier geometry",
    )
    bundle.add(
        "design_condition_fixed",
        condition_number(fixed[inlier]),
        unit="ratio",
        reason="insufficient inlier geometry",
    )
    return bundle


register_signal(
    SignalSpec(
        family="B",
        description="correspondence counts, scores, residuals, coverage, and conditioning",
        requires=("forward_transform", "correspondences", "grid"),
        compute=compute_correspondence_geometry,
        definition_version="1",
        cost_class="cheap",
        notes="Fit residual is in-sample agreement, not true registration error.",
    )
)
