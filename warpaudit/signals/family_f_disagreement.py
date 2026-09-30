"""Family F: cross-pipeline disagreement (specification §7.1).

The difference between this pipeline's map and a preselected second pipeline's
map for the same pair, summarised on the shared prespecified grid.

Two properties of the spec entry drive the implementation. "Missing output
explicitly recorded": a comparison pipeline that produced no transform yields
an availability reason and a recorded count, never a quietly smaller
denominator. And "correlated errors can make agreement misleading": two
pipelines that share a matcher family, a checkpoint lineage, or the same
fitting policy can agree precisely because they fail the same way, so a small
disagreement is not evidence of correctness and this family is not a ceiling
on either pipeline's error.

The comparison results arrive through ``auxiliary_results`` under the
``pipeline:<id>`` key convention, alongside the ``e2:`` perturbation entries
consumed by family E2.
"""

from __future__ import annotations

import numpy as np

from ..types import FeatureBundle, RegistrationResult, SignalContext
from .family_d_plausibility import moving_frame_lattice
from .registry import SignalSpec, register_signal

__all__ = ["PIPELINE_PREFIX", "compute_cross_pipeline_disagreement"]

PIPELINE_PREFIX = "pipeline:"


def _comparison_results(ctx: SignalContext) -> dict[str, RegistrationResult]:
    return {
        key[len(PIPELINE_PREFIX) :]: result
        for key, result in sorted(ctx.auxiliary_results.items())
        if key.startswith(PIPELINE_PREFIX)
        and key[len(PIPELINE_PREFIX) :] != ctx.result.pipeline_id
    }


def compute_cross_pipeline_disagreement(ctx: SignalContext) -> FeatureBundle:
    bundle = FeatureBundle(family="F")
    forward = ctx.result.forward_moving_to_fixed
    comparisons = _comparison_results(ctx)
    names = (
        "disagreement_mean",
        "disagreement_median",
        "disagreement_p95",
        "disagreement_invalid_fraction",
    )
    bundle.add("comparison_pipelines", float(len(comparisons)), unit="count")
    if forward is None or not comparisons:
        reason = (
            "no comparison pipeline result for this pair"
            if forward is not None
            else "this pipeline returned no transform"
        )
        for name in names:
            bundle.add(name, None, reason=reason)
        bundle.add("comparison_without_transform", float(len(comparisons)), unit="count")
        return bundle

    # The two maps are compared as functions, so both are evaluated on one
    # identical set of moving-frame points: the same deterministic lattice
    # convention family D uses, which depends only on the pair and therefore
    # cannot differ between pipelines. Distances are taken in the fixed frame
    # and normalised by its working diagonal.
    diagonal = ctx.pair.coordinates.fixed.working_diagonal
    size = max(int(ctx.config.grid_size), 2)
    sample = moving_frame_lattice(ctx, size)

    ours = np.asarray(forward.apply(sample), dtype=np.float64)
    without_transform = 0
    distances: list[np.ndarray] = []
    for result in comparisons.values():
        other = result.forward_moving_to_fixed
        if other is None:
            without_transform += 1
            continue
        theirs = np.asarray(other.apply(sample), dtype=np.float64)
        valid = np.isfinite(ours).all(axis=1) & np.isfinite(theirs).all(axis=1)
        distance = np.full(len(sample), np.nan)
        distance[valid] = np.linalg.norm(theirs[valid] - ours[valid], axis=1) / diagonal
        distances.append(distance)

    bundle.add("comparison_without_transform", float(without_transform), unit="count")
    if not distances:
        for name in names:
            bundle.add(name, None, reason="every comparison pipeline returned no transform")
        return bundle

    stacked = np.concatenate(distances)
    finite = stacked[np.isfinite(stacked)]
    bundle.add(
        "disagreement_invalid_fraction",
        float(1.0 - finite.size / stacked.size),
        unit="fraction",
    )
    reason = "no finite disagreement sample"
    bundle.add(
        "disagreement_mean",
        float(finite.mean()) if finite.size else None,
        unit="frame diagonal",
        reason=reason,
    )
    bundle.add(
        "disagreement_median",
        float(np.median(finite)) if finite.size else None,
        unit="frame diagonal",
        reason=reason,
    )
    bundle.add(
        "disagreement_p95",
        float(np.percentile(finite, 95)) if finite.size else None,
        unit="frame diagonal",
        reason=reason,
    )
    return bundle


register_signal(
    SignalSpec(
        family="F",
        description="disagreement with a preselected second pipeline on a shared grid",
        requires=("forward_transform", "auxiliary_results", "grid"),
        compute=compute_cross_pipeline_disagreement,
        definition_version="1",
        cost_class="cheap",
        notes=(
            "Cheap only because both pipelines are already cached; the comparison "
            "pipeline's registration cost is real and is budgeted with it. Agreement "
            "between correlated pipelines is not evidence of correctness."
        ),
    )
)
