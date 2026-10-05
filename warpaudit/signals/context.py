"""Assembly of the signal context from cached registration rows (spec §7.3, §12.3).

This lives inside :mod:`warpaudit.signals` on purpose. What a family is *given*
decides what it can compute, so a mistake here turns a computable feature into
a permanently cached "unmet requirement" that a later run will not retry. Being
part of the signals package puts this code under the feature code identity, so
correcting it invalidates exactly the rows it could have spoiled.

Two rules the assembly must not break:

* selectors choose which jobs to compute, never what a job may see. The
  reverse direction (family C) and the other pipeline's row for the same pair
  (family F) stay visible even when the sweep is narrowed to one pipeline or
  one direction;
* the context still has no vocabulary for annotations, labels, or
  ground-truth-derived masks, and nothing here introduces one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..geometry.grids import prespecified_grid
from ..geometry.support import intersect_support
from ..types import PairInput, RegistrationResult, SignalConfig, SignalContext
from .family_f_disagreement import PIPELINE_PREFIX
from .registry import get_signal

__all__ = ["ContextSources", "build_signal_context", "required_images"]

_IMAGE_REQUIREMENTS = {"moving_image": "moving", "fixed_image": "fixed"}


def required_images(families: Iterable[str]) -> set[str]:
    """Which images the requested families declare they need.

    Driven by the registry rather than a hard-coded family list, so adding an
    image-using family cannot silently produce unmet requirements.
    """
    needed: set[str] = set()
    for family in families:
        for requirement in get_signal(family).requires:
            if requirement in _IMAGE_REQUIREMENTS:
                needed.add(_IMAGE_REQUIREMENTS[requirement])
    return needed


@dataclass(frozen=True)
class ContextSources:
    """Cached rows a job's context is assembled from.

    ``reverse`` and ``same_pair_other_pipelines`` are the pool-derived lookups
    that selectors must not narrow.
    """

    reverse: RegistrationResult | None
    perturbations: Mapping[str, RegistrationResult]
    same_pair_other_pipelines: Mapping[str, RegistrationResult]


def build_signal_context(
    *,
    pair: PairInput,
    result: RegistrationResult,
    sources: ContextSources,
    families: Iterable[str],
    grid_size: int,
    bootstrap_B: int,
    perturbation_B: int,
    seed: int,
    fitting_policy: Mapping[str, object],
    perturbation_B_sensitivity: Sequence[int] = (),
    load_image: Callable[[str], np.ndarray],
) -> SignalContext:
    """Build the context for one cached registration job."""
    families = tuple(families)
    transform = result.forward_moving_to_fixed
    support = (
        None
        if transform is None
        else intersect_support(transform, pair.coordinates.moving, pair.coordinates.fixed)
    )
    grid = prespecified_grid(pair.coordinates.fixed, size=grid_size, support=support)

    auxiliaries: dict[str, RegistrationResult] = dict(sources.perturbations)
    for pipeline_id, other in sources.same_pair_other_pipelines.items():
        auxiliaries[f"{PIPELINE_PREFIX}{pipeline_id}"] = other

    images = {name: load_image(name) for name in sorted(required_images(families))}
    config = SignalConfig(
        bootstrap_B=bootstrap_B,
        perturbation_B=perturbation_B,
        grid_size=grid_size,
        seed=seed,
        options={
            "fitting_policy": dict(fitting_policy),
            "perturbation_B_sensitivity": tuple(int(b) for b in perturbation_B_sensitivity),
        },
    )
    return SignalContext(
        pair=pair,
        result=result,
        reverse_estimate=sources.reverse,
        auxiliary_results=auxiliaries,
        grid=grid,
        config=config,
        images=images,
    )
