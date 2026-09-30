"""Feature registry: access requirements, units, missingness, cost (spec §7).

Every signal declares what it needs *before* it runs. That declaration is how
a missing input becomes an explicit availability reason instead of a silent
``nan``, and how the capability table of §7.3 is generated rather than
maintained by hand.

The registry also enforces the hard rule of §7.3/§12.3: a signal's declared
requirements can name images, the registration result, a reverse *estimate*,
auxiliary pipeline results, and the ground-truth-free grid. They can never
name annotations, evaluation masks derived from ground truth, errors, or
labels -- there is no vocabulary for those here, and ``tests/test_label_access``
walks the import graph to confirm none is smuggled in.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

from ..types import FeatureBundle, SignalContext

__all__ = [
    "REQUIREMENTS",
    "SignalSpec",
    "available_families",
    "compute_families",
    "get_signal",
    "register_signal",
]

Requirement = Literal[
    "moving_image",
    "fixed_image",
    "forward_transform",
    "correspondences",
    "match_scores",
    "reverse_estimate",
    "auxiliary_results",
    "grid",
    "refit_capable",
]

REQUIREMENTS: tuple[str, ...] = (
    "moving_image",
    "fixed_image",
    "forward_transform",
    "correspondences",
    "match_scores",
    "reverse_estimate",
    "auxiliary_results",
    "grid",
    "refit_capable",
)


@dataclass(frozen=True)
class SignalSpec:
    family: str
    description: str
    requires: tuple[str, ...]
    compute: Callable[[SignalContext], FeatureBundle]
    definition_version: str = "1"
    cost_class: str = "cheap"  # cheap | moderate | expensive
    notes: str = ""

    def missing_requirements(self, ctx: SignalContext) -> list[str]:
        missing: list[str] = []
        for req in self.requires:
            if (
                req == "moving_image"
                and "moving" not in ctx.images
                or req == "fixed_image"
                and "fixed" not in ctx.images
                or req == "forward_transform"
                and ctx.result.forward_moving_to_fixed is None
                or req == "correspondences"
                and ctx.result.matches_moving is None
                or req == "match_scores"
                and ctx.result.match_scores is None
                or req == "reverse_estimate"
                and ctx.reverse_estimate is None
                or req == "auxiliary_results"
                and not ctx.auxiliary_results
            ):
                missing.append(req)
        return missing


_REGISTRY: dict[str, SignalSpec] = {}


def register_signal(spec: SignalSpec) -> SignalSpec:
    unknown = sorted(set(spec.requires) - set(REQUIREMENTS))
    if unknown:
        raise ValueError(
            f"signal {spec.family!r} declares unknown requirement(s) {unknown}. "
            "Annotations, labels, and ground-truth-derived masks have no requirement "
            "vocabulary here by design (§7.3)."
        )
    _REGISTRY[spec.family] = spec
    return spec


def get_signal(family: str) -> SignalSpec:
    try:
        return _REGISTRY[family]
    except KeyError as exc:
        raise KeyError(
            f"no signal registered for family {family!r}; available: {sorted(_REGISTRY)}"
        ) from exc


def available_families() -> list[str]:
    return sorted(_REGISTRY)


def compute_families(ctx: SignalContext, families: Iterable[str]) -> dict[str, FeatureBundle]:
    """Compute several families, timing each and recording unmet requirements.

    A family whose requirements are unmet still produces a bundle: an empty
    one carrying the reason. A dropped row would make missingness invisible,
    and §7.3 requires missingness to be reported, not hidden.
    """
    out: dict[str, FeatureBundle] = {}
    for family in families:
        spec = get_signal(family)
        missing = spec.missing_requirements(ctx)
        if missing:
            bundle = FeatureBundle(family=family)
            bundle.add(
                "family_available",
                None,
                reason=f"unmet requirements: {', '.join(missing)}",
            )
            out[family] = bundle
            continue
        wall0, cpu0 = time.perf_counter(), time.process_time()
        bundle = spec.compute(ctx)
        bundle.wall_time_s = time.perf_counter() - wall0
        bundle.cpu_time_s = time.process_time() - cpu0
        out[family] = bundle
    return out
