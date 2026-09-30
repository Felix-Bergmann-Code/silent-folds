"""The H4 intersection-union decision rule (specification §10.2, §10.3).

The primary confirmatory claim is an **intersection-union test** over three
linked quantities, for one M2-selected direction, learner, budget, nominal
policy coverage, and common pipeline block:

1. the lower one-sided 95% bound for **transferred target AUROC** exceeds
   ``auroc_floor`` (0.70) -- useful absolute ranking;
2. the upper one-sided 95% bound for **Gap_AUC** is below
   ``gap_margin`` (0.05) -- ranking noninferiority to the matched-budget
   reference, not merely failure to detect a difference;
3. the lower one-sided 95% bound for **Delta_policy** exceeds
   ``policy_margin`` (0.05) -- material excess accepted failure risk from the
   transferred source policy relative to equally budgeted target supervision
   on the same target population.

Each component is tested at one-sided alpha = 0.05. **No Bonferroni division
is applied**, because the global claim is made only when all three component
nulls are rejected -- that is what makes it an intersection-union test rather
than a family of separate claims. :func:`decide` refuses to run with a
corrected alpha so the rule cannot drift.

The module also writes the *conclusion sentence*. Which sentence is permitted
depends on which components passed, and letting code choose it removes the
temptation to write the strong one after seeing a partial result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bootstrap import BootstrapResult

__all__ = ["ComponentTest", "JointClaim", "decide"]


@dataclass(frozen=True)
class ComponentTest:
    name: str
    estimate: float
    bound: float
    bound_kind: str  # "lower" | "upper"
    margin: float
    passed: bool
    reliable: bool
    ci: tuple[float, float]
    note: str = ""

    def as_row(self) -> dict[str, float | bool | str]:
        return {
            "component": self.name,
            "estimate": self.estimate,
            "one_sided_bound": self.bound,
            "bound_kind": self.bound_kind,
            "margin": self.margin,
            "passed": self.passed,
            "interval_reliable": self.reliable,
            "ci_low": self.ci[0],
            "ci_high": self.ci[1],
            "note": self.note,
        }


@dataclass
class JointClaim:
    components: list[ComponentTest]
    alpha_one_sided: float
    multiplicity_rule: str
    joint_supported: bool
    conclusion: str
    warnings: list[str] = field(default_factory=list)

    def as_rows(self) -> list[dict]:
        return [c.as_row() for c in self.components]

    def summary(self) -> str:
        lines = [
            f"Intersection-union test at one-sided alpha = {self.alpha_one_sided} "
            f"({self.multiplicity_rule}; no Bonferroni division applied).",
            "",
        ]
        for c in self.components:
            mark = "PASS" if c.passed else "fail"
            rel = "" if c.reliable else "  [interval unreliable]"
            lines.append(
                f"  [{mark}] {c.name}: estimate {c.estimate:.4f}, "
                f"{c.bound_kind} one-sided bound {c.bound:.4f} vs margin {c.margin:.4f}"
                f"{rel}"
            )
        lines += ["", f"Joint claim supported: {self.joint_supported}", "", self.conclusion]
        if self.warnings:
            lines += ["", "Warnings:"] + [f"  - {w}" for w in self.warnings]
        return "\n".join(lines)


def _conclusion(ranking_useful: bool, noninferior: bool, policy_gap: bool, reliable: bool) -> str:
    """The sentence the evidence permits (§10.3)."""
    if not reliable:
        return (
            "Inconclusive. At least one interval is unreliable, so no component "
            "conclusion is drawn. Expand independent data or narrow the claim; "
            "multiplying corruptions is not a remedy."
        )
    if ranking_useful and noninferior and policy_gap:
        return (
            "Useful target ranking survived while the frozen source policy incurred "
            "materially greater accepted failure risk than equally budgeted target "
            "supervision, in this specified transfer. The claim is limited to the "
            "named datasets, pipelines, budget, and nominal acceptance target."
        )
    if policy_gap and not (ranking_useful and noninferior):
        return (
            "Policy deterioration is supported: the frozen source policy accepted "
            "materially more failures than the matched-budget target policy. Preserved "
            "ranking is NOT claimed, because the ranking components did not pass."
        )
    if ranking_useful and noninferior and not policy_gap:
        return (
            "Ranking transfer is supported: transferred target discrimination is useful "
            "and noninferior to the matched-budget reference. The policy component did "
            "not pass, so no claim of excess accepted risk is made. Failure to reject "
            "zero is not evidence of policy equivalence; that needs its own predeclared "
            "margin."
        )
    return (
        "The joint dissociation claim is not supported. Report the component estimates "
        "and intervals under their actual bounds; an inconclusive interval or a weak "
        "detector does not support the headline."
    )


def decide(
    target_auroc: BootstrapResult,
    gap_auc: BootstrapResult,
    delta_policy: BootstrapResult,
    *,
    auroc_floor: float = 0.70,
    gap_margin: float = 0.05,
    policy_margin: float = 0.05,
    alpha_one_sided: float = 0.05,
    multiplicity_rule: str = "intersection_union",
    realised_coverages: dict[str, float] | None = None,
    accepted_group_counts: dict[str, int] | None = None,
    min_accepted_groups: int = 3,
) -> JointClaim:
    """Evaluate the three-component joint criterion.

    ``target_auroc``, ``gap_auc`` and ``delta_policy`` must come from the same
    bootstrap procedure so that the components are paired. Their alpha is
    overridden here to ``alpha_one_sided`` to guarantee every bound uses the
    declared level.
    """
    if multiplicity_rule != "intersection_union":
        raise ValueError(
            "§10.2 fixes the multiplicity rule to an intersection-union test; "
            f"got {multiplicity_rule!r}"
        )
    if not 0.0 < alpha_one_sided < 0.5:
        raise ValueError("alpha_one_sided must lie in (0, 0.5)")

    for result in (target_auroc, gap_auc, delta_policy):
        result.alpha = alpha_one_sided

    warnings: list[str] = []
    for result in (target_auroc, gap_auc, delta_policy):
        if not result.reliable:
            warnings.append(
                f"{result.name}: {result.n_invalid}/{result.n_resamples} invalid resamples "
                "-- interval flagged unreliable (§10.2)"
            )

    c1 = ComponentTest(
        name="transferred target AUROC > floor",
        estimate=target_auroc.estimate,
        bound=target_auroc.lower_one_sided(),
        bound_kind="lower",
        margin=auroc_floor,
        passed=bool(target_auroc.lower_one_sided() > auroc_floor),
        reliable=target_auroc.reliable,
        ci=target_auroc.two_sided(),
        note="useful absolute ranking on the target test population",
    )
    c2 = ComponentTest(
        name="Gap_AUC < noninferiority margin",
        estimate=gap_auc.estimate,
        bound=gap_auc.upper_one_sided(),
        bound_kind="upper",
        margin=gap_margin,
        passed=bool(gap_auc.upper_one_sided() < gap_margin),
        reliable=gap_auc.reliable,
        ci=gap_auc.two_sided(),
        note="ranking noninferiority to the matched-budget target reference",
    )
    c3 = ComponentTest(
        name="Delta_policy > policy margin",
        estimate=delta_policy.estimate,
        bound=delta_policy.lower_one_sided(),
        bound_kind="lower",
        margin=policy_margin,
        passed=bool(delta_policy.lower_one_sided() > policy_margin),
        reliable=delta_policy.reliable,
        ci=delta_policy.two_sided(),
        note="excess accepted failure risk of the frozen source policy",
    )

    for label, count in (accepted_group_counts or {}).items():
        if count < min_accepted_groups:
            warnings.append(
                f"{label} accepts cases from only {count} group(s): the policy endpoint "
                "is uninformative, not safe (§10.3)"
            )
    for label, coverage in (realised_coverages or {}).items():
        if not np.isfinite(coverage) or coverage <= 0:
            warnings.append(f"{label} realised coverage is {coverage}: the endpoint is undefined")

    components = [c1, c2, c3]
    reliable = all(c.reliable for c in components)
    joint = all(c.passed for c in components) and reliable and not warnings

    if joint is False and all(c.passed for c in components) and warnings:
        warnings.append(
            "all three component bounds passed but a validity warning is outstanding; "
            "the joint claim is withheld until it is resolved"
        )

    return JointClaim(
        components=components,
        alpha_one_sided=alpha_one_sided,
        multiplicity_rule=multiplicity_rule,
        joint_supported=bool(joint),
        conclusion=_conclusion(c1.passed, c2.passed, c3.passed, reliable),
        warnings=warnings,
    )
