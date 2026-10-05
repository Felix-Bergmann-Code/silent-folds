"""The three policy estimands (specification §9.2, §10.1, §10.3).

    Delta_policy    = R_Y(pi_X) - R_Y(pi_Y,n)      PRIMARY, target-internal
    Delta_domain    = R_Y(pi_X) - R_X(pi_X)        secondary, deployment shift
    Delta_threshold = R_Y(d_X,t_X) - R_Y(d_X,t_Y)  secondary, threshold transport

``Delta_policy`` compares the frozen source policy with an independently
target-trained and target-calibrated reference **at matched supervision
budget**, on the **same untouched target test groups**. Positive values mean
excess accepted failure risk from using the transferred source policy rather
than equally budgeted target supervision.

Two things this module refuses to do, both of which would silently invalidate
the primary claim:

* force the two policies to the same realised target coverage by picking a
  threshold on target-test scores -- each threshold comes from its own
  calibration groups at the same *nominal* target, and both realised coverages
  are reported;
* reinterpret ``Delta_policy`` through the prevalence decomposition, which
  applies to ``Delta_domain`` only (§9.2a).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..predictors.policy import FrozenPolicy
from .riskcoverage import AcceptanceOutcome, CasePopulation, risk_coverage_curve

__all__ = [
    "PolicyComparison",
    "delta_domain",
    "delta_policy",
    "delta_threshold",
    "shared_coverage_risk_difference",
]


def _difference(a: AcceptanceOutcome, b: AcceptanceOutcome) -> float:
    if not (a.risk_defined and b.risk_defined):
        return float("nan")
    return float(a.risk - b.risk)


@dataclass
class PolicyComparison:
    """One evaluated contrast, with everything §10.1 requires as context."""

    name: str
    estimate: float
    left: AcceptanceOutcome
    right: AcceptanceOutcome
    left_label: str
    right_label: str
    defined: bool
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, float | int | bool | str]:
        row: dict[str, float | int | bool | str] = {
            "contrast": self.name,
            "estimate": self.estimate,
            "defined": self.defined,
            "notes": "; ".join(self.notes),
        }
        for label, outcome in ((self.left_label, self.left), (self.right_label, self.right)):
            for key, value in outcome.as_row().items():
                row[f"{label}__{key}"] = value
        return row


def _check_same_population(a: CasePopulation, b: CasePopulation, contrast: str) -> list[str]:
    """Both policies must be scored on identical target test cases (§9.2.3)."""
    notes: list[str] = []
    if len(a) != len(b):
        raise ValueError(
            f"{contrast}: the two policies were evaluated on populations of different "
            f"size ({len(a)} vs {len(b)}); the contrast requires identical test cases"
        )
    if not np.array_equal(a.group, b.group):
        raise ValueError(f"{contrast}: group identities differ between the two arms")
    if not np.array_equal(a.failure, b.failure, equal_nan=True):
        raise ValueError(f"{contrast}: failure labels differ between the two arms")
    if not np.allclose(a.weight, b.weight):
        notes.append("evaluation weights differ between arms; check the weighting scheme")
    return notes


def delta_policy(
    source_policy: FrozenPolicy,
    target_reference_policy: FrozenPolicy,
    target_test_source_scores: CasePopulation,
    target_test_reference_scores: CasePopulation,
) -> PolicyComparison:
    """PRIMARY estimand ``R_Y(pi_X) - R_Y(pi_Y,n)`` (§9.2.3, §10.3).

    The two populations carry the *same* target test cases with different
    scores: one from the transferred source detector, one from the
    independently trained target reference.
    """
    notes = _check_same_population(
        target_test_source_scores, target_test_reference_scores, "Delta_policy"
    )
    left = source_policy.apply(target_test_source_scores)
    right = target_reference_policy.apply(target_test_reference_scores)

    if source_policy.nominal_acceptance != target_reference_policy.nominal_acceptance:
        notes.append(
            "the two policies use different nominal acceptance targets; §9.2 requires "
            "the same prespecified nominal target for this contrast"
        )
    for label, outcome in (("source", left), ("target-reference", right)):
        if outcome.n_accepted_groups < 3:
            notes.append(
                f"{label} policy accepts cases from only {outcome.n_accepted_groups} group(s): "
                "the policy endpoint is uninformative, not safe (§10.3)"
            )
    return PolicyComparison(
        name="Delta_policy",
        estimate=_difference(left, right),
        left=left,
        right=right,
        left_label="pi_X_on_Y",
        right_label="pi_Y_on_Y",
        defined=left.risk_defined and right.risk_defined,
        notes=notes,
    )


def delta_domain(
    source_policy: FrozenPolicy,
    target_test: CasePopulation,
    source_test: CasePopulation,
) -> PolicyComparison:
    """SECONDARY estimand ``R_Y(pi_X) - R_X(pi_X)`` (§9.2.2, §9.2a).

    The identical frozen policy on two different populations. This is a
    deployment/domain-shift endpoint. It does not isolate a causal mechanism,
    and it is never substituted for ``Delta_policy``.
    """
    left = source_policy.apply(target_test)
    right = source_policy.apply(source_test)
    notes = [
        "secondary deployment endpoint: the two populations differ in prevalence, "
        "difficulty, and group composition; see the §9.2a decomposition"
    ]
    return PolicyComparison(
        name="Delta_domain",
        estimate=_difference(left, right),
        left=left,
        right=right,
        left_label="pi_X_on_Y",
        right_label="pi_X_on_X",
        defined=left.risk_defined and right.risk_defined,
        notes=notes,
    )


def delta_threshold(
    source_policy: FrozenPolicy,
    target_recalibrated_policy: FrozenPolicy,
    target_test: CasePopulation,
) -> PolicyComparison:
    """SECONDARY estimand ``R_Y(d_X,t_X) - R_Y(d_X,t_Y)`` (§9.2).

    The transferred detector is held fixed; only the threshold changes, chosen
    on separate target calibration scores under the same nominal coverage
    rule. This arm uses target adaptation and is **not** part of the strict
    frozen-source result; it separates score-distribution/threshold transport
    from detector-training transport.
    """
    if (
        source_policy.detector_hash
        and target_recalibrated_policy.detector_hash
        and source_policy.detector_hash != target_recalibrated_policy.detector_hash
    ):
        raise ValueError(
            "Delta_threshold requires the SAME detector with a different threshold; "
            "the two policies carry different detector hashes"
        )
    left = source_policy.apply(target_test)
    right = target_recalibrated_policy.apply(target_test)
    return PolicyComparison(
        name="Delta_threshold",
        estimate=_difference(left, right),
        left=left,
        right=right,
        left_label="d_X_t_X",
        right_label="d_X_t_Y",
        defined=left.risk_defined and right.risk_defined,
        notes=["target-adapted arm; excluded from the strict frozen-source result"],
    )


def shared_coverage_risk_difference(
    left: CasePopulation, right: CasePopulation, coverage: float
) -> dict[str, float]:
    """Supporting ranking-based comparison at a prespecified shared coverage (§9.2).

    Returns ``nan`` risks when the requested coverage exceeds either arm's
    attainable maximum, rather than reporting the closest achievable point as
    though it were the requested one.
    """
    lc = risk_coverage_curve(left)
    rc = risk_coverage_curve(right)
    r_left, r_right = lc.risk_at(coverage), rc.risk_at(coverage)
    return {
        "requested_coverage": float(coverage),
        "risk_left": r_left,
        "risk_right": r_right,
        "difference": float(r_left - r_right)
        if np.isfinite(r_left) and np.isfinite(r_right)
        else float("nan"),
        "attainable_left": lc.max_attainable_coverage,
        "attainable_right": rc.max_attainable_coverage,
    }
