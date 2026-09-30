"""Frozen acceptance policies (specification §9.2).

A *policy* is the whole deployable object: detector, preprocessing, probability
map, threshold, and tie rule, hashed together. Once frozen it is applied
unchanged to every evaluation population. The verification requirement is
blunt (§12.5.8): frozen policy thresholds are identical before and after
target evaluation, and :meth:`FrozenPolicy.assert_unchanged` exists so a test
can prove it rather than assert it in prose.

Threshold selection uses only the policy's own calibration groups and the
prespecified nominal acceptance target. It never sees test scores. In
particular, relabelling the target's best-ranked 70% as "the performance of a
source-calibrated threshold" is exactly the error this module is built to make
impossible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..cache.hashing import short_hash
from ..evaluation.riskcoverage import AcceptanceOutcome, CasePopulation, accept_by_threshold

__all__ = ["FrozenPolicy", "ThresholdSelection", "select_threshold"]


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    nominal_acceptance: float
    realised_calibration_coverage: float
    n_calibration_groups: int
    n_calibration_cases: int
    constant_scores: bool
    attainable: bool
    note: str = ""


def select_threshold(
    calibration: CasePopulation,
    *,
    nominal_acceptance: float = 0.70,
    tie_rule: str = "conservative_reject",
) -> ThresholdSelection:
    """Choose the acceptance threshold on calibration data alone.

    Candidate thresholds are the distinct observed scores among acceptable
    calibration cases, plus ``-inf`` (accept nothing). The candidate whose
    realised coverage is closest to ``nominal_acceptance`` is chosen, with ties
    broken toward the *lower* coverage so the rule is conservative and fully
    deterministic.

    Coverage is measured against all eligible attempted calibration cases,
    including explicit failures which are auto-rejected. When the explicit
    failure rate alone puts the nominal target out of reach, the selection is
    marked ``attainable=False`` and the maximum attainable coverage is
    recorded -- the policy is not quietly redefined to a coverage it can meet.
    """
    if not 0.0 < nominal_acceptance < 1.0:
        raise ValueError("nominal_acceptance must lie strictly in (0, 1)")
    if tie_rule != "conservative_reject":
        raise ValueError("deployed threshold selection requires conservative_reject tie handling")

    acceptable = calibration.acceptable
    total = calibration.total_weight
    n_groups = int(len(set(calibration.group.tolist())))

    if not acceptable.any() or total <= 0:
        return ThresholdSelection(
            threshold=float("-inf"),
            nominal_acceptance=nominal_acceptance,
            realised_calibration_coverage=0.0,
            n_calibration_groups=n_groups,
            n_calibration_cases=len(calibration),
            constant_scores=False,
            attainable=False,
            note="no acceptable calibration case; the policy accepts nothing",
        )

    scores = np.unique(calibration.score[acceptable])
    constant = len(scores) == 1
    candidates = np.concatenate(([-np.inf], scores))

    coverages = np.array(
        [
            float(calibration.weight[acceptable & (calibration.score <= t)].sum() / total)
            for t in candidates
        ]
    )
    distance = np.abs(coverages - nominal_acceptance)
    # Ties in distance are broken toward the smaller coverage: mergesort on
    # coverage first, then a stable argmin on distance.
    order = np.argsort(coverages, kind="mergesort")
    best = order[int(np.argmin(distance[order]))]

    max_cov = float(coverages[-1])
    attainable = max_cov + 1e-12 >= nominal_acceptance
    notes = []
    if not attainable:
        notes.append(
            f"nominal acceptance {nominal_acceptance:.2f} exceeds the maximum attainable "
            f"calibration coverage {max_cov:.3f} "
            f"(explicit failure weight {calibration.explicit_failure_fraction:.3f})"
        )
    if constant:
        notes.append(
            "all acceptable calibration scores are identical; report the seeded random "
            "reference of §9.2 with this policy"
        )
    if n_groups < 10:
        notes.append(
            f"only {n_groups} calibration group(s): threshold variability is high and the "
            "policy endpoint may be uninformative (§3.3, §10.3)"
        )

    return ThresholdSelection(
        threshold=float(candidates[best]),
        nominal_acceptance=nominal_acceptance,
        realised_calibration_coverage=float(coverages[best]),
        n_calibration_groups=n_groups,
        n_calibration_cases=len(calibration),
        constant_scores=constant,
        attainable=attainable,
        note="; ".join(notes),
    )


@dataclass
class FrozenPolicy:
    """Detector + probability map + threshold + tie rule, frozen together."""

    policy_id: str
    threshold: float
    nominal_acceptance: float
    tie_rule: str = "conservative_reject"
    detector_hash: str = ""
    preprocessing_hash: str = ""
    calibration_hash: str = ""
    selection: ThresholdSelection | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    _frozen_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self._frozen_fingerprint:
            self._frozen_fingerprint = self.fingerprint()

    def fingerprint(self) -> str:
        return short_hash(
            {
                "policy_id": self.policy_id,
                "threshold": None if not np.isfinite(self.threshold) else self.threshold,
                "threshold_is_neg_inf": bool(self.threshold == float("-inf")),
                "nominal_acceptance": self.nominal_acceptance,
                "tie_rule": self.tie_rule,
                "detector_hash": self.detector_hash,
                "preprocessing_hash": self.preprocessing_hash,
                "calibration_hash": self.calibration_hash,
            }
        )

    def assert_unchanged(self) -> None:
        """Verification hook for §12.5.8."""
        if self.fingerprint() != self._frozen_fingerprint:
            raise RuntimeError(
                f"policy {self.policy_id!r} changed after freezing: "
                f"{self._frozen_fingerprint} -> {self.fingerprint()}"
            )

    def apply(self, population: CasePopulation) -> AcceptanceOutcome:
        """Evaluate this frozen policy on a population. Never refits anything."""
        self.assert_unchanged()
        return accept_by_threshold(population, self.threshold, tie_rule=self.tie_rule)

    @classmethod
    def from_calibration(
        cls,
        policy_id: str,
        calibration: CasePopulation,
        *,
        nominal_acceptance: float = 0.70,
        tie_rule: str = "conservative_reject",
        detector_hash: str = "",
        preprocessing_hash: str = "",
        calibration_hash: str = "",
        provenance: dict[str, Any] | None = None,
    ) -> FrozenPolicy:
        sel = select_threshold(
            calibration, nominal_acceptance=nominal_acceptance, tie_rule=tie_rule
        )
        return cls(
            policy_id=policy_id,
            threshold=sel.threshold,
            nominal_acceptance=nominal_acceptance,
            tie_rule=tie_rule,
            detector_hash=detector_hash,
            preprocessing_hash=preprocessing_hash,
            calibration_hash=calibration_hash,
            selection=sel,
            provenance=dict(provenance or {}),
        )
