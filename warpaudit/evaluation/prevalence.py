"""Prevalence-matched secondary decomposition (specification §9.2a).

This applies to ``Delta_domain`` only. The primary ``Delta_policy`` contrast
already evaluates both policies on one target population, so prevalence
standardisation is neither needed nor permitted there; :func:`standardise`
refuses a population pair that is actually the same population.

Construction, exactly as specified:

* choose ``pi_star`` as the group-weighted **source calibration** failure
  prevalence, before opening source or target test labels;
* reweight with ``w'_i = w_i * pi_star / pi_D`` for failures and
  ``w'_i = w_i * (1 - pi_star) / (1 - pi_D)`` for successes, on both
  independent source and target test sets;
* report class-conditional acceptance ``a1 = P(accept | failure)`` and
  ``a0 = P(accept | success)``, then

      coverage* = pi_star * a1 + (1 - pi_star) * a0
      risk*     = pi_star * a1 / coverage*

Labels are used only *after* prediction, for evaluation. This is not a
deployable adaptation and it does not make two populations causally
equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..predictors.policy import FrozenPolicy
from .riskcoverage import CasePopulation

__all__ = [
    "PrevalenceDecomposition",
    "StandardisedOutcome",
    "reference_prevalence",
    "reweight",
    "standardise",
]


def reference_prevalence(source_calibration: CasePopulation) -> float:
    """``pi_star``: group-weighted source-calibration failure prevalence.

    Computed over all attempted calibration cases, so explicit failures are
    part of the failure population (§9.2a).
    """
    return source_calibration.operational_prevalence


def reweight(population: CasePopulation, pi_star: float) -> tuple[np.ndarray, dict[str, float]]:
    """Label-reweighted evaluation weights at a common prevalence.

    Weights are deliberately **not clipped**: §9.2a forbids clipping without a
    prespecified sensitivity rule. Extreme weights and effective sample size
    are returned so the diagnostic can be marked unavailable instead.
    """
    pi_D = population.operational_prevalence
    diagnostics = {
        "pi_star": float(pi_star),
        "pi_observed": float(pi_D),
        "effective_sample_size": float("nan"),
        "max_weight_ratio": float("nan"),
    }
    if not np.isfinite(pi_D) or pi_D <= 0.0 or pi_D >= 1.0:
        diagnostics["reason"] = "degenerate observed prevalence"
        return np.full(len(population), np.nan), diagnostics
    if not np.isfinite(pi_star) or pi_star <= 0.0 or pi_star >= 1.0:
        diagnostics["reason"] = "degenerate reference prevalence pi_star"
        return np.full(len(population), np.nan), diagnostics

    ratio_fail = pi_star / pi_D
    ratio_succ = (1.0 - pi_star) / (1.0 - pi_D)
    w = population.weight * np.where(population.failure == 1.0, ratio_fail, ratio_succ)

    total = w.sum()
    diagnostics["effective_sample_size"] = (
        float(total**2 / np.sum(w**2)) if np.sum(w**2) > 0 else float("nan")
    )
    diagnostics["max_weight_ratio"] = float(max(ratio_fail, ratio_succ))
    return w, diagnostics


@dataclass(frozen=True)
class StandardisedOutcome:
    population: str
    pi_star: float
    a1: float
    a0: float
    coverage_star: float
    risk_star: float
    coverage_raw: float
    risk_raw: float
    effective_sample_size: float
    max_weight_ratio: float
    available: bool
    reason: str = ""

    def as_row(self) -> dict[str, float | bool | str]:
        return {
            "population": self.population,
            "pi_star": self.pi_star,
            "accept_given_failure_a1": self.a1,
            "accept_given_success_a0": self.a0,
            "coverage_standardised": self.coverage_star,
            "risk_standardised": self.risk_star,
            "coverage_raw": self.coverage_raw,
            "risk_raw": self.risk_raw,
            "effective_sample_size": self.effective_sample_size,
            "max_weight_ratio": self.max_weight_ratio,
            "available": self.available,
            "reason": self.reason,
        }


def _standardise_one(
    policy: FrozenPolicy, population: CasePopulation, pi_star: float, name: str
) -> StandardisedOutcome:
    raw = policy.apply(population)
    _, diag = reweight(population, pi_star)

    a1, a0 = raw.accept_given_failure, raw.accept_given_success
    if not (np.isfinite(a1) and np.isfinite(a0)) or "reason" in diag:
        return StandardisedOutcome(
            population=name,
            pi_star=float(pi_star),
            a1=a1,
            a0=a0,
            coverage_star=float("nan"),
            risk_star=float("nan"),
            coverage_raw=raw.coverage,
            risk_raw=raw.risk,
            effective_sample_size=diag["effective_sample_size"],
            max_weight_ratio=diag["max_weight_ratio"],
            available=False,
            reason=diag.get(
                "reason", "a class is absent, so class-conditional " "acceptance is undefined"
            ),
        )

    coverage_star = pi_star * a1 + (1.0 - pi_star) * a0
    risk_star = (pi_star * a1 / coverage_star) if coverage_star > 0 else float("nan")
    return StandardisedOutcome(
        population=name,
        pi_star=float(pi_star),
        a1=float(a1),
        a0=float(a0),
        coverage_star=float(coverage_star),
        risk_star=float(risk_star),
        coverage_raw=raw.coverage,
        risk_raw=raw.risk,
        effective_sample_size=diag["effective_sample_size"],
        max_weight_ratio=diag["max_weight_ratio"],
        available=coverage_star > 0,
        reason="" if coverage_star > 0 else "zero standardised coverage: risk undefined",
    )


@dataclass
class PrevalenceDecomposition:
    pi_star: float
    target: StandardisedOutcome
    source: StandardisedOutcome
    delta_domain_raw: float
    delta_domain_standardised: float
    delta_coverage_raw: float
    delta_coverage_standardised: float
    delta_a1: float
    delta_a0: float
    available: bool
    notes: list[str] = field(default_factory=list)

    def interpretation(self) -> str:
        """The sentence §9.2a permits, chosen by the numbers themselves."""
        if not self.available:
            return "Prevalence decomposition unavailable; report Delta_domain raw only."
        if not np.isfinite(self.delta_domain_standardised):
            return "Standardised risk is undefined; report the raw deployment gap only."
        if abs(self.delta_domain_standardised) < 0.01 <= abs(self.delta_domain_raw):
            return (
                "The raw deployment gap largely disappears at common prevalence. It "
                "remains operationally relevant, but under this decomposition it is "
                "attributed to the observed prevalence composition rather than to a "
                "further mechanism."
            )
        return (
            "A residual standardised Delta_domain remains at common prevalence. This "
            "suggests differences beyond the marginal failure rate -- potentially "
            "failure severity, covariate shift, or group composition -- and does not "
            "isolate a causal mechanism by itself."
        )

    def as_rows(self) -> list[dict]:
        return [self.target.as_row(), self.source.as_row()]


def standardise(
    policy: FrozenPolicy,
    target_test: CasePopulation,
    source_test: CasePopulation,
    source_calibration: CasePopulation,
    *,
    valid_outputs_only: bool = False,
) -> PrevalenceDecomposition:
    """Run the §9.2a decomposition for one frozen policy.

    ``valid_outputs_only=True`` repeats the decomposition after removing
    explicit failures, which separates explicit-failure shifts from
    silent-error shifts.
    """
    if target_test is source_test:
        raise ValueError(
            "prevalence standardisation applies to the source-versus-target "
            "Delta_domain endpoint; it must not be used on the target-internal "
            "Delta_policy contrast (§9.2a)"
        )

    notes: list[str] = []
    if valid_outputs_only:
        target_test = target_test.subset(~target_test.explicit_failure)
        source_test = source_test.subset(~source_test.explicit_failure)
        notes.append("restricted to valid outputs: explicit-failure shifts are excluded")

    pi_star = reference_prevalence(source_calibration)
    tgt = _standardise_one(policy, target_test, pi_star, "target_test")
    src = _standardise_one(policy, source_test, pi_star, "source_test")

    available = tgt.available and src.available
    if not available:
        notes.append("decomposition marked unavailable: " + (tgt.reason or src.reason))
    for outcome in (tgt, src):
        if np.isfinite(outcome.max_weight_ratio) and outcome.max_weight_ratio > 5.0:
            notes.append(
                f"{outcome.population}: extreme reweighting factor "
                f"{outcome.max_weight_ratio:.1f}; weights are not clipped (§9.2a)"
            )

    def diff(a: float, b: float) -> float:
        return float(a - b) if np.isfinite(a) and np.isfinite(b) else float("nan")

    return PrevalenceDecomposition(
        pi_star=pi_star,
        target=tgt,
        source=src,
        delta_domain_raw=diff(tgt.risk_raw, src.risk_raw),
        delta_domain_standardised=diff(tgt.risk_star, src.risk_star),
        delta_coverage_raw=diff(tgt.coverage_raw, src.coverage_raw),
        delta_coverage_standardised=diff(tgt.coverage_star, src.coverage_star),
        delta_a1=diff(tgt.a1, src.a1),
        delta_a0=diff(tgt.a0, src.a0),
        available=available,
        notes=notes,
    )
