"""Development-only information planning for the M2 direction decision.

This module deliberately separates facts from projections:

* role and pair counts come from the immutable pair manifest;
* failure-bearing and success-bearing probabilities come only from groups in
  the development reserve;
* H4 power numbers are labelled design scenarios, not estimates of unseen
  confirmatory effects.

The planning approximation is intentionally conservative about independence:
groups, rather than pairs, are the effective units in the interval-width
calculation.  Confirmatory inference still requires the complete group-refit
bootstrap in :mod:`warpaudit.evaluation.bootstrap`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .splits import TransferSplit, make_transfer_split

__all__ = [
    "AcceptedGroupProjection",
    "ClassSupportProjection",
    "DevelopmentEvidence",
    "DirectionPlan",
    "IUTScenario",
    "choose_fold_count",
    "development_evidence",
    "make_direction_plan",
    "project_accepted_groups",
    "project_class_support",
    "simulate_iut_scenario",
]


@dataclass(frozen=True)
class DevelopmentEvidence:
    dataset_id: str
    pipeline_id: str
    n_groups: int
    n_pairs: int
    n_failures: int
    n_successes: int
    failure_bearing_groups: int
    success_bearing_groups: int
    group_weighted_failure_prevalence: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ClassSupportProjection:
    n_test_groups: int
    failure_bearing_mean: float
    failure_bearing_p05: float
    failure_bearing_p95: float
    success_bearing_mean: float
    success_bearing_p05: float
    success_bearing_p95: float
    probability_both_at_least_minimum: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AcceptedGroupProjection:
    n_test_groups: int
    n_calibration_groups: int
    nominal_acceptance: float
    realised_coverage_mean: float
    realised_coverage_p05: float
    realised_coverage_p95: float
    threshold_quantile_sd: float
    accepted_groups_mean: float
    accepted_groups_p05: float
    accepted_groups_p95: float
    probability_at_least_minimum: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class IUTScenario:
    name: str
    assumed_target_auroc: float
    assumed_gap_auc: float
    assumed_delta_policy: float
    target_auroc_se: float
    gap_auc_se: float
    delta_policy_se: float
    probability_target_auc_passes: float
    probability_gap_passes: float
    probability_delta_policy_passes: float
    probability_joint_passes: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DirectionPlan:
    source_dataset: str
    target_dataset: str
    n_folds: int
    n_train_groups: int
    n_calibration_groups: int
    folds: tuple[TransferSplit, ...]

    @property
    def direction(self) -> str:
        return f"{self.source_dataset}->{self.target_dataset}"


def choose_fold_count(
    n_target_groups: int,
    *,
    preferred: int,
    fallback: int,
    minimum_test_groups: int,
) -> int:
    """Use the configured fallback when five-way tests are structurally too small."""
    if preferred < 2 or fallback < 2:
        raise ValueError("fold counts must be at least two")
    if n_target_groups < preferred:
        preferred = fallback
    if n_target_groups < preferred:
        raise ValueError("target groups cannot support either configured fold count")
    if n_target_groups // preferred < minimum_test_groups:
        if n_target_groups < fallback:
            raise ValueError("target groups cannot support the fallback fold count")
        return fallback
    return preferred


def _matched_budget(
    source_groups: int,
    target_groups: int,
    *,
    n_folds: int,
    source_test_reserve: int,
) -> tuple[int, int]:
    """Largest balanced train/calibration budget feasible in every outer fold."""
    largest_target_test = int(np.ceil(target_groups / n_folds))
    smallest_target_pool = target_groups - largest_target_test
    source_supervision_capacity = source_groups - source_test_reserve
    capacity = min(smallest_target_pool, source_supervision_capacity)
    if capacity < 2:
        raise ValueError("no matched budget remains after the independent source-test reserve")
    each = capacity // 2
    return each, each


def make_direction_plan(
    *,
    source_dataset: str,
    target_dataset: str,
    source_groups: list[str],
    target_groups: list[str],
    development_groups: list[str],
    preferred_folds: int,
    fallback_folds: int,
    low_information_threshold: int,
    seed: int,
) -> DirectionPlan:
    """Allocate every outer fold with matched group budgets and independent tests."""
    source = sorted(set(source_groups) - set(development_groups))
    target = sorted(set(target_groups) - set(development_groups))
    n_folds = choose_fold_count(
        len(target),
        preferred=preferred_folds,
        fallback=fallback_folds,
        minimum_test_groups=low_information_threshold,
    )
    n_train, n_calibration = _matched_budget(
        len(source),
        len(target),
        n_folds=n_folds,
        source_test_reserve=low_information_threshold,
    )
    folds = tuple(
        make_transfer_split(
            protocol="P3",
            fold=fold,
            n_folds=n_folds,
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            source_groups=source_groups,
            target_groups=target_groups,
            development_groups=development_groups,
            seed=seed,
            n_train_groups=n_train,
            n_calibration_groups=n_calibration,
        )
        for fold in range(n_folds)
    )
    return DirectionPlan(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        n_folds=n_folds,
        n_train_groups=n_train,
        n_calibration_groups=n_calibration,
        folds=folds,
    )


def development_evidence(
    rows: pd.DataFrame,
    *,
    dataset_id: str,
    pipeline_id: str,
) -> DevelopmentEvidence:
    """Summarise observed outcomes from an already restricted development frame."""
    required = {"group_id", "operational_failure"}
    missing = required - set(rows)
    if missing:
        raise ValueError(f"development evidence lacks columns: {sorted(missing)}")
    if rows.empty:
        raise ValueError("development evidence is empty")
    failure = rows["operational_failure"].astype(bool)
    groups = rows["group_id"].astype(str)
    group_rates = failure.groupby(groups).mean()
    group_has_failure = failure.groupby(groups).any()
    group_has_success = (~failure).groupby(groups).any()
    return DevelopmentEvidence(
        dataset_id=dataset_id,
        pipeline_id=pipeline_id,
        n_groups=int(groups.nunique()),
        n_pairs=int(len(rows)),
        n_failures=int(failure.sum()),
        n_successes=int((~failure).sum()),
        failure_bearing_groups=int(group_has_failure.sum()),
        success_bearing_groups=int(group_has_success.sum()),
        group_weighted_failure_prevalence=float(group_rates.mean()),
    )


def project_class_support(
    evidence: DevelopmentEvidence,
    *,
    n_test_groups: int,
    minimum: int,
    n_simulations: int = 20_000,
    seed: int = 20260907,
) -> ClassSupportProjection:
    """Jeffreys-posterior projection for unseen class-bearing group counts."""
    if n_test_groups < 1 or minimum < 1 or n_simulations < 100:
        raise ValueError("projection sizes are too small")
    rng = np.random.default_rng(seed)
    n = evidence.n_groups
    p_failure = rng.beta(
        evidence.failure_bearing_groups + 0.5,
        n - evidence.failure_bearing_groups + 0.5,
        size=n_simulations,
    )
    p_success = rng.beta(
        evidence.success_bearing_groups + 0.5,
        n - evidence.success_bearing_groups + 0.5,
        size=n_simulations,
    )
    failure_counts = rng.binomial(n_test_groups, p_failure)
    success_counts = rng.binomial(n_test_groups, p_success)
    qf = np.quantile(failure_counts, (0.05, 0.95))
    qs = np.quantile(success_counts, (0.05, 0.95))
    return ClassSupportProjection(
        n_test_groups=n_test_groups,
        failure_bearing_mean=float(failure_counts.mean()),
        failure_bearing_p05=float(qf[0]),
        failure_bearing_p95=float(qf[1]),
        success_bearing_mean=float(success_counts.mean()),
        success_bearing_p05=float(qs[0]),
        success_bearing_p95=float(qs[1]),
        probability_both_at_least_minimum=float(
            np.mean((failure_counts >= minimum) & (success_counts >= minimum))
        ),
    )


def project_accepted_groups(
    *,
    n_test_groups: int,
    n_calibration_groups: int,
    nominal_acceptance: float,
    minimum: int,
    n_simulations: int = 20_000,
    seed: int = 20260907,
) -> AcceptedGroupProjection:
    """Project quantile-threshold variability on the independent-group scale.

    The calibration threshold is represented by its percentile under a
    continuous score distribution.  Its finite-sample distribution is the
    corresponding beta order statistic.  This is a planning approximation,
    not a replacement for refitting the declared policy.
    """
    if n_calibration_groups < 1 or n_test_groups < 1:
        raise ValueError("calibration and test groups must be positive")
    if not 0.0 < nominal_acceptance < 1.0:
        raise ValueError("nominal acceptance must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    order = int(np.floor(nominal_acceptance * (n_calibration_groups + 1)))
    order = min(max(order, 1), n_calibration_groups)
    realised = rng.beta(
        order,
        n_calibration_groups + 1 - order,
        size=n_simulations,
    )
    accepted = rng.binomial(n_test_groups, realised)
    qc = np.quantile(realised, (0.05, 0.95))
    qa = np.quantile(accepted, (0.05, 0.95))
    return AcceptedGroupProjection(
        n_test_groups=n_test_groups,
        n_calibration_groups=n_calibration_groups,
        nominal_acceptance=nominal_acceptance,
        realised_coverage_mean=float(realised.mean()),
        realised_coverage_p05=float(qc[0]),
        realised_coverage_p95=float(qc[1]),
        threshold_quantile_sd=float(realised.std(ddof=1)),
        accepted_groups_mean=float(accepted.mean()),
        accepted_groups_p05=float(qa[0]),
        accepted_groups_p95=float(qa[1]),
        probability_at_least_minimum=float(np.mean(accepted >= minimum)),
    )


def _auc_standard_error(auc: float, n_failure: float, n_success: float) -> float:
    """Hanley-McNeil planning SE with class-bearing groups as effective counts."""
    n1 = max(float(n_failure), 1.0)
    n0 = max(float(n_success), 1.0)
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    variance = (
        auc * (1.0 - auc)
        + (n1 - 1.0) * (q1 - auc * auc)
        + (n0 - 1.0) * (q2 - auc * auc)
    ) / (n1 * n0)
    return float(np.sqrt(max(variance, 0.0)))


def simulate_iut_scenario(
    name: str,
    *,
    assumed_target_auroc: float,
    assumed_gap_auc: float,
    assumed_delta_policy: float,
    failure_bearing_groups: float,
    success_bearing_groups: float,
    accepted_groups: float,
    auroc_floor: float,
    gap_margin: float,
    delta_policy_margin: float,
    alpha: float,
    n_simulations: int = 20_000,
    seed: int = 20260907,
    paired_correlation: float = 0.5,
    accepted_failure_risk: float = 0.15,
) -> IUTScenario:
    """Simulate whether one-sided planning bounds clear all H4 margins.

    AUROC uncertainty uses class-bearing groups as the effective sample size.
    The two target scores are correlated, as are the two target-policy risks;
    therefore the gap simulations are paired.  This normal approximation is
    only a feasibility screen.  It must never be reported as confirmatory
    bootstrap inference.
    """
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie in (0, 0.5)")
    if not -1.0 < paired_correlation < 1.0:
        raise ValueError("paired correlation must lie in (-1, 1)")
    reference_auc = float(np.clip(assumed_target_auroc + assumed_gap_auc, 1e-6, 1 - 1e-6))
    source_se = _auc_standard_error(
        assumed_target_auroc, failure_bearing_groups, success_bearing_groups
    )
    reference_se = _auc_standard_error(
        reference_auc, failure_bearing_groups, success_bearing_groups
    )
    gap_se = float(
        np.sqrt(
            max(
                source_se**2
                + reference_se**2
                - 2.0 * paired_correlation * source_se * reference_se,
                0.0,
            )
        )
    )
    risk_source = float(
        np.clip(accepted_failure_risk + assumed_delta_policy / 2.0, 1e-6, 1 - 1e-6)
    )
    risk_reference = float(
        np.clip(accepted_failure_risk - assumed_delta_policy / 2.0, 1e-6, 1 - 1e-6)
    )
    effective_accepted = max(float(accepted_groups), 1.0)
    policy_var = (
        risk_source * (1.0 - risk_source)
        + risk_reference * (1.0 - risk_reference)
        - 2.0
        * paired_correlation
        * np.sqrt(
            risk_source
            * (1.0 - risk_source)
            * risk_reference
            * (1.0 - risk_reference)
        )
    ) / effective_accepted
    policy_se = float(np.sqrt(max(policy_var, 0.0)))

    rng = np.random.default_rng(seed)
    covariance = np.array(
        [
            [source_se**2, paired_correlation * source_se * reference_se],
            [paired_correlation * source_se * reference_se, reference_se**2],
        ]
    )
    auc_draws = rng.multivariate_normal(
        [assumed_target_auroc, reference_auc], covariance, size=n_simulations
    )
    delta_draws = rng.normal(assumed_delta_policy, policy_se, size=n_simulations)
    z = float(norm.ppf(1.0 - alpha))
    target_pass = auc_draws[:, 0] - z * source_se > auroc_floor
    gap_draws = auc_draws[:, 1] - auc_draws[:, 0]
    gap_pass = gap_draws + z * gap_se < gap_margin
    policy_pass = delta_draws - z * policy_se > delta_policy_margin
    return IUTScenario(
        name=name,
        assumed_target_auroc=assumed_target_auroc,
        assumed_gap_auc=assumed_gap_auc,
        assumed_delta_policy=assumed_delta_policy,
        target_auroc_se=source_se,
        gap_auc_se=gap_se,
        delta_policy_se=policy_se,
        probability_target_auc_passes=float(target_pass.mean()),
        probability_gap_passes=float(gap_pass.mean()),
        probability_delta_policy_passes=float(policy_pass.mean()),
        probability_joint_passes=float((target_pass & gap_pass & policy_pass).mean()),
    )
