"""One evaluated fold of the transfer protocol (specification §8, §9, §10).

This is the procedure the confirmatory claim rests on, so its structure is
deliberately literal about the separations the specification demands:

* the transferred detector sees source training groups only, and its
  probability map and threshold come from source *calibration* groups. It
  never touches target labels;
* the matched-budget target reference is fitted and calibrated on its own
  assigned target groups, at the same ``(n_train, n_calibration)`` budget;
* both are evaluated on the same untouched target test groups, so the primary
  policy contrast is paired case by case;
* the independent source test groups estimate the frozen policy's in-domain
  risk without calibration-set optimism, and feed only the secondary
  ``Delta_domain``.

The whole fold is expressed as one function of its group assignment, which is
what lets the complete refit bootstrap of §10.2 rerun it -- fitting, threshold
selection, and evaluation together -- on resampled groups.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..cache.hashing import short_hash
from ..predictors.model import (
    LogisticRiskDetector,
    fit_lightgbm_detector,
    fit_logistic_detector,
    lightgbm_available,
)
from ..predictors.policy import FrozenPolicy
from .metrics import auroc, average_precision, brier_score, calibration_in_the_large
from .policy_eval import PolicyComparison, delta_domain, delta_policy, delta_threshold
from .prevalence import PrevalenceDecomposition, standardise
from .riskcoverage import CasePopulation, risk_coverage_curve
from .weights import group_first_weights, inverse_group_size_weights

__all__ = [
    "ArmScores",
    "FoldOutcome",
    "control_scores",
    "evaluate_fold",
    "macro_auroc",
    "macro_average_precision",
    "select_rows",
]

#: Roles whose groups must stay disjoint inside one evaluated fold (§8.1).
ROLES = (
    "source_train",
    "source_calibration",
    "source_test",
    "target_train",
    "target_calibration",
    "target_test",
)


def select_rows(cases: pd.DataFrame, groups: Sequence[str]) -> pd.DataFrame:
    """Rows belonging to ``groups``, in the deterministic order of the table.

    A group may legitimately appear more than once when the refit bootstrap
    resamples with replacement, and every copy must contribute, so this
    concatenates per-group blocks rather than filtering with ``isin``.
    """
    wanted = list(groups)
    if not wanted:
        return cases.iloc[0:0]
    by_group = {key: frame for key, frame in cases.groupby("group_id", sort=False)}
    blocks = [by_group[g] for g in wanted if g in by_group]
    if not blocks:
        return cases.iloc[0:0]
    return pd.concat(blocks, ignore_index=True)


def _population(rows: pd.DataFrame, scores: np.ndarray) -> CasePopulation:
    """A scored evaluation cell under group-first sampling weights (§10.1)."""
    weight = group_first_weights(
        rows["group_id"].astype(str).tolist(),
        rows["pair_id"].astype(str).tolist(),
        rows["pipeline_id"].astype(str).tolist(),
    )
    return CasePopulation(
        score=np.asarray(scores, dtype=np.float64),
        failure=rows["operational_failure"].to_numpy(dtype=np.float64),
        weight=weight,
        group=rows["group_id"].astype(str).to_numpy(),
        explicit_failure=rows["explicit_failure"].to_numpy(dtype=bool),
        loss=rows["bounded_loss"].to_numpy(dtype=np.float64)
        if "bounded_loss" in rows
        else None,
    )


def macro_auroc(rows: pd.DataFrame, scores: np.ndarray) -> float:
    """AUROC among valid outputs, macro-averaged over the common pipelines.

    §10.1 requires the macro-average over the fixed common block with separate
    cells retained, and requires a cell that lacks both classes to be marked
    undefined rather than dropped: returning ``nan`` here propagates that into
    the bootstrap as an invalid resample instead of silently averaging the
    remaining pipelines.
    """
    return _macro(rows, scores, auroc)


def macro_average_precision(rows: pd.DataFrame, scores: np.ndarray) -> float:
    return _macro(rows, scores, average_precision)


def _macro(rows: pd.DataFrame, scores: np.ndarray, statistic) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores) & (~rows["explicit_failure"].to_numpy(dtype=bool))
    if not valid.any():
        return float("nan")
    per_pipeline: list[float] = []
    pipelines = rows["pipeline_id"].astype(str).to_numpy()
    for pipeline in sorted(set(pipelines)):
        cell = valid & (pipelines == pipeline)
        if cell.sum() < 2:
            return float("nan")
        y = rows.loc[cell, "operational_failure"].to_numpy(dtype=np.float64)
        if len(np.unique(y)) < 2:
            return float("nan")
        weight = group_first_weights(
            rows.loc[cell, "group_id"].astype(str).tolist(),
            rows.loc[cell, "pair_id"].astype(str).tolist(),
        )
        per_pipeline.append(float(statistic(scores[cell], y, weight)))
    finite = [v for v in per_pipeline if np.isfinite(v)]
    if len(finite) != len(per_pipeline) or not finite:
        return float("nan")
    return float(np.mean(finite))


@dataclass(frozen=True)
class ArmScores:
    """One scoring arm evaluated on one population."""

    arm: str
    scores: np.ndarray
    probabilities: np.ndarray | None = None
    detector_hash: str = ""
    preprocessing_hash: str = ""
    calibration_hash: str = ""


def control_scores(
    name: str,
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    seed: int,
) -> np.ndarray:
    """The required comparison arms of §7.4.

    ``constant`` uses the source failure rate, so it ranks nothing; ``random``
    is a seeded fixed ordering; ``missingness_only`` and ``pipeline_identity``
    are the identity controls that a detector must beat to claim it learned
    registration quality rather than which pipeline or dataset produced a row.
    """
    n = len(test_rows)
    if name == "constant":
        rate = float(train_rows["operational_failure"].mean())
        return np.full(n, rate, dtype=np.float64)
    if name == "random":
        return np.random.default_rng(seed).permutation(n).astype(np.float64) / max(n, 1)
    if name == "missingness_only":
        design = ~np.isfinite(
            test_rows.loc[:, list(feature_names)].to_numpy(dtype=np.float64)
        )
        train_design = ~np.isfinite(
            train_rows.loc[:, list(feature_names)].to_numpy(dtype=np.float64)
        )
        return _fit_simple_logistic(
            train_design.astype(np.float64),
            train_rows,
            design.astype(np.float64),
            seed=seed,
        )
    if name == "pipeline_identity":
        pipelines = sorted(set(train_rows["pipeline_id"].astype(str)))
        train_design = np.stack(
            [(train_rows["pipeline_id"].astype(str) == p).to_numpy(float) for p in pipelines],
            axis=1,
        )
        design = np.stack(
            [(test_rows["pipeline_id"].astype(str) == p).to_numpy(float) for p in pipelines],
            axis=1,
        )
        return _fit_simple_logistic(train_design, train_rows, design, seed=seed)
    raise ValueError(f"unknown control arm {name!r}")


def _fit_simple_logistic(
    train_design: np.ndarray, train_rows: pd.DataFrame, design: np.ndarray, *, seed: int
) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    y = train_rows["operational_failure"].to_numpy(dtype=np.float64)
    if len(np.unique(y)) < 2 or train_design.shape[1] == 0:
        return np.full(len(design), float("nan"))
    weight = inverse_group_size_weights(
        train_rows["group_id"].astype(str).tolist(),
        train_rows["pipeline_id"].astype(str).tolist(),
    )
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)
    model.fit(train_design, y.astype(int), sample_weight=weight)
    return np.asarray(model.decision_function(design))


@dataclass
class FoldOutcome:
    """Everything one evaluated fold produces."""

    fold: int
    estimands: dict[str, float] = field(default_factory=dict)
    comparisons: dict[str, PolicyComparison] = field(default_factory=dict)
    prediction_rows: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    prevalence: PrevalenceDecomposition | None = None
    source_policy: FrozenPolicy | None = None
    target_policy: FrozenPolicy | None = None
    budget: dict[str, int] = field(default_factory=dict)


def _fit_arm(
    rows: pd.DataFrame,
    calibration_rows: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    C_values: tuple[float, ...],
    seed: int,
    calibration_method: str,
) -> LogisticRiskDetector:
    detector = fit_logistic_detector(
        rows.loc[:, list(feature_names)],
        rows["operational_failure"].to_numpy(dtype=np.float64),
        rows["group_id"].astype(str).to_numpy(),
        feature_names=tuple(feature_names),
        C_values=C_values,
        seed=seed,
    )
    detector.calibrate(
        calibration_rows.loc[:, list(feature_names)],
        calibration_rows["operational_failure"].to_numpy(dtype=np.float64),
        method=calibration_method,
        sample_weight=inverse_group_size_weights(
            calibration_rows["group_id"].astype(str).tolist(),
            calibration_rows["pipeline_id"].astype(str).tolist(),
        ),
    )
    return detector


def evaluate_fold(
    cases: pd.DataFrame,
    role_groups: Mapping[str, Sequence[str]],
    feature_names: Sequence[str],
    *,
    fold: int = 0,
    protocol: str = "P3",
    experiment_id: str = "",
    nominal_acceptance: float = 0.70,
    tie_rule: str = "conservative_reject",
    calibration_method: str = "platt",
    C_values: tuple[float, ...] = (0.1, 1.0, 10.0),
    seed: int = 20260907,
    requested_coverages: Sequence[float] = (0.5, 0.7, 0.8, 0.9),
    secondary_learner: str = "",
    lightgbm_leaves: tuple[int, ...] = (7, 15),
    lightgbm_min_child_samples: tuple[int, ...] = (20, 50),
    with_predictions: bool = True,
    with_controls: bool = True,
) -> FoldOutcome:
    """Fit, freeze, and evaluate one fold; return its estimands and rows."""
    outcome = FoldOutcome(fold=fold)
    parts = {role: select_rows(cases, role_groups.get(role, ())) for role in ROLES}
    for role, frame in parts.items():
        if frame.empty:
            outcome.notes.append(f"{role} has no scored rows in this fold")
            return outcome

    source_detector = _fit_arm(
        parts["source_train"],
        parts["source_calibration"],
        feature_names,
        C_values=C_values,
        seed=seed,
        calibration_method=calibration_method,
    )
    target_detector = _fit_arm(
        parts["target_train"],
        parts["target_calibration"],
        feature_names,
        C_values=C_values,
        seed=seed,
        calibration_method=calibration_method,
    )

    def probabilities(detector: LogisticRiskDetector, rows: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            detector.predict_probability(rows.loc[:, list(feature_names)]), dtype=np.float64
        )

    source_calibration_pop = _population(
        parts["source_calibration"], probabilities(source_detector, parts["source_calibration"])
    )
    source_policy = FrozenPolicy.from_calibration(
        "pi_X",
        source_calibration_pop,
        nominal_acceptance=nominal_acceptance,
        tie_rule=tie_rule,
        detector_hash=source_detector.fingerprint,
        preprocessing_hash=source_detector.preprocessor.fingerprint,
        calibration_hash=short_hash(sorted(set(role_groups.get("source_calibration", ())))),
        provenance={"fold": fold, "protocol": protocol},
    )
    target_calibration_pop = _population(
        parts["target_calibration"], probabilities(target_detector, parts["target_calibration"])
    )
    target_policy = FrozenPolicy.from_calibration(
        "pi_Y_n",
        target_calibration_pop,
        nominal_acceptance=nominal_acceptance,
        tie_rule=tie_rule,
        detector_hash=target_detector.fingerprint,
        preprocessing_hash=target_detector.preprocessor.fingerprint,
        calibration_hash=short_hash(sorted(set(role_groups.get("target_calibration", ())))),
        provenance={"fold": fold, "protocol": protocol},
    )

    test_rows = parts["target_test"]
    source_on_target = probabilities(source_detector, test_rows)
    target_on_target = probabilities(target_detector, test_rows)
    source_on_source_test = probabilities(source_detector, parts["source_test"])

    target_test_source = _population(test_rows, source_on_target)
    target_test_reference = _population(test_rows, target_on_target)
    source_test_pop = _population(parts["source_test"], source_on_source_test)

    outcome.comparisons["Delta_policy"] = delta_policy(
        source_policy, target_policy, target_test_source, target_test_reference
    )
    outcome.comparisons["Delta_domain"] = delta_domain(
        source_policy, target_test_source, source_test_pop
    )

    # Secondary threshold-transport arm: the same transferred detector, a
    # threshold chosen on separate target calibration scores.
    target_calibration_source_scores = _population(
        parts["target_calibration"], probabilities(source_detector, parts["target_calibration"])
    )
    recalibrated = FrozenPolicy.from_calibration(
        "d_X_t_Y",
        target_calibration_source_scores,
        nominal_acceptance=nominal_acceptance,
        tie_rule=tie_rule,
        detector_hash=source_detector.fingerprint,
        preprocessing_hash=source_detector.preprocessor.fingerprint,
        calibration_hash=short_hash(sorted(set(role_groups.get("target_calibration", ())))),
        provenance={"fold": fold, "arm": "threshold_transport"},
    )
    outcome.comparisons["Delta_threshold"] = delta_threshold(
        source_policy, recalibrated, target_test_source
    )

    transferred_auroc = macro_auroc(test_rows, source_on_target)
    reference_auroc = macro_auroc(test_rows, target_on_target)
    outcome.estimands = {
        "target_auroc_transferred": transferred_auroc,
        "target_auroc_reference": reference_auroc,
        "gap_auc": reference_auroc - transferred_auroc,
        "target_ap_transferred": macro_average_precision(test_rows, source_on_target),
        "target_ap_reference": macro_average_precision(test_rows, target_on_target),
        "source_test_auroc_transferred": macro_auroc(parts["source_test"], source_on_source_test),
        "delta_policy": outcome.comparisons["Delta_policy"].estimate,
        "delta_domain": outcome.comparisons["Delta_domain"].estimate,
        "delta_threshold": outcome.comparisons["Delta_threshold"].estimate,
        "realised_coverage_source_on_target": outcome.comparisons["Delta_policy"].left.coverage,
        "realised_coverage_reference_on_target": outcome.comparisons["Delta_policy"].right.coverage,
        "accepted_groups_source_on_target": float(
            outcome.comparisons["Delta_policy"].left.n_accepted_groups
        ),
        "accepted_groups_reference_on_target": float(
            outcome.comparisons["Delta_policy"].right.n_accepted_groups
        ),
        "target_test_prevalence": target_test_source.operational_prevalence,
        "target_test_explicit_failure_fraction": target_test_source.explicit_failure_fraction,
        "policy_risk_source_on_target": outcome.comparisons["Delta_policy"].left.risk,
        "policy_risk_reference_on_target": outcome.comparisons["Delta_policy"].right.risk,
        "policy_risk_source_on_source_test": outcome.comparisons["Delta_domain"].right.risk,
        "source_test_coverage": outcome.comparisons["Delta_domain"].right.coverage,
        # Probability calibration is reported beside ranking and kept separate
        # from it (§9.1): a well-ranked detector can be badly calibrated.
        "brier_transferred": brier_score(
            source_on_target, test_rows["operational_failure"].to_numpy(dtype=np.float64),
            target_test_source.weight,
        ),
        "brier_reference": brier_score(
            target_on_target, test_rows["operational_failure"].to_numpy(dtype=np.float64),
            target_test_reference.weight,
        ),
        "calibration_in_the_large_transferred": calibration_in_the_large(
            source_on_target, test_rows["operational_failure"].to_numpy(dtype=np.float64),
            target_test_source.weight,
        ),
        "calibration_in_the_large_reference": calibration_in_the_large(
            target_on_target, test_rows["operational_failure"].to_numpy(dtype=np.float64),
            target_test_reference.weight,
        ),
    }
    # Ranking utility (§9.2.1, §10.1): risk at the prespecified requested
    # coverages, plus the area over the attainable interval with its endpoints,
    # so two runs with different explicit-failure rates are not compared through
    # the number alone.
    curve = risk_coverage_curve(target_test_source, tie_rule=tie_rule)
    area, lower, upper = curve.area()
    outcome.estimands["risk_coverage_area"] = area
    outcome.estimands["attainable_coverage_lower"] = lower
    outcome.estimands["attainable_coverage_upper"] = upper
    for coverage in requested_coverages:
        outcome.estimands[f"risk_at_coverage_{coverage:g}"] = curve.risk_at(coverage)

    # Secondary prevalence-standardised deployment diagnostic (§9.2a). It uses
    # labels only after prediction, and never reinterprets Delta_policy, whose
    # two policies already share one target population.
    decomposition = standardise(
        source_policy, target_test_source, source_test_pop, source_calibration_pop
    )
    outcome.prevalence = decomposition
    outcome.estimands["prevalence_reference_pi_star"] = decomposition.pi_star
    outcome.estimands["delta_domain_standardised"] = decomposition.delta_domain_standardised
    outcome.estimands["delta_coverage_standardised"] = decomposition.delta_coverage_standardised
    outcome.estimands["delta_accept_given_failure"] = decomposition.delta_a1
    outcome.estimands["delta_accept_given_success"] = decomposition.delta_a0
    if not decomposition.available:
        outcome.notes.extend(decomposition.notes)

    outcome.source_policy = source_policy
    outcome.target_policy = target_policy
    outcome.budget = {
        "source_train_groups": len(set(role_groups.get("source_train", ()))),
        "source_calibration_groups": len(set(role_groups.get("source_calibration", ()))),
        "target_train_groups": len(set(role_groups.get("target_train", ()))),
        "target_calibration_groups": len(set(role_groups.get("target_calibration", ()))),
        "source_train_pairs": int(parts["source_train"]["pair_id"].nunique()),
        "target_train_pairs": int(parts["target_train"]["pair_id"].nunique()),
    }
    outcome.notes.extend(outcome.comparisons["Delta_policy"].notes)

    if with_controls:
        for name in ("constant", "random", "missingness_only", "pipeline_identity"):
            scores = control_scores(
                name, parts["source_train"], test_rows, feature_names, seed=seed
            )
            outcome.estimands[f"control_{name}_auroc"] = macro_auroc(test_rows, scores)
        # Each applicable single score, and the strongest chosen on source
        # validation rather than on the target test cells.
        best_name, best_score = "", float("-inf")
        for feature in feature_names:
            single_source = parts["source_calibration"][feature].to_numpy(dtype=np.float64)
            source_auc = macro_auroc(parts["source_calibration"], single_source)
            oriented = 1.0 if not np.isfinite(source_auc) or source_auc >= 0.5 else -1.0
            values = test_rows[feature].to_numpy(dtype=np.float64) * oriented
            outcome.estimands[f"single_{feature}_auroc"] = macro_auroc(test_rows, values)
            if np.isfinite(source_auc) and max(source_auc, 1.0 - source_auc) > best_score:
                best_score = max(source_auc, 1.0 - source_auc)
                best_name = feature
        if best_name:
            outcome.estimands["best_single_score_feature_auroc"] = outcome.estimands[
                f"single_{best_name}_auroc"
            ]
            outcome.notes.append(
                f"strongest single score selected on source calibration: {best_name}"
            )

    if secondary_learner:
        # §7.4's secondary learner is reported beside the primary one, never
        # substituted for it: the frozen claim names one learner, so a second
        # model class is evidence about the feature set, not a second chance at
        # the same hypothesis.
        available, reason = lightgbm_available()
        if not available:
            outcome.notes.append(
                f"secondary learner {secondary_learner!r} unavailable ({reason}); "
                "install the optional gbm extra to report it"
            )
        elif secondary_learner != "lightgbm":
            outcome.notes.append(f"unknown secondary learner {secondary_learner!r}; skipped")
        else:
            try:
                secondary = fit_lightgbm_detector(
                    parts["source_train"].loc[:, list(feature_names)],
                    parts["source_train"]["operational_failure"].to_numpy(dtype=np.float64),
                    parts["source_train"]["group_id"].astype(str).to_numpy(),
                    feature_names=tuple(feature_names),
                    num_leaves=lightgbm_leaves,
                    min_child_samples=lightgbm_min_child_samples,
                    seed=seed,
                )
                outcome.estimands["secondary_learner_target_auroc"] = macro_auroc(
                    test_rows, secondary.decision_function(test_rows.loc[:, list(feature_names)])
                )
            except ValueError as exc:
                outcome.notes.append(f"secondary learner could not be fitted: {exc}")

    if with_predictions:
        outcome.prediction_rows = _prediction_rows(
            experiment_id=experiment_id,
            protocol=protocol,
            fold=fold,
            rows=test_rows,
            arms=(
                ArmScores(
                    "transferred",
                    source_on_target,
                    source_on_target,
                    source_detector.fingerprint,
                    source_detector.preprocessor.fingerprint,
                    source_policy.calibration_hash,
                ),
                ArmScores(
                    "target_reference",
                    target_on_target,
                    target_on_target,
                    target_detector.fingerprint,
                    target_detector.preprocessor.fingerprint,
                    target_policy.calibration_hash,
                ),
            ),
            policies={"transferred": source_policy, "target_reference": target_policy},
            role_groups=role_groups,
            budget=outcome.budget,
        )
    return outcome


def _prediction_rows(
    *,
    experiment_id: str,
    protocol: str,
    fold: int,
    rows: pd.DataFrame,
    arms: Sequence[ArmScores],
    policies: Mapping[str, FrozenPolicy],
    role_groups: Mapping[str, Sequence[str]],
    budget: Mapping[str, int],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    training_hash = short_hash(
        {role: sorted(set(role_groups.get(role, ()))) for role in ("source_train", "target_train")}
    )
    for arm in arms:
        policy = policies[arm.arm]
        accepted = (
            (~rows["explicit_failure"].to_numpy(dtype=bool))
            & np.isfinite(arm.scores)
            & (arm.scores <= policy.threshold)
        )
        for i, (_, row) in enumerate(rows.iterrows()):
            out.append(
                {
                    "job_id": row["job_id"],
                    "experiment_id": experiment_id,
                    "protocol": protocol,
                    "arm": arm.arm,
                    "fold": fold,
                    "training_groups_hash": training_hash,
                    "annotation_budget_groups": int(
                        budget.get("source_train_groups", 0)
                        + budget.get("source_calibration_groups", 0)
                    ),
                    "annotation_budget_pairs": int(budget.get("source_train_pairs", 0)),
                    "learner_config_hash": arm.detector_hash,
                    "preprocessing_hash": arm.preprocessing_hash,
                    "calibration_hash": arm.calibration_hash,
                    "score": float(arm.scores[i]),
                    "probability": float(
                        arm.probabilities[i] if arm.probabilities is not None else np.nan
                    ),
                    "threshold": float(policy.threshold),
                    "accepted": bool(accepted[i]),
                    "policy_id": policy.policy_id,
                }
            )
    return out
