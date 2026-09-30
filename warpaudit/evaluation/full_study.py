"""Locked multi-domain evaluation for the factorized-stability study."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..evaluation.metrics import (
    auroc,
    average_precision,
    brier_score,
    calibration_in_the_large,
    class_bearing_groups,
)
from ..evaluation.riskcoverage import CasePopulation
from ..evaluation.weights import inverse_group_size_weights
from ..parallel import ordered_map
from ..predictors.model import fit_logistic_detector
from ..predictors.policy import FrozenPolicy
from ..protocols.splits import make_outer_folds


def _columns(feature_names: Sequence[str], families: Sequence[str]) -> tuple[str, ...]:
    prefixes = tuple(f"{family}:" for family in families)
    return tuple(name for name in feature_names if name.startswith(prefixes))


def _population(frame: pd.DataFrame, probability: np.ndarray) -> CasePopulation:
    groups = frame["group_id"].astype(str).to_numpy()
    return CasePopulation(
        score=probability,
        failure=frame["operational_failure"].to_numpy(float),
        weight=inverse_group_size_weights(groups),
        group=groups,
        explicit_failure=frame["explicit_failure"].astype(bool).to_numpy(),
        loss=frame["bounded_loss"].to_numpy(float),
    )


def _interval(values: list[float], alpha: float = 0.05) -> list[float]:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(finite):
        return [float("nan"), float("nan")]
    return [float(np.quantile(finite, alpha / 2)), float(np.quantile(finite, 1 - alpha / 2))]


def _cluster_bootstrap(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    augmented: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, list[float]]:
    groups = sorted(set(frame["group_id"].astype(str)))
    by_group = {
        group: np.flatnonzero(frame["group_id"].astype(str).to_numpy() == group)
        for group in groups
    }
    rng = np.random.default_rng(seed)
    brier_delta: list[float] = []
    auroc_delta: list[float] = []
    labels = frame["operational_failure"].to_numpy(float)
    for _ in range(n_resamples):
        drawn = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([by_group[group] for group in drawn])
        synthetic_groups = np.concatenate(
            [np.repeat(f"draw-{i}", len(by_group[group])) for i, group in enumerate(drawn)]
        )
        weights = inverse_group_size_weights(synthetic_groups)
        y = labels[indices]
        brier_delta.append(
            brier_score(baseline[indices], y, weights)
            - brier_score(augmented[indices], y, weights)
        )
        auroc_delta.append(
            auroc(augmented[indices], y, weights) - auroc(baseline[indices], y, weights)
        )
    return {
        "brier_improvement_95ci": _interval(brier_delta),
        "auroc_improvement_95ci": _interval(auroc_delta),
    }


def evaluate_full_study(
    cases: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    development_datasets: Sequence[str],
    external_datasets: Sequence[str],
    confirmatory_datasets: Sequence[str],
    pipelines: Sequence[str],
    baseline_families: Sequence[str],
    augmented_families: Sequence[str],
    calibration_method: str,
    nominal_acceptance: float,
    high_confidence_cutoff: float,
    min_class_bearing_groups: int,
    min_brier_improvement: float,
    bootstrap_resamples: int,
    C_values: tuple[float, ...],
    seed: int,
    workers: int = 1,
    worker_threads: int = 1,
) -> list[dict]:
    """Fit on development datasets and evaluate every external cell."""
    if workers < 1 or worker_threads < 0:
        raise ValueError("workers must be positive and worker threads nonnegative")
    if workers > 1:
        options = dict(locals())
        options.pop("cases")
        options.pop("feature_names")
        options["workers"] = 1
        tasks = ((cases[cases["pipeline_id"].astype(str) == pipeline].copy(),
                  feature_names, {**options, "pipelines": (pipeline,)}) for pipeline in pipelines)
        return [row for batch in ordered_map(_evaluate_pipeline, tasks, workers=workers,
                                              threads=worker_threads) for row in batch]
    baseline_columns = _columns(feature_names, baseline_families)
    augmented_columns = _columns(feature_names, augmented_families)
    if not baseline_columns or set(baseline_columns) >= set(augmented_columns):
        raise ValueError("augmented feature columns must strictly extend the baseline")
    results: list[dict] = []
    for pipeline in pipelines:
        dev = cases[
            cases["dataset_id"].astype(str).isin(development_datasets)
            & (cases["pipeline_id"].astype(str) == pipeline)
        ].copy()
        groups = sorted(set(dev["group_id"].astype(str)))
        if len(groups) < 5:
            raise ValueError(f"{pipeline}: development data have fewer than five groups")
        assignment = make_outer_folds(groups, n_folds=5, seed=seed)
        calibration_mask = dev["group_id"].astype(str).map(assignment).to_numpy() == 0
        train, calibration = dev.loc[~calibration_mask], dev.loc[calibration_mask]
        models = {}
        policies = {}
        for arm, columns in (("baseline", baseline_columns), ("augmented", augmented_columns)):
            model = fit_logistic_detector(
                train.loc[:, columns],
                train["operational_failure"].to_numpy(float),
                train["group_id"].astype(str).to_numpy(),
                feature_names=columns,
                C_values=C_values,
                seed=seed,
            )
            cal_weights = inverse_group_size_weights(calibration["group_id"].astype(str))
            model.calibrate(
                calibration.loc[:, columns],
                calibration["operational_failure"].to_numpy(float),
                method=calibration_method,
                sample_weight=cal_weights,
            )
            cal_probability = model.predict_probability(calibration.loc[:, columns])
            policy = FrozenPolicy.from_calibration(
                f"full-{pipeline}-{arm}",
                _population(calibration, cal_probability),
                nominal_acceptance=nominal_acceptance,
                detector_hash=model.fingerprint,
                preprocessing_hash=model.preprocessor.fingerprint,
            )
            models[arm], policies[arm] = model, policy

        for dataset in external_datasets:
            cell = cases[
                (cases["dataset_id"].astype(str) == dataset)
                & (cases["pipeline_id"].astype(str) == pipeline)
            ].copy()
            if cell.empty:
                results.append({"dataset": dataset, "pipeline": pipeline, "eligible": False,
                                "reason": "no scored cases"})
                continue
            y = cell["operational_failure"].to_numpy(float)
            groups_array = cell["group_id"].astype(str).to_numpy()
            weights = inverse_group_size_weights(groups_array)
            failure_groups, success_groups = class_bearing_groups(y, groups_array)
            probabilities = {
                arm: models[arm].predict_probability(
                    cell.loc[:, baseline_columns if arm == "baseline" else augmented_columns]
                )
                for arm in ("baseline", "augmented")
            }
            row: dict = {
                "dataset": dataset,
                "pipeline": pipeline,
                "confirmatory": dataset in set(confirmatory_datasets),
                "n_cases": len(cell),
                "n_groups": len(set(groups_array)),
                "failure_bearing_groups": failure_groups,
                "success_bearing_groups": success_groups,
                "eligible": failure_groups >= min_class_bearing_groups
                and success_groups >= min_class_bearing_groups,
            }
            for arm in ("baseline", "augmented"):
                p = probabilities[arm]
                outcome = policies[arm].apply(_population(cell, p))
                low = p <= high_confidence_cutoff
                row[arm] = {
                    "auroc": auroc(p, y, weights),
                    "average_precision": average_precision(p, y, weights),
                    "brier": brier_score(p, y, weights),
                    "calibration_in_the_large": calibration_in_the_large(p, y, weights),
                    "coverage": outcome.coverage,
                    "accepted_risk": outcome.risk,
                    "high_confidence_failures": int(np.sum(low & (y == 1))),
                    "high_confidence_cases": int(np.sum(low)),
                }
            row["contrast"] = {
                "brier_improvement": row["baseline"]["brier"] - row["augmented"]["brier"],
                "auroc_improvement": row["augmented"]["auroc"] - row["baseline"]["auroc"],
                "accepted_risk_improvement": row["baseline"]["accepted_risk"]
                - row["augmented"]["accepted_risk"],
            }
            # Sparse descriptive cohorts cannot support population intervals.
            row["contrast"].update(
                _cluster_bootstrap(
                    cell.reset_index(drop=True),
                    probabilities["baseline"],
                    probabilities["augmented"],
                    n_resamples=bootstrap_resamples,
                    seed=seed ^ int.from_bytes(f"{dataset}/{pipeline}".encode()[:8], "little"),
                ) if row["eligible"] else {
                    "brier_improvement_95ci": [float("nan"), float("nan")],
                    "auroc_improvement_95ci": [float("nan"), float("nan")],
                }
            )
            lower = row["contrast"]["brier_improvement_95ci"][0]
            row["primary_passed"] = bool(
                row["confirmatory"] and row["eligible"] and np.isfinite(lower)
                and lower > min_brier_improvement
            )
            if not row["eligible"]:
                row["reason"] = "class-bearing independent-group gate failed; descriptive only"
            results.append(row)
    return results


def _evaluate_pipeline(task):
    cases, names, options = task
    return evaluate_full_study(cases, names, **options)
