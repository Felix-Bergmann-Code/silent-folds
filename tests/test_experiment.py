"""One evaluated fold: the separations the confirmatory claim depends on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from warpaudit.evaluation.experiment import ROLES, evaluate_fold, macro_auroc, select_rows


def _cases(seed: int = 5, n_groups: int = 24, pairs_per_group: int = 3) -> pd.DataFrame:
    """Two pipelines per pair, with a signal feature and a pure-noise feature."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        dataset = "X" if g < n_groups // 2 else "Y"
        for p in range(pairs_per_group):
            # Failure is a group-level tendency plus per-pair noise.
            base = rng.uniform(0.1, 0.9)
            failure = float(rng.uniform() < base)
            for pipeline in ("p1", "p2"):
                rows.append(
                    {
                        "job_id": f"{dataset}{g}-{p}-{pipeline}",
                        "dataset_id": dataset,
                        "pair_id": f"{dataset}/{g}/{p}",
                        "group_id": f"{dataset}/g{g:02d}",
                        "pipeline_id": pipeline,
                        "operational_failure": failure,
                        "silent_failure": failure,
                        "explicit_failure": False,
                        "bounded_loss": failure * 4.0,
                        "tre_norm": 0.01 + 0.02 * failure,
                        # A feature that genuinely tracks failure, plus noise.
                        "A:signal": failure + rng.normal(0.0, 0.55),
                        "B:noise": rng.normal(0.0, 1.0),
                    }
                )
    return pd.DataFrame(rows)


def _roles(cases: pd.DataFrame) -> dict[str, list[str]]:
    source = sorted(set(cases.loc[cases.dataset_id == "X", "group_id"]))
    target = sorted(set(cases.loc[cases.dataset_id == "Y", "group_id"]))
    return {
        "source_train": source[:5],
        "source_calibration": source[5:9],
        "source_test": source[9:],
        "target_train": target[:5],
        "target_calibration": target[5:9],
        "target_test": target[9:],
    }


def test_fold_roles_stay_disjoint_and_target_test_is_scored_by_both_arms() -> None:
    cases = _cases()
    roles = _roles(cases)
    outcome = evaluate_fold(cases, roles, ("A:signal", "B:noise"), fold=0, experiment_id="e")

    assert outcome.estimands["target_auroc_transferred"] > 0.5
    assert np.isfinite(outcome.estimands["gap_auc"])
    # Both arms score the same untouched target test cases, so the primary
    # policy contrast is paired case by case.
    test_jobs = set(select_rows(cases, roles["target_test"])["job_id"])
    by_arm: dict[str, set[str]] = {}
    for row in outcome.prediction_rows:
        by_arm.setdefault(row["arm"], set()).add(row["job_id"])
    assert by_arm["transferred"] == by_arm["target_reference"] == test_jobs
    # No prediction row may come from a fitting or calibration group.
    fitted = set()
    for role in ("source_train", "source_calibration", "target_train", "target_calibration"):
        fitted |= set(select_rows(cases, roles[role])["job_id"])
    assert not (test_jobs & fitted)


def test_the_two_policies_are_frozen_from_their_own_calibration_groups() -> None:
    cases = _cases()
    outcome = evaluate_fold(cases, _roles(cases), ("A:signal", "B:noise"), experiment_id="e")
    source, target = outcome.source_policy, outcome.target_policy
    assert source is not None and target is not None
    # Different detectors, different calibration sets, same nominal target.
    assert source.detector_hash != target.detector_hash
    assert source.calibration_hash != target.calibration_hash
    assert source.nominal_acceptance == target.nominal_acceptance
    # Freezing is enforced, not documented.
    source.assert_unchanged()
    target.assert_unchanged()


def test_an_undefined_pipeline_cell_makes_the_macro_average_undefined() -> None:
    """Dropping the difficult cell and averaging the rest is forbidden (§10.1)."""
    cases = _cases()
    one_class = cases.copy()
    one_class.loc[one_class.pipeline_id == "p2", "operational_failure"] = 1.0
    scores = one_class["A:signal"].to_numpy(dtype=float)
    assert np.isnan(macro_auroc(one_class, scores))
    # The other pipeline alone is perfectly estimable, which is the point: the
    # aggregate is undefined rather than quietly equal to that one cell.
    assert np.isfinite(macro_auroc(one_class[one_class.pipeline_id == "p1"], 
                                   one_class.loc[one_class.pipeline_id == "p1", "A:signal"].to_numpy(float)))


def test_controls_and_single_scores_are_reported_alongside_the_detector() -> None:
    cases = _cases()
    outcome = evaluate_fold(cases, _roles(cases), ("A:signal", "B:noise"), experiment_id="e")
    for name in ("constant", "random", "missingness_only", "pipeline_identity"):
        assert f"control_{name}_auroc" in outcome.estimands
    assert "single_A:signal_auroc" in outcome.estimands
    assert "best_single_score_feature_auroc" in outcome.estimands
    # A constant score ranks nothing, so its AUROC is undefined or exactly 0.5.
    constant = outcome.estimands["control_constant_auroc"]
    assert not np.isfinite(constant) or constant == pytest.approx(0.5)


def test_a_fold_missing_a_role_reports_it_instead_of_estimating() -> None:
    cases = _cases()
    roles = _roles(cases)
    roles["source_calibration"] = []
    outcome = evaluate_fold(cases, roles, ("A:signal", "B:noise"))
    assert outcome.estimands == {}
    assert any("source_calibration" in note for note in outcome.notes)


def test_every_declared_role_is_required_by_the_fold() -> None:
    assert set(ROLES) == {
        "source_train",
        "source_calibration",
        "source_test",
        "target_train",
        "target_calibration",
        "target_test",
    }

def test_the_secondary_learner_is_reported_or_its_absence_is(monkeypatch) -> None:
    """§7.4's secondary learner must never silently vanish from the report."""
    from warpaudit.evaluation import experiment

    cases = _cases()
    roles = _roles(cases)
    available, _ = experiment.lightgbm_available()
    outcome = evaluate_fold(
        cases, roles, ("A:signal", "B:noise"), secondary_learner="lightgbm"
    )
    if available:
        assert np.isfinite(outcome.estimands["secondary_learner_target_auroc"])
    else:
        assert any("unavailable" in note for note in outcome.notes)

    # An unknown learner is named in the notes rather than ignored.
    unknown = evaluate_fold(cases, roles, ("A:signal", "B:noise"), secondary_learner="mlp")
    assert "secondary_learner_target_auroc" not in unknown.estimands
    assert any("mlp" in note for note in unknown.notes)


def test_ranking_utility_and_prevalence_standardisation_are_reported() -> None:
    cases = _cases()
    outcome = evaluate_fold(cases, _roles(cases), ("A:signal", "B:noise"))
    for coverage in (0.5, 0.7, 0.8, 0.9):
        assert f"risk_at_coverage_{coverage:g}" in outcome.estimands
    # The attainable interval travels with the area, so two runs with different
    # explicit-failure rates cannot be compared through the number alone.
    assert "attainable_coverage_lower" in outcome.estimands
    assert "attainable_coverage_upper" in outcome.estimands
    assert outcome.prevalence is not None
    assert np.isfinite(outcome.estimands["prevalence_reference_pi_star"])
