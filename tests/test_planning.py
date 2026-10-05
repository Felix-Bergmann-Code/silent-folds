from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from warpaudit.cli import _assess_direction
from warpaudit.config import Config, ConfigError, load_config
from warpaudit.protocols.planning import (
    DevelopmentEvidence,
    choose_fold_count,
    development_evidence,
    make_direction_plan,
    project_accepted_groups,
    project_class_support,
    simulate_iut_scenario,
)


def test_direction_plan_falls_back_and_keeps_all_roles_disjoint() -> None:
    source = [f"X/g{i:02d}" for i in range(44)]
    target = [f"Y/g{i:02d}" for i in range(33)]
    plan = make_direction_plan(
        source_dataset="X",
        target_dataset="Y",
        source_groups=source,
        target_groups=target,
        development_groups=[],
        preferred_folds=5,
        fallback_folds=3,
        low_information_threshold=10,
        seed=7,
    )
    assert plan.n_folds == 3
    assert plan.n_train_groups == plan.n_calibration_groups == 11
    for fold in plan.folds:
        assert fold.budget() == {
            "source_train_groups": 11,
            "source_calibration_groups": 11,
            "target_train_groups": 11,
            "target_calibration_groups": 11,
        }
        role_sets = [set(getattr(fold, role)) for role in fold.ROLES]
        assert all(
            not left & right
            for i, left in enumerate(role_sets)
            for right in role_sets[i + 1 :]
        )
        assert len(fold.source_test) == 22
        assert len(fold.target_test) == 11


def test_development_projections_are_group_based_and_reproducible() -> None:
    rows = pd.DataFrame(
        {
            "group_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "operational_failure": [False, True, False, False, True, True, False, True],
        }
    )
    evidence = development_evidence(rows, dataset_id="D", pipeline_id="P")
    assert evidence.n_groups == 4
    assert evidence.failure_bearing_groups == 3
    assert evidence.success_bearing_groups == 3
    assert evidence.group_weighted_failure_prevalence == 0.5

    first = project_class_support(evidence, n_test_groups=15, minimum=10, seed=11)
    second = project_class_support(evidence, n_test_groups=15, minimum=10, seed=11)
    assert first == second
    assert 0.0 <= first.probability_both_at_least_minimum <= 1.0

    accepted = project_accepted_groups(
        n_test_groups=15,
        n_calibration_groups=11,
        nominal_acceptance=0.70,
        minimum=5,
        seed=11,
    )
    assert accepted.threshold_quantile_sd > 0
    assert accepted.accepted_groups_p05 <= accepted.accepted_groups_mean
    assert accepted.accepted_groups_mean <= accepted.accepted_groups_p95


def test_iut_scenario_requires_every_one_sided_component() -> None:
    boundary = simulate_iut_scenario(
        "boundary",
        assumed_target_auroc=0.70,
        assumed_gap_auc=0.05,
        assumed_delta_policy=0.05,
        failure_bearing_groups=15,
        success_bearing_groups=15,
        accepted_groups=12,
        auroc_floor=0.70,
        gap_margin=0.05,
        delta_policy_margin=0.05,
        alpha=0.05,
        n_simulations=5_000,
        seed=3,
    )
    strong = simulate_iut_scenario(
        "strong",
        assumed_target_auroc=0.90,
        assumed_gap_auc=-0.10,
        assumed_delta_policy=0.35,
        failure_bearing_groups=40,
        success_bearing_groups=40,
        accepted_groups=35,
        auroc_floor=0.70,
        gap_margin=0.05,
        delta_policy_margin=0.05,
        alpha=0.05,
        n_simulations=5_000,
        seed=3,
    )
    assert boundary.probability_joint_passes < 0.05
    assert strong.probability_joint_passes > 0.80
    assert np.isfinite(strong.gap_auc_se)


def test_fold_choice_keeps_five_when_each_test_has_ten_groups() -> None:
    assert (
        choose_fold_count(
            50,
            preferred=5,
            fallback=3,
            minimum_test_groups=10,
        )
        == 5
    )


def test_planning_parameters_stay_outside_the_cache_reuse_contract(tmp_path: Path) -> None:
    """Planning knobs are report-only, so tuning them must not invalidate caches."""
    base = """
project: fixture
datasets:
  - {id: FIRE, version: fixture, root: data/FIRE}
pipelines:
  - {id: fixture, version: fixture-1, matcher: fixture, checkpoint: local}
  - {id: fixture2, version: fixture-2, matcher: fixture2, checkpoint: local}
"""
    plain = tmp_path / "plain.yaml"
    plain.write_text(base, encoding="utf-8")
    tuned = tmp_path / "tuned.yaml"
    tuned.write_text(
        base
        + """
planning:
  min_class_bearing_groups: 3
  feasibility_probability: 0.5
  n_simulations: 500
  scenarios:
    only: [0.9, 0.01, 0.2]
""",
        encoding="utf-8",
    )
    unchanged = load_config(plain)
    changed = load_config(tuned)
    assert changed.planning.min_class_bearing_groups == 3
    assert changed.planning.scenarios == {"only": (0.9, 0.01, 0.2)}
    assert changed.hash == unchanged.hash
    assert "planning" in Config.HASH_EXCLUDED_SECTIONS

    broken = tmp_path / "broken.yaml"
    broken.write_text(
        base + "\nplanning: {feasibility_probability: 1.5, scenarios: {bad: [0.9, 0.01]}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as excinfo:
        load_config(broken)
    message = str(excinfo.value)
    assert "planning.feasibility_probability" in message
    assert "planning.scenarios.bad" in message


def _evidence(dataset_id: str, *, failure_groups: int, n_groups: int) -> DevelopmentEvidence:
    rows = pd.DataFrame(
        {
            "group_id": [f"{dataset_id}/g{i:02d}" for i in range(n_groups) for _ in range(4)],
            "operational_failure": [
                (i < failure_groups) and bool(j % 2) for i in range(n_groups) for j in range(4)
            ],
        }
    )
    return development_evidence(rows, dataset_id=dataset_id, pipeline_id="P")


def test_direction_assessment_blocks_on_scarce_class_bearing_groups() -> None:
    """A target whose development split rarely fails cannot carry the ranking claim."""
    cfg = load_config(Path("configs/pilot.yaml"))
    groups = {
        "COph100": [f"COph100/g{i:02d}" for i in range(56)],
        "FIRE": [f"FIRE/g{i:02d}" for i in range(42)],
    }
    development = groups["COph100"][:12] + groups["FIRE"][:9]
    evidence = {
        ("COph100", "xfeat_h"): _evidence("COph100", failure_groups=10, n_groups=12),
        ("FIRE", "xfeat_h"): _evidence("FIRE", failure_groups=1, n_groups=9),
    }

    to_fire = _assess_direction(
        cfg,
        source_dataset="COph100",
        target_dataset="FIRE",
        groups_by_dataset=groups,
        development_groups=development,
        evidence=evidence,
        common_pipelines=["xfeat_h"],
    )
    to_coph = _assess_direction(
        cfg,
        source_dataset="FIRE",
        target_dataset="COph100",
        groups_by_dataset=groups,
        development_groups=development,
        evidence=evidence,
        common_pipelines=["xfeat_h"],
    )

    scarce = to_fire["pipelines"][0]["class_support"]["probability_both_at_least_minimum"]
    plentiful = to_coph["pipelines"][0]["class_support"]["probability_both_at_least_minimum"]
    assert scarce < plentiful
    assert any("bearing each class" in reason for reason in to_fire["blocking"])
    assert not any("bearing each class" in reason for reason in to_coph["blocking"])
    # Neither direction may be silently called feasible while a stated minimum fails.
    for assessment in (to_fire, to_coph):
        assert assessment["feasible"] == (not assessment["blocking"])
        roles_per_fold = assessment["target_test_groups_per_fold"]
        assert min(roles_per_fold) == assessment["smallest_target_test_groups"]
