"""Grouped partitions, budget matching, and leakage checks."""

from .budgets import BudgetReport, SideBudget, compare_budgets, count_budget
from .leakage import LeakageError, check_split, enforce
from .planning import (
    AcceptedGroupProjection,
    ClassSupportProjection,
    DevelopmentEvidence,
    DirectionPlan,
    IUTScenario,
    choose_fold_count,
    development_evidence,
    make_direction_plan,
    project_accepted_groups,
    project_class_support,
    simulate_iut_scenario,
)
from .splits import PROTOCOLS, FoldAssignment, TransferSplit, make_outer_folds, make_transfer_split

__all__ = [
    "BudgetReport",
    "AcceptedGroupProjection",
    "ClassSupportProjection",
    "DevelopmentEvidence",
    "DirectionPlan",
    "FoldAssignment",
    "LeakageError",
    "IUTScenario",
    "PROTOCOLS",
    "SideBudget",
    "TransferSplit",
    "check_split",
    "choose_fold_count",
    "compare_budgets",
    "count_budget",
    "development_evidence",
    "enforce",
    "make_outer_folds",
    "make_direction_plan",
    "make_transfer_split",
    "project_accepted_groups",
    "project_class_support",
    "simulate_iut_scenario",
]
