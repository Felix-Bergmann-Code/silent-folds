"""Annotation-budget accounting (specification §8.1, §8.3).

    Match source and target-reference supervision as
    ``(n_training_groups, n_calibration_groups)``, count all labels used for
    fitting/tuning/calibration, and report pair/class counts and label-reuse
    conventions. Because groups contain different numbers of pairs, group
    matching is not pair-count matching; add a pair-budget sensitivity
    comparison.

So this module counts three things separately -- groups, pairs, and *labels
actually consumed* -- and never lets one stand in for another. Probability-
calibration labels are budgeted too: a constant baseline needs no misleading
training-cost claim, but a Platt map fitted on 200 labelled pairs has spent
them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .splits import TransferSplit

__all__ = ["BudgetReport", "SideBudget", "compare_budgets", "count_budget"]


@dataclass(frozen=True)
class SideBudget:
    side: str
    train_groups: int
    calibration_groups: int
    train_pairs: int
    calibration_pairs: int
    train_labels: int
    calibration_labels: int

    @property
    def total_groups(self) -> int:
        return self.train_groups + self.calibration_groups

    @property
    def total_labels(self) -> int:
        return self.train_labels + self.calibration_labels

    def as_row(self) -> dict[str, int | str]:
        return {
            "side": self.side,
            "train_groups": self.train_groups,
            "calibration_groups": self.calibration_groups,
            "train_pairs": self.train_pairs,
            "calibration_pairs": self.calibration_pairs,
            "train_labels": self.train_labels,
            "calibration_labels": self.calibration_labels,
            "total_groups": self.total_groups,
            "total_labels": self.total_labels,
        }


@dataclass
class BudgetReport:
    source: SideBudget
    target_reference: SideBudget
    groups_matched: bool
    pairs_matched: bool
    labels_matched: bool
    notes: list[str] = field(default_factory=list)

    @property
    def pair_ratio(self) -> float:
        denom = self.target_reference.train_pairs + self.target_reference.calibration_pairs
        num = self.source.train_pairs + self.source.calibration_pairs
        return float(num) / float(denom) if denom else float("nan")

    def as_rows(self) -> list[dict]:
        return [self.source.as_row(), self.target_reference.as_row()]

    def summary(self) -> str:
        lines = [
            f"Group budget matched: {self.groups_matched}",
            f"Pair budget matched:  {self.pairs_matched} (source/target pair ratio "
            f"{self.pair_ratio:.2f})",
            f"Label budget matched: {self.labels_matched}",
        ]
        return "\n".join(lines + [f"  - {n}" for n in self.notes])


def count_budget(
    side: str,
    train_groups: Sequence[str],
    calibration_groups: Sequence[str],
    pairs_per_group: Mapping[str, int],
    *,
    labels_per_pair: int = 1,
) -> SideBudget:
    """Count groups, pairs, and consumed labels for one arm.

    ``labels_per_pair`` is 1 for a pair-level failure label. Landmark
    annotations behind that label are dataset provenance, not a per-arm
    supervision cost, and are reported in the data manifest instead.
    """

    def pairs(groups: Sequence[str]) -> int:
        return int(sum(pairs_per_group.get(g, 0) for g in groups))

    train_pairs = pairs(train_groups)
    cal_pairs = pairs(calibration_groups)
    return SideBudget(
        side=side,
        train_groups=len(train_groups),
        calibration_groups=len(calibration_groups),
        train_pairs=train_pairs,
        calibration_pairs=cal_pairs,
        train_labels=train_pairs * labels_per_pair,
        calibration_labels=cal_pairs * labels_per_pair,
    )


def compare_budgets(
    split: TransferSplit,
    pairs_per_group: Mapping[str, int],
    *,
    pair_tolerance: float = 0.20,
) -> BudgetReport:
    """Compare the source and target-reference supervision budgets of one fold."""
    source = count_budget(
        f"source:{split.source_dataset}",
        split.source_train,
        split.source_calibration,
        pairs_per_group,
    )
    target = count_budget(
        f"target-reference:{split.target_dataset}",
        split.target_train,
        split.target_calibration,
        pairs_per_group,
    )

    groups_matched = (
        source.train_groups == target.train_groups
        and source.calibration_groups == target.calibration_groups
    )
    denom = target.train_pairs + target.calibration_pairs
    ratio = (source.train_pairs + source.calibration_pairs) / denom if denom else float("inf")
    pairs_matched = abs(ratio - 1.0) <= pair_tolerance

    notes: list[str] = []
    if groups_matched and not pairs_matched:
        notes.append(
            f"groups are matched but pair counts differ by a factor of {ratio:.2f}; "
            "§8.1 requires reporting the pair-budget sensitivity comparison alongside "
            "the group-matched result"
        )
    if not groups_matched:
        notes.append(
            "group budgets are NOT matched; this cell cannot be presented as a "
            "matched-budget comparison"
        )
    if source.calibration_groups == 0 or target.calibration_groups == 0:
        notes.append("an arm has no calibration groups: its policy cannot be frozen")

    return BudgetReport(
        source=source,
        target_reference=target,
        groups_matched=groups_matched,
        pairs_matched=pairs_matched,
        labels_matched=source.total_labels == target.total_labels,
        notes=notes,
    )
