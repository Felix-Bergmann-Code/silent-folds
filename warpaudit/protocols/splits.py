"""Grouped partitions and transfer protocols (specification §8).

Structure of one evaluated fold, from §8.1:

* **target test** is fold ``k`` of the non-development target groups, and
  nothing else touches it;
* the remaining target groups are partitioned, without overlap, into
  **reference-training** and **target-calibration** groups;
* the source side is partitioned into **source-training**,
  **source-calibration**, and an **independent source test**, from
  non-development source groups with fixed seeds and outer folds.

Source test outcomes never set the source threshold; they estimate its
in-domain risk without calibration-set optimism. Calibration groups are
separate from model-fitting and test groups everywhere.

The protocols are named P1-P6 exactly as in §8.2 so that a result table cell
can be traced back to the design it came from.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal

from ..cache.hashing import short_hash

__all__ = [
    "PROTOCOLS",
    "FoldAssignment",
    "TransferSplit",
    "make_outer_folds",
    "make_transfer_split",
]

Protocol = Literal["P1", "P2", "P3", "P4", "P5", "P6"]

PROTOCOLS: dict[str, str] = {
    "P1": "in-domain: grouped training/calibration/test within one dataset",
    "P2": "pipeline transfer: train on pipelines A/B, test pipeline C on unseen subjects",
    "P3": "pairwise dataset transfer: train/calibrate on X, score held-out groups of Y",
    "P4": "leave-one-dataset-out: train on all sources except Y, test Y",
    "P5": "clean-to-corrupted: fit on clean training groups, test corruptions of unseen groups",
    "P6": "combined transfer: hold out dataset and pipeline simultaneously (exploratory)",
}


def _stable_key(group_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def make_outer_folds(group_ids: Iterable[str], *, n_folds: int, seed: int) -> dict[str, int]:
    """Deterministic grouped fold assignment.

    Folds are balanced by size and depend only on the group ids and the seed,
    so adding a dataset cannot reshuffle an existing assignment. Class balance
    is deliberately *not* used: fold membership must not depend on outcomes.
    """
    ids = sorted(set(group_ids))
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if len(ids) < n_folds:
        raise ValueError(
            f"{len(ids)} group(s) cannot support {n_folds} folds; §8.1 requires falling "
            "back to a smaller fixed fold count before confirmation, not fewer groups"
        )
    ordered = sorted(ids, key=lambda g: _stable_key(g, seed))
    return {g: i % n_folds for i, g in enumerate(ordered)}


@dataclass(frozen=True)
class FoldAssignment:
    fold: int
    n_folds: int
    assignment: dict[str, int]

    def groups_in(self, fold: int) -> list[str]:
        return sorted(g for g, f in self.assignment.items() if f == fold)

    def groups_not_in(self, fold: int) -> list[str]:
        return sorted(g for g, f in self.assignment.items() if f != fold)


@dataclass
class TransferSplit:
    """One evaluated fold of a source -> target transfer protocol."""

    protocol: str
    fold: int
    source_dataset: str
    target_dataset: str
    source_train: list[str] = field(default_factory=list)
    source_calibration: list[str] = field(default_factory=list)
    source_test: list[str] = field(default_factory=list)
    target_train: list[str] = field(default_factory=list)
    target_calibration: list[str] = field(default_factory=list)
    target_test: list[str] = field(default_factory=list)
    development: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    ROLES = (
        "source_train",
        "source_calibration",
        "source_test",
        "target_train",
        "target_calibration",
        "target_test",
    )

    def role_of(self, group_id: str) -> str | None:
        for role in self.ROLES:
            if group_id in getattr(self, role):
                return role
        return "development" if group_id in self.development else None

    def budget(self) -> dict[str, int]:
        """Supervision budget as ``(n_training_groups, n_calibration_groups)``.

        Both sides are reported so a reader can see whether the matched-budget
        requirement of §8.1 was met, rather than inferring it.
        """
        return {
            "source_train_groups": len(self.source_train),
            "source_calibration_groups": len(self.source_calibration),
            "target_train_groups": len(self.target_train),
            "target_calibration_groups": len(self.target_calibration),
        }

    @property
    def hash(self) -> str:
        return short_hash(asdict(self))

    def to_dict(self) -> dict:
        data = asdict(self)
        data["budget"] = self.budget()
        data["hash"] = self.hash
        return data


def _partition(groups: Sequence[str], sizes: dict[str, int], seed: int) -> dict[str, list[str]]:
    """Deterministically cut ``groups`` into named parts of the requested sizes."""
    ordered = sorted(set(groups), key=lambda g: _stable_key(g, seed))
    out: dict[str, list[str]] = {}
    cursor = 0
    for name, size in sizes.items():
        out[name] = sorted(ordered[cursor : cursor + size])
        cursor += size
    out["_remaining"] = sorted(ordered[cursor:])
    return out


def make_transfer_split(
    *,
    protocol: str,
    fold: int,
    n_folds: int,
    source_dataset: str,
    target_dataset: str,
    source_groups: Sequence[str],
    target_groups: Sequence[str],
    development_groups: Sequence[str] = (),
    seed: int = 20260907,
    n_train_groups: int | None = None,
    n_calibration_groups: int | None = None,
) -> TransferSplit:
    """Build one fold of P3/P4 with matched source and target-reference budgets.

    ``n_train_groups`` and ``n_calibration_groups`` define the shared
    supervision budget ``n``. When omitted, the budget is taken from whichever
    side is scarcer, so the two arms are matched by construction and the
    matching is visible in :meth:`TransferSplit.budget`. Group matching is not
    pair-count matching -- §8.1 requires a pair-budget sensitivity comparison
    alongside it, which :mod:`warpaudit.protocols.budgets` computes.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; known: {sorted(PROTOCOLS)}")

    dev = set(development_groups)
    src = [g for g in sorted(set(source_groups)) if g not in dev]
    tgt = [g for g in sorted(set(target_groups)) if g not in dev]
    if set(src) & set(tgt):
        raise ValueError("source and target group ids overlap; ids must be dataset-namespaced")

    folds = make_outer_folds(tgt, n_folds=n_folds, seed=seed)
    assignment = FoldAssignment(fold=fold, n_folds=n_folds, assignment=folds)
    target_test = assignment.groups_in(fold)
    target_pool = assignment.groups_not_in(fold)

    # Budget: default to half the smaller pool for training, a quarter for
    # calibration, then match both sides to the same numbers.
    pool = min(len(target_pool), len(src))
    n_train = n_train_groups if n_train_groups is not None else max(1, pool // 2)
    n_cal = n_calibration_groups if n_calibration_groups is not None else max(1, pool // 4)

    notes: list[str] = []
    if n_train + n_cal > len(target_pool):
        raise ValueError(
            f"target pool has {len(target_pool)} group(s) but the requested budget needs "
            f"{n_train + n_cal}; reduce the budget or the fold count before confirmation"
        )
    if n_train + n_cal > len(src):
        raise ValueError(
            f"source pool has {len(src)} group(s) but the matched budget needs "
            f"{n_train + n_cal}"
        )

    tgt_parts = _partition(target_pool, {"train": n_train, "calibration": n_cal}, seed + 101)
    src_parts = _partition(src, {"train": n_train, "calibration": n_cal}, seed + 202)
    source_test = src_parts["_remaining"]

    if not source_test:
        notes.append(
            "no independent source test groups remain: the in-domain risk of the frozen "
            "source policy cannot be estimated without calibration-set optimism (§8.1)"
        )
    if n_cal < 19:
        notes.append(
            f"{n_cal} calibration group(s) per side: below the 19-group split-conformal "
            "quantile requirement at alpha = 0.05 (§9.3). That requirement is separate "
            "from ordinary policy-calibration adequacy, which may need more."
        )
    if tgt_parts["_remaining"]:
        notes.append(
            f"{len(tgt_parts['_remaining'])} target group(s) are unused in this fold, held "
            "out of every role to keep the budget matched"
        )

    return TransferSplit(
        protocol=protocol,
        fold=fold,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        source_train=src_parts["train"],
        source_calibration=src_parts["calibration"],
        source_test=source_test,
        target_train=tgt_parts["train"],
        target_calibration=tgt_parts["calibration"],
        target_test=target_test,
        development=sorted(dev),
        notes=notes,
    )
