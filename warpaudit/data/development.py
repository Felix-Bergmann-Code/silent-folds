"""Development reserve and outcome-inspection register (specification §3.3).

Subject groups are created *before* feature-error relationships are inspected.
A deterministic, seeded 20% of identifiable groups is reserved for
development. Any group whose outcomes influenced a design decision is marked
development-only and can never re-enter a confirmatory summary -- including
groups inspected before this code existed, which is why
:func:`register_inspected` takes a free-text reason and is stored in the
manifest rather than in a comment.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..cache.store import atomic_write_text

__all__ = ["DevelopmentManifest", "select_development_groups"]


def _stable_rank(group_id: str, seed: int) -> str:
    """Seeded, order-independent ranking key for one group."""
    return hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()


def select_development_groups(
    group_ids: Iterable[str],
    *,
    fraction: float = 0.20,
    seed: int = 20260907,
    always_development: Iterable[str] = (),
) -> list[str]:
    """Deterministically choose the development reserve.

    Selection depends only on the group ids and the seed, never on their
    order, their outcomes, or how many datasets were loaded, so re-running
    after adding a dataset cannot reshuffle an existing reserve. The count is
    ``ceil(fraction * n)`` per §3.3.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("development fraction must lie strictly in (0, 1)")
    ids = sorted(set(group_ids))
    forced = sorted(set(always_development) & set(ids))
    remaining = [g for g in ids if g not in set(forced)]

    target = math.ceil(fraction * len(ids))
    n_extra = max(0, target - len(forced))
    chosen = sorted(remaining, key=lambda g: _stable_rank(g, seed))[:n_extra]
    return sorted(set(forced) | set(chosen))


@dataclass
class DevelopmentManifest:
    """``manifests/development_groups.json``."""

    seed: int
    fraction: float
    development_groups: list[str] = field(default_factory=list)
    confirmatory_groups: list[str] = field(default_factory=list)
    #: group_id -> why its outcomes were inspected. Populated by hand or by
    #: any command that reads outcomes outside the confirmatory protocol.
    inspected_outcomes: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        group_ids: Sequence[str],
        *,
        fraction: float,
        seed: int,
        inspected: dict[str, str] | None = None,
    ) -> DevelopmentManifest:
        inspected = dict(inspected or {})
        dev = select_development_groups(
            group_ids, fraction=fraction, seed=seed, always_development=inspected
        )
        confirm = sorted(set(group_ids) - set(dev))
        notes = []
        if inspected:
            notes.append(
                f"{len(inspected)} group(s) were force-assigned to development because "
                "their outcomes had already been inspected (spec §3.3)."
            )
        if not confirm:
            notes.append(
                "No confirmatory groups remain. Under §3.3 the study must be described "
                "as exploratory unless an independent test set is secured."
            )
        return cls(
            seed=seed,
            fraction=fraction,
            development_groups=dev,
            confirmatory_groups=confirm,
            inspected_outcomes=inspected,
            notes=notes,
        )

    def register_inspected(self, group_id: str, reason: str) -> None:
        """Move a group into development because its outcomes were seen."""
        self.inspected_outcomes[group_id] = reason
        if group_id not in self.development_groups:
            self.development_groups.append(group_id)
            self.development_groups.sort()
        if group_id in self.confirmatory_groups:
            self.confirmatory_groups.remove(group_id)

    def is_development(self, group_id: str) -> bool:
        return group_id in set(self.development_groups)

    def to_json(self) -> str:
        return json.dumps(
            {
                "seed": self.seed,
                "fraction": self.fraction,
                "n_development": len(self.development_groups),
                "n_confirmatory": len(self.confirmatory_groups),
                "development_groups": sorted(self.development_groups),
                "confirmatory_groups": sorted(self.confirmatory_groups),
                "inspected_outcomes": self.inspected_outcomes,
                "notes": self.notes,
            },
            indent=2,
            sort_keys=False,
        )

    def write(self, path: str | Path) -> Path:
        return atomic_write_text(path, self.to_json() + "\n")

    @classmethod
    def read(cls, path: str | Path) -> DevelopmentManifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            seed=int(data["seed"]),
            fraction=float(data["fraction"]),
            development_groups=list(data["development_groups"]),
            confirmatory_groups=list(data["confirmatory_groups"]),
            inspected_outcomes=dict(data.get("inspected_outcomes", {})),
            notes=list(data.get("notes", [])),
        )
