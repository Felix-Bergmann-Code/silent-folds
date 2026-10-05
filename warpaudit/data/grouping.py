"""Independent-unit construction (specification §4.1).

The rule this module implements, and the reason it exists:

    Build an image-pair graph and group connected components whenever images
    are reused. Patient grouping overrides smaller eye or component groupings
    when known. For FIRE, never replace an unavailable patient ID with a
    unique pair ID and then claim subject independence.

So a pair ID can never become a group ID. If no subject identity is available,
the fallback is the *connected component of shared images*, which is
conservative: it merges pairs, it never splits them. Every group carries the
basis it was derived from, and :func:`grouping_claim` produces the sentence
that may honestly be written about it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..types import GROUP_BASIS_PRIORITY, GroupBasis

__all__ = [
    "GroupAssignment",
    "GroupingReport",
    "assign_groups",
    "connected_components",
    "grouping_claim",
]


@dataclass(frozen=True)
class GroupAssignment:
    pair_id: str
    group_id: str
    group_basis: GroupBasis
    component_id: str
    subject_evidence: str = ""


@dataclass
class GroupingReport:
    dataset_id: str
    n_pairs: int
    n_images: int
    n_components: int
    n_groups: int
    basis: GroupBasis
    n_pairs_without_subject_id: int
    unresolved_pairs: list[str] = field(default_factory=list)
    duplicate_image_content: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def patient_disjoint_claimable(self) -> bool:
        """True only when every pair carries recovered subject identity."""
        return self.basis == "patient" and self.n_pairs_without_subject_id == 0


class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:  # path compression
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic merge order so component ids do not depend on
            # dictionary iteration order.
            lo, hi = sorted((ra, rb))
            self._parent[hi] = lo


def connected_components(
    pairs: Sequence[tuple[str, str, str]],
    *,
    duplicate_groups: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, str]:
    """Component id per pair from the image-reuse graph.

    ``pairs`` is a sequence of ``(pair_id, moving_image_id, fixed_image_id)``.
    ``duplicate_groups`` maps a content hash to image ids with identical
    content; duplicates are merged so a repeated image cannot appear on both
    sides of a split (spec §4.1, §12.5.4).
    """
    uf = UnionFind()
    for pair_id, moving, fixed in pairs:
        uf.union(f"img:{moving}", f"img:{fixed}")
        uf.union(f"pair:{pair_id}", f"img:{moving}")
    for images in (duplicate_groups or {}).values():
        images = list(images)
        for other in images[1:]:
            uf.union(f"img:{images[0]}", f"img:{other}")

    roots: dict[str, str] = {}
    ordered: dict[str, str] = {}
    for pair_id, _, _ in pairs:
        root = uf.find(f"pair:{pair_id}")
        if root not in roots:
            roots[root] = f"cc{len(roots):04d}"
        ordered[pair_id] = roots[root]
    return ordered


def assign_groups(
    dataset_id: str,
    pairs: Sequence[tuple[str, str, str]],
    *,
    subject_ids: Mapping[str, str] | None = None,
    subject_basis: GroupBasis = "patient",
    subject_bases: Mapping[str, GroupBasis] | None = None,
    duplicate_groups: Mapping[str, Sequence[str]] | None = None,
    subject_evidence: str = "",
) -> tuple[list[GroupAssignment], GroupingReport]:
    """Assign the highest known independent unit to every pair.

    ``subject_ids`` maps ``pair_id -> subject id`` and may be partial. A pair
    with no subject id falls back to its shared-image connected component; the
    dataset's reported basis is then the *weaker* of the two, because a mixed
    assignment cannot support a subject-disjoint claim.

    ``subject_bases`` optionally records the evidence level per pair.  This is
    needed for archives in which some rows expose patient identity while
    others expose only eye identity.  Missing entries use ``subject_basis``.

    All ids are namespaced by dataset (spec §4.1), so ``FIRE/patient/07`` and
    ``COph100/patient/07`` can never collide.
    """
    if subject_basis not in GROUP_BASIS_PRIORITY:
        raise ValueError(f"unknown subject basis {subject_basis!r}")
    subject_bases = dict(subject_bases or {})
    invalid_bases = sorted(set(subject_bases.values()) - set(GROUP_BASIS_PRIORITY))
    if invalid_bases:
        raise ValueError(f"unknown per-pair subject bases {invalid_bases!r}")

    components = connected_components(pairs, duplicate_groups=duplicate_groups)
    subject_ids = dict(subject_ids or {})

    # A subject id must not split a shared-image component: two pairs sharing
    # an image cannot be independent even if they are labelled with different
    # subjects. Where that happens the component wins and the conflict is
    # reported rather than silently resolved.
    by_component: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for pair_id, comp in components.items():
        if pair_id in subject_ids:
            basis = subject_bases.get(pair_id, subject_basis)
            by_component[comp].add((basis, subject_ids[pair_id]))
    conflicts = {c: sorted(s) for c, s in by_component.items() if len(s) > 1}

    assignments: list[GroupAssignment] = []
    missing: list[str] = []
    for pair_id, _, _ in pairs:
        comp = components[pair_id]
        subject = subject_ids.get(pair_id)
        if subject is not None and comp not in conflicts:
            basis = subject_bases.get(pair_id, subject_basis)
            group_id = f"{dataset_id}/{basis}/{subject}"
        else:
            group_id = f"{dataset_id}/image_component/{comp}"
            basis = "image_component"
            if subject is None:
                missing.append(pair_id)
        assignments.append(
            GroupAssignment(
                pair_id=pair_id,
                group_id=group_id,
                group_basis=basis,
                component_id=comp,
                subject_evidence=subject_evidence if subject is not None else "",
            )
        )

    bases = {a.group_basis for a in assignments}
    weakest = max(bases, key=GROUP_BASIS_PRIORITY.index) if bases else "image_component"

    notes: list[str] = []
    if duplicate_groups:
        notes.append(
            f"{len(duplicate_groups)} exact image-content duplicate set(s) were merged "
            "before group assignment"
        )
    if conflicts:
        notes.append(
            f"{len(conflicts)} shared-image component(s) span multiple subject ids; "
            "the component grouping was kept because it is the conservative unit: "
            + ", ".join(f"{c}={ids}" for c, ids in sorted(conflicts.items())[:5])
        )
    if missing:
        notes.append(
            f"{len(missing)} pair(s) have no recovered subject id and fall back to "
            "shared-image components; this dataset is NOT patient-disjoint"
        )

    report = GroupingReport(
        dataset_id=dataset_id,
        n_pairs=len(pairs),
        n_images=len({i for _, m, f in pairs for i in (m, f)}),
        n_components=len(set(components.values())),
        n_groups=len({a.group_id for a in assignments}),
        basis=weakest,  # type: ignore[arg-type]
        n_pairs_without_subject_id=len(missing),
        unresolved_pairs=sorted(missing),
        duplicate_image_content={k: list(v) for k, v in (duplicate_groups or {}).items()},
        notes=notes,
    )
    return assignments, report


def grouping_claim(report: GroupingReport) -> str:
    """The strongest sentence the evidence supports. Used verbatim in reports."""
    if report.patient_disjoint_claimable:
        return (
            f"{report.dataset_id}: grouped by recovered patient identity "
            f"({report.n_groups} groups over {report.n_pairs} pairs); "
            "splits are patient-disjoint."
        )
    if report.basis == "image_component":
        return (
            f"{report.dataset_id}: patient identity is not resolved for "
            f"{report.n_pairs_without_subject_id} of {report.n_pairs} pairs. "
            f"Grouping falls back to shared-image connected components "
            f"({report.n_groups} groups). Splits are image-disjoint but MUST NOT "
            "be described as patient-disjoint."
        )
    return (
        f"{report.dataset_id}: grouped by {report.basis} "
        f"({report.n_groups} groups over {report.n_pairs} pairs). "
        f"Dependence between {report.basis} units of one patient is unresolved; "
        "splits are not claimed to be patient-disjoint."
    )
