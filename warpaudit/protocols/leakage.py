"""Leakage checks (specification §12.5.4, §8.1).

    Features and source fitting cannot access target labels; groups and shared
    images cannot cross training/calibration/test within any fold. All
    derivative rows inherit groups.

These are executable assertions, not review guidance. Every check returns a
list of concrete violations so a failing sweep names the offending ids instead
of raising a generic error.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from .splits import TransferSplit

__all__ = [
    "LeakageError",
    "check_derivative_groups",
    "check_group_disjoint",
    "check_no_shared_images",
    "check_split",
    "enforce",
]


class LeakageError(AssertionError):
    """Raised when a split would let information cross a role boundary."""


def check_group_disjoint(split: TransferSplit) -> list[str]:
    """No group may hold two roles inside one fold."""
    seen: dict[str, str] = {}
    violations: list[str] = []
    for role in TransferSplit.ROLES:
        for group in getattr(split, role):
            if group in seen:
                violations.append(f"group {group!r} appears in both {seen[group]!r} and {role!r}")
            else:
                seen[group] = role
    for group in split.development:
        if group in seen:
            violations.append(
                f"development group {group!r} also appears in {seen[group]!r}; development "
                "outcomes must never enter a confirmatory role (§3.3)"
            )
    return violations


def check_no_shared_images(
    split: TransferSplit,
    group_of_pair: Mapping[str, str],
    images_of_pair: Mapping[str, tuple[str, str]],
) -> list[str]:
    """No image may appear on both sides of a role boundary.

    Shared-image connected components already guarantee this when grouping is
    component-based, but a *subject-based* grouping can still let a duplicated
    image cross roles, which is exactly the case §4.1 asks to audit.
    """
    role_of_group: dict[str, str] = {}
    for role in TransferSplit.ROLES:
        for group in getattr(split, role):
            role_of_group[group] = role

    roles_of_image: dict[str, set[str]] = defaultdict(set)
    for pair_id, group in group_of_pair.items():
        role = role_of_group.get(group)
        if role is None:
            continue
        for image in images_of_pair.get(pair_id, ()):
            roles_of_image[image].add(role)

    return [
        f"image {image!r} appears in multiple roles: {sorted(roles)}"
        for image, roles in sorted(roles_of_image.items())
        if len(roles) > 1
    ]


def check_derivative_groups(
    rows: Iterable[Mapping[str, object]],
    *,
    pair_key: str = "pair_id",
    group_key: str = "group_id",
) -> list[str]:
    """Every derivative row inherits its pair's group (§4.1).

    Reverse directions, corruption severities, and seed repeats are
    derivatives. A derivative that acquired a new group id would create a fake
    independent subject.
    """
    groups_of_pair: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups_of_pair[str(row[pair_key])].add(str(row[group_key]))
    return [
        f"pair {pair!r} carries multiple group ids across its derivative rows: {sorted(groups)}"
        for pair, groups in sorted(groups_of_pair.items())
        if len(groups) > 1
    ]


def check_split(
    split: TransferSplit,
    *,
    group_of_pair: Mapping[str, str] | None = None,
    images_of_pair: Mapping[str, tuple[str, str]] | None = None,
    rows: Sequence[Mapping[str, object]] | None = None,
) -> list[str]:
    """Run every applicable check and return all violations."""
    violations = check_group_disjoint(split)
    if group_of_pair is not None and images_of_pair is not None:
        violations += check_no_shared_images(split, group_of_pair, images_of_pair)
    if rows is not None:
        violations += check_derivative_groups(rows)
    return violations


def enforce(violations: Sequence[str], context: str = "") -> None:
    if violations:
        header = f"{len(violations)} leakage violation(s)" + (f" in {context}" if context else "")
        raise LeakageError(header + ":\n  - " + "\n  - ".join(violations))
