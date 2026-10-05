"""Surgical edits to the configuration file (spec §12.1).

Re-pinning matcher provenance on a new machine is a protocol change, not a
formatting change, so it must alter exactly the recorded values and nothing
else. Round-tripping the YAML through a parser would discard the comments that
carry the reasoning behind every pin, which is most of the file's value, so
this edits the specific lines instead and refuses anything it cannot locate
unambiguously.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["ConfigEditError", "replace_pipeline_provenance"]


class ConfigEditError(RuntimeError):
    """Raised when the block to edit cannot be located unambiguously."""


def _pipeline_span(lines: list[str], pipeline_id: str) -> tuple[int, int]:
    """The line range of one pipeline entry in the ``pipelines:`` sequence."""
    starts = [
        index
        for index, line in enumerate(lines)
        if re.fullmatch(rf"\s*-\s+id:\s*{re.escape(pipeline_id)}\s*", line.rstrip("\n"))
    ]
    if len(starts) != 1:
        raise ConfigEditError(
            f"expected exactly one '- id: {pipeline_id}' entry, found {len(starts)}"
        )
    start = starts[0]
    indent = len(lines[start]) - len(lines[start].lstrip())
    for index in range(start + 1, len(lines)):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            continue
        current = len(lines[index]) - len(lines[index].lstrip())
        if current <= indent and stripped.startswith("- "):
            return start, index
        if current < indent:
            return start, index
    return start, len(lines)


def replace_pipeline_provenance(
    path: str | Path, pipeline_id: str, provenance: dict[str, str]
) -> bool:
    """Replace one pipeline's ``provenance:`` mapping. Returns whether it changed.

    Every value is quoted so a version string such as ``2.3.1+cu121`` cannot be
    reinterpreted by the YAML scalar rules.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    start, end = _pipeline_span(lines, pipeline_id)

    block_starts = [
        index
        for index in range(start, end)
        if re.fullmatch(r"\s*provenance:\s*", lines[index].rstrip("\n"))
    ]
    if len(block_starts) != 1:
        raise ConfigEditError(
            f"pipeline {pipeline_id!r} has {len(block_starts)} 'provenance:' blocks; "
            "expected exactly one"
        )
    block_start = block_starts[0]
    key_indent = len(lines[block_start]) - len(lines[block_start].lstrip())

    block_end = block_start + 1
    entry_indent: int | None = None
    while block_end < end:
        stripped = lines[block_end].strip()
        if not stripped:
            break
        current = len(lines[block_end]) - len(lines[block_end].lstrip())
        if current <= key_indent:
            break
        if entry_indent is None:
            entry_indent = current
        block_end += 1
    if entry_indent is None:
        entry_indent = key_indent + 2

    replacement = [
        f'{" " * entry_indent}{key}: "{value}"\n' for key, value in sorted(provenance.items())
    ]
    if lines[block_start + 1 : block_end] == replacement:
        return False
    path.write_text(
        "".join(lines[: block_start + 1] + replacement + lines[block_end:]), encoding="utf-8"
    )
    return True
