"""Status ledger for resumable sweeps (specification §12.1).

The ledger answers one question cheaply: *which jobs are done, which failed,
and why?* It is append-only JSON Lines so that an interrupted process leaves a
readable file, and it is de-duplicated by ``job_id`` on load with last-write-
wins.

Infrastructure failures (timeout, OOM, transport) are retried under a fixed
policy; scientific no-output statuses are not. Unresolved technical failures
are reported separately from scientific failures (spec §6.2).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["JobRecord", "StatusLedger"]


@dataclass
class JobRecord:
    job_id: str
    kind: str  # register | features | evaluate
    state: str  # pending | running | done | failed
    status: str = ""  # RegistrationStatus value when applicable
    attempt: int = 1
    message: str = ""
    runtime_s: float = float("nan")
    started_at: float = field(default_factory=time.time)
    finished_at: float = float("nan")
    extra: dict[str, Any] = field(default_factory=dict)


class StatusLedger:
    """Append-only job ledger backed by one JSON Lines file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: JobRecord) -> None:
        line = json.dumps(asdict(record), sort_keys=True, default=str)
        # A hard kill can leave the previous JSON object without its newline.
        # Start the resumed record on a fresh line so the loader can skip only
        # the torn object instead of losing the first valid retry with it.
        needs_separator = False
        if self.path.exists() and self.path.stat().st_size:
            with open(self.path, "rb") as previous:
                previous.seek(-1, os.SEEK_END)
                needs_separator = previous.read(1) != b"\n"
        with open(self.path, "a", encoding="utf-8") as fh:
            if needs_separator:
                fh.write("\n")
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def __iter__(self) -> Iterator[JobRecord]:
        if not self.path.exists():
            return iter(())
        records: list[JobRecord] = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(JobRecord(**json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    # A torn final line from a hard kill is skipped, not fatal.
                    continue
        return iter(records)

    def latest(self) -> dict[str, JobRecord]:
        out: dict[str, JobRecord] = {}
        for rec in self:
            out[rec.job_id] = rec
        return out

    def completed(self, kind: str | None = None) -> set[str]:
        return {
            jid
            for jid, rec in self.latest().items()
            if rec.state == "done" and (kind is None or rec.kind == kind)
        }

    def failures(self, kind: str | None = None) -> dict[str, JobRecord]:
        return {
            jid: rec
            for jid, rec in self.latest().items()
            if rec.state == "failed" and (kind is None or rec.kind == kind)
        }

    def attempts(self, job_id: str) -> int:
        rec = self.latest().get(job_id)
        return rec.attempt if rec else 0

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rec in self.latest().values():
            counts[rec.state] = counts.get(rec.state, 0) + 1
            if rec.status:
                counts[f"status:{rec.status}"] = counts.get(f"status:{rec.status}", 0) + 1
        return counts
