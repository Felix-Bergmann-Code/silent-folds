"""Atomic, resumable cache writes (specification §12.1, §12.4).

Design constraints taken from the sprint acceptance criterion "a deliberately
interrupted fixture job resumes without duplicate rows":

* every write goes to a temporary file in the destination directory and is
  then ``os.replace``-d into place, so a reader never sees a partial file;
* rows are keyed by ``job_id`` and de-duplicated on load, keeping the last
  complete write;
* an empty Parquet file is not acceptance -- :meth:`ShardedTable.count` and
  the ledger are what a caller checks.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "PARQUET_UNAVAILABLE_REASON",
    "ShardedTable",
    "atomic_write_bytes",
    "atomic_write_text",
    "parquet_available",
    "read_table",
]

_PARQUET_STATE: dict[str, Any] = {}
_NULLABLE_OBJECT_COLUMNS = frozenset(
    {"transform_params", "matches_moving", "matches_fixed", "inlier_mask", "match_scores"}
)


def parquet_available() -> bool:
    """Probe the Parquet engine once.

    Parquet is the declared release format (§12.4). Some environments ship a
    pyarrow build that is binary-incompatible with the installed pandas; that
    is an environment defect, not a reason to lose data, so the store falls
    back to newline-delimited JSON and records the reason. ``estimate-cost``
    and the sprint handoff surface the fallback so it is never silent.
    """
    if "ok" not in _PARQUET_STATE:
        try:
            import pyarrow.parquet  # noqa: F401

            pd.DataFrame({"a": [1]}).to_parquet(None)
            _PARQUET_STATE["ok"] = True
            _PARQUET_STATE["reason"] = ""
        except Exception as exc:  # pragma: no cover - environment dependent
            _PARQUET_STATE["ok"] = False
            _PARQUET_STATE["reason"] = f"{type(exc).__name__}: {exc}"
    return bool(_PARQUET_STATE["ok"])


def PARQUET_UNAVAILABLE_REASON() -> str:
    parquet_available()
    return str(_PARQUET_STATE.get("reason", ""))


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".jsonl":
        frame = pd.read_json(path, lines=True)
    elif path.suffix in (".csv", ".tsv"):
        frame = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    else:
        raise ValueError(f"unsupported table format: {path.suffix}")
    # Arrow/pandas combinations differ in whether an all-null nested column is
    # returned as None or floating NaN. Restore the versioned cache contract at
    # the I/O boundary so behavior is identical on the Windows study machine.
    for column in _NULLABLE_OBJECT_COLUMNS & set(frame.columns):
        values = frame[column].astype(object)
        frame[column] = values.where(pd.notna(values), None)
    return frame


@dataclass
class ShardedTable:
    """A logical table stored as append-only Parquet shards.

    Sharding keeps writes cheap and interruption-safe: a crashed run loses at
    most the shard it was writing, and reruns produce a new shard whose rows
    supersede earlier ones with the same ``key_column``.
    """

    root: Path
    name: str
    key_column: str = "job_id"
    _next_index: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        return self.root / self.name

    @property
    def suffix(self) -> str:
        return ".parquet" if parquet_available() else ".jsonl"

    def shards(self) -> list[Path]:
        """Every shard, in write order, across both permitted formats."""
        return sorted(
            list(self.directory.glob("part-*.parquet")) + list(self.directory.glob("part-*.jsonl")),
            key=lambda p: p.name,
        )

    def append(self, rows: Sequence[Mapping[str, Any]], *, shard_hint: str = "") -> Path | None:
        """Write one shard. Returns ``None`` for an empty write."""
        if not rows:
            return None
        frame = pd.DataFrame(list(rows))
        # The study has one publishing parent. Scan once per writer, rather
        # than walking an ever-growing directory for every registration.
        if self._next_index is None:
            self._next_index = 1 + max(
                (int(path.name.split("-")[1].split(".")[0]) for path in self.shards()),
                default=-1,
            )
        index = self._next_index
        self._next_index += 1
        suffix = f"-{shard_hint}" if shard_hint else ""
        path = self.directory / f"part-{index:06d}{suffix}{self.suffix}"
        if path.suffix == ".parquet":
            buf = frame.to_parquet(index=False)
            assert buf is not None  # to_parquet(path=None) returns bytes
            return atomic_write_bytes(path, buf)
        return atomic_write_text(path, frame.to_json(orient="records", lines=True))

    def load(self) -> pd.DataFrame:
        """Read every shard, keeping the last row per key."""
        frames = [read_table(p) for p in self.shards()]
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        if self.key_column in out.columns:
            out = out.drop_duplicates(subset=[self.key_column], keep="last").reset_index(drop=True)
        return out

    def existing_keys(self) -> set[str]:
        frame = self.load()
        if frame.empty or self.key_column not in frame.columns:
            return set()
        return set(frame[self.key_column].astype(str))

    def count(self) -> int:
        return int(len(self.load()))

    def missing(self, keys: Iterable[str]) -> list[str]:
        """Keys not yet present -- the basis of a resumable sweep."""
        done = self.existing_keys()
        return [k for k in keys if k not in done]
