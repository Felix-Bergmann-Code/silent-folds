"""Deterministic content and configuration hashes (specification §12.1, §12.4).

A cached row is reusable only when the dataset version, pipeline version,
feature implementation, and preprocessing hashes all match. That rule is
useless unless the hashes are stable across processes and machines, so this
module fixes one canonical serialisation and uses it everywhere.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "HASH_LENGTH",
    "canonical_json",
    "file_digest",
    "hash_mapping",
    "job_id",
    "short_hash",
]

HASH_LENGTH = 16


def _default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set | frozenset):
        return sorted(obj, key=repr)
    if hasattr(obj, "value"):  # Enum
        return obj.value
    raise TypeError(f"not JSON-serialisable for hashing: {type(obj)!r}")


def canonical_json(payload: Any) -> str:
    """Sorted-key, compact, NaN-free JSON used as hash input."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_default,
    )


def short_hash(payload: Any, length: int = HASH_LENGTH) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:length]


def hash_mapping(mapping: Mapping[str, Any], length: int = HASH_LENGTH) -> str:
    return short_hash(dict(mapping), length)


def file_digest(path: str | Path, chunk: int = 1 << 20) -> str:
    """Full SHA-256 of a file. Used for archive and image checksums."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def job_id(
    *,
    dataset_version: str,
    pair_id: str,
    pipeline_version: str,
    direction: str,
    condition: str,
    severity: int,
    seed: int,
) -> str:
    """Deterministic job identifier -- the cache primary key of §12.4.

    One registration record per
    ``(dataset_version, pair_id, pipeline_version, direction, condition,
    severity, seed)``.
    """
    return short_hash(
        {
            "dataset_version": dataset_version,
            "pair_id": pair_id,
            "pipeline_version": pipeline_version,
            "direction": direction,
            "condition": condition,
            "severity": int(severity),
            "seed": int(seed),
        }
    )
