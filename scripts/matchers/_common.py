"""Dependency-light helpers shared by isolated matcher workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_VERSION = "1.0"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    return parser.parse_args()


def read_request(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"expected WarpAudit matcher protocol {PROTOCOL_VERSION}")
    if payload.get("pixel_center_convention") != "integer-centre":
        raise ValueError("worker only supports the frozen integer-centre convention")
    return payload


def write_response(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def load_working_image(frame: dict[str, Any]) -> np.ndarray:
    path = Path(frame["working_array_path"])
    image = np.load(path, allow_pickle=False)
    expected = tuple(int(v) for v in frame["working_hw"])
    if image.ndim != 3 or image.shape[:2] != expected or image.shape[2] != 3:
        raise ValueError(f"working image shape {image.shape} does not match {expected} RGB")
    if image.dtype != np.float32 or not np.isfinite(image).all():
        raise ValueError("working image must be a finite float32 RGB array")
    return image


def seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device(requested: str, torch: Any) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return requested


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repository: Path) -> str:
    process = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"cannot identify upstream commit: {process.stderr.strip()}")
    return process.stdout.strip()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def add_upstream_to_path(path: Path) -> None:
    sys.path.insert(0, str(path))


def peak_vram_bytes(device: str, torch: Any) -> int:
    if device == "cuda":
        return int(torch.cuda.max_memory_allocated())
    return 0


def base_response(
    request: dict[str, Any],
    *,
    moving: np.ndarray,
    fixed: np.ndarray,
    scores: np.ndarray | None,
    provenance: dict[str, str],
    runtime_s: float,
    cpu_time_s: float,
    peak_vram: int,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "pipeline_id": request["pipeline_id"],
        "coordinate_frame": "working_pixel_centres",
        "matches_moving": np.asarray(moving, dtype=float).tolist(),
        "matches_fixed": np.asarray(fixed, dtype=float).tolist(),
        "match_scores": None if scores is None else np.asarray(scores, dtype=float).tolist(),
        "provenance": provenance,
        "runtime_s": float(runtime_s),
        "cpu_time_s": float(cpu_time_s),
        "peak_vram_bytes": int(peak_vram),
        "diagnostics": diagnostics,
    }
