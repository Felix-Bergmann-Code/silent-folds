"""Resumable execution of isolated registration adapters (specification §12.1).

The runner depends only on the typed ``Registrar`` protocol.  XFeat,
SuperPoint/LightGlue, and future native methods may live in incompatible
environments as long as their adapter returns a ``RegistrationResult`` or
writes the equivalent schema.  Scientific no-output statuses are completed
attempts; infrastructure failures remain retryable under the fixed attempt
policy.
"""

from __future__ import annotations

import platform
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..cache.hashing import canonical_json, short_hash
from ..cache.hashing import job_id as make_job_id
from ..cache.ledger import JobRecord, StatusLedger
from ..cache.store import ShardedTable
from ..geometry.transforms import (
    AffineTransform,
    ComposedTransform,
    HomographyTransform,
    ThinPlateSplineTransform,
    Transform,
)
from ..parallel import ordered_map
from ..types import PairInput, Registrar, RegistrationResult, RegistrationStatus

__all__ = [
    "RegistrationJob",
    "RegistrationRunner",
    "deserialise_transform",
    "serialise_transform",
]


def _array(value: np.ndarray | None) -> str | None:
    return None if value is None else canonical_json(np.asarray(value))


def _diagnostic_values(value: Any) -> Any:
    """Encode undefined diagnostic measurements as JSON null, never as zero.

    This applies only to descriptive diagnostics. Transform parameters and
    correspondence arrays still pass through the strict finite JSON writer.
    The explicit registration status and reason retain no-output semantics.
    """
    if isinstance(value, dict):
        return {key: _diagnostic_values(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _diagnostic_values(value.tolist())
    if isinstance(value, list | tuple):
        return [_diagnostic_values(item) for item in value]
    if isinstance(value, float | np.floating) and not np.isfinite(value):
        return None
    return value


def serialise_transform(transform: Transform | None) -> dict[str, Any] | None:
    """Lossless JSON-compatible parameters for supported canonical transforms."""
    if transform is None:
        return None
    if isinstance(transform, HomographyTransform):
        return {"family": "homography", "matrix": transform.matrix.tolist()}
    if isinstance(transform, AffineTransform):
        return {"family": "affine", "matrix": transform.matrix.tolist()}
    if isinstance(transform, ThinPlateSplineTransform):
        return {
            "family": "tps",
            "control_points": transform.control_points.tolist(),
            "weights": transform.weights.tolist(),
            "affine": transform.affine.tolist(),
            "normalisation": transform.normalisation.matrix.tolist(),
            "denormalisation": transform.denormalisation.matrix.tolist(),
            "regularisation": transform.regularisation,
        }
    if isinstance(transform, ComposedTransform):
        return {"family": "composed", "steps": [serialise_transform(s) for s in transform.steps]}
    raise TypeError(
        f"transform family {type(transform).__name__} has no lossless cache representation"
    )


def deserialise_transform(payload: dict[str, Any] | str | None) -> Transform | None:
    """Inverse of :func:`serialise_transform` for exact cache replay."""
    if payload is None:
        return None
    if isinstance(payload, str):
        import json

        payload = json.loads(payload)
    family = payload.get("family")
    if family == "homography":
        return HomographyTransform(np.asarray(payload["matrix"], dtype=np.float64))
    if family == "affine":
        return AffineTransform(np.asarray(payload["matrix"], dtype=np.float64))
    if family == "tps":
        return ThinPlateSplineTransform(
            control_points=np.asarray(payload["control_points"], dtype=np.float64),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            affine=np.asarray(payload["affine"], dtype=np.float64),
            normalisation=AffineTransform(np.asarray(payload["normalisation"], dtype=np.float64)),
            denormalisation=AffineTransform(
                np.asarray(payload["denormalisation"], dtype=np.float64)
            ),
            regularisation=float(payload["regularisation"]),
        )
    if family == "composed":
        steps = tuple(deserialise_transform(step) for step in payload["steps"])
        if any(step is None for step in steps):
            raise ValueError("composed transform contains an empty step")
        return ComposedTransform(steps)  # type: ignore[arg-type]
    raise ValueError(f"unknown cached transform family {family!r}")


@dataclass(frozen=True)
class RegistrationJob:
    pair: PairInput
    dataset_version: str
    pipeline_version: str
    seed: int
    fold: int = -1
    is_development: bool = False
    sampling_probability: float = float("nan")
    code_hash: str = ""
    env_hash: str = ""
    checkpoint_hash: str = ""
    config_hash: str = ""
    input_corrections: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def job_id(self) -> str:
        return make_job_id(
            dataset_version=self.dataset_version,
            pair_id=self.pair.pair_id,
            pipeline_version=self.pipeline_version,
            direction=self.pair.direction,
            condition=self.pair.condition,
            severity=self.pair.severity,
            seed=self.seed,
        )


class RegistrationRunner:
    """Execute one registrar with an append-only ledger and sharded cache."""

    def __init__(
        self,
        registrar: Registrar,
        table: ShardedTable,
        ledger: StatusLedger,
        *,
        max_infrastructure_attempts: int = 3,
    ) -> None:
        if max_infrastructure_attempts < 1:
            raise ValueError("max_infrastructure_attempts must be positive")
        self.registrar = registrar
        self.table = table
        self.ledger = ledger
        self.max_infrastructure_attempts = max_infrastructure_attempts
        self._latest = ledger.latest()
        cached = self.table.load()
        self._cached_by_job = (
            {}
            if cached.empty or "job_id" not in cached
            else {str(row["job_id"]): row for _, row in cached.iterrows()}
        )

    def _cache_contract_matches(self, job: RegistrationJob) -> bool:
        row = self._cached_by_job.get(job.job_id)
        if row is None:
            return False
        expected = {
            "dataset_version": job.dataset_version,
            "pipeline_version": job.pipeline_version,
            "code_hash": job.code_hash,
            "env_hash": job.env_hash,
            "checkpoint_hash": job.checkpoint_hash,
            "config_hash": job.config_hash,
        }
        return all(str(row.get(key, "")) == str(value) for key, value in expected.items())

    def _failed_result(self, status: RegistrationStatus, exc: BaseException) -> RegistrationResult:
        return RegistrationResult(
            pipeline_id=self.registrar.pipeline_id,
            status=status,
            forward_moving_to_fixed=None,
            diagnostics={"exception_type": type(exc).__name__, "message": str(exc)},
        )

    def _row(self, job: RegistrationJob, result: RegistrationResult) -> dict[str, Any]:
        pair = job.pair
        coords = pair.coordinates
        params = serialise_transform(result.forward_moving_to_fixed)
        diagnostic_code = str(
            result.diagnostics.get("code") or result.diagnostics.get("reason") or ""
        )
        return {
            "job_id": job.job_id,
            "dataset_id": pair.dataset_id,
            "dataset_version": job.dataset_version,
            "pair_id": pair.pair_id,
            "moving_image_id": pair.moving_image_id,
            "fixed_image_id": pair.fixed_image_id,
            "group_id": pair.group_id,
            "group_basis": pair.group_basis,
            "fold": job.fold,
            "is_development": job.is_development,
            "sampling_probability": job.sampling_probability,
            "pipeline_id": result.pipeline_id,
            "pipeline_version": job.pipeline_version,
            "direction": pair.direction,
            "condition": pair.condition,
            "severity": pair.severity,
            "seed": job.seed,
            "original_moving_hw": [coords.moving.original_height, coords.moving.original_width],
            "original_fixed_hw": [coords.fixed.original_height, coords.fixed.original_width],
            "working_moving_hw": [coords.moving.working_height, coords.moving.working_width],
            "working_fixed_hw": [coords.fixed.working_height, coords.fixed.working_width],
            "A_moving": coords.A_m.tolist(),
            "A_fixed": coords.A_f.tolist(),
            "valid_support_convention": "fixed content intersect backward-warped moving content",
            "transform_family": None if params is None else params["family"],
            "transform_params": None if params is None else canonical_json(params),
            "transform_checksum": "" if params is None else short_hash(params),
            "matches_moving": _array(result.matches_moving),
            "matches_fixed": _array(result.matches_fixed),
            "inlier_mask": _array(result.inlier_mask),
            "match_scores": _array(result.match_scores),
            "status": result.status.value,
            "diagnostic_code": diagnostic_code,
            "diagnostics": canonical_json(_diagnostic_values(result.diagnostics)),
            "explicit_failure": result.is_explicit_failure,
            "n_matches": result.n_matches,
            "n_inliers": result.n_inliers,
            "overlap_fraction": result.diagnostics.get("overlap_fraction", np.nan),
            "runtime_s": result.runtime_s,
            "cpu_time_s": result.cpu_time_s,
            "peak_vram_bytes": result.peak_vram_bytes,
            "code_hash": job.code_hash,
            "env_hash": job.env_hash,
            "checkpoint_hash": job.checkpoint_hash,
            "config_hash": job.config_hash,
            "hardware": f"{platform.system()} {platform.machine()}",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def run_one(self, job: RegistrationJob) -> RegistrationResult | None:
        attempt = self._next_attempt(job)
        if attempt is None:
            return None
        self._record(JobRecord(job.job_id, "register", "running", attempt=attempt))
        results = _execute_registration((self.registrar, job, attempt, attempt))
        return self._publish(job, results)

    def _record(self, record: JobRecord) -> None:
        self.ledger.append(record)
        self._latest[record.job_id] = record

    def _next_attempt(self, job: RegistrationJob) -> int | None:
        latest = self._latest.get(job.job_id)
        cache_is_current = self._cache_contract_matches(job)
        if latest is not None and latest.state == "done" and cache_is_current:
            return None
        stale_completed = latest is not None and latest.state == "done" and not cache_is_current
        attempt = 1 if latest is None or stale_completed else latest.attempt + 1
        if not stale_completed and latest is not None and attempt > self.max_infrastructure_attempts:
            return None

        return attempt

    def _publish(self, job, results):
        for attempt, started, finished, result in results:
            row = self._row(job, result)
            self.table.append([row], shard_hint=job.job_id)
            self._cached_by_job[job.job_id] = row
            state = "failed" if result.status.is_infrastructure else "done"
            self._record(JobRecord(
                job.job_id,
                "register",
                state,
                status=result.status.value,
                attempt=attempt,
                message=str(
                    result.diagnostics.get("message") or result.diagnostics.get("reason") or ""
                ),
                runtime_s=result.runtime_s,
                started_at=started,
                finished_at=finished,
            ))
        return result

    def run_parallel(self, jobs, *, workers=1, worker_threads=1):
        """Isolate estimators; only this parent publishes cache and ledger rows.

        Yield in manifest order. Duplicate jobs are rejected before submission,
        and technical retries retain the original seed and fixed attempt cap.
        """
        seen = set()
        scheduled = deque()

        def tasks():
            for job in jobs:
                if job.job_id in seen:
                    raise ValueError(f"duplicate registration job {job.job_id}")
                seen.add(job.job_id)
                attempt = self._next_attempt(job)
                if attempt is None:
                    continue
                self._record(JobRecord(job.job_id, "register", "running", attempt=attempt))
                scheduled.append(job)
                yield self.registrar, job, attempt, self.max_infrastructure_attempts

        for results in ordered_map(_execute_registration, tasks(), workers=workers,
                                   threads=worker_threads):
            job = scheduled.popleft()
            yield job, self._publish(job, results)

    def run(self, jobs: list[RegistrationJob]) -> dict[str, int]:
        for job in jobs:
            self.run_one(job)
        return self.ledger.summary()

    def incomplete_jobs(self, jobs: list[RegistrationJob]) -> list[RegistrationJob]:
        """Requested jobs without a completed, provenance-matching result."""
        latest = self._latest
        return [
            job for job in jobs
            if job.job_id not in latest
            or latest[job.job_id].state != "done"
            or not self._cache_contract_matches(job)
        ]


def _execute_registration(task):
    registrar, job, first_attempt, max_attempts = task
    if job.input_corrections is not None:
        from ..evaluation.perturbations import CoordinateCorrectingRegistrar
        registrar = CoordinateCorrectingRegistrar(
            registrar, {(job.pair.pair_id, job.pair.condition): job.input_corrections}
        )
    results = []
    for attempt in range(first_attempt, max_attempts + 1):
        started = time.time()
        try:
            result = registrar.register(job.pair, job.seed)
        except Exception as exc:
            status = (RegistrationStatus.TIMEOUT if isinstance(exc, TimeoutError) else
                      RegistrationStatus.OOM if isinstance(exc, MemoryError) else
                      RegistrationStatus.INFRASTRUCTURE_ERROR)
            result = RegistrationResult(
                registrar.pipeline_id, status, None,
                diagnostics={"exception_type": type(exc).__name__, "message": str(exc)},
            )
        finished = time.time()
        if not np.isfinite(result.runtime_s):
            result.runtime_s = finished - started
        results.append((attempt, started, finished, result))
        if not result.status.is_infrastructure:
            break
    return results
