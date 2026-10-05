"""Isolated matcher-process adapter and versioned JSON exchange protocol.

Only correspondence extraction runs in the pipeline environment. Transform
fitting remains in :mod:`warpaudit.registration.fitting`, which guarantees an
addressable seeded estimator and supports the controlled homography/TPS block.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..config import PipelineConfig
from ..geometry.resample import resample_to_frame
from ..types import PairInput, RegistrationResult
from .fitting import FittingPolicy, fit_correspondences

PROTOCOL_VERSION = "1.0"
MAX_RESPONSE_BYTES = 64 * 1024 * 1024


class AdapterProtocolError(RuntimeError):
    """An isolated matcher returned an invalid or scientifically ambiguous payload."""


def _frame_payload(pair: PairInput, which: str) -> dict[str, Any]:
    frame = getattr(pair.coordinates, which)
    path = getattr(pair, f"{which}_path")
    return {
        "image_id": frame.image_id,
        "path": str(path.resolve()),
        "original_hw": [frame.original_height, frame.original_width],
        "working_hw": [frame.working_height, frame.working_width],
        "to_working": frame.to_working.tolist(),
        "padding": [frame.pad_left, frame.pad_top, frame.pad_right, frame.pad_bottom],
        "resample_note": frame.resample_note,
    }


def _as_points(payload: Any, name: str) -> np.ndarray:
    points = np.asarray(payload, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise AdapterProtocolError(f"{name} must have shape (N, 2), got {points.shape}")
    if not np.isfinite(points).all():
        raise AdapterProtocolError(f"{name} contains non-finite coordinates")
    return points


def _clean_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AdapterProtocolError("diagnostics must be a JSON object")
    return json.loads(json.dumps(value, allow_nan=False))


def _venv_interpreter(declared: Path) -> Path:
    """Map a declared POSIX venv interpreter onto this platform's layout.

    Virtual environments put the interpreter at ``bin/python`` on POSIX and
    ``Scripts/python.exe`` on Windows. One configuration should describe the
    same isolated environment on both, so the declared POSIX path is translated
    rather than duplicated per platform. Only the layout changes: the venv named
    by the configuration is still the one that runs, and the declared command --
    not the resolved path -- remains what ``env_hash`` records, so this cannot
    move a cached row between environments.
    """
    if declared.exists():
        return declared
    if declared.parent.name == "bin":
        candidate = declared.parent.parent / "Scripts" / f"{declared.name}.exe"
        if candidate.exists():
            return candidate
    return declared


class SubprocessMatcherRegistrar:
    """Run a matcher in an isolated interpreter and fit the configured transform.

    The command is executed without a shell and receives two appended options:
    ``--request <json> --response <json>``. The response must report working-
    frame pixel-centre coordinates. Ground truth is intentionally absent from
    the request schema.
    """

    def __init__(
        self,
        pipeline: PipelineConfig,
        *,
        project_root: Path | None = None,
        strict_provenance: bool = True,
    ) -> None:
        if not pipeline.command:
            raise ValueError(f"pipeline {pipeline.id!r} has no subprocess command")
        self.pipeline_id = pipeline.id
        self.pipeline = pipeline
        #: Only ``inspect-provenance`` relaxes this, to report what a machine
        #: actually resolved. Every path that writes a cached registration
        #: keeps it on, so a mismatched stack can never produce cached rows.
        self.strict_provenance = bool(strict_provenance)
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.policy = FittingPolicy(
            threshold_px=pipeline.ransac_threshold_px,
            max_iters=pipeline.max_iters,
            confidence=pipeline.confidence,
            min_matches=pipeline.min_matches,
            name=pipeline.fit,
        )

    def _command(self) -> list[str]:
        command = list(self.pipeline.command)
        executable = Path(command[0])
        if not executable.is_absolute() and ("/" in command[0] or "\\" in command[0]):
            # Do not call Path.resolve() here: venv Python launchers are often
            # symlinks, and dereferencing one silently escapes the isolated
            # environment by changing sys.prefix/site-packages.
            executable = self.project_root / executable
            command[0] = str(_venv_interpreter(executable))
        return command

    def _request(
        self, pair: PairInput, seed: int, *, moving_array: Path, fixed_array: Path
    ) -> dict[str, Any]:
        moving = _frame_payload(pair, "moving")
        fixed = _frame_payload(pair, "fixed")
        moving["working_array_path"] = str(moving_array)
        fixed["working_array_path"] = str(fixed_array)
        return {
            "protocol_version": PROTOCOL_VERSION,
            "pipeline_id": self.pipeline.id,
            "pipeline_version": self.pipeline.version,
            "matcher": self.pipeline.matcher,
            "seed": int(seed),
            "pixel_center_convention": pair.coordinates.pixel_center_convention,
            "moving": moving,
            "fixed": fixed,
            "options": self.pipeline.options,
            "expected_provenance": self.pipeline.provenance,
        }

    def _read_response(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise AdapterProtocolError("matcher exited without writing its response")
        size = path.stat().st_size
        if size > MAX_RESPONSE_BYTES:
            raise AdapterProtocolError(f"matcher response exceeds {MAX_RESPONSE_BYTES} bytes")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AdapterProtocolError(f"cannot read matcher response: {exc}") from exc
        if not isinstance(payload, dict):
            raise AdapterProtocolError("matcher response must be a JSON object")
        return payload

    def _validate_response(
        self, payload: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, str]]:
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise AdapterProtocolError(
                f"protocol version {payload.get('protocol_version')!r} does not match "
                f"{PROTOCOL_VERSION!r}"
            )
        if payload.get("pipeline_id") != self.pipeline.id:
            raise AdapterProtocolError(
                f"response pipeline {payload.get('pipeline_id')!r} does not match "
                f"{self.pipeline.id!r}"
            )
        if payload.get("coordinate_frame") != "working_pixel_centres":
            raise AdapterProtocolError(
                "matcher must return 'working_pixel_centres' coordinates explicitly"
            )

        moving = _as_points(payload.get("matches_moving", []), "matches_moving")
        fixed = _as_points(payload.get("matches_fixed", []), "matches_fixed")
        if moving.shape != fixed.shape:
            raise AdapterProtocolError("moving and fixed correspondence shapes differ")

        score_payload = payload.get("match_scores")
        scores = None if score_payload is None else np.asarray(score_payload, dtype=np.float64)
        if scores is not None:
            if scores.shape != (len(moving),):
                raise AdapterProtocolError(
                    f"match_scores must have shape ({len(moving)},), got {scores.shape}"
                )
            if not np.isfinite(scores).all():
                raise AdapterProtocolError("match_scores contains non-finite values")

        actual = payload.get("provenance")
        if not isinstance(actual, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in actual.items()
        ):
            raise AdapterProtocolError("provenance must be a string-to-string mapping")
        mismatches = {
            key: (expected, actual.get(key))
            for key, expected in self.pipeline.provenance.items()
            if actual.get(key) != expected
        }
        if mismatches and self.strict_provenance:
            raise AdapterProtocolError(f"matcher provenance does not match config: {mismatches}")
        return moving, fixed, scores, actual

    @staticmethod
    def _materialise_working_image(source: Path, frame: Any, destination: Path) -> None:
        if not source.is_file():
            raise FileNotFoundError(source)
        with Image.open(source) as image:
            original = np.asarray(image.convert("RGB"), dtype=np.float64)
        working = resample_to_frame(original, frame).astype(np.float32)
        np.save(destination, working, allow_pickle=False)

    def register(self, pair: PairInput, seed: int) -> RegistrationResult:
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="warpaudit-matcher-") as tmp:
            tmp_path = Path(tmp)
            moving_array = tmp_path / "moving.npy"
            fixed_array = tmp_path / "fixed.npy"
            self._materialise_working_image(
                pair.moving_path, pair.coordinates.moving, moving_array
            )
            self._materialise_working_image(pair.fixed_path, pair.coordinates.fixed, fixed_array)
            request = self._request(
                pair, seed, moving_array=moving_array, fixed_array=fixed_array
            )
            request_path = tmp_path / "request.json"
            response_path = tmp_path / "response.json"
            request_path.write_text(
                json.dumps(request, sort_keys=True, allow_nan=False), encoding="utf-8"
            )
            command = self._command() + [
                "--request",
                str(request_path),
                "--response",
                str(response_path),
            ]
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = str(seed)
            try:
                process = subprocess.run(
                    command,
                    cwd=self.project_root,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=self.pipeline.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    f"matcher exceeded {self.pipeline.timeout_s:g}s timeout"
                ) from exc
            if process.returncode != 0:
                stderr = process.stderr[-4000:].strip()
                raise RuntimeError(
                    f"matcher process exited {process.returncode}"
                    + (f": {stderr}" if stderr else "")
                )
            response = self._read_response(response_path)

        moving, fixed, scores, provenance = self._validate_response(response)
        fit = fit_correspondences(
            moving,
            fixed,
            self.policy,
            seed=seed,
            transform_family=self.pipeline.transform_family,
            tps_regularisation=float(self.pipeline.options.get("tps_regularisation", 1e-3)),
        )
        wall_time = time.perf_counter() - started
        diagnostics = {
            "adapter_protocol": PROTOCOL_VERSION,
            "matcher": self.pipeline.matcher,
            "matcher_diagnostics": _clean_diagnostics(response.get("diagnostics", {})),
            "matcher_runtime_s": float(response.get("runtime_s", float("nan"))),
            "provenance": provenance,
            "fit": fit.diagnostics,
            "fit_iterations": fit.n_iterations,
            "fit_design_condition": fit.design_condition,
        }
        return RegistrationResult(
            pipeline_id=self.pipeline.id,
            status=fit.status,
            forward_moving_to_fixed=fit.transform,
            matches_moving=moving,
            matches_fixed=fixed,
            inlier_mask=fit.inlier_mask,
            match_scores=scores,
            runtime_s=wall_time,
            cpu_time_s=float(response.get("cpu_time_s", float("nan"))),
            peak_vram_bytes=int(response.get("peak_vram_bytes", 0)),
            diagnostics=diagnostics,
        )


def subprocess_factory(
    pipeline: PipelineConfig,
    *,
    project_root: Path | None = None,
    strict_provenance: bool = True,
) -> SubprocessMatcherRegistrar:
    return SubprocessMatcherRegistrar(
        pipeline, project_root=project_root, strict_provenance=strict_provenance
    )
