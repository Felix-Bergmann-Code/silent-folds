from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from warpaudit.cache.ledger import StatusLedger
from warpaudit.cache.schema import REGISTRATION_COLUMNS, validate_columns
from warpaudit.cache.store import ShardedTable
from warpaudit.config import PipelineConfig
from warpaudit.geometry.transforms import HomographyTransform
from warpaudit.registration.fitting import FittingPolicy, dlt_homography, fit_homography
from warpaudit.registration.runner import RegistrationJob, RegistrationRunner
from warpaudit.registration.subprocess_adapter import (
    AdapterProtocolError,
    SubprocessMatcherRegistrar,
)
from warpaudit.types import RegistrationResult, RegistrationStatus


class FixtureRegistrar:
    pipeline_id = "fixture"

    def __init__(self, src: np.ndarray, dst: np.ndarray) -> None:
        self.src, self.dst, self.calls = src, dst, 0

    def register(self, pair, seed):
        self.calls += 1
        fit = fit_homography(
            self.src, self.dst, FittingPolicy(max_iters=100, threshold_px=0.1), seed=seed
        )
        return RegistrationResult(
            self.pipeline_id, fit.status, fit.transform, self.src, self.dst, fit.inlier_mask
        )


def test_seeded_homography_fit_and_resumable_runner(tmp_path, pair_input, correspondences) -> None:
    src, dst = correspondences
    fit = fit_homography(src, dst, FittingPolicy(max_iters=100), seed=4)
    assert fit.ok
    np.testing.assert_allclose(fit.transform.apply(src), dst, atol=1e-9)

    registrar = FixtureRegistrar(src, dst)
    table = ShardedTable(tmp_path / "cache", "registrations")
    runner = RegistrationRunner(registrar, table, StatusLedger(tmp_path / "ledger.jsonl"))
    job = RegistrationJob(pair_input, "fixture-v1", "pipeline-v1", 4)
    runner.run([job, job])
    loaded = table.load()
    assert registrar.calls == 1 and len(loaded) == 1
    assert validate_columns("registrations", loaded.columns) == []
    assert set(REGISTRATION_COLUMNS) <= set(loaded.columns)


def test_runner_recomputes_completed_job_when_cache_contract_changes(
    tmp_path, pair_input, correspondences
) -> None:
    src, dst = correspondences
    registrar = FixtureRegistrar(src, dst)
    table = ShardedTable(tmp_path / "cache", "registrations")
    ledger = StatusLedger(tmp_path / "ledger.jsonl")

    first = RegistrationJob(
        pair_input,
        "fixture-v1",
        "pipeline-v1",
        4,
        code_hash="code-v1",
        env_hash="env-v1",
        checkpoint_hash="weights-v1",
        config_hash="config-v1",
    )
    RegistrationRunner(registrar, table, ledger).run([first])
    updated = RegistrationJob(
        pair_input,
        "fixture-v1",
        "pipeline-v1",
        4,
        code_hash="code-v2",
        env_hash="env-v1",
        checkpoint_hash="weights-v1",
        config_hash="config-v1",
    )
    RegistrationRunner(registrar, table, ledger).run([updated])

    loaded = table.load()
    assert registrar.calls == 2
    assert len(loaded) == 1
    assert loaded.iloc[0]["code_hash"] == "code-v2"


def test_rank_deficient_fit_is_explicit() -> None:
    src = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], float)
    result = fit_homography(src, src)
    assert result.status is RegistrationStatus.DEGENERATE_FIT
    assert result.transform is None


def test_empty_matches_survive_cache_roundtrip(tmp_path, pair_input):
    from warpaudit.cli import _result_from_row

    class EmptyRegistrar:
        pipeline_id = "fixture"

        def register(self, pair, seed):
            return RegistrationResult(
                self.pipeline_id, RegistrationStatus.NO_MATCHES, None,
                np.empty((0, 2)), np.empty((0, 2)),
                match_scores=np.empty(0),
            )

    table = ShardedTable(tmp_path, "registrations")
    runner = RegistrationRunner(EmptyRegistrar(), table, StatusLedger(tmp_path / "ledger.jsonl"))
    runner.run_one(RegistrationJob(pair_input, "v1", "p1", 4))
    restored = _result_from_row(table.load().iloc[0])
    assert restored.status is RegistrationStatus.NO_MATCHES
    assert restored.matches_moving.shape == restored.matches_fixed.shape == (0, 2)
    assert restored.match_scores.shape == (0,)
    assert restored.is_explicit_failure and restored.forward_moving_to_fixed is None


@pytest.mark.parametrize("status", [RegistrationStatus.NO_MATCHES,
                                   RegistrationStatus.DEGENERATE_FIT])
def test_undefined_fit_diagnostics_are_saved_and_not_retried(
    tmp_path, pair_input, status
) -> None:
    class NoOutput:
        pipeline_id = "fixture"
        calls = 0

        def register(self, pair, seed):
            self.calls += 1
            return RegistrationResult(
                self.pipeline_id, status, None,
                diagnostics={"fit_design_condition": float("nan"),
                             "nested": [np.float64(np.inf), np.array([-np.inf, 2.0])],
                             "reason": "insufficient or degenerate correspondences"},
            )

    registrar = NoOutput()
    table = ShardedTable(tmp_path / "cache", "registrations")
    ledger = StatusLedger(tmp_path / "ledger.jsonl")
    job = RegistrationJob(pair_input, "v1", "p1", 4)
    runner = RegistrationRunner(registrar, table, ledger)
    runner.run([job, job])
    row = table.load().iloc[0]
    assert registrar.calls == 1
    assert row.status == status.value and row.explicit_failure
    assert row.transform_params is None
    diagnostics = json.loads(row.diagnostics)
    assert diagnostics["fit_design_condition"] is None
    assert diagnostics["nested"] == [None, [None, 2.0]]
    assert ledger.latest()[job.job_id].state == "done"
    assert not runner.incomplete_jobs([job])
    assert RegistrationRunner(registrar, table, ledger).run_one(job) is None


@pytest.mark.parametrize("recover", [True, False])
def test_progress_retries_technical_failures_and_blocks_exhausted_jobs(
    tmp_path, pair_input, recover
) -> None:
    from warpaudit.cli import CommandError, _run_with_progress

    class Flaky:
        pipeline_id = "fixture"
        calls = 0
        seeds = []

        def register(self, pair, seed):
            self.calls += 1
            self.seeds.append(seed)
            if self.calls == 1 or not recover:
                raise RuntimeError("worker process crashed")
            return RegistrationResult(self.pipeline_id, RegistrationStatus.NO_MATCHES, None)

    registrar = Flaky()
    runner = RegistrationRunner(registrar, ShardedTable(tmp_path, "registrations"),
                                StatusLedger(tmp_path / "ledger.jsonl"))
    job = RegistrationJob(pair_input, "v1", "p1", 4)
    if recover:
        _run_with_progress(runner, [job], label="fixture")
        assert registrar.calls == 2
        assert not runner.incomplete_jobs([job])
    else:
        with pytest.raises(CommandError, match="downstream stages blocked"):
            _run_with_progress(runner, [job], label="fixture")
        assert registrar.calls == 3
        with pytest.raises(CommandError, match="downstream stages blocked"):
            _run_with_progress(runner, [job], label="fixture")
        assert registrar.calls == 3
    assert set(registrar.seeds) == {4}


def test_minimal_dlt_uses_one_full_svd_and_retains_null_vector(monkeypatch) -> None:
    src = np.array([[0, 0], [10, 0], [10, 8], [0, 8]], dtype=float)
    dst = src + np.array([3.0, -2.0])
    original_svd = np.linalg.svd
    calls: list[tuple[int, int]] = []

    def counted_svd(array, *args, **kwargs):
        calls.append(array.shape)
        return original_svd(array, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", counted_svd)
    matrix, _ = dlt_homography(src, dst)

    assert calls == [(8, 9)]
    assert matrix is not None
    np.testing.assert_allclose(
        HomographyTransform(matrix).apply(src),
        dst,
        atol=1e-10,
    )


def _subprocess_pipeline(**options) -> PipelineConfig:
    worker = Path(__file__).parent / "fixtures" / "matcher_worker.py"
    return PipelineConfig(
        id="fixture",
        version="fixture-v1",
        matcher="fixture",
        checkpoint="none",
        adapter="subprocess",
        command=(sys.executable, str(worker)),
        max_iters=100,
        ransac_threshold_px=0.1,
        options=options,
        provenance={"fixture_commit": "fixed"},
    )


def test_isolated_matcher_protocol_fits_shared_homography(pair_input) -> None:
    Image.fromarray(np.zeros((100, 120, 3), dtype=np.uint8)).save(pair_input.moving_path)
    Image.fromarray(np.zeros((100, 120, 3), dtype=np.uint8)).save(pair_input.fixed_path)
    registrar = SubprocessMatcherRegistrar(
        _subprocess_pipeline(offset=[3.0, -2.0]), project_root=Path(__file__).parents[1]
    )

    result = registrar.register(pair_input, seed=7)

    assert result.status is RegistrationStatus.OK
    assert result.n_matches == result.n_inliers == 8
    np.testing.assert_allclose(
        result.forward_moving_to_fixed.apply(result.matches_moving),
        result.matches_fixed,
        atol=1e-9,
    )
    assert result.diagnostics["adapter_protocol"] == "1.0"
    assert result.diagnostics["matcher_diagnostics"]["moving_shape"] == [100, 120, 3]


def test_isolated_matcher_rejects_ambiguous_coordinate_frame(pair_input) -> None:
    Image.fromarray(np.zeros((100, 120, 3), dtype=np.uint8)).save(pair_input.moving_path)
    Image.fromarray(np.zeros((100, 120, 3), dtype=np.uint8)).save(pair_input.fixed_path)
    registrar = SubprocessMatcherRegistrar(
        _subprocess_pipeline(coordinate_frame="original"),
        project_root=Path(__file__).parents[1],
    )
    with pytest.raises(AdapterProtocolError, match="working_pixel_centres"):
        registrar.register(pair_input, seed=7)


def test_isolated_python_symlink_is_not_resolved_out_of_environment(tmp_path) -> None:
    environment_python = tmp_path / "pipeline-env" / "bin" / "python"
    environment_python.parent.mkdir(parents=True)
    try:
        environment_python.symlink_to(sys.executable)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows account lacks symlink creation privilege")
        raise
    pipeline = _subprocess_pipeline()
    pipeline.command = ("pipeline-env/bin/python", "worker.py")

    registrar = SubprocessMatcherRegistrar(pipeline, project_root=tmp_path)

    assert registrar._command()[0] == str(environment_python)


def test_provenance_inspection_reports_without_ever_relaxing_the_gate(tmp_path) -> None:
    """The escape hatch must report a mismatch, never let one produce a row."""
    from warpaudit.config import PipelineConfig
    from warpaudit.registration.subprocess_adapter import (
        AdapterProtocolError,
        SubprocessMatcherRegistrar,
    )

    pipeline = PipelineConfig(
        id="fx",
        version="v1",
        matcher="fx",
        adapter="subprocess",
        command=("python", "worker.py"),
        provenance={"torch_version": "2.3.1"},
    )
    payload = {
        "protocol_version": "1.0",
        "pipeline_id": "fx",
        "coordinate_frame": "working_pixel_centres",
        "matches_moving": [[1.0, 1.0], [2.0, 2.0]],
        "matches_fixed": [[1.0, 1.0], [2.0, 2.0]],
        "provenance": {"torch_version": "2.3.1+cu121"},
    }

    strict = SubprocessMatcherRegistrar(pipeline, project_root=tmp_path)
    with pytest.raises(AdapterProtocolError, match="does not match config"):
        strict._validate_response(dict(payload))

    # Reporting mode returns the observed values instead of raising...
    relaxed = SubprocessMatcherRegistrar(
        pipeline, project_root=tmp_path, strict_provenance=False
    )
    *_, observed = relaxed._validate_response(dict(payload))
    assert observed["torch_version"] == "2.3.1+cu121"
    # ...and strictness is the default, so nothing that writes a cached row
    # can acquire the relaxed behaviour by omission.
    assert SubprocessMatcherRegistrar(pipeline, project_root=tmp_path).strict_provenance


def test_a_posix_venv_path_resolves_to_the_windows_layout(tmp_path) -> None:
    """One configuration must describe the same isolated environment on both."""
    from warpaudit.registration.subprocess_adapter import _venv_interpreter

    posix = tmp_path / "env" / "bin" / "python"
    posix.parent.mkdir(parents=True)
    posix.write_text("", encoding="utf-8")
    assert _venv_interpreter(posix) == posix

    windows_env = tmp_path / "winenv"
    (windows_env / "Scripts").mkdir(parents=True)
    (windows_env / "Scripts" / "python.exe").write_text("", encoding="utf-8")
    declared = windows_env / "bin" / "python"
    assert _venv_interpreter(declared) == windows_env / "Scripts" / "python.exe"

    # An environment that exists under neither layout is reported as declared,
    # so the error names what the configuration actually asked for.
    absent = tmp_path / "missing" / "bin" / "python"
    assert _venv_interpreter(absent) == absent
