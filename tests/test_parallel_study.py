"""Scientific equivalence, failure policy and E2 pair isolation under spawn."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from warpaudit.cache.ledger import StatusLedger
from warpaudit.cache.store import ShardedTable
from warpaudit.cli import CommandError, _run_with_progress
from warpaudit.evaluation.perturbations import CoordinateCorrectingRegistrar
from warpaudit.geometry.transforms import HomographyTransform
from warpaudit.parallel import ordered_map
from warpaudit.registration.runner import RegistrationJob, RegistrationRunner
from warpaudit.types import RegistrationResult, RegistrationStatus


class SeededRegistrar:
    pipeline_id = "fixture"

    def register(self, pair, seed):
        matrix = np.eye(3)
        matrix[:2, 2] = np.random.default_rng(seed).normal(size=2)
        return RegistrationResult(self.pipeline_id, RegistrationStatus.OK,
                                  HomographyTransform(matrix))


class BrokenRegistrar:
    pipeline_id = "fixture"

    def register(self, pair, seed):
        raise RuntimeError("technical failure")


class RecoveringRegistrar:
    pipeline_id = "fixture"

    def __init__(self):
        self.calls = {}

    def register(self, pair, seed):
        self.calls[pair.pair_id] = self.calls.get(pair.pair_id, 0) + 1
        if self.calls[pair.pair_id] == 1:
            raise TimeoutError("temporary transport failure")
        return RegistrationResult(self.pipeline_id, RegistrationStatus.NO_MATCHES, None,
                                  diagnostics={"seed": seed})


def _runner(root, registrar):
    return RegistrationRunner(registrar, ShardedTable(root, "registrations"),
                              StatusLedger(root / "status.jsonl"))


@pytest.mark.parametrize("perturbed", [False, True])
def test_registration_spawn_matches_serial_and_resumes(tmp_path, pair_input, perturbed):
    jobs = [RegistrationJob(replace(pair_input, pair_id=f"pair-{i}"), "v1", "v1", i,
                            input_corrections=(np.eye(3) * (i + 1), np.eye(3))
                            if perturbed else None)
            for i in range(5)]
    serial = _runner(tmp_path / "serial", SeededRegistrar())
    parallel = _runner(tmp_path / "parallel", SeededRegistrar())
    _run_with_progress(serial, jobs, label="serial")
    _run_with_progress(parallel, jobs, label="parallel", workers=2)
    columns = ["job_id", "seed", "status", "transform_params", "transform_checksum",
               "group_id", "fold", "is_development", "config_hash", "diagnostics"]
    pd.testing.assert_frame_equal(serial.table.load()[columns], parallel.table.load()[columns])
    before = parallel.ledger.path.read_bytes()
    _run_with_progress(_runner(tmp_path / "parallel", SeededRegistrar()), jobs,
                       label="resume", workers=2)
    assert parallel.ledger.path.read_bytes() == before


@pytest.mark.parametrize("registrar,recover", [(BrokenRegistrar, False),
                                               (RecoveringRegistrar, True)])
def test_parallel_retry_cap_and_no_output(tmp_path, pair_input, registrar, recover):
    runner = _runner(tmp_path, registrar())
    jobs = [RegistrationJob(replace(pair_input, pair_id=f"pair-{i}"), "v1", "v1", 4)
            for i in range(3)]
    if recover:
        _run_with_progress(runner, jobs, label="retry", workers=2)
        assert all(record.attempt == 2 and record.state == "done"
                   for record in runner.ledger.latest().values())
    else:
        with pytest.raises(CommandError, match="downstream stages blocked"):
            _run_with_progress(runner, jobs, label="retry", workers=2)
        assert all(record.attempt == 3 and record.state == "failed"
                   for record in runner.ledger.latest().values())
    before = runner.ledger.path.read_bytes()
    if recover:
        _run_with_progress(runner, jobs, label="resume", workers=2)
    else:
        with pytest.raises(CommandError, match="downstream stages blocked"):
            _run_with_progress(runner, jobs, label="resume", workers=2)
    assert runner.ledger.path.read_bytes() == before


def test_e2_corrections_are_pair_specific_even_for_same_draw_number(pair_input):
    from warpaudit.signals.family_e2_perturbation import MOVING_CORRECTION_KEY

    first, second = np.eye(3), np.eye(3)
    first[0, 2], second[0, 2] = 2, -7
    registrar = CoordinateCorrectingRegistrar(SeededRegistrar(), {
        ("first", "e2:translation:000"): (first, np.eye(3)),
        ("second", "e2:translation:000"): (second, np.eye(3)),
    })
    for name, matrix in (("first", first), ("second", second)):
        pair = replace(pair_input, pair_id=name, condition="e2:translation:000")
        result = registrar.register(pair, 17)
        np.testing.assert_array_equal(result.diagnostics[MOVING_CORRECTION_KEY], matrix)
    with pytest.raises(ValueError, match="no E2 correction"):
        registrar.register(replace(pair_input, condition="e2:translation:000"), 17)


def test_ordered_map_propagates_worker_errors():
    with pytest.raises(ValueError):
        list(ordered_map(int, ["1", "invalid", "3"], workers=2))


def test_full_evaluation_parallel_equals_serial():
    from test_full_study_v3 import _cases

    from warpaudit.evaluation.full_study import evaluate_full_study

    cases, names = _cases()
    cases = pd.concat([cases, cases.assign(pipeline_id="second")], ignore_index=True)
    options = dict(development_datasets=("DEV",), external_datasets=("EXT",),
                   confirmatory_datasets=("EXT",), pipelines=("pipe", "second"),
                   baseline_families=("A",), augmented_families=("A", "E1", "E2"),
                   calibration_method="platt", nominal_acceptance=0.7,
                   high_confidence_cutoff=0.2, min_class_bearing_groups=2,
                   min_brier_improvement=0.01, bootstrap_resamples=50,
                   C_values=(0.1, 1.0), seed=17)
    serial = evaluate_full_study(cases, names, **options)
    parallel = evaluate_full_study(cases, names, workers=2, **options)
    assert serial == parallel
