from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pytest

from warpaudit import cli
from warpaudit.config import load_config
from warpaudit.study import build_plan

CONFIG = "configs/pilot.yaml"


def test_registration_changes_invalidate_features_but_reports_do_not(tmp_path) -> None:
    registration = tmp_path / "warpaudit" / "registration" / "adapter.py"
    registration.parent.mkdir(parents=True)
    registration.write_text("version = 1\n", encoding="utf-8")
    first = cli._feature_code_identity(tmp_path)
    (tmp_path / "report.md").write_text("report edit", encoding="utf-8")
    assert cli._feature_code_identity(tmp_path) == first
    registration.write_text("version = 2\n", encoding="utf-8")
    assert cli._feature_code_identity(tmp_path) != first


def test_unrelated_cli_commands_do_not_invalidate_features(tmp_path) -> None:
    cli_path = tmp_path / "warpaudit" / "cli.py"
    cli_path.parent.mkdir(parents=True)
    relevant = """
def _pair_from_row(row):
    return row

def _prepare_e2_pair(task):
    return task
"""
    cli_path.write_text(relevant, encoding="utf-8")
    first = cli._feature_code_identity(tmp_path)
    cli_path.write_text(relevant + "\ndef unrelated_report():\n    return 1\n", encoding="utf-8")
    assert cli._feature_code_identity(tmp_path) == first
    cli_path.write_text(
        relevant.replace("return task", "return tuple(task)"), encoding="utf-8"
    )
    assert cli._feature_code_identity(tmp_path) != first


def _plan(**kwargs):
    return build_plan(load_config(Path(CONFIG)), config_path=CONFIG, **kwargs)


def test_parallel_feature_workers_only_change_execution_options():
    serial, parallel = _plan(), _plan(feature_workers=4)
    assert serial.ids() == parallel.ids()
    for left, right in zip(serial, parallel, strict=True):
        for original, configured in zip(left.commands, right.commands, strict=True):
            if original[0] == "features":
                assert configured == (*original, "--workers", "4", "--flush-every", "1")
            else:
                assert configured == original


def test_isolated_plan_places_probe_and_environment_outputs_in_run_directory():
    cfg = load_config(Path(CONFIG))
    cfg = replace(cfg, paths=replace(cfg.paths, reports="study_outputs/run1/reports"))
    plan = build_plan(cfg, config_path=CONFIG, feature_workers=32, feature_worker_threads=1)
    for stage_id in ("environment", "probe-adapters"):
        for command in plan.get(stage_id).commands:
            if "--output" in command:
                assert command[command.index("--output") + 1].startswith("study_outputs/run1/reports/")
    features = plan.get("features-development").commands[0]
    assert features[-2:] == ("--worker-threads", "1")


def test_plan_orders_cheap_verification_before_expensive_acquisition() -> None:
    """A wrong matcher environment must fail in seconds, not after a 3 GB download."""
    ids = _plan().ids()
    assert ids.index("probe-adapters") < ids.index("prepare-data")
    assert ids.index("audit-data") < ids.index("register-development")
    assert ids.index("diagnose-development") < ids.index("features-confirmatory")
    for stage in _plan():
        for requirement in stage.requires:
            assert ids.index(requirement) < ids.index(stage.id)


def test_confirmatory_sweeps_stay_ground_truth_free_and_exclude_frozen_e1() -> None:
    plan = _plan()
    confirmatory = [
        argv for stage in plan for argv in stage.commands if "confirmatory" in argv
    ]
    assert confirmatory
    # Label access is the boundary: no stage may score the reserve.
    assert not any(argv[0] == "labels" for argv in confirmatory)
    features = [argv for argv in confirmatory if argv[0] == "features"]
    assert features and all("E1" not in argv for argv in features)
    # E1 is still recomputed on development, where the degeneracy gate is decided.
    development_features = [
        argv
        for stage in plan
        for argv in stage.commands
        if argv[0] == "features" and "development" in argv
    ]
    assert development_features and all("E1" in argv for argv in development_features)

    assert not any(
        "confirmatory" in argv
        for stage in _plan(include_confirmatory=False)
        for argv in stage.commands
    )


def test_stage_selection_rejects_unknown_ids_and_keeps_declared_order() -> None:
    plan = _plan()
    narrowed = plan.select(start="audit-data", skip=("e2",))
    assert narrowed.ids()[0] == "audit-data"
    assert "e2" not in narrowed.ids()
    assert "environment" not in narrowed.ids()
    assert list(narrowed.ids()) == [i for i in plan.ids() if i in set(narrowed.ids())]
    with pytest.raises(ValueError, match="unknown stage id"):
        plan.select(only=("no-such-stage",))


def test_failed_stage_skips_only_its_dependents_and_reports_a_failing_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "configs" / "pilot.yaml"
    config.parent.mkdir()
    config.write_text(
        """
project: fixture
datasets:
  - {id: FIRE, version: fixture, root: data/FIRE}
pipelines:
  - {id: fixture, version: fixture-1, matcher: fixture, checkpoint: local}
  - {id: fixture2, version: fixture-2, matcher: fixture2, checkpoint: local}
""",
        encoding="utf-8",
    )
    attempted: list[tuple[str, ...]] = []

    def fake_run(argv, *, root, log_path, echo):
        attempted.append(argv)
        log_path.write_text("fake\n", encoding="utf-8")
        return 1 if argv[0] == "register" else 0

    monkeypatch.setattr(cli, "_run_stage_command", fake_run)
    args = argparse.Namespace(
        config=str(config),
        download=False,
        acknowledge_fire_terms_unresolved=False,
        development_only=True,
        e2_limit=1,
        e2_sample_cap=0,
        signed_off_by="",
        review_note="",
        acknowledge_infeasible=False,
        refit_bootstrap=0,
        only=[],
        skip=["environment", "probe-adapters", "prepare-data", "audit-data"],
        start_at="",
        keep_going=True,
        quiet=True,
        dry_run=False,
        min_free_gb=0.0,
    )
    assert cli.command_run_study(args) == cli.EXIT_STAGE_FAILED

    started = {argv[0] for argv in attempted}
    assert "register" in started
    # Everything downstream of the failed registrations must be skipped, and
    # nothing may score labels that were never produced.
    assert "labels" not in started
    assert "features" not in started
    # An independent stage still runs under --keep-going.
    assert "plan-study" not in started  # it depends on labels-development
    summary = (tmp_path / "reports" / "STUDY_RUN.md").read_text(encoding="utf-8")
    assert "register-development" in summary
    assert "unmet prerequisite" in summary

def test_confirmatory_outcome_stages_appear_only_with_a_reviewed_sign_off() -> None:
    """G1 is a reviewed gate, so an unattended run cannot walk through it."""
    unsigned = _plan().ids()
    assert "freeze" not in unsigned
    assert "labels-confirmatory" not in unsigned
    assert "evaluate" not in unsigned
    # The label-free confirmatory work is still planned: it is permitted before G1.
    assert "register-confirmatory" in unsigned
    assert "features-confirmatory" in unsigned

    signed = _plan(signed_off_by="reviewer", review_note="checked").ids()
    for stage in ("freeze", "labels-confirmatory", "evaluate"):
        assert stage in signed
    assert signed.index("freeze") < signed.index("labels-confirmatory")
    assert signed.index("labels-confirmatory") < signed.index("evaluate")


def test_the_reserve_sweep_omits_the_families_the_gate_can_remove() -> None:
    """E1 over the reserve is the run's largest avoidable cost."""
    plan = _plan()
    confirmatory = [
        argv
        for stage in plan
        for argv in stage.commands
        if argv[0] == "features" and "confirmatory" in argv
    ]
    assert confirmatory
    assert all("E1" not in argv and "E2" not in argv for argv in confirmatory)
    # Everything cheap the configuration asks for is still extracted.
    assert any("D" in argv and "G" in argv for argv in confirmatory)
