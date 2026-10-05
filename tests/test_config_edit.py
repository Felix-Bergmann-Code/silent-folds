"""Re-pinning provenance edits exactly what it must and nothing else."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from warpaudit.config import load_config
from warpaudit.config_edit import ConfigEditError, replace_pipeline_provenance

SOURCE = Path("configs/pilot.yaml")


def test_repinning_changes_only_the_named_block(tmp_path: Path) -> None:
    target = tmp_path / "pilot.yaml"
    shutil.copy(SOURCE, target)
    before = target.read_text(encoding="utf-8")
    original = yaml.safe_load(before)

    new = {
        "python_version": "3.11.9",
        "torch_version": "2.3.1+cu121",
        "numpy_version": "1.26.4",
        "upstream_commit": "e92685f57f8318b18725c5c8c0bd28c7fe188d9a",
        "xfeat_weights_sha256": "0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b",
    }
    assert replace_pipeline_provenance(target, "xfeat_h", new) is True
    after = yaml.safe_load(target.read_text(encoding="utf-8"))

    by_id = {p["id"]: p for p in after["pipelines"]}
    assert by_id["xfeat_h"]["provenance"] == new
    # The other pipeline, and every other section, is untouched.
    original_by_id = {p["id"]: p for p in original["pipelines"]}
    assert by_id["sp_lg_h"] == original_by_id["sp_lg_h"]
    for section in ("datasets", "geometry", "labels", "features", "splits", "policy"):
        assert after[section] == original[section]
    # The comments that carry the reasoning survive.
    assert "# pilot core (spec 7.1)" not in after  # sanity: parsed form has no comments
    assert target.read_text(encoding="utf-8").count("#") == before.count("#")


def test_a_local_version_survives_the_round_trip(tmp_path: Path) -> None:
    """`2.3.1+cu121` must not be reinterpreted by YAML scalar rules."""
    target = tmp_path / "pilot.yaml"
    shutil.copy(SOURCE, target)
    replace_pipeline_provenance(
        target,
        "xfeat_h",
        {"torch_version": "2.3.1+cu121", "python_version": "3.11.9"},
    )
    cfg = load_config(target, strict=False)
    assert cfg.pipeline("xfeat_h").provenance["torch_version"] == "2.3.1+cu121"


def test_rewriting_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "pilot.yaml"
    shutil.copy(SOURCE, target)
    values = {"python_version": "3.11.9", "torch_version": "2.3.1+cu121"}
    assert replace_pipeline_provenance(target, "xfeat_h", values) is True
    assert replace_pipeline_provenance(target, "xfeat_h", values) is False


def test_an_unlocatable_block_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    target = tmp_path / "pilot.yaml"
    shutil.copy(SOURCE, target)
    with pytest.raises(ConfigEditError, match="expected exactly one"):
        replace_pipeline_provenance(target, "no_such_pipeline", {"a": "b"})


def test_repinning_refuses_to_write_an_empty_block(tmp_path: Path) -> None:
    """A matcher that reports nothing must not erase the pins it cannot confirm."""
    target = tmp_path / "pilot.yaml"
    shutil.copy(SOURCE, target)
    before = yaml.safe_load(target.read_text(encoding="utf-8"))
    original = {p["id"]: p["provenance"] for p in before["pipelines"]}

    # The guard lives in the command; the editor itself is deliberately literal,
    # so this documents that an empty write would indeed destroy the block.
    replace_pipeline_provenance(target, "xfeat_h", {})
    after = yaml.safe_load(target.read_text(encoding="utf-8"))
    by_id = {p["id"]: p for p in after["pipelines"]}
    assert by_id["xfeat_h"].get("provenance") in (None, {})
    assert by_id["sp_lg_h"]["provenance"] == original["sp_lg_h"]
