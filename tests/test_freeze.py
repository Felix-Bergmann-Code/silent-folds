"""G1 gating: confirmatory outcomes are unreadable until the protocol is frozen."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from warpaudit.protocols.freeze import (
    FREEZE_FILENAME,
    FreezeError,
    FreezeRecord,
    load_freeze,
    require_confirmatory_access,
)


def _record(**overrides) -> FreezeRecord:
    base = dict(
        generated_at="2026-09-08T00:00:00+00:00",
        config_hash="cfg0",
        git_commit="abc",
        registration_code_identity="reg0",
        feature_code_identity="feat0",
        primary_direction="X->Y",
        source_dataset="X",
        target_dataset="Y",
        n_folds=3,
        n_train_groups=11,
        n_calibration_groups=11,
        fold_hashes=("f0", "f1", "f2"),
        frozen_feature_families=("A", "B"),
        common_block_pipelines=("p1", "p2"),
        learner="logistic",
        nominal_acceptance=0.7,
        auroc_floor=0.7,
        gap_noninferiority_margin=0.05,
        delta_policy_margin=0.05,
        alpha_one_sided=0.05,
        signed_off_by="reviewer",
        review_note="checked",
    )
    base.update(overrides)
    return FreezeRecord(**base)


def _write(path: Path, record: FreezeRecord) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / FREEZE_FILENAME).write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )


def test_confirmatory_access_is_refused_without_a_freeze(tmp_path: Path) -> None:
    with pytest.raises(FreezeError, match="G1 gates"):
        require_confirmatory_access(tmp_path, config_hash="cfg0", what="scoring")


def test_access_is_refused_when_the_protocol_changed_after_the_freeze(tmp_path: Path) -> None:
    """Changing the protocol after G1 is exactly what the gate exists to stop."""
    _write(tmp_path, _record())
    granted = require_confirmatory_access(tmp_path, config_hash="cfg0", what="scoring")
    assert granted.primary_direction == "X->Y"
    with pytest.raises(FreezeError, match="not the one being run"):
        require_confirmatory_access(tmp_path, config_hash="cfg1", what="scoring")


def test_an_edited_freeze_is_detected_rather_than_trusted(tmp_path: Path) -> None:
    _write(tmp_path, _record())
    path = tmp_path / FREEZE_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["nominal_acceptance"] = 0.9  # quietly widen acceptance after freezing
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(FreezeError, match="edited since it was written"):
        load_freeze(tmp_path)


def test_the_freeze_hash_covers_every_frozen_choice(tmp_path: Path) -> None:
    baseline = _record().hash
    assert _record(primary_direction="Y->X").hash != baseline
    assert _record(frozen_feature_families=("A", "B", "C")).hash != baseline
    assert _record(delta_policy_margin=0.01).hash != baseline
    assert _record(n_calibration_groups=19).hash != baseline
    assert _record(learner="lightgbm").hash != baseline
    # The timestamp is not a frozen choice, so it must not change the hash.
    assert _record(generated_at="2027-01-01T00:00:00+00:00").hash == baseline
