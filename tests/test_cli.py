from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from warpaudit.cache.schema import FEATURE_COLUMNS, validate_columns
from warpaudit.cache.store import ShardedTable
from warpaudit.cli import main
from warpaudit.geometry.transforms import HomographyTransform
from warpaudit.registration.adapters import register_adapter
from warpaudit.types import RegistrationResult, RegistrationStatus

ROOT = Path(__file__).parents[1]


def test_module_help_and_config_validation() -> None:
    help_run = subprocess.run(
        [sys.executable, "-m", "warpaudit", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_run.returncode == 0
    assert "audit-data" in help_run.stdout and "estimate-cost" in help_run.stdout
    assert main(["validate-config", "--config", str(ROOT / "configs/pilot.yaml")]) == 0


def test_missing_data_fails_without_writing_empty_manifest(tmp_path: Path) -> None:
    config = tmp_path / "configs" / "pilot.yaml"
    config.parent.mkdir()
    config.write_text(
        """
project: fixture
datasets:
  - {id: FIRE, version: fixture, root: data/FIRE}
pipelines:
  - {id: fixture, version: fixture-1, matcher: fixture}
  - {id: fixture2, version: fixture-2, matcher: fixture}
""",
        encoding="utf-8",
    )
    assert main(["audit-data", "--config", str(config)]) == 3
    assert not (tmp_path / "manifests" / "pairs.parquet").exists()


def test_invalid_config_types_are_collected_without_traceback(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(
        """
datasets: [{id: D, version: v}]
pipelines:
  - {id: one, version: v}
  - {id: two, version: v}
geometry: {working_long_edge: nope}
primary_direction: 42
""",
        encoding="utf-8",
    )
    assert main(["validate-config", "--config", str(config)]) == 2


def test_audit_data_writes_real_manifest_for_fixture(tmp_path: Path) -> None:
    config = tmp_path / "configs" / "pilot.yaml"
    images = tmp_path / "data" / "FIRE" / "Images"
    annotations = tmp_path / "data" / "FIRE" / "Ground Truth"
    config.parent.mkdir()
    images.mkdir(parents=True)
    annotations.mkdir(parents=True)
    Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8)).save(images / "S01_1.jpg")
    Image.fromarray(np.full((12, 16, 3), 64, dtype=np.uint8)).save(images / "S01_2.jpg")
    (annotations / "control_points_S01_1_2.txt").write_text("1 1 1 1\n2 2 2 2\n", encoding="utf-8")
    config.write_text(
        """
project: fixture
geometry: {working_long_edge: 32, grid_size: 4}
features: {families: [A, B, E1], bootstrap_B: 4}
datasets:
  - {id: FIRE, version: fixture, root: data/FIRE}
pipelines:
  - {id: fixture, version: fixture-1, matcher: fixture, checkpoint: local, max_iters: 20}
  - {id: fixture2, version: fixture-2, matcher: fixture2, checkpoint: local, max_iters: 20}
""",
        encoding="utf-8",
    )
    assert main(["audit-data", "--config", str(config)]) == 0
    manifest_dir = tmp_path / "manifests"
    assert (manifest_dir / "development_groups.json").exists()
    assert (manifest_dir / "provenance.yaml").exists()
    assert (manifest_dir / "data.json").exists()
    assert (manifest_dir / "pairs.parquet").exists() or (manifest_dir / "pairs.jsonl").exists()

    class Registrar:
        pipeline_id = "fixture"

        def register(self, pair, seed):
            points = np.array(
                [[1, 1], [6, 1], [11, 1], [1, 5], [6, 5], [11, 5], [1, 9], [11, 9]],
                dtype=float,
            )
            scale = pair.coordinates.moving.to_working[0, 0]
            working_points = (points + 0.5) * scale - 0.5
            return RegistrationResult(
                "fixture",
                RegistrationStatus.OK,
                HomographyTransform(np.eye(3)),
                working_points,
                working_points,
                np.ones(len(points), bool),
                np.ones(len(points)),
            )

    register_adapter("fixture", lambda pipeline: Registrar())
    assert (
        main(
            [
                "register",
                "--config",
                str(config),
                "--split",
                "development",
                "--pipeline",
                "fixture",
                "--direction",
                "both",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "labels",
                "--config",
                str(config),
                "--split",
                "development",
                "--pipeline",
                "fixture",
                "--direction",
                "canonical",
            ]
        )
        == 0
    )
    labels = ShardedTable(tmp_path / "cache", "labels").load()
    assert len(labels) == 1
    assert validate_columns("labels", labels.columns) == []
    assert labels.iloc[0]["tre_defined"]
    assert labels.iloc[0]["tre_px"] == 0.0
    assert (
        main(
            [
                "features",
                "--workers",
                "2",
                "--config",
                str(config),
                "--split",
                "development",
                "--families",
                "A",
                "B",
                "C",
                "E1",
                "--direction",
                "canonical",
            ]
        )
        == 0
    )
    feature_cache = ShardedTable(tmp_path / "cache", "features", key_column="feature_id")
    feature_rows = feature_cache.load()
    assert set(feature_rows["family"]) == {"A", "B", "C", "E1"}
    assert "cycle_median" in set(feature_rows["feature_name"])
    assert set(FEATURE_COLUMNS) <= set(feature_rows.columns)

    # A resumed sweep must skip completed work rather than recompute and discard
    # it: the overnight run depends on an interrupted feature stage being cheap
    # to restart.
    shards_before = len(feature_cache.shards())
    assert (
        main(
            [
                "features",
                "--workers",
                "2",
                "--config",
                str(config),
                "--split",
                "development",
                "--families",
                "A",
                "B",
                "C",
                "E1",
                "--direction",
                "canonical",
            ]
        )
        == 0
    )
    assert len(feature_cache.shards()) == shards_before
    assert len(feature_cache.load()) == len(feature_rows)


def _rekey_config(path: Path, *, min_outer_folds: int, tau_primary: float = 0.005) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
project: fixture
datasets:
  - {{id: FIRE, version: fixture, root: data/FIRE}}
pipelines:
  - {{id: one, version: v1, matcher: fixture}}
  - {{id: two, version: v2, matcher: fixture}}
labels: {{tau_primary: {tau_primary}}}
splits: {{min_outer_folds: {min_outer_folds}}}
""",
        encoding="utf-8",
    )


def _seed_feature_cache(config: Path, rows: int = 3) -> tuple[ShardedTable, str]:
    from warpaudit.cache.hashing import short_hash
    from warpaudit.config import load_config

    cfg = load_config(config)
    cache_root = cfg.paths.resolve(config.parent.parent)["cache_root"]
    table = ShardedTable(cache_root, "features", key_column="feature_id")
    table.append(
        [
            {
                "feature_id": short_hash(
                    {"job_id": f"job{i}", "feature_hash": f"fh{i}", "config_hash": cfg.hash}
                ),
                "job_id": f"job{i}",
                "feature_hash": f"fh{i}",
                "config_hash": cfg.hash,
                "value": float(i),
            }
            for i in range(rows)
        ]
    )
    return table, cfg.hash


def test_rekey_cache_carries_features_across_a_fold_count_change(tmp_path: Path) -> None:
    from warpaudit.config import load_config

    previous = tmp_path / "configs" / "before.yaml"
    current = tmp_path / "configs" / "after.yaml"
    _rekey_config(previous, min_outer_folds=3)
    _rekey_config(current, min_outer_folds=2)
    table, old_hash = _seed_feature_cache(previous)
    new_hash = load_config(current).hash
    assert old_hash != new_hash

    assert main(
        ["rekey-cache", "--config", str(current), "--previous-config", str(previous)]
    ) == 0

    frame = table.load()
    carried = frame[frame["config_hash"].astype(str) == new_hash]
    assert len(carried) == 3
    # Values survive untouched; only the cache key moves.
    assert sorted(carried["value"]) == [0.0, 1.0, 2.0]
    assert sorted(carried["job_id"]) == ["job0", "job1", "job2"]


def test_rekey_cache_refuses_a_change_that_could_alter_a_value(tmp_path: Path) -> None:
    previous = tmp_path / "configs" / "before.yaml"
    current = tmp_path / "configs" / "after.yaml"
    _rekey_config(previous, min_outer_folds=3)
    # A label threshold does change downstream cached values, so re-keying is refused.
    _rekey_config(current, min_outer_folds=2, tau_primary=0.01)
    _seed_feature_cache(previous)

    assert main(
        ["rekey-cache", "--config", str(current), "--previous-config", str(previous)]
    ) == 3


def test_rekey_cache_is_a_noop_when_the_hash_is_unchanged(tmp_path: Path) -> None:
    config = tmp_path / "configs" / "same.yaml"
    _rekey_config(config, min_outer_folds=3)
    _seed_feature_cache(config)
    assert main(
        ["rekey-cache", "--config", str(config), "--previous-config", str(config)]
    ) == 0


def test_rekey_cache_moves_registrations_onto_the_current_contract(tmp_path: Path) -> None:
    from warpaudit.config import load_config

    previous = tmp_path / "configs" / "before.yaml"
    current = tmp_path / "configs" / "after.yaml"
    _rekey_config(previous, min_outer_folds=3)
    _rekey_config(current, min_outer_folds=2)
    old_cfg = load_config(previous)
    cache_root = old_cfg.paths.resolve(tmp_path)["cache_root"]
    registrations = ShardedTable(cache_root, "registrations")
    registrations.append(
        [
            {"job_id": f"job{i}", "config_hash": old_cfg.hash, "n_inliers": i}
            for i in range(4)
        ]
    )
    _seed_feature_cache(previous)
    new_hash = load_config(current).hash

    assert main(
        ["rekey-cache", "--config", str(current), "--previous-config", str(previous)]
    ) == 0

    frame = registrations.load()
    # job_id is the key, so re-keyed rows supersede rather than duplicate.
    assert len(frame) == 4
    assert set(frame["config_hash"].astype(str)) == {new_hash}
    assert sorted(frame["n_inliers"]) == [0, 1, 2, 3]
