from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from warpaudit.cli import _feature_results
from warpaudit.config import load_config
from warpaudit.geometry.transforms import HomographyTransform
from warpaudit.signals.context import ContextSources
from warpaudit.types import RegistrationResult, RegistrationStatus


@pytest.mark.parametrize("worker_threads", [0, 1])
def test_spawned_feature_workers_match_serial_values_and_order(
    pair_input, correspondences, worker_threads
):
    cfg = load_config(Path(__file__).parents[1] / "configs/pilot.yaml")
    for path in (pair_input.moving_path, pair_input.fixed_path):
        Image.fromarray(np.random.default_rng(4).integers(
            0, 256, (100, 120, 3), dtype=np.uint8
        )).save(path)
    src, dst = correspondences
    result = RegistrationResult(
        "xfeat_h", RegistrationStatus.OK, HomographyTransform(np.eye(3)),
        src, dst, np.ones(len(src), bool), np.ones(len(src)),
    )
    failure = RegistrationResult("xfeat_h", RegistrationStatus.NO_MATCHES, None)
    sources = ContextSources(result, {}, {"sp_lg_h": result})
    tasks = [
        (i, pd.Series({"seed": 4, "pair_id": pair_input.pair_id}), pair_input,
         r, sources, ("A", "B", "C", "D", "E1", "F", "G"), cfg)
        for i, r in enumerate((result, failure, result))
    ]
    serial = list(_feature_results(iter(tasks), 1))
    parallel = list(_feature_results(iter(tasks), 2, worker_threads))
    assert [item[0] for item in parallel] == [0, 1, 2]
    for left, right in zip(serial, parallel, strict=True):
        assert left[3].keys() == right[3].keys()
        for family in left[3]:
            lvalues, rvalues = left[3][family].values, right[3][family].values
            assert lvalues.keys() == rvalues.keys()
            for name in lvalues:
                lvalue, rvalue = asdict(lvalues[name]), asdict(rvalues[name])
                np.testing.assert_equal(lvalue.pop("value"), rvalue.pop("value"))
                assert lvalue == rvalue
