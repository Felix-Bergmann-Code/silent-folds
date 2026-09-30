from __future__ import annotations

import json
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts import pole_guard_ablation as ablation
from scripts.pole_guard_ablation import (
    add_pole_guard_features,
    guarded_columns,
    treatment_frames,
)


def test_pole_guard_sets_bending_missing_and_adds_indicator():
    frame = pd.DataFrame(
        {
            "transform_family": ["homography", "homography"],
            "status": ["ok", "ok"],
            "transform_params": [
                json.dumps(
                    {
                        "matrix": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [1.0, 0.0, -49.5],
                        ]
                    }
                ),
                json.dumps(
                    {
                        "matrix": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.001, 0.0, 1.0],
                        ]
                    }
                ),
            ],
            "working_moving_hw": [np.array([80, 100]), np.array([80, 100])],
            "working_fixed_hw": [np.array([80, 100]), np.array([80, 100])],
            "A_moving": [np.eye(3), np.eye(3)],
            "A_fixed": [np.eye(3), np.eye(3)],
            "D:bending_energy": [1e9, 0.2],
        }
    )
    guarded, guarded_name, indicator = add_pole_guard_features(
        frame, ("D:bending_energy",)
    )
    assert guarded.pole_crosses_image.tolist() == [True, False]
    assert guarded.pole_clearance_diagonal_fraction.iloc[0] == 0.0
    assert guarded.pole_clearance_diagonal_fraction.iloc[1] > 0.0
    assert guarded.pole_crosses_inscribed_circle.tolist() == [True, False]
    assert guarded["D:projective_pole_in_fov"].tolist() == [1.0, 0.0]
    assert np.isnan(guarded.loc[0, guarded_name])
    assert guarded.loc[1, guarded_name] == 0.2
    assert guarded[indicator].tolist() == [1.0, 0.0]
    assert np.isfinite(guarded["D:bending_energy_fov_restricted"]).all()
    assert guarded_columns(
        ("A:score", "D:bending_energy"),
        "D:bending_energy",
        guarded_name,
        indicator,
    ) == ("A:score", guarded_name)

    columns = ("A:score", "D:bending_energy")
    expected = {
        "fov_guard_on_rectangle": (
            "A:score",
            "D:bending_energy_fov_pole_guarded",
        ),
        "rectangle_guard_on_fov": (
            "A:score",
            "D:bending_energy_fov_restricted_rectangle_guarded",
        ),
        "fov_guard_on_fov": (
            "A:score",
            "D:bending_energy_fov_restricted_fov_guarded",
        ),
    }
    for treatment, expected_columns in expected.items():
        _, _, _, treated_columns = treatment_frames(
            guarded,
            guarded,
            guarded,
            treatment=treatment,
            columns=columns,
            source_name="D:bending_energy",
            guarded_name=guarded_name,
            indicator_name=indicator,
            train_min=0.0,
            train_max=1.0,
        )
        assert treated_columns == expected_columns


def test_warning_audit_captures_and_serializes_fit_warning(tmp_path, monkeypatch):
    fake_model = SimpleNamespace(
        model=SimpleNamespace(n_iter_=np.asarray([3])),
        calibrator=SimpleNamespace(n_iter_=np.asarray([4])),
        selected_C=1.0,
    )

    monkeypatch.setattr(ablation, "load_config", lambda path: SimpleNamespace(
        full_study=SimpleNamespace(development_datasets=("FIRE", "COph100")),
        splits=SimpleNamespace(seed=0),
    ))
    monkeypatch.setattr(ablation, "assemble", lambda *args: (pd.DataFrame(), ("D:x",)))
    monkeypatch.setattr(
        ablation,
        "select_arms",
        lambda *args: {"non_stability": ("D:x",)},
    )
    partition = pd.DataFrame({"operational_failure": [0, 1]})
    monkeypatch.setattr(
        ablation.dd,
        "split_cell",
        lambda *args: (partition, partition, partition),
    )

    def fake_fit(*args, **kwargs):
        warnings.warn("line search stopped", RuntimeWarning, stacklevel=2)
        return fake_model, object(), 0.01

    monkeypatch.setattr(ablation.ps, "fit_arm", fake_fit)
    monkeypatch.setattr(
        ablation.audit,
        "model_diagnostics",
        lambda *args: {"platt_slope": -0.1, "ranking_reversed_by_calibration": True},
    )

    result = ablation.warning_audit(tmp_path)

    assert result["warning_free"] is False
    assert result["warnings"] == [
        {"category": "RuntimeWarning", "message": "line search stopped"}
    ]
    written = json.loads((tmp_path / "optimizer_warning_audit.json").read_text())
    assert written["model_iterations"] == [3]
    assert written["calibrator_iterations"] == [4]
