import json

import numpy as np
import pandas as pd
import pytest

from warpaudit.cli import main
from warpaudit.evaluation.card import evaluate_scores, generalized_risk_area


def fixture():
    return pd.DataFrame(
        {
            "score": [0.1, 0.2, 0.8, np.nan],
            "probability": [0.05, 0.10, 0.90, 1.0],
            "failure": [0, 0, 1, 1],
            "group": ["a", "a", "b", "c"],
            "explicit_failure": [False, False, False, True],
        }
    )


def test_evaluation_card_reports_three_layers():
    result = evaluate_scores(fixture())
    assert result.summary["ranking"]["auroc"] == 1.0
    assert result.summary["acceptance"]["augrc"] == pytest.approx(1 / 9)
    assert result.summary["acceptance"]["maximum_attainable_coverage"] == 2 / 3
    assert result.summary["probability"]["brier_all_attempts"] < 0.01
    assert len(result.curve) == 4


def test_generalized_risk_area_references():
    assert generalized_risk_area(0.2, 1.0) == pytest.approx(0.02)
    assert generalized_risk_area(0.2, 0.5) == pytest.approx(0.1)


def test_evaluation_card_cli(tmp_path):
    source, output = tmp_path / "scores.csv", tmp_path / "card.json"
    fixture().to_csv(source, index=False)
    assert main(["evaluation-card", "--input", str(source), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["support"]["cases"] == 4
    assert output.with_name("card_curve.csv").is_file()


def test_evaluation_card_parses_csv_boolean_strings():
    frame = fixture().astype({"explicit_failure": str})
    result = evaluate_scores(frame)
    assert result.summary["support"]["explicit_failures"] == 1


def test_evaluation_card_rejects_ambiguous_boolean_strings():
    frame = fixture().astype({"explicit_failure": object})
    frame.loc[0, "explicit_failure"] = "no"
    with pytest.raises(ValueError, match="explicit_failure"):
        evaluate_scores(frame)
