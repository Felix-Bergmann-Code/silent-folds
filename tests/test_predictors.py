from __future__ import annotations

import numpy as np
import pandas as pd

from warpaudit.predictors.model import fit_logistic_detector


def test_logistic_uses_frozen_source_preprocessing_and_separate_calibration() -> None:
    rng = np.random.default_rng(4)
    groups = np.repeat([f"g{i}" for i in range(12)], 4)
    labels = np.tile([0, 0, 1, 1], 12)
    signal = labels + rng.normal(0, 0.1, len(labels))
    signal[::11] = np.nan
    features = pd.DataFrame({"signal": signal, "constant": 1.0})
    detector = fit_logistic_detector(features, labels, groups, n_folds=4)
    median_before = detector.preprocessor.medians.copy()
    detector.calibrate(features.iloc[:24], labels[:24], method="platt")
    probability = detector.predict_probability(features.iloc[24:])
    np.testing.assert_array_equal(detector.preprocessor.medians, median_before)
    assert np.all((probability >= 0) & (probability <= 1))
    assert detector.fingerprint


def test_isotonic_calibration_is_non_decreasing_in_raw_score() -> None:
    rng = np.random.default_rng(7)
    groups = np.repeat([f"g{i}" for i in range(12)], 4)
    labels = np.tile([0, 0, 1, 1], 12)
    features = pd.DataFrame(
        {"signal": labels + rng.normal(0, 0.25, len(labels))}
    )
    detector = fit_logistic_detector(features, labels, groups, n_folds=4)
    detector.calibrate(features, labels, method="isotonic")
    raw = detector.decision_function(features)
    probability = detector.predict_probability(features)
    order = np.argsort(raw, kind="stable")
    assert np.all(np.diff(probability[order]) >= 0)
