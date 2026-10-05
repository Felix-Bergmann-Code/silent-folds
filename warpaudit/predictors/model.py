"""Small source-supervised pilot learner (specification §7.3-§7.4).

All preprocessing is fitted inside the source training fold.  Missing feature
values are median-imputed and accompanied by explicit missingness indicators;
the same frozen transform is then used for calibration and target scoring.
Probability calibration is a separate operation on separate calibration
groups.  The implementation never accepts annotations or a target-domain flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..cache.hashing import short_hash
from ..evaluation.metrics import auroc
from ..evaluation.weights import inverse_group_size_weights
from ..protocols.splits import make_outer_folds

__all__ = [
    "LogisticRiskDetector",
    "SourcePreprocessor",
    "fit_lightgbm_detector",
    "fit_logistic_detector",
    "lightgbm_available",
]


def _binary_labels(labels: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(labels, dtype=np.float64)
    if y.shape != (n,) or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("labels must be a binary vector matching the feature rows")
    if len(np.unique(y)) < 2:
        raise ValueError("both classes are required to fit or calibrate a detector")
    return y.astype(np.int64)


@dataclass
class SourcePreprocessor:
    """Frozen source-fitted median imputation and standardisation."""

    feature_names: tuple[str, ...]
    medians: np.ndarray | None = None
    scaler: StandardScaler = field(default_factory=StandardScaler)
    all_missing_features: tuple[str, ...] = ()

    def _matrix(self, features: pd.DataFrame) -> np.ndarray:
        missing = [name for name in self.feature_names if name not in features.columns]
        if missing:
            raise ValueError(f"feature frame lacks frozen columns: {missing}")
        return features.loc[:, self.feature_names].to_numpy(dtype=np.float64, copy=True)

    def fit(self, features: pd.DataFrame) -> SourcePreprocessor:
        values = self._matrix(features)
        if len(values) == 0:
            raise ValueError("cannot fit preprocessing on zero rows")
        medians = np.array(
            [
                np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else np.nan
                for column in values.T
            ],
            dtype=np.float64,
        )
        all_missing = ~np.isfinite(medians)
        medians[all_missing] = 0.0
        missing = ~np.isfinite(values)
        imputed = np.where(missing, medians, values)
        design = np.concatenate((imputed, missing.astype(np.float64)), axis=1)
        self.medians = medians
        self.all_missing_features = tuple(
            name
            for name, is_missing in zip(self.feature_names, all_missing, strict=True)
            if is_missing
        )
        self.scaler.fit(design)
        return self

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        if self.medians is None:
            raise RuntimeError("preprocessor has not been fitted")
        values = self._matrix(features)
        missing = ~np.isfinite(values)
        imputed = np.where(missing, self.medians, values)
        design = np.concatenate((imputed, missing.astype(np.float64)), axis=1)
        return self.scaler.transform(design)

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.feature_names + tuple(f"{name}__missing" for name in self.feature_names)

    @property
    def fingerprint(self) -> str:
        if self.medians is None:
            raise RuntimeError("preprocessor has not been fitted")
        return short_hash(
            {
                "feature_names": self.feature_names,
                "medians": self.medians,
                "mean": self.scaler.mean_,
                "scale": self.scaler.scale_,
                "all_missing_features": self.all_missing_features,
            }
        )


@dataclass
class LogisticRiskDetector:
    """Fitted ranking model plus an optional calibration map."""

    preprocessor: SourcePreprocessor
    model: LogisticRegression
    selected_C: float
    cv_scores: dict[float, float]
    calibration_method: str = ""
    calibrator: LogisticRegression | IsotonicRegression | None = None

    def decision_function(self, features: pd.DataFrame) -> np.ndarray:
        design = self.preprocessor.transform(features)
        if hasattr(self.model, "decision_function"):
            return np.asarray(self.model.decision_function(design))
        # Tree ensembles expose probabilities rather than a margin. The log-odds
        # is the monotone score matching the logistic arm, so ranking, threshold
        # selection, and Platt scaling behave identically across learners.
        probability = np.clip(
            np.asarray(self.model.predict_proba(design))[:, 1], 1e-12, 1 - 1e-12
        )
        return np.log(probability / (1.0 - probability))

    def calibrate(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        *,
        method: str = "platt",
        sample_weight: np.ndarray | None = None,
    ) -> LogisticRiskDetector:
        raw = self.decision_function(features)
        y = _binary_labels(labels, len(raw))
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if weights is not None and weights.shape != (len(raw),):
            raise ValueError("calibration sample_weight must match feature rows")
        if method == "platt":
            calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000, random_state=0)
            calibrator.fit(raw[:, None], y, sample_weight=weights)
        elif method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(raw, y, sample_weight=weights)
        else:
            raise ValueError("calibration method must be 'platt' or 'isotonic'")
        self.calibration_method = method
        self.calibrator = calibrator
        return self

    def predict_probability(self, features: pd.DataFrame) -> np.ndarray:
        if self.calibrator is None:
            raise RuntimeError("probability map has not been fitted on calibration groups")
        raw = self.decision_function(features)
        if self.calibration_method == "platt":
            return np.asarray(self.calibrator.predict_proba(raw[:, None])[:, 1])
        return np.asarray(self.calibrator.predict(raw))

    @property
    def fingerprint(self) -> str:
        calibration: dict[str, object] = {"method": self.calibration_method}
        if isinstance(self.calibrator, LogisticRegression):
            calibration.update(coef=self.calibrator.coef_, intercept=self.calibrator.intercept_)
        elif isinstance(self.calibrator, IsotonicRegression):
            calibration.update(x=self.calibrator.X_thresholds_, y=self.calibrator.y_thresholds_)
        parameters: dict[str, object] = {"class": type(self.model).__name__}
        if hasattr(self.model, "coef_"):
            parameters["coef"] = self.model.coef_
            parameters["intercept"] = self.model.intercept_
        else:
            # A booster has no coefficient vector; its dumped structure is the
            # equivalent identity of the fitted model.
            booster = getattr(self.model, "booster_", None)
            parameters["model"] = booster.model_to_string() if booster is not None else repr(
                self.model.get_params()
            )
        return short_hash(
            {
                "preprocessor": self.preprocessor.fingerprint,
                "C": self.selected_C,
                "parameters": short_hash(parameters),
                "calibration": calibration,
            }
        )


def fit_logistic_detector(
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    feature_names: tuple[str, ...] | None = None,
    C_values: tuple[float, ...] = (0.1, 1.0, 10.0),
    n_folds: int = 5,
    seed: int = 20260907,
) -> LogisticRiskDetector:
    """Select ``C`` by grouped inner validation and refit on all source rows."""
    if feature_names is None:
        feature_names = tuple(str(name) for name in features.columns)
    if not feature_names:
        raise ValueError("at least one feature is required")
    y = _binary_labels(labels, len(features))
    group = np.asarray(groups).astype(str)
    if group.shape != (len(features),):
        raise ValueError("groups must match feature rows")
    unique = np.unique(group)
    folds_n = min(int(n_folds), len(unique))
    if folds_n < 2:
        raise ValueError("grouped inner validation requires at least two groups")
    assignment = make_outer_folds(unique, n_folds=folds_n, seed=seed)

    cv_scores: dict[float, float] = {}
    for C in sorted(set(float(value) for value in C_values)):
        fold_scores: list[float] = []
        for fold in range(folds_n):
            valid = np.array([assignment[g] == fold for g in group])
            train = ~valid
            if len(np.unique(y[train])) < 2 or len(np.unique(y[valid])) < 2:
                continue
            pre = SourcePreprocessor(feature_names).fit(features.loc[train])
            model = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=seed)
            train_weight = inverse_group_size_weights(group[train])
            model.fit(pre.transform(features.loc[train]), y[train], sample_weight=train_weight)
            scores = model.decision_function(pre.transform(features.loc[valid]))
            fold_scores.append(auroc(scores, y[valid], inverse_group_size_weights(group[valid])))
        cv_scores[C] = float(np.mean(fold_scores)) if fold_scores else float("nan")

    valid_C = [C for C, score in cv_scores.items() if np.isfinite(score)]
    if not valid_C:
        raise ValueError("every grouped validation fold was one-class; C cannot be selected")
    selected = min(valid_C, key=lambda C: (-cv_scores[C], C))
    preprocessor = SourcePreprocessor(feature_names).fit(features)
    model = LogisticRegression(C=selected, solver="lbfgs", max_iter=2000, random_state=seed)
    model.fit(
        preprocessor.transform(features),
        y,
        sample_weight=inverse_group_size_weights(group),
    )
    return LogisticRiskDetector(preprocessor, model, selected, cv_scores)

def lightgbm_available() -> tuple[bool, str]:
    """Probe the optional secondary learner (spec §7.4).

    LightGBM is an optional extra, not a core dependency: it is the *secondary*
    learner, so its absence must cost the run that one reported arm rather than
    the primary claim.
    """
    try:
        import lightgbm  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def fit_lightgbm_detector(
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    feature_names: tuple[str, ...] | None = None,
    num_leaves: tuple[int, ...] = (7, 15),
    min_child_samples: tuple[int, ...] = (20, 50),
    n_folds: int = 5,
    seed: int = 20260907,
) -> LogisticRiskDetector:
    """Shallow LightGBM over the same frozen source preprocessing.

    The grid is the frozen one from §7.4 and is selected by the same grouped
    inner validation as the primary learner, so the two arms differ in model
    class alone rather than in how they were tuned. The returned object reuses
    :class:`LogisticRiskDetector`'s calibration and fingerprinting so a policy
    can be frozen on it unchanged.
    """
    import lightgbm as lgb

    if feature_names is None:
        feature_names = tuple(str(name) for name in features.columns)
    y = _binary_labels(labels, len(features))
    group = np.asarray(groups).astype(str)
    unique = np.unique(group)
    folds_n = min(int(n_folds), len(unique))
    if folds_n < 2:
        raise ValueError("grouped inner validation requires at least two groups")
    assignment = make_outer_folds(unique, n_folds=folds_n, seed=seed)

    grid = [(int(leaves), int(child)) for leaves in num_leaves for child in min_child_samples]
    cv_scores: dict[tuple[int, int], float] = {}
    for leaves, child in grid:
        fold_scores: list[float] = []
        for fold in range(folds_n):
            valid = np.array([assignment[g] == fold for g in group])
            train = ~valid
            if len(np.unique(y[train])) < 2 or len(np.unique(y[valid])) < 2:
                continue
            pre = SourcePreprocessor(feature_names).fit(features.loc[train])
            model = lgb.LGBMClassifier(
                num_leaves=leaves,
                min_child_samples=child,
                n_estimators=200,
                learning_rate=0.05,
                random_state=seed,
                verbose=-1,
            )
            model.fit(
                pre.transform(features.loc[train]),
                y[train],
                sample_weight=inverse_group_size_weights(group[train]),
            )
            scores = model.predict_proba(pre.transform(features.loc[valid]))[:, 1]
            fold_scores.append(auroc(scores, y[valid], inverse_group_size_weights(group[valid])))
        cv_scores[(leaves, child)] = float(np.mean(fold_scores)) if fold_scores else float("nan")

    usable = [key for key, score in cv_scores.items() if np.isfinite(score)]
    if not usable:
        raise ValueError("every grouped validation fold was one-class; no grid point is usable")
    leaves, child = min(usable, key=lambda k: (-cv_scores[k], k))
    preprocessor = SourcePreprocessor(feature_names).fit(features)
    model = lgb.LGBMClassifier(
        num_leaves=leaves,
        min_child_samples=child,
        n_estimators=200,
        learning_rate=0.05,
        random_state=seed,
        verbose=-1,
    )
    model.fit(
        preprocessor.transform(features),
        y,
        sample_weight=inverse_group_size_weights(group),
    )
    return LogisticRiskDetector(
        preprocessor=preprocessor,
        model=model,
        selected_C=float(leaves),
        cv_scores={float(k[0] * 1000 + k[1]): v for k, v in cv_scores.items()},
    )
