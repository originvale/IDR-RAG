"""Frozen estimator configurations used by the paper."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from .features import FEATURE_NAMES

PAPER_SEED = 20260829
PAPER_TREES = 350
PAPER_MIN_LEAF = 10


def _xy(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        [[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=float
    )
    y = np.asarray([float(row["observed_delta"]) for row in rows], dtype=float)
    if x.shape != (len(rows), 32) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("invalid feature or target table")
    return x, y


def fit_idr(rows: list[dict[str, Any]], n_jobs: int = 1) -> ExtraTreesRegressor:
    x, y = _xy(rows)
    model = ExtraTreesRegressor(
        n_estimators=PAPER_TREES,
        min_samples_leaf=PAPER_MIN_LEAF,
        random_state=PAPER_SEED,
        max_features=1.0,
        bootstrap=False,
        n_jobs=n_jobs,
    )
    return model.fit(x, y)


def fit_dynamic_benefit(
    rows: list[dict[str, Any]], n_jobs: int = 1
) -> ExtraTreesClassifier:
    x, y = _xy(rows)
    model = ExtraTreesClassifier(
        n_estimators=PAPER_TREES,
        min_samples_leaf=PAPER_MIN_LEAF,
        random_state=PAPER_SEED,
        max_features=1.0,
        bootstrap=False,
        class_weight=None,
        n_jobs=n_jobs,
    )
    return model.fit(x, y > 0)


def predict_idr(model: ExtraTreesRegressor, rows: list[dict[str, Any]]) -> np.ndarray:
    x, _ = _xy(rows)
    return np.asarray(model.predict(x), dtype=float)


def predict_dynamic_benefit(
    model: ExtraTreesClassifier, rows: list[dict[str, Any]]
) -> np.ndarray:
    x, _ = _xy(rows)
    positive = list(model.classes_).index(True)
    return np.asarray(model.predict_proba(x)[:, positive], dtype=float)
