"""LightGBM training and prediction contract shared by notebook 2 and the serving layer.

The model consumes a float32 matrix whose columns follow model_feature_columns() exactly.
Predictions are clipped at zero because sales are non-negative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from demand_features import feature_columns

CATEGORICAL_FEATURES = [
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
    "day_of_week",
]
SAME_DAY_DISCOUNT_FEATURES = ["discount", "discount_vs_recent"]

BASE_PARAMS = {
    "metric": "l1",
    "learning_rate": 0.05,
    "num_leaves": 255,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "seed": 20240626,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
    "verbose": -1,
}
OBJECTIVES = {
    "l2": {"objective": "regression"},
    "l1": {"objective": "regression_l1"},
    "tweedie": {"objective": "tweedie", "tweedie_variance_power": 1.2},
}


def model_feature_columns(weather_mode: str = "none", same_day_discount: bool = True) -> list[str]:
    cols = feature_columns(weather_mode)
    if not same_day_discount:
        cols = [c for c in cols if c not in SAME_DAY_DISCOUNT_FEATURES]
    return cols


def model_params(objective: str) -> dict:
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(OBJECTIVES)}")
    return {**BASE_PARAMS, **OBJECTIVES[objective]}


def to_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"feature columns missing: {missing}")
    return frame[columns].to_numpy(dtype=np.float32)


def make_dataset(x: np.ndarray, y: np.ndarray, columns: list[str], reference=None):
    import lightgbm as lgb

    if x.shape[1] != len(columns):
        raise ValueError("matrix width does not match column list")
    return lgb.Dataset(
        x,
        label=y,
        feature_name=list(columns),
        categorical_feature=[c for c in CATEGORICAL_FEATURES if c in columns],
        reference=reference,
        free_raw_data=True,
    )


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    columns: list[str],
    objective: str,
    num_boost_round: int,
    x_valid: np.ndarray | None = None,
    y_valid: np.ndarray | None = None,
    early_stopping_rounds: int | None = None,
    num_threads: int | None = None,
):
    import lightgbm as lgb

    params = model_params(objective)
    if num_threads is not None:
        params["num_threads"] = num_threads
    dtrain = make_dataset(x_train, y_train, columns)
    callbacks = []
    valid_sets = []
    if x_valid is not None:
        valid_sets = [make_dataset(x_valid, y_valid, columns, reference=dtrain)]
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False))
    booster = lgb.train(params, dtrain, num_boost_round=num_boost_round, valid_sets=valid_sets, callbacks=callbacks)
    return booster


def predict(booster, frame_or_matrix, columns: list[str] | None = None, num_iteration: int | None = None) -> np.ndarray:
    names = booster.feature_name()
    if isinstance(frame_or_matrix, pd.DataFrame):
        x = to_matrix(frame_or_matrix, names)
    else:
        if columns is None or list(columns) != names:
            raise ValueError("matrix input requires columns equal to the booster feature order")
        x = np.asarray(frame_or_matrix, dtype=np.float32)
    if num_iteration is None:
        best = booster.best_iteration
        num_iteration = best if best and best > 0 else None
    return np.clip(booster.predict(x, num_iteration=num_iteration), 0.0, None)
