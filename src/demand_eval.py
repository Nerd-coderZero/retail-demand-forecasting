"""Baselines and metrics for fixed-origin 7-day forecasts.

All baselines receive the target panel with every day after the forecast origin
set to NaN, and fail if any prediction is non-finite, so a baseline that reads
past the origin cannot silently produce numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from demand_features import HORIZON

BASELINES = ("seasonal_naive_7", "moving_average_7", "same_dow_mean_4w")


def _masked(y: np.ndarray, origin_idx: int) -> np.ndarray:
    m = y.astype(np.float64, copy=True)
    m[:, origin_idx + 1 :] = np.nan
    return m


def baseline_forecast(y: np.ndarray, origin_idx: int, name: str, horizon: int = HORIZON) -> np.ndarray:
    """Forecast of shape (n_series, horizon) for days origin_idx+1 .. origin_idx+horizon."""
    if origin_idx + 1 < 28:
        raise ValueError("origin needs at least 28 observed days")
    hist = _masked(y, origin_idx)
    targets = origin_idx + 1 + np.arange(horizon)
    if name == "seasonal_naive_7":
        pred = hist[:, targets - 7]
    elif name == "moving_average_7":
        pred = np.repeat(hist[:, origin_idx - 6 : origin_idx + 1].mean(axis=1, keepdims=True), horizon, axis=1)
    elif name == "same_dow_mean_4w":
        pred = np.mean(np.stack([hist[:, targets - k] for k in (7, 14, 21, 28)]), axis=0)
    else:
        raise ValueError(f"unknown baseline {name}")
    if not np.isfinite(pred).all():
        raise RuntimeError(f"{name} produced non-finite values; it read data after the origin")
    return pred


def point_metrics(actual, pred) -> dict[str, float]:
    a = np.asarray(actual, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    err = p - a
    total = a.sum()
    return {
        "n": int(a.size),
        "sum_actual": float(total),
        "wape": float(np.abs(err).sum() / total) if total > 0 else float("nan"),
        "wpe": float(err.sum() / total) if total > 0 else float("nan"),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err * err).mean())),
    }


def metrics_by(frame: pd.DataFrame, by: str, actual: str = "actual", pred: str = "pred") -> pd.DataFrame:
    rows = []
    for key, g in frame.groupby(by, observed=True, sort=True):
        rows.append({by: key, **point_metrics(g[actual].to_numpy(), g[pred].to_numpy())})
    return pd.DataFrame(rows)


def stockout_bucket(hours) -> pd.Categorical:
    h = np.asarray(hours)
    labels = np.where(h == 0, "0h", np.where(h >= 16, "16h_full_day", "1-15h"))
    return pd.Categorical(labels, categories=["0h", "1-15h", "16h_full_day"], ordered=True)
