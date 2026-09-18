"""Leakage-safe daily demand features for FreshRetailNet-50K.

Every history-derived feature for target day t reads only days <= t - MIN_LAG.
With MIN_LAG equal to the forecast horizon, one feature definition serves every
horizon step 1..HORIZON from a single forecast origin (direct strategy), and the
same code path is used for training rows and for serving requests.

Same-day columns are split into two groups:
- KNOWN_FUTURE_COLS: planned in advance (calendar, discount, promo activity).
  Treated as known for the target day. Discount is an assumption, see notebook 1.
- Realized same-day columns (target, stockout hours, weather): never used for
  the target day. Weather is only available through the explicit "oracle" mode,
  which exists for a labelled sensitivity comparison and not for serving.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY_COLS = ["store_id", "product_id"]
DATE_COL = "dt"
TARGET_COL = "sale_amount"
STOCK_COL = "stock_hour6_22_cnt"
STATIC_COLS = [
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
]
KNOWN_FUTURE_COLS = ["discount", "activity_flag", "holiday_flag"]
WEATHER_COLS = ["precpt", "avg_temperature", "avg_humidity", "avg_wind_level"]
REALIZED_COLS = [TARGET_COL, STOCK_COL] + WEATHER_COLS

HORIZON = 7
MIN_LAG = HORIZON
SALES_LAGS = (7, 8, 9, 10, 11, 12, 13, 14, 21, 28)
SALES_MEAN_WINDOWS = (7, 14, 28)
SALES_STD_WINDOWS = (7, 28)
STOCK_MEAN_WINDOWS = (7, 28)
FULL_DAY_STOCKOUT_HOURS = 16
WEATHER_MODES = ("none", "oracle")

REQUIRED_HISTORY_DAYS = max(max(SALES_LAGS), MIN_LAG + max(SALES_MEAN_WINDOWS) - 1)


class PanelError(ValueError):
    pass


def to_panel(df: pd.DataFrame, value_cols: list[str]) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[str, np.ndarray]]:
    """Reshape a long frame into (n_series, n_days) arrays.

    Requires a complete, duplicate-free panel over a contiguous daily range.
    """
    missing = [c for c in KEY_COLS + [DATE_COL] + value_cols if c not in df.columns]
    if missing:
        raise PanelError(f"missing columns: {missing}")
    dates = pd.to_datetime(df[DATE_COL])
    start, end = dates.min(), dates.max()
    grid = pd.date_range(start, end, freq="D")
    n_days = len(grid)
    keys = df[KEY_COLS].drop_duplicates().sort_values(KEY_COLS).reset_index(drop=True)
    n_series = len(keys)
    if len(df) != n_series * n_days:
        raise PanelError(
            f"incomplete or duplicated panel: {len(df)} rows, expected {n_series} x {n_days} = {n_series * n_days}"
        )
    order = np.lexsort((dates.to_numpy(), df["product_id"].to_numpy(), df["store_id"].to_numpy()))
    sorted_dates = dates.to_numpy()[order].reshape(n_series, n_days)
    if not (sorted_dates == grid.to_numpy()[None, :]).all():
        raise PanelError("panel dates are not a complete contiguous grid for every series")
    sorted_keys = df[KEY_COLS].to_numpy()[order].reshape(n_series, n_days, len(KEY_COLS))
    if not (sorted_keys == sorted_keys[:, :1, :]).all():
        raise PanelError("series keys are not constant along the date axis after sorting")
    arrays = {c: df[c].to_numpy(dtype=np.float64)[order].reshape(n_series, n_days) for c in value_cols}
    return keys, grid, arrays


def lag(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    if k < x.shape[1]:
        out[:, k:] = x[:, : x.shape[1] - k]
    return out


def _window_sums(x: np.ndarray, window: int, offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sums over days (t - offset - window + 1 .. t - offset), accumulated window element by element.

    Each output value depends only on the values inside its own window, so the result is bit-identical
    regardless of how much history precedes the window (required for serving parity).
    """
    s1 = np.zeros(x.shape, dtype=np.float64)
    s2 = np.zeros(x.shape, dtype=np.float64)
    n = np.zeros(x.shape, dtype=np.int32)
    for j in range(window):
        v = lag(x, offset + j)
        valid = ~np.isnan(v)
        v0 = np.where(valid, v, 0.0)
        s1 += v0
        s2 += v0 * v0
        n += valid
    return s1, s2, n


def rolling_mean(x: np.ndarray, window: int, offset: int) -> np.ndarray:
    """Mean over days (t - offset - window + 1 .. t - offset). NaN unless the full window is observed."""
    s1, _, n = _window_sums(x, window, offset)
    return np.where(n == window, s1 / window, np.nan)


def rolling_std(x: np.ndarray, window: int, offset: int) -> np.ndarray:
    """Population standard deviation over the same window as rolling_mean."""
    s1, s2, n = _window_sums(x, window, offset)
    mean = s1 / window
    var = np.maximum(s2 / window - mean * mean, 0.0)
    return np.where(n == window, np.sqrt(var), np.nan)


def feature_columns(weather_mode: str = "none") -> list[str]:
    if weather_mode not in WEATHER_MODES:
        raise ValueError(f"weather_mode must be one of {WEATHER_MODES}")
    cols = list(STATIC_COLS) + ["day_of_week"] + list(KNOWN_FUTURE_COLS)
    cols += [f"sales_lag_{k}" for k in SALES_LAGS]
    cols += [f"sales_mean_{w}_off{MIN_LAG}" for w in SALES_MEAN_WINDOWS]
    cols += [f"sales_std_{w}_off{MIN_LAG}" for w in SALES_STD_WINDOWS]
    cols += ["sales_same_dow_mean_4w", f"sales_zero_share_28_off{MIN_LAG}"]
    cols += [f"stock_hours_lag_{MIN_LAG}"]
    cols += [f"stock_hours_mean_{w}_off{MIN_LAG}" for w in STOCK_MEAN_WINDOWS]
    cols += [f"stock_fullday_share_28_off{MIN_LAG}"]
    cols += [f"discount_mean_7_off{MIN_LAG}", "discount_vs_recent", f"activity_share_28_off{MIN_LAG}"]
    if weather_mode == "oracle":
        cols += [f"oracle_{c}" for c in WEATHER_COLS]
    return cols


def build_feature_arrays(
    arrays: dict[str, np.ndarray],
    grid: pd.DatetimeIndex,
    weather_mode: str = "none",
) -> dict[str, np.ndarray]:
    """History and known-future features as (n_series, n_days) arrays. Static columns excluded."""
    if weather_mode not in WEATHER_MODES:
        raise ValueError(f"weather_mode must be one of {WEATHER_MODES}")
    y = arrays[TARGET_COL]
    stock = arrays[STOCK_COL]
    n_series, n_days = y.shape
    f: dict[str, np.ndarray] = {}

    f["day_of_week"] = np.broadcast_to(grid.dayofweek.to_numpy(dtype=np.float64)[None, :], (n_series, n_days))
    for c in KNOWN_FUTURE_COLS:
        f[c] = arrays[c]

    for k in SALES_LAGS:
        f[f"sales_lag_{k}"] = lag(y, k)
    for w in SALES_MEAN_WINDOWS:
        f[f"sales_mean_{w}_off{MIN_LAG}"] = rolling_mean(y, w, MIN_LAG)
    for w in SALES_STD_WINDOWS:
        f[f"sales_std_{w}_off{MIN_LAG}"] = rolling_std(y, w, MIN_LAG)
    f["sales_same_dow_mean_4w"] = (lag(y, 7) + lag(y, 14) + lag(y, 21) + lag(y, 28)) / 4.0
    zero = np.where(np.isnan(y), np.nan, (y == 0).astype(np.float64))
    f[f"sales_zero_share_28_off{MIN_LAG}"] = rolling_mean(zero, 28, MIN_LAG)

    f[f"stock_hours_lag_{MIN_LAG}"] = lag(stock, MIN_LAG)
    for w in STOCK_MEAN_WINDOWS:
        f[f"stock_hours_mean_{w}_off{MIN_LAG}"] = rolling_mean(stock, w, MIN_LAG)
    fullday = np.where(np.isnan(stock), np.nan, (stock >= FULL_DAY_STOCKOUT_HOURS).astype(np.float64))
    f[f"stock_fullday_share_28_off{MIN_LAG}"] = rolling_mean(fullday, 28, MIN_LAG)

    disc_recent = rolling_mean(arrays["discount"], 7, MIN_LAG)
    f[f"discount_mean_7_off{MIN_LAG}"] = disc_recent
    f["discount_vs_recent"] = arrays["discount"] - disc_recent
    f[f"activity_share_28_off{MIN_LAG}"] = rolling_mean(arrays["activity_flag"], 28, MIN_LAG)

    if weather_mode == "oracle":
        for c in WEATHER_COLS:
            f[f"oracle_{c}"] = arrays[c]
    return f


def mask_realized(arrays: dict[str, np.ndarray], grid: pd.DatetimeIndex, first_masked_date) -> dict[str, np.ndarray]:
    """Copy of arrays with realized columns set to NaN from first_masked_date onward."""
    idx = int(np.searchsorted(grid.to_numpy(), np.datetime64(pd.Timestamp(first_masked_date))))
    out = dict(arrays)
    for c in REALIZED_COLS:
        if c in out:
            a = out[c].copy()
            a[:, idx:] = np.nan
            out[c] = a
    return out


def panel_value_cols(weather_mode: str = "none") -> list[str]:
    cols = [TARGET_COL, STOCK_COL] + list(KNOWN_FUTURE_COLS)
    if weather_mode == "oracle":
        cols += list(WEATHER_COLS)
    return cols


def build_features(
    df: pd.DataFrame,
    weather_mode: str = "none",
    first_masked_date=None,
    rows_from_date=None,
) -> pd.DataFrame:
    """Long feature frame: keys, dt, feature_columns(weather_mode).

    first_masked_date: realized columns from this date on are removed before any
    feature is computed. rows_from_date: only rows on or after this date are returned.
    """
    keys, grid, arrays = to_panel(df, panel_value_cols(weather_mode))
    if first_masked_date is not None:
        arrays = mask_realized(arrays, grid, first_masked_date)
    feats = build_feature_arrays(arrays, grid, weather_mode)

    static = df.drop_duplicates(KEY_COLS)[STATIC_COLS].copy()
    static = keys.merge(static, on=KEY_COLS, how="left", validate="one_to_one")
    if static[STATIC_COLS].isna().any().any():
        raise PanelError("static attributes missing for some series")

    start = 0
    if rows_from_date is not None:
        start = int(np.searchsorted(grid.to_numpy(), np.datetime64(pd.Timestamp(rows_from_date))))
    n_series = len(keys)
    days = grid[start:]
    n_days = len(days)

    out = {}
    for c in STATIC_COLS:
        out[c] = np.repeat(static[c].to_numpy(), n_days)
    out[DATE_COL] = np.tile(days.to_numpy(), n_series)
    for name in feature_columns(weather_mode):
        if name in STATIC_COLS:
            continue
        out[name] = np.ascontiguousarray(feats[name][:, start:]).reshape(-1).astype(np.float32)
    frame = pd.DataFrame(out)
    return frame[STATIC_COLS + [DATE_COL] + [c for c in feature_columns(weather_mode) if c not in STATIC_COLS]]
