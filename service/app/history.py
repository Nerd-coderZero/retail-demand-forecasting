"""Recent-history snapshot and per-request feature construction.

The snapshot holds the last REQUIRED_HISTORY_DAYS of every series. A request names series, dates
and the known-in-advance columns; features are built with the same demand_features code path used
in training, so serving features cannot drift from training features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import demand_features as dfe

PANEL_COLS = dfe.panel_value_cols("none")


class HistoryError(ValueError):
    pass


@dataclass(frozen=True)
class Defaults:
    activity_flag: int = 0
    discount: float = float("nan")


class HistoryStore:
    def __init__(self, frame: pd.DataFrame, horizon_days: int = dfe.HORIZON, history_days: int = dfe.REQUIRED_HISTORY_DAYS):
        frame = frame.copy()
        frame[dfe.DATE_COL] = pd.to_datetime(frame[dfe.DATE_COL]).dt.normalize()
        frame = frame.sort_values(dfe.KEY_COLS + [dfe.DATE_COL]).reset_index(drop=True)
        self.dates = pd.DatetimeIndex(sorted(frame[dfe.DATE_COL].unique()))
        if len(self.dates) != history_days:
            raise HistoryError(f"snapshot must hold exactly {history_days} days, found {len(self.dates)}")
        keys = frame[dfe.KEY_COLS].drop_duplicates().reset_index(drop=True)
        self.n_series = len(keys)
        if len(frame) != self.n_series * history_days:
            raise HistoryError("snapshot is not a complete panel")
        self.keys = keys
        self.index = {(int(s), int(p)): i for i, (s, p) in enumerate(zip(keys["store_id"], keys["product_id"]))}
        self.static = {c: frame[c].to_numpy()[:: history_days] for c in dfe.STATIC_COLS}
        self.values = {c: frame[c].to_numpy(dtype=np.float64).reshape(self.n_series, history_days) for c in PANEL_COLS}
        self.history_days = history_days
        self.horizon_days = horizon_days
        self.horizon_dates = pd.DatetimeIndex(
            [self.dates[-1] + pd.Timedelta(days=i + 1) for i in range(horizon_days)]
        )
        self.horizon_holiday = (self.horizon_dates.dayofweek >= 5).astype(np.float64)

    def series_row(self, store_id: int, product_id: int) -> int:
        row = self.index.get((int(store_id), int(product_id)))
        if row is None:
            raise HistoryError(f"unknown series store_id={store_id} product_id={product_id}")
        return row

    def date_position(self, date: pd.Timestamp) -> int:
        matches = np.flatnonzero(self.horizon_dates == date)
        if not len(matches):
            first, last = self.horizon_dates[0].date(), self.horizon_dates[-1].date()
            raise HistoryError(f"date {date.date()} outside the forecast window {first}..{last}")
        return int(matches[0])

    def build_features(self, rows: list[int], known: dict[tuple[int, int], dict] | None = None) -> pd.DataFrame:
        """Features for every horizon day of the given series rows, in (series, date) order."""
        rows = list(dict.fromkeys(rows))
        n, h, d = len(rows), self.horizon_days, self.history_days
        total_days = d + h
        idx = np.asarray(rows, dtype=np.int64)
        frame = {c: np.repeat(self.static[c][idx], total_days) for c in dfe.STATIC_COLS}
        frame[dfe.DATE_COL] = np.tile(
            np.concatenate([self.dates.to_numpy(), self.horizon_dates.to_numpy()]), n
        )
        for c in PANEL_COLS:
            history = self.values[c][idx]
            future = np.full((n, h), np.nan)
            if c == "holiday_flag":
                future[:] = self.horizon_holiday
            elif c == "activity_flag":
                future[:] = Defaults.activity_flag
            frame[c] = np.concatenate([history, future], axis=1).reshape(-1)
        long = pd.DataFrame(frame)
        if known:
            offsets = {(i, j): i * total_days + d + j for i in range(n) for j in range(h)}
            row_of = {r: i for i, r in enumerate(rows)}
            for (series_row, day), overrides in known.items():
                position = offsets[(row_of[series_row], day)]
                for column, value in overrides.items():
                    long.loc[position, column] = value
        features = dfe.build_features(long, weather_mode="none", rows_from_date=self.horizon_dates[0])
        return features
