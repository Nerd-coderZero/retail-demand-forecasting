import sys
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path(__file__).resolve().parent
for _candidate in (_here, _here.parent / "src"):
    if (_candidate / "demand_features.py").exists():
        sys.path.insert(0, str(_candidate))
        break

import demand_eval as de
import demand_features as dfe


def synthetic_long(n_stores=3, n_products=4, n_days=70, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for s in range(n_stores):
        for p in range(n_products):
            rows.append(
                pd.DataFrame(
                    {
                        "city_id": s % 2,
                        "store_id": s,
                        "management_group_id": p % 3,
                        "first_category_id": p % 2,
                        "second_category_id": p % 3,
                        "third_category_id": p,
                        "product_id": p,
                        "dt": dates.strftime("%Y-%m-%d"),
                        "sale_amount": rng.gamma(2.0, 1.0, n_days).round(1),
                        "stock_hour6_22_cnt": rng.integers(0, 17, n_days),
                        "discount": rng.uniform(0.5, 1.0, n_days).round(3),
                        "activity_flag": rng.integers(0, 2, n_days),
                        "holiday_flag": (dates.dayofweek >= 5).astype(int),
                        "precpt": rng.uniform(0, 10, n_days),
                        "avg_temperature": rng.uniform(15, 30, n_days),
                        "avg_humidity": rng.uniform(40, 90, n_days),
                        "avg_wind_level": rng.uniform(0, 4, n_days),
                    }
                )
            )
    df = pd.concat(rows, ignore_index=True)
    first = (df["store_id"] == 0) & (df["product_id"] == 0)
    day = np.arange(first.sum())
    df.loc[first, "sale_amount"] = np.where(day >= len(day) - 50, 0.1, np.round(rng.gamma(5.0, 7.3, first.sum()), 1))
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _features(df, mode="none"):
    keys, grid, arrays = dfe.to_panel(df, dfe.panel_value_cols(mode))
    return keys, grid, arrays, dfe.build_feature_arrays(arrays, grid, mode)


def test_lag_and_window_values():
    df = synthetic_long()
    keys, grid, arrays, f = _features(df)
    y = arrays["sale_amount"]
    t = 40
    assert np.allclose(f["sales_lag_7"][:, t], y[:, t - 7])
    assert np.allclose(f["sales_lag_28"][:, t], y[:, t - 28])
    assert np.allclose(f["sales_mean_7_off7"][:, t], y[:, t - 13 : t - 6].mean(axis=1))
    assert np.allclose(f["sales_mean_28_off7"][:, t], y[:, t - 34 : t - 6].mean(axis=1))
    assert np.allclose(f["sales_std_7_off7"][:, t], y[:, t - 13 : t - 6].std(axis=1), atol=1e-6)
    assert np.isnan(f["sales_mean_28_off7"][:, 33]).all()
    assert np.isfinite(f["sales_mean_28_off7"][:, 34]).all()


def test_target_perturbation_does_not_reach_next_six_days():
    df = synthetic_long()
    keys, grid, arrays, base = _features(df)
    d = 45
    pert = dict(arrays)
    pert["sale_amount"] = arrays["sale_amount"].copy()
    pert["sale_amount"][:, d] += 100.0
    f = dfe.build_feature_arrays(pert, grid, "none")
    for name in base:
        assert np.array_equal(base[name][:, d : d + 7], f[name][:, d : d + 7], equal_nan=True), name
    changed = [n for n in base if not np.array_equal(base[n][:, d + 7], f[n][:, d + 7], equal_nan=True)]
    assert "sales_lag_7" in changed and "sales_mean_7_off7" in changed


def test_same_day_realized_columns_are_not_features():
    df = synthetic_long()
    keys, grid, arrays, base = _features(df, "none")
    d = 50
    pert = {k: v.copy() for k, v in arrays.items()}
    pert["stock_hour6_22_cnt"][:, d:] = 16.0
    pert["sale_amount"][:, d:] = 999.0
    f = dfe.build_feature_arrays(pert, grid, "none")
    for name in base:
        assert np.array_equal(base[name][:, d : d + 7], f[name][:, d : d + 7], equal_nan=True), name


def test_oracle_weather_is_the_only_mode_reading_same_day_weather():
    df = synthetic_long()
    keys, grid, arrays, base = _features(df, "oracle")
    pert = {k: v.copy() for k, v in arrays.items()}
    pert["precpt"][:, 50] += 5.0
    f = dfe.build_feature_arrays(pert, grid, "oracle")
    assert not np.array_equal(base["oracle_precpt"][:, 50], f["oracle_precpt"][:, 50])
    assert not any(c.startswith("oracle_") for c in dfe.feature_columns("none"))


def test_masking_matches_unmasked_for_horizon_rows():
    df = synthetic_long()
    first_masked = pd.Timestamp("2024-01-01") + pd.Timedelta(days=63)
    full = dfe.build_features(df, rows_from_date=first_masked)
    masked = dfe.build_features(df, first_masked_date=first_masked, rows_from_date=first_masked)
    pd.testing.assert_frame_equal(full, masked, check_exact=True)


def test_truncated_history_gives_identical_horizon_features():
    df = synthetic_long(n_days=200)
    horizon_start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=193)
    full = dfe.build_features(df, rows_from_date=horizon_start)
    keep_from = horizon_start - pd.Timedelta(days=dfe.REQUIRED_HISTORY_DAYS)
    short = df[pd.to_datetime(df["dt"]) >= keep_from]
    trunc = dfe.build_features(short, rows_from_date=horizon_start)
    pd.testing.assert_frame_equal(full, trunc, check_exact=True)
    too_short = df[pd.to_datetime(df["dt"]) >= keep_from + pd.Timedelta(days=1)]
    trunc2 = dfe.build_features(too_short, rows_from_date=horizon_start)
    assert trunc2.isna().sum().sum() > full.isna().sum().sum()


def test_incomplete_panel_raises():
    df = synthetic_long()
    try:
        dfe.to_panel(df.iloc[1:], dfe.panel_value_cols())
    except dfe.PanelError:
        return
    raise AssertionError("expected PanelError")


def test_baselines_ignore_data_after_origin():
    df = synthetic_long()
    keys, grid, arrays = dfe.to_panel(df, dfe.panel_value_cols())
    y = arrays["sale_amount"]
    origin = 55
    pert = y.copy()
    pert[:, origin + 1 :] = -1.0e6
    for name in de.BASELINES:
        a = de.baseline_forecast(y, origin, name)
        b = de.baseline_forecast(pert, origin, name)
        assert np.array_equal(a, b), name
    assert np.array_equal(de.baseline_forecast(y, origin, "seasonal_naive_7")[:, 0], y[:, origin - 6])


def test_metrics_values():
    m = de.point_metrics([1.0, 3.0], [2.0, 1.0])
    assert np.isclose(m["wape"], 3.0 / 4.0)
    assert np.isclose(m["wpe"], -1.0 / 4.0)
    assert np.isclose(m["mae"], 1.5)
    assert np.isclose(m["rmse"], np.sqrt(2.5))


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print(f"PASS {n}")
    print(f"{len(names)} passed")
