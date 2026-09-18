"""Assemble notebooks/01_data_features_baselines.ipynb from src/ and tests/ so the
module code inside the notebook is byte-identical to the source files."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


def writefile(name, path):
    code(f"%%writefile {name}\n" + (ROOT / path).read_text())


md("""
# Notebook 1: data validation, leakage-safe features, baselines

Project 4, retail demand forecasting on FreshRetailNet-50K (CC BY 4.0, Dingdong Inc., arXiv 2505.16319).
Target: daily `sale_amount` per store-product series. Forecast: 7 days ahead from a fixed origin.

This notebook:
1. Downloads the dataset at a pinned Hugging Face revision and verifies SHA-256 checksums.
2. Validates structure: row counts, nulls, panel completeness, date contiguity, static attributes.
3. Verifies the meaning of the hourly and stockout columns against each other.
4. Profiles demand (weekday, stockout, holiday, promotion, discount effects).
5. Defines the time-based split and writes the leakage-safe feature and evaluation modules.
6. Runs leakage tests: synthetic unit tests, then invariance tests on the real validation horizon.
7. Scores three baselines on the validation horizon with WAPE, WPE, MAE, RMSE and error breakdowns.

Not in this notebook: model training (notebook 2) and any scoring on the official eval split.
The eval split is loaded only for structural checks; its targets are not used for any metric here.

Runtime: CPU only, no GPU needed. Kaggle setting required: Internet on.
""")

md("## 1. Configuration")
code("""
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DATASET_REPO = "Dingdong-Inc/FreshRetailNet-50K"
DATASET_REVISION = "08c1fab7f9257bc73679d415d65d644165d351d4"
DATASET_FILES = {
    "train": ("data/train.parquet", "6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d"),
    "eval": ("data/eval.parquet", "1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e"),
}
CACHE_DIR = Path.home() / ".cache" / "freshretailnet-50k" / DATASET_REVISION
KAGGLE_WORKING = Path("/kaggle/working")
OUTPUT_DIR = KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path.cwd() / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODULE_DIR = Path.cwd()
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

EXPECTED_TRAIN_ROWS = 4_500_000
EXPECTED_EVAL_ROWS = 350_000
EXPECTED_SERIES = 50_000
HORIZON_DAYS = 7
SEED = 20240626
LIST_COLS = ["hours_sale", "hours_stock_status"]

environment = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "pyarrow": pa.__version__,
    "matplotlib": matplotlib.__version__,
    "on_kaggle": KAGGLE_WORKING.exists(),
}
print(json.dumps(environment, indent=2))
print("output dir:", OUTPUT_DIR)
""")

md("""
## 2. Download at a pinned revision

The revision hash fixes the dataset version; the SHA-256 check fails the notebook if the bytes differ.
""")
code("""
def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(split):
    rel, expected = DATASET_FILES[split]
    dest = CACHE_DIR / rel
    if not (dest.exists() and sha256_of(dest) == expected):
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}/{rel}"
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out, 1 << 20)
        tmp.replace(dest)
    actual = sha256_of(dest)
    assert actual == expected, f"{split}: sha256 mismatch {actual}"
    print(f"{split}: {dest} {dest.stat().st_size:,} bytes, sha256 ok")
    return dest


t0 = time.time()
TRAIN_PATH = fetch("train")
EVAL_PATH = fetch("eval")
print(f"{time.time() - t0:.1f}s")
""")

md("## 3. Load and validate structure")
code("""
profile = {"dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION}}


def check(name, condition, detail=None):
    profile.setdefault("checks", {})[name] = {"passed": bool(condition), "detail": detail}
    print(("PASS " if condition else "FAIL ") + name + ("" if detail is None else f" | {detail}"))
    assert condition, name


train_schema = pq.read_schema(TRAIN_PATH)
eval_schema = pq.read_schema(EVAL_PATH)
check("train and eval schemas identical", train_schema.equals(eval_schema, check_metadata=False))
scalar_cols = [c for c in train_schema.names if c not in LIST_COLS]

train = pd.read_parquet(TRAIN_PATH, columns=scalar_cols)
evald = pd.read_parquet(EVAL_PATH, columns=scalar_cols)
check("train row count", len(train) == EXPECTED_TRAIN_ROWS, len(train))
check("eval row count", len(evald) == EXPECTED_EVAL_ROWS, len(evald))
check("no nulls in train scalar columns", int(train.isna().sum().sum()) == 0)
check("no nulls in eval scalar columns (including eval sale_amount)", int(evald.isna().sum().sum()) == 0)

key = ["store_id", "product_id"]
for name, frame in (("train", train), ("eval", evald)):
    check(f"{name}: no duplicate (store, product, dt)", not frame.duplicated(key + ["dt"]).any())

train_series = train[key].drop_duplicates()
eval_series = evald[key].drop_duplicates()
check("series count train", len(train_series) == EXPECTED_SERIES, len(train_series))
check("same series set in train and eval",
      len(train_series.merge(eval_series, on=key, how="inner")) == EXPECTED_SERIES)

train_dates = pd.DatetimeIndex(sorted(pd.to_datetime(train["dt"].unique())))
eval_dates = pd.DatetimeIndex(sorted(pd.to_datetime(evald["dt"].unique())))
check("train dates contiguous", train_dates.equals(pd.date_range(train_dates[0], train_dates[-1], freq="D")),
      f"{train_dates[0].date()}..{train_dates[-1].date()} ({len(train_dates)} days)")
check("eval dates contiguous", eval_dates.equals(pd.date_range(eval_dates[0], eval_dates[-1], freq="D")),
      f"{eval_dates[0].date()}..{eval_dates[-1].date()} ({len(eval_dates)} days)")
check("eval horizon equals forecast horizon", len(eval_dates) == HORIZON_DAYS)
check("eval starts the day after train ends", eval_dates[0] == train_dates[-1] + pd.Timedelta(days=1))
check("complete train panel", (train.groupby(key).size() == len(train_dates)).all())
check("complete eval panel", (evald.groupby(key).size() == len(eval_dates)).all())

both = pd.concat([train, evald], ignore_index=True)
static_cols = ["city_id", "management_group_id", "first_category_id", "second_category_id", "third_category_id"]
check("static attributes constant per series",
      int(both.groupby(key)[static_cols].nunique().max().max()) == 1)
check("city constant per store", int(both.groupby("store_id")["city_id"].nunique().max()) == 1)
check("category and management group constant per product",
      int(both.groupby("product_id")[["third_category_id", "management_group_id"]].nunique().max().max()) == 1)
check("holiday_flag constant per date", int(both.groupby("dt")["holiday_flag"].nunique().max()) == 1)
check("weather constant per store and date",
      int(both.groupby(["store_id", "dt"])[["precpt", "avg_temperature"]].nunique().max().max()) == 1)
check("sale_amount non-negative", bool((both["sale_amount"] >= 0).all()))
check("stock_hour6_22_cnt within 0..16", bool(both["stock_hour6_22_cnt"].between(0, 16).all()))
check("activity_flag and holiday_flag binary",
      set(both["activity_flag"].unique()) <= {0, 1} and set(both["holiday_flag"].unique()) <= {0, 1})

profile["shape"] = {
    "train_rows": len(train), "eval_rows": len(evald), "series": len(train_series),
    "stores": int(both["store_id"].nunique()), "products": int(both["product_id"].nunique()),
    "cities": int(both["city_id"].nunique()),
    "train_dates": [str(train_dates[0].date()), str(train_dates[-1].date())],
    "eval_dates": [str(eval_dates[0].date()), str(eval_dates[-1].date())],
}
print(json.dumps(profile["shape"], indent=2))
del both
""")

md("""
## 4. Hourly and stockout column semantics

Checked over every training row, in batches to bound memory:
- `sale_amount` equals the sum of the 24 values in `hours_sale`.
- `stock_hour6_22_cnt` equals the number of 1s in `hours_stock_status` for hours 6..21 (16 hours).
- Hours with status 1 have lower sales than hours with status 0, which identifies 1 as out of stock.
""")
code("""
rows = 0
sum_mismatch = 0
count_mismatch = 0
bad_lengths = 0
status_values = set()
sales_by_status = np.zeros(2)
hours_by_status = np.zeros(2)
pf = pq.ParquetFile(TRAIN_PATH)
for batch in pf.iter_batches(batch_size=500_000, columns=["sale_amount", "stock_hour6_22_cnt"] + LIST_COLS):
    lengths_ok = all(
        pc.min_max(pc.list_value_length(batch.column(c))).as_py() == {"min": 24, "max": 24} for c in LIST_COLS
    )
    bad_lengths += 0 if lengths_ok else 1
    hs = batch.column("hours_sale").flatten().to_numpy(zero_copy_only=False).reshape(-1, 24)
    st = batch.column("hours_stock_status").flatten().to_numpy(zero_copy_only=False).reshape(-1, 24)
    sa = batch.column("sale_amount").to_numpy()
    cnt = batch.column("stock_hour6_22_cnt").to_numpy()
    sum_mismatch += int((~np.isclose(hs.sum(axis=1), sa, atol=1e-6)).sum())
    count_mismatch += int((st[:, 6:22].sum(axis=1) != cnt).sum())
    status_values |= set(np.unique(st).tolist())
    for s in (0, 1):
        mask = st == s
        sales_by_status[s] += hs[mask].sum()
        hours_by_status[s] += mask.sum()
    rows += len(sa)
    del hs, st

mean_hourly = sales_by_status / hours_by_status
check("all hourly lists have 24 values", bad_lengths == 0)
check("sale_amount equals sum of hours_sale", sum_mismatch == 0, f"{sum_mismatch} mismatches in {rows:,} rows")
check("stock_hour6_22_cnt equals stockout hours 6..21", count_mismatch == 0, f"{count_mismatch} mismatches")
check("stock status values are 0/1", status_values <= {0, 1}, sorted(status_values))
check("status 1 hours sell less than status 0 hours", mean_hourly[1] < mean_hourly[0],
      f"mean hourly sales status0={mean_hourly[0]:.4f} status1={mean_hourly[1]:.4f}")
profile["hourly_semantics"] = {"rows_checked": rows, "mean_hourly_sales_in_stock": float(mean_hourly[0]),
                               "mean_hourly_sales_out_of_stock": float(mean_hourly[1])}
""")

md("## 5. Demand profile (training period only)")
code("""
tr = train.assign(date=pd.to_datetime(train["dt"]))
eda = {
    "zero_sales_share": float((tr["sale_amount"] == 0).mean()),
    "rows_with_any_stockout_hour_share": float((tr["stock_hour6_22_cnt"] > 0).mean()),
    "rows_full_day_stockout_share": float((tr["stock_hour6_22_cnt"] == 16).mean()),
    "activity_flag_share": float(tr["activity_flag"].mean()),
    "holiday_date_count": int(tr.groupby("dt")["holiday_flag"].first().sum()),
    "mean_sales_by_day_of_week": tr.groupby(tr["date"].dt.dayofweek)["sale_amount"].mean().round(4).to_dict(),
    "mean_sales_by_holiday_flag": tr.groupby("holiday_flag")["sale_amount"].mean().round(4).to_dict(),
    "mean_sales_by_activity_flag": tr.groupby("activity_flag")["sale_amount"].mean().round(4).to_dict(),
    "mean_sales_by_stockout_hours": tr.groupby("stock_hour6_22_cnt")["sale_amount"].mean().round(4).to_dict(),
    "sale_amount_quantiles": tr["sale_amount"].quantile([0.5, 0.9, 0.99, 1.0]).round(3).to_dict(),
}
disc_bins = pd.cut(tr["discount"], [-0.001, 0.5, 0.7, 0.9, 0.99, 1.0, np.inf],
                   labels=["<=0.5", "0.5-0.7", "0.7-0.9", "0.9-0.99", "0.99-1.0", ">1.0"])
eda["mean_sales_by_discount_bin"] = tr.groupby(disc_bins, observed=True)["sale_amount"].mean().round(4).to_dict()
eda["rows_by_discount_bin"] = disc_bins.value_counts(sort=False).to_dict()
profile["eda"] = eda
for k, v in eda.items():
    print(f"{k}: {v}")
""")
code("""
SURFACE, INK, INK_2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

dow = tr.groupby(tr["date"].dt.dayofweek)["sale_amount"].mean()
axes[0].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow.values, color=SERIES, width=0.6)
axes[0].set_title("Mean daily sales by day of week", loc="left")
axes[0].set_ylabel("normalized sales per series-day")

soh = tr.groupby("stock_hour6_22_cnt")["sale_amount"].mean()
axes[1].bar(soh.index, soh.values, color=SERIES, width=0.7)
axes[1].set_title("Mean daily sales by stockout hours (6-22h)", loc="left")
axes[1].set_xlabel("stockout hours in the day")
axes[1].set_xticks(range(0, 17, 2))

daily = tr.groupby("date").agg(total=("sale_amount", "sum"), holiday=("holiday_flag", "first"))
axes[2].plot(daily.index, daily["total"], color=SERIES, linewidth=2)
hol = daily[daily["holiday"] == 1]
axes[2].scatter(hol.index, hol["total"], s=18, color=INK_2, zorder=3, label="holiday_flag = 1")
axes[2].set_title("Total daily sales, training period", loc="left")
axes[2].legend(frameon=False, loc="upper left")
plt.setp(axes[2].get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "nb1_demand_profile.png", dpi=130)
plt.show()
""")

md("""
Reading the stockout panel: rows with 1 to 4 stockout hours have higher mean sales than rows with none, because
fast-selling items are the ones that run out. Sales only fall as stockout hours grow toward a full day. Stockout
hours therefore carry both a demand signal and a censoring signal, which is why they enter the model only as lagged
history and never for the target day.
""")

md("""
## 6. Which same-day columns are known in advance

| Column | Treated as | Basis |
|---|---|---|
| `holiday_flag` | known in advance | constant per date across all stores (checked above); calendar |
| `activity_flag` | known in advance | planned promotion; same assumption as the dataset paper |
| `discount` | known in advance, **unverified** | see evidence below |
| `stock_hour6_22_cnt`, `hours_stock_status` | realized, never a same-day feature | stock state is only observed after the day |
| weather columns | realized, excluded by default | actual weather, not a forecast; available only in the labelled `oracle` mode |
| `sale_amount`, `hours_sale` | target | |

If `discount` were computed from transactions (price paid over list price), it would carry same-day information.
The cell below prints the evidence available in the data. The known-in-advance treatment follows the dataset paper,
and a with/without same-day discount comparison belongs in notebook 2.
""")
code("""
zero_day = train["sale_amount"] == 0
full_stockout = train["stock_hour6_22_cnt"] == 16
discount_evidence = {
    "share_discount_eq_1_on_zero_sale_days": float((train.loc[zero_day, "discount"] == 1).mean()),
    "share_discount_eq_1_on_positive_sale_days": float((train.loc[~zero_day, "discount"] == 1).mean()),
    "share_discount_eq_1_on_full_stockout_days": float((train.loc[full_stockout, "discount"] == 1).mean()),
    "rows_discount_eq_0": int((train["discount"] == 0).sum()),
    "rows_discount_gt_1": int((train["discount"] > 1).sum()),
    "distinct_discount_values_per_series_median": float(train.groupby(key)["discount"].nunique().median()),
}
hol_by_date = pd.concat([train, evald]).groupby("dt")["holiday_flag"].first()
hol_by_date.index = pd.to_datetime(hol_by_date.index)
calendar_evidence = {
    "weekday_dates_flagged_holiday": [str(d.date()) for d in hol_by_date[(hol_by_date == 1) & (hol_by_date.index.dayofweek < 5)].index],
    "weekend_dates_not_flagged": [str(d.date()) for d in hol_by_date[(hol_by_date == 0) & (hol_by_date.index.dayofweek >= 5)].index],
}
profile["discount_evidence"] = discount_evidence
profile["calendar_evidence"] = calendar_evidence
print(json.dumps(discount_evidence, indent=2))
print(json.dumps(calendar_evidence, indent=2))
""")

md("""
## 7. Time-based split

Random splits put later days into training and earlier days into testing, which lets the model learn from the
future it is scored on. All splits here are by date:

| Window | Dates | Use |
|---|---|---|
| fit | first 83 training days | feature history and model fitting |
| validation | last 7 training days | model selection and baseline comparison (this notebook) |
| eval | official 7-day eval split | scored once, at the end, for the final report |

Forecasts are made from a fixed origin (the last day before the 7-day window) for all 7 days at once.
""")
code("""
VALIDATION_START = train_dates[-HORIZON_DAYS]
VALIDATION_ORIGIN = VALIDATION_START - pd.Timedelta(days=1)
split = {
    "fit": [str(train_dates[0].date()), str(VALIDATION_ORIGIN.date())],
    "validation": [str(VALIDATION_START.date()), str(train_dates[-1].date())],
    "eval": [str(eval_dates[0].date()), str(eval_dates[-1].date())],
    "horizon_days": HORIZON_DAYS,
}
profile["split"] = split
print(json.dumps(split, indent=2))
""")

md("""
## 8. Feature and evaluation modules

Written as files so notebook 2 and the serving container import the same code (no train/serve drift).

Feature rule: every history feature for target day `t` reads only days `<= t - 7`. With a 7-day horizon, one feature
definition is valid for every horizon step from a single origin (direct strategy, no recursive feeding of
predictions). Cost: the nearest horizon days do not use the most recent 6 days of history.
""")
writefile("demand_features.py", "src/demand_features.py")
writefile("demand_eval.py", "src/demand_eval.py")
writefile("test_demand_features.py", "tests/test_demand_features.py")

md("### 8.1 Unit tests on a synthetic panel")
code("""
result = subprocess.run([sys.executable, "test_demand_features.py"], capture_output=True, text=True, cwd=MODULE_DIR)
print(result.stdout)
print(result.stderr[-3000:])
check("synthetic unit tests pass", result.returncode == 0)
""")

md("""
## 9. Leakage invariance tests on the real validation horizon

Three versions of the training panel are built:
- `original`: data as downloaded.
- `masked`: target, stockout hours and weather set to NaN for all 7 validation days.
- `perturbed`: the same columns replaced with random values for all 7 validation days.

Features for the validation days must be identical in all three. A positive control then changes the target on the
origin day only and confirms that `sales_lag_7` on the last validation day does change, so the test can fail.
The serving-parity test rebuilds validation features from only the last `REQUIRED_HISTORY_DAYS` of history.
""")
code("""
import importlib

import demand_eval as de
import demand_features as dfe

importlib.reload(dfe)
importlib.reload(de)

keys, grid, arrays = dfe.to_panel(train, dfe.panel_value_cols("oracle"))
v0 = int(np.searchsorted(grid.to_numpy(), VALIDATION_START.to_datetime64()))
check("validation window indices", grid[v0] == VALIDATION_START and len(grid) - v0 == HORIZON_DAYS, (v0, len(grid)))

rng = np.random.default_rng(SEED)
masked = dfe.mask_realized(arrays, grid, VALIDATION_START)
perturbed = {k: v.copy() for k, v in arrays.items()}
for c in dfe.REALIZED_COLS:
    perturbed[c][:, v0:] = rng.uniform(0, 100, size=perturbed[c][:, v0:].shape)

def horizon_slices(feature_arrays):
    return {c: np.ascontiguousarray(a[:, v0:]) for c, a in feature_arrays.items()}


for mode in dfe.WEATHER_MODES:
    ref = horizon_slices(dfe.build_feature_arrays(arrays, grid, mode))
    oracle_cols = [c for c in ref if c.startswith("oracle_")]
    history_cols = [c for c in ref if c not in oracle_cols]
    for variant_name, variant in (("masked", masked), ("perturbed", perturbed)):
        other = horizon_slices(dfe.build_feature_arrays(variant, grid, mode))
        identical = all(np.array_equal(ref[c], other[c], equal_nan=True) for c in history_cols)
        check(f"[{mode}] validation-day features unchanged when realized validation data is {variant_name}",
              identical, f"{len(history_cols)} feature arrays compared")
        if oracle_cols and variant_name == "perturbed":
            differs = all(not np.array_equal(ref[c], other[c], equal_nan=True) for c in oracle_cols)
            check("[oracle] oracle weather columns do read same-day weather (labelled exception)", differs)
        del other
    del ref

control = {k: v.copy() for k, v in arrays.items()}
control["sale_amount"][:, v0 - 1] += 1000.0
ref = horizon_slices(dfe.build_feature_arrays(arrays, grid, "none"))
ctl = horizon_slices(dfe.build_feature_arrays(control, grid, "none"))
check("positive control: origin-day change reaches sales_lag_7 on the last validation day",
      not np.array_equal(ref["sales_lag_7"][:, -1], ctl["sales_lag_7"][:, -1]))
check("positive control: origin-day change does not reach any feature on the first validation day",
      all(np.array_equal(ref[c][:, 0], ctl[c][:, 0], equal_nan=True) for c in ref))
del ctl, ref, control, masked, perturbed

full_val = dfe.build_features(train, first_masked_date=VALIDATION_START, rows_from_date=VALIDATION_START)
short_hist = train[pd.to_datetime(train["dt"]) >= VALIDATION_START - pd.Timedelta(days=dfe.REQUIRED_HISTORY_DAYS)]
short_val = dfe.build_features(short_hist, first_masked_date=VALIDATION_START, rows_from_date=VALIDATION_START)
pd.testing.assert_frame_equal(full_val, short_val, check_exact=True)
check("serving parity: last REQUIRED_HISTORY_DAYS of history reproduce validation features exactly",
      True, f"REQUIRED_HISTORY_DAYS={dfe.REQUIRED_HISTORY_DAYS}, rows={len(full_val):,}")
del short_hist, short_val
""")
code("""
feature_nan_share = full_val[dfe.feature_columns("none")].isna().mean().round(4)
print("validation-day feature NaN share (non-zero only):")
print(feature_nan_share[feature_nan_share > 0].to_string() if (feature_nan_share > 0).any() else "none")
print(full_val.head(3).T)
feature_spec = {
    "horizon_days": dfe.HORIZON,
    "min_lag_days": dfe.MIN_LAG,
    "required_history_days": dfe.REQUIRED_HISTORY_DAYS,
    "weather_modes": list(dfe.WEATHER_MODES),
    "feature_columns": {m: dfe.feature_columns(m) for m in dfe.WEATHER_MODES},
    "known_future_columns": dfe.KNOWN_FUTURE_COLS,
    "realized_columns_never_same_day": dfe.REALIZED_COLS,
    "unverified_assumptions": ["discount is known in advance for the target day"],
}
""")

md("""
## 10. Baselines on the validation horizon

- `seasonal_naive_7`: value from the same weekday one week earlier.
- `moving_average_7`: mean of the 7 days up to the origin, repeated for all horizon days.
- `same_dow_mean_4w`: mean of the same weekday over the previous 4 weeks.

Metrics: WAPE = sum|pred - actual| / sum actual; WPE = sum(pred - actual) / sum actual (negative means
under-forecast); MAE; RMSE. Actuals on stockout days are censored (sales capped by availability), so errors are also
broken down by stockout hours on the target day. That column is used only to slice the errors, never as an input.
""")
code("""
y = arrays["sale_amount"]
origin_idx = v0 - 1
actual = y[:, v0:]
horizon_dates = grid[v0:]

val_frame = pd.DataFrame({
    "store_id": np.repeat(keys["store_id"].to_numpy(), HORIZON_DAYS),
    "product_id": np.repeat(keys["product_id"].to_numpy(), HORIZON_DAYS),
    "dt": np.tile(horizon_dates.to_numpy(), len(keys)),
    "horizon_day": np.tile(np.arange(1, HORIZON_DAYS + 1), len(keys)),
    "actual": actual.reshape(-1),
    "stockout_hours": arrays["stock_hour6_22_cnt"][:, v0:].reshape(-1),
    "activity_flag": arrays["activity_flag"][:, v0:].reshape(-1).astype(int),
})
val_frame = val_frame.merge(
    train.drop_duplicates(key)[key + ["first_category_id", "city_id"]], on=key, how="left", validate="many_to_one"
)
val_frame["stockout_bucket"] = de.stockout_bucket(val_frame["stockout_hours"])

baseline_results = {}
overall_rows = []
for name in de.BASELINES:
    pred = de.baseline_forecast(y, origin_idx, name)
    frame = val_frame.assign(pred=pred.reshape(-1))
    overall = de.point_metrics(frame["actual"], frame["pred"])
    overall_rows.append({"baseline": name, **overall})
    baseline_results[name] = {
        "overall": overall,
        "by_horizon_day": de.metrics_by(frame, "horizon_day").to_dict(orient="records"),
        "by_stockout_bucket": de.metrics_by(frame, "stockout_bucket").astype({"stockout_bucket": str}).to_dict(orient="records"),
        "by_activity_flag": de.metrics_by(frame, "activity_flag").to_dict(orient="records"),
        "by_first_category": de.metrics_by(frame, "first_category_id").to_dict(orient="records"),
    }

overall_table = pd.DataFrame(overall_rows).set_index("baseline")
print(overall_table[["wape", "wpe", "mae", "rmse", "n"]].round(4).to_string())
""")
code("""
for name in de.BASELINES:
    print(f"\\n{name}")
    for part in ("by_horizon_day", "by_stockout_bucket", "by_activity_flag"):
        t = pd.DataFrame(baseline_results[name][part])
        print(t.drop(columns=["sum_actual"]).round(4).to_string(index=False))
best = overall_table["wape"].idxmin()
print(f"\\nlowest validation WAPE: {best} ({overall_table.loc[best, 'wape']:.4f})")
cat = pd.DataFrame(baseline_results[best]["by_first_category"]).sort_values("sum_actual", ascending=False)
print("\\nby first_category_id for", best, "(largest 10 by volume)")
print(cat.head(10).round(4).to_string(index=False))
""")

md("""
Reading the breakdowns: on full-day stockout days the recorded actuals are close to zero, so WAPE and WPE for that
bucket divide by a tiny total and are not meaningful; MAE is the comparable number there. The negative WPE across
all baselines means validation-week demand was higher than the history the baselines average over.
""")

md("## 11. Save artifacts")
code("""
for module in ("demand_features.py", "demand_eval.py", "test_demand_features.py"):
    src = MODULE_DIR / module
    if src.resolve() != (OUTPUT_DIR / module).resolve():
        shutil.copy2(src, OUTPUT_DIR / module)

environment["module_sha256"] = {m: sha256_of(MODULE_DIR / m) for m in ("demand_features.py", "demand_eval.py")}
artifacts = {
    "nb1_data_profile.json": profile,
    "nb1_feature_spec.json": feature_spec,
    "nb1_baseline_validation_metrics.json": {"split": split, "baselines": baseline_results},
    "nb1_environment.json": environment,
}
for fname, payload in artifacts.items():
    (OUTPUT_DIR / fname).write_text(json.dumps(payload, indent=2, default=str))
for p in sorted(OUTPUT_DIR.glob("nb1_*")) + [OUTPUT_DIR / "demand_features.py", OUTPUT_DIR / "demand_eval.py"]:
    print(f"{p.name}: {p.stat().st_size:,} bytes")
failed = [k for k, v in profile["checks"].items() if not v["passed"]]
print(f"checks: {len(profile['checks'])} run, {len(failed)} failed")
""")

nb = nbf.v4.new_notebook()
for i, cell in enumerate(cells):
    cell["id"] = f"cell-{i:02d}"
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = ROOT / "notebooks" / "01_data_features_baselines.ipynb"
nbf.write(nb, out)
print(out)
