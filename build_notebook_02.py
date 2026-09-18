"""Assemble notebooks/02_lightgbm_training_evaluation.ipynb from src/ and tests/ so the
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
# Notebook 2: LightGBM training, ablations, final evaluation, exports

Project 4, retail demand forecasting on FreshRetailNet-50K. Continues notebook 1 but runs on its own: it downloads the
same pinned dataset and writes the same feature and evaluation modules.

This notebook:
1. Re-checks the data and confirms the baselines reproduce notebook 1's validation numbers exactly.
2. Builds features once with the serving code path and confirms masking realized data changes nothing.
3. Compares three LightGBM objectives on the validation week (squared error, absolute error, Tweedie).
4. Runs two ablations on the best objective: without same-day discount, and with oracle (actual) weather.
5. Retrains the served configuration on all 90 training days and scores the official eval week once.
6. Breaks errors down by horizon day, stockout hours, promotion flag, category, store and product.
7. Verifies serving parity end to end: predictions from a 34-day history snapshot equal the evaluation predictions.
8. Exports the model files, model card, metrics, parity sample and the serving history snapshot.

Runtime: CPU only. Kaggle setting required: Internet on.
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

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa

DATASET_REPO = "Dingdong-Inc/FreshRetailNet-50K"
DATASET_REVISION = "08c1fab7f9257bc73679d415d65d644165d351d4"
DATASET_FILES = {
    "train": ("data/train.parquet", "6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d"),
    "eval": ("data/eval.parquet", "1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e"),
}
CACHE_DIR = Path.home() / ".cache" / "freshretailnet-50k" / DATASET_REVISION
KAGGLE_WORKING = Path("/kaggle/working")
OUTPUT_DIR = KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path.cwd() / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODULE_DIR = Path.cwd()
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

EXPECTED_TRAIN_ROWS = 4_500_000
EXPECTED_EVAL_ROWS = 350_000
EXPECTED_SERIES = 50_000
HORIZON_DAYS = 7
SEED = 20240626
MAX_BOOST_ROUNDS = 3000
EARLY_STOPPING_ROUNDS = 100
NUM_THREADS = 4
SERVE_SAME_DAY_DISCOUNT = False
PARITY_SAMPLE_ROWS = 2000

NB1_VALIDATION_BASELINES = {
    "seasonal_naive_7": {"wape": 0.4123924065855388, "wpe": -0.11586909447557808, "mae": 0.49337008, "rmse": 0.8283414784702537},
    "moving_average_7": {"wape": 0.36563863424327187, "wpe": -0.11586909447557808, "mae": 0.43743570285714284, "rmse": 0.7387430908984557},
    "same_dow_mean_4w": {"wape": 0.39023538779530315, "wpe": -0.1232945722498342, "mae": 0.4668622928571428, "rmse": 0.7986253394610553},
}

environment = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "pyarrow": pa.__version__,
    "lightgbm": lgb.__version__,
    "on_kaggle": KAGGLE_WORKING.exists(),
}
print(json.dumps(environment, indent=2))
print("output dir:", OUTPUT_DIR)

report = {"checks": {}}


def check(name, condition, detail=None):
    report["checks"][name] = {"passed": bool(condition), "detail": detail}
    print(("PASS " if condition else "FAIL ") + name + ("" if detail is None else f" | {detail}"))
    assert condition, name
""")

md("## 2. Download at the pinned revision")
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
    print(f"{split}: {dest.stat().st_size:,} bytes, sha256 ok")
    return dest


TRAIN_PATH = fetch("train")
EVAL_PATH = fetch("eval")
""")

md("## 3. Feature, evaluation and model modules")
writefile("demand_features.py", "src/demand_features.py")
writefile("demand_eval.py", "src/demand_eval.py")
writefile("demand_model.py", "src/demand_model.py")
writefile("test_demand_features.py", "tests/test_demand_features.py")
writefile("test_demand_model.py", "tests/test_demand_model.py")
code("""
for test_file in ("test_demand_features.py", "test_demand_model.py"):
    result = subprocess.run([sys.executable, test_file], capture_output=True, text=True, cwd=MODULE_DIR)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr[-3000:])
    check(f"{test_file} passes", result.returncode == 0)

import demand_eval as de
import demand_features as dfe
import demand_model as dm
""")

md("## 4. Load, structural re-check, feature build")
code("""
load_cols = list(dict.fromkeys(dfe.STATIC_COLS + [dfe.DATE_COL] + dfe.panel_value_cols("oracle")))
train = pd.read_parquet(TRAIN_PATH, columns=load_cols)
evald = pd.read_parquet(EVAL_PATH, columns=load_cols)
check("train row count", len(train) == EXPECTED_TRAIN_ROWS, len(train))
check("eval row count", len(evald) == EXPECTED_EVAL_ROWS, len(evald))
check("no nulls", int(train.isna().sum().sum()) == 0 and int(evald.isna().sum().sum()) == 0)

combined = pd.concat([train, evald], ignore_index=True)
del train, evald
keys, grid, panel = dfe.to_panel(combined, dfe.panel_value_cols("oracle"))
n_series, n_days = len(keys), len(grid)
check("combined panel is complete", n_series == EXPECTED_SERIES and n_days == 97, (n_series, n_days))

EVAL_START_IDX = n_days - HORIZON_DAYS
VALIDATION_START_IDX = EVAL_START_IDX - HORIZON_DAYS
FIRST_TRAIN_IDX = dfe.REQUIRED_HISTORY_DAYS
EVAL_START, VALIDATION_START = grid[EVAL_START_IDX], grid[VALIDATION_START_IDX]
split = {
    "fit_target_days": [str(grid[FIRST_TRAIN_IDX].date()), str(grid[VALIDATION_START_IDX - 1].date())],
    "validation_days": [str(VALIDATION_START.date()), str(grid[EVAL_START_IDX - 1].date())],
    "final_train_target_days": [str(grid[FIRST_TRAIN_IDX].date()), str(grid[EVAL_START_IDX - 1].date())],
    "eval_days": [str(EVAL_START.date()), str(grid[-1].date())],
    "first_train_day_reason": f"first day with a complete {dfe.REQUIRED_HISTORY_DAYS}-day feature history",
}
check("split dates", split["validation_days"] == ["2024-06-19", "2024-06-25"] and split["eval_days"] == ["2024-06-26", "2024-07-02"], split)
report["split"] = split

t0 = time.time()
features = dfe.build_features(combined, weather_mode="oracle")
print(f"feature build {time.time() - t0:.1f}s, shape {features.shape}")
check("feature rows are series-major and aligned with the panel",
      np.array_equal(features["store_id"].to_numpy(), np.repeat(keys["store_id"].to_numpy(), n_days))
      and np.array_equal(features["product_id"].to_numpy(), np.repeat(keys["product_id"].to_numpy(), n_days))
      and np.array_equal(features[dfe.DATE_COL].to_numpy(), np.tile(grid.to_numpy(), n_series)))

non_oracle = [c for c in dfe.feature_columns("none")]
for label, start in (("validation", VALIDATION_START), ("eval", EVAL_START)):
    end = start + pd.Timedelta(days=HORIZON_DAYS)
    masked = dfe.build_features(combined, weather_mode="none", first_masked_date=start, rows_from_date=start)
    masked = masked[masked[dfe.DATE_COL] < end].reset_index(drop=True)
    in_window = (features[dfe.DATE_COL] >= start) & (features[dfe.DATE_COL] < end)
    unmasked = features.loc[in_window, list(masked.columns)].reset_index(drop=True)
    pd.testing.assert_frame_equal(masked, unmasked, check_exact=True)
    check(f"{label} week features identical with realized data masked from {start.date()}", True, f"{len(masked):,} rows")
    del masked, unmasked
""")
code("""
ALL_MODEL_COLUMNS = dm.model_feature_columns("oracle", True)
X_all = dm.to_matrix(features, ALL_MODEL_COLUMNS)
col_index = {c: i for i, c in enumerate(ALL_MODEL_COLUMNS)}
meta = pd.DataFrame({
    "store_id": features["store_id"].to_numpy(),
    "product_id": features["product_id"].to_numpy(),
    "first_category_id": features["first_category_id"].to_numpy(),
    "dt": features[dfe.DATE_COL].to_numpy(),
    "day_idx": np.tile(np.arange(n_days), n_series),
})
y_all = panel[dfe.TARGET_COL].reshape(-1)
stock_all = panel[dfe.STOCK_COL].reshape(-1)
activity_all = panel["activity_flag"].reshape(-1).astype(int)
del features

day_idx = meta["day_idx"].to_numpy()
rows_fit = np.flatnonzero((day_idx >= FIRST_TRAIN_IDX) & (day_idx < VALIDATION_START_IDX))
rows_val = np.flatnonzero((day_idx >= VALIDATION_START_IDX) & (day_idx < EVAL_START_IDX))
rows_final = np.flatnonzero((day_idx >= FIRST_TRAIN_IDX) & (day_idx < EVAL_START_IDX))
rows_eval = np.flatnonzero(day_idx >= EVAL_START_IDX)
none_cols = [col_index[c] for c in dm.model_feature_columns("none", True)]
check("no missing feature values on any training, validation or eval row",
      not np.isnan(X_all[np.concatenate([rows_final, rows_eval])][:, none_cols]).any())
check("row counts", (len(rows_fit), len(rows_val), len(rows_final), len(rows_eval)) ==
      (49 * n_series, 7 * n_series, 56 * n_series, 7 * n_series),
      {"fit": len(rows_fit), "validation": len(rows_val), "final_train": len(rows_final), "eval": len(rows_eval)})
""")

md("""
## 5. Baselines on validation, regression check against notebook 1

The baselines must reproduce notebook 1's validation metrics. A mismatch means the data or code differ between the two
notebooks.
""")
code("""
def baseline_predictions(origin_idx):
    y_panel = panel[dfe.TARGET_COL]
    return {name: de.baseline_forecast(y_panel, origin_idx, name).reshape(-1) for name in de.BASELINES}


def frame_for(rows, pred):
    return pd.DataFrame({
        "store_id": meta["store_id"].to_numpy()[rows],
        "product_id": meta["product_id"].to_numpy()[rows],
        "first_category_id": meta["first_category_id"].to_numpy()[rows],
        "horizon_day": day_idx[rows] - (day_idx[rows].min() - 1),
        "stockout_bucket": de.stockout_bucket(stock_all[rows]),
        "activity_flag": activity_all[rows],
        "actual": y_all[rows],
        "pred": pred,
    })


def breakdowns(frame):
    out = {}
    for by in ("horizon_day", "stockout_bucket", "activity_flag", "first_category_id"):
        t = de.metrics_by(frame, by)
        t[by] = t[by].astype(str)
        out[f"by_{by}"] = t.to_dict(orient="records")
    return out


def worst_groups(frame, by, top=10):
    g = frame.assign(abs_err=(frame["pred"] - frame["actual"]).abs()).groupby(by).agg(
        rows=("actual", "size"), sum_actual=("actual", "sum"), sum_pred=("pred", "sum"), abs_error=("abs_err", "sum"))
    g["wape"] = g["abs_error"] / g["sum_actual"]
    g["wpe"] = (g["sum_pred"] - g["sum_actual"]) / g["sum_actual"]
    return g.sort_values("abs_error", ascending=False).head(top).reset_index()


val_baselines = baseline_predictions(VALIDATION_START_IDX - 1)
validation_results = {}
for name, pred in val_baselines.items():
    m = de.point_metrics(y_all[rows_val], pred)
    validation_results[name] = {"kind": "baseline", "overall": m}
    expected = NB1_VALIDATION_BASELINES[name]
    same = all(abs(m[k] - expected[k]) <= 1e-9 * max(1.0, abs(expected[k])) for k in expected)
    check(f"{name} reproduces notebook 1 validation metrics", same, {k: round(m[k], 6) for k in expected})
BEST_BASELINE = min(de.BASELINES, key=lambda n: validation_results[n]["overall"]["wape"])
print("best baseline on validation:", BEST_BASELINE)
""")

md("""
## 6. Validation experiments

All LightGBM runs train on target days 2024-05-01 to 2024-06-18 and use the validation week for early stopping
(metric: absolute error, which ranks models the same way as WAPE on a fixed set). Because the validation week is used
both for early stopping and for choosing between runs, validation numbers are slightly optimistic; the eval week in
section 8 is the unbiased number.

Runs:
- `l2`, `l1`, `tweedie`: objective comparison with the full feature set (weather excluded).
- `<best>_no_same_day_discount`: best objective without `discount` and `discount_vs_recent`.
- `<best>_oracle_weather`: best objective plus actual same-day weather. Labelled oracle: not available at forecast time.
""")
code("""
validation_predictions = {}
validation_boosters = {}


def run_validation(name, objective, weather_mode, same_day_discount):
    cols = dm.model_feature_columns(weather_mode, same_day_discount)
    idx = [col_index[c] for c in cols]
    t0 = time.time()
    booster = dm.train(
        X_all[np.ix_(rows_fit, idx)], y_all[rows_fit], cols, objective, MAX_BOOST_ROUNDS,
        X_all[np.ix_(rows_val, idx)], y_all[rows_val], early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        num_threads=NUM_THREADS,
    )
    seconds = time.time() - t0
    pred = dm.predict(booster, X_all[np.ix_(rows_val, idx)], columns=cols)
    m = de.point_metrics(y_all[rows_val], pred)
    validation_results[name] = {
        "kind": "lightgbm", "objective": objective, "weather_mode": weather_mode,
        "same_day_discount": same_day_discount, "best_iteration": int(booster.best_iteration),
        "train_seconds": round(seconds, 1), "n_features": len(cols), "overall": m,
    }
    validation_predictions[name] = pred
    validation_boosters[name] = booster
    print(f"{name}: wape={m['wape']:.4f} wpe={m['wpe']:+.4f} best_iteration={booster.best_iteration} {seconds:.0f}s")


for objective in ("l2", "l1", "tweedie"):
    run_validation(objective, objective, "none", True)

BEST_OBJECTIVE = min(("l2", "l1", "tweedie"), key=lambda n: validation_results[n]["overall"]["wape"])
print("best objective on validation:", BEST_OBJECTIVE)
for name in list(validation_boosters):
    if name != BEST_OBJECTIVE:
        del validation_boosters[name]
""")
code("""
NO_DISCOUNT_RUN = f"{BEST_OBJECTIVE}_no_same_day_discount"
ORACLE_RUN = f"{BEST_OBJECTIVE}_oracle_weather"
run_validation(NO_DISCOUNT_RUN, BEST_OBJECTIVE, "none", False)
run_validation(ORACLE_RUN, BEST_OBJECTIVE, "oracle", True)

table = pd.DataFrame(
    [{"run": k, "kind": v["kind"], **{m: v["overall"][m] for m in ("wape", "wpe", "mae", "rmse")},
      "best_iteration": v.get("best_iteration")} for k, v in validation_results.items()]
).set_index("run")
print(table.round(4).to_string())
""")

md("""
## 7. Served configuration

Rule fixed before any eval scoring (constant `SERVE_SAME_DAY_DISCOUNT` in section 1):

- Objective: the lowest validation WAPE among `l2`, `l1`, `tweedie`.
- Same-day discount: **excluded** from the served model. Notebook 1 could not confirm that `discount` is set before the
  day (zero-sale and full-stockout days carry `discount = 1.0` far more often), so a model that depends on it may be
  reading realized information. The with-discount model is retrained and scored alongside as a comparison, and the gap
  between the two is the amount of accuracy that rests on that unverified assumption.
- Weather: excluded. The oracle-weather run is reported on validation only.
""")
code("""
SERVED_RUN = BEST_OBJECTIVE if SERVE_SAME_DAY_DISCOUNT else NO_DISCOUNT_RUN
COMPARISON_RUN = NO_DISCOUNT_RUN if SERVE_SAME_DAY_DISCOUNT else BEST_OBJECTIVE
served_cfg = validation_results[SERVED_RUN]
comparison_cfg = validation_results[COMPARISON_RUN]
print("served:", SERVED_RUN, "| comparison:", COMPARISON_RUN)

val_frame_served = frame_for(rows_val, validation_predictions[SERVED_RUN])
val_frame_base = frame_for(rows_val, val_baselines[BEST_BASELINE])
validation_results[SERVED_RUN]["breakdowns"] = breakdowns(val_frame_served)
validation_results[BEST_BASELINE]["breakdowns"] = breakdowns(val_frame_base)
for part in ("by_horizon_day", "by_stockout_bucket", "by_activity_flag"):
    a = pd.DataFrame(validation_results[SERVED_RUN]["breakdowns"][part])
    b = pd.DataFrame(validation_results[BEST_BASELINE]["breakdowns"][part])
    key_col = a.columns[0]
    merged = a[[key_col, "n", "wape", "wpe", "mae"]].merge(
        b[[key_col, "wape", "wpe", "mae"]], on=key_col, suffixes=(f"_{SERVED_RUN}", f"_{BEST_BASELINE}"))
    print(f"\\nvalidation {part}")
    print(merged.round(4).to_string(index=False))
""")

md("""
## 8. Final retrain and eval scoring (once)

The served and comparison configurations are retrained on target days 2024-05-01 to 2024-06-25 with the number of
boosting rounds found on validation, then scored on 2024-06-26 to 2024-07-02 together with the three baselines.
No choice in this notebook depends on these numbers.
""")
code("""
final_boosters = {}
for name, cfg in ((SERVED_RUN, served_cfg), (COMPARISON_RUN, comparison_cfg)):
    cols = dm.model_feature_columns(cfg["weather_mode"], cfg["same_day_discount"])
    idx = [col_index[c] for c in cols]
    t0 = time.time()
    final_boosters[name] = dm.train(X_all[np.ix_(rows_final, idx)], y_all[rows_final], cols, cfg["objective"],
                                    cfg["best_iteration"], num_threads=NUM_THREADS)
    print(f"final {name}: {cfg['best_iteration']} rounds, {time.time() - t0:.0f}s")

eval_predictions = {f"lightgbm_{SERVED_RUN}": None, f"lightgbm_{COMPARISON_RUN}": None}
for name, booster in final_boosters.items():
    cols = booster.feature_name()
    eval_predictions[f"lightgbm_{name}"] = dm.predict(booster, X_all[np.ix_(rows_eval, [col_index[c] for c in cols])], columns=cols)
eval_predictions.update(baseline_predictions(EVAL_START_IDX - 1))

eval_results = {}
for name, pred in eval_predictions.items():
    eval_results[name] = {"overall": de.point_metrics(y_all[rows_eval], pred)}
eval_table = pd.DataFrame([{"model": k, **v["overall"]} for k, v in eval_results.items()]).set_index("model")
print(eval_table[["wape", "wpe", "mae", "rmse", "n"]].round(4).to_string())
SERVED_KEY = f"lightgbm_{SERVED_RUN}"
""")
code("""
eval_frame_served = frame_for(rows_eval, eval_predictions[SERVED_KEY])
eval_frame_base = frame_for(rows_eval, eval_predictions[BEST_BASELINE])
eval_results[SERVED_KEY]["breakdowns"] = breakdowns(eval_frame_served)
eval_results[BEST_BASELINE]["breakdowns"] = breakdowns(eval_frame_base)
for part in ("by_horizon_day", "by_stockout_bucket", "by_activity_flag"):
    a = pd.DataFrame(eval_results[SERVED_KEY]["breakdowns"][part])
    b = pd.DataFrame(eval_results[BEST_BASELINE]["breakdowns"][part])
    key_col = a.columns[0]
    merged = a[[key_col, "n", "wape", "wpe", "mae"]].merge(
        b[[key_col, "wape", "wpe", "mae"]], on=key_col, suffixes=("_served", f"_{BEST_BASELINE}"))
    print(f"\\neval {part}")
    print(merged.round(4).to_string(index=False))

cat_table = pd.DataFrame(eval_results[SERVED_KEY]["breakdowns"]["by_first_category_id"]).sort_values("sum_actual", ascending=False)
print("\\neval by first_category_id, served model (largest 10 by volume)")
print(cat_table.head(10).drop(columns=["sum_actual"]).round(4).to_string(index=False))
worst_stores = worst_groups(eval_frame_served, "store_id")
worst_products = worst_groups(eval_frame_served, "product_id")
eval_results[SERVED_KEY]["worst_stores_by_abs_error"] = worst_stores.to_dict(orient="records")
eval_results[SERVED_KEY]["worst_products_by_abs_error"] = worst_products.to_dict(orient="records")
print("\\nstores with the largest total absolute error (served model)")
print(worst_stores.round(4).to_string(index=False))
print("\\nproducts with the largest total absolute error (served model)")
print(worst_products.round(4).to_string(index=False))
""")

md("""
Reading the eval results: the full-day stockout bucket has near-zero recorded actuals, so its WAPE and WPE are not
meaningful and MAE is the comparable number. Recorded sales on stockout days understate true demand, so a model that
matches them is also learning that understatement; this bias is reported, not corrected, in this version.
""")

md("## 9. Feature importance (served model)")
code("""
served_booster = final_boosters[SERVED_RUN]
importance = pd.DataFrame({
    "feature": served_booster.feature_name(),
    "gain": served_booster.feature_importance("gain"),
    "splits": served_booster.feature_importance("split"),
}).sort_values("gain", ascending=False)
importance["gain_share"] = importance["gain"] / importance["gain"].sum()
print(importance.head(20).round(4).to_string(index=False))
""")

md("""
## 10. Serving parity, end to end

The server will hold only the last `REQUIRED_HISTORY_DAYS` of history and receive the known-in-advance columns for
the requested days. This cell rebuilds eval features from exactly that input, with the eval week's realized columns
removed, runs the saved model file, and requires predictions identical to section 8.
""")
code("""
SERVED_MODEL_PATH = MODEL_DIR / "lgbm_served.txt"
COMPARISON_MODEL_PATH = MODEL_DIR / "lgbm_comparison_with_same_day_discount.txt" if not SERVE_SAME_DAY_DISCOUNT else MODEL_DIR / "lgbm_comparison_without_same_day_discount.txt"
served_booster.save_model(str(SERVED_MODEL_PATH))
final_boosters[COMPARISON_RUN].save_model(str(COMPARISON_MODEL_PATH))
reloaded = lgb.Booster(model_file=str(SERVED_MODEL_PATH))

history_start = EVAL_START - pd.Timedelta(days=dfe.REQUIRED_HISTORY_DAYS)
dates = pd.to_datetime(combined[dfe.DATE_COL])
serving_input = combined[(dates >= history_start)].copy()
serving_input.loc[pd.to_datetime(serving_input[dfe.DATE_COL]) >= EVAL_START, dfe.REALIZED_COLS] = np.nan
serving_features = dfe.build_features(serving_input, weather_mode="none", rows_from_date=EVAL_START)
parity_pred = dm.predict(reloaded, serving_features)
check("serving parity: reloaded model on 34-day history reproduces eval predictions exactly",
      np.array_equal(parity_pred, eval_predictions[SERVED_KEY]), f"{len(parity_pred):,} predictions")

rng = np.random.default_rng(SEED)
sample_idx = np.sort(rng.choice(len(serving_features), size=PARITY_SAMPLE_ROWS, replace=False))
parity_sample = serving_features.iloc[sample_idx][[dfe.KEY_COLS[0], dfe.KEY_COLS[1], dfe.DATE_COL] +
                                                  [c for c in reloaded.feature_name() if c not in dfe.KEY_COLS]].reset_index(drop=True)
parity_sample["prediction"] = parity_pred[sample_idx]

snapshot_start = grid[-1] - pd.Timedelta(days=dfe.REQUIRED_HISTORY_DAYS - 1)
snapshot_cols = list(dict.fromkeys(dfe.STATIC_COLS + [dfe.DATE_COL] + dfe.panel_value_cols("none")))
serving_history = combined.loc[dates >= snapshot_start, snapshot_cols].sort_values(dfe.KEY_COLS + [dfe.DATE_COL]).reset_index(drop=True)
future_dates = pd.date_range(grid[-1] + pd.Timedelta(days=1), periods=HORIZON_DAYS, freq="D")
future = serving_history.drop_duplicates(dfe.KEY_COLS)[dfe.STATIC_COLS].merge(
    pd.DataFrame({dfe.DATE_COL: future_dates.strftime("%Y-%m-%d")}), how="cross")
for c in dfe.REALIZED_COLS:
    if c in snapshot_cols:
        future[c] = np.nan
future["activity_flag"] = 0
future["holiday_flag"] = (future_dates.dayofweek[np.tile(np.arange(HORIZON_DAYS), n_series)] >= 5).astype(int)
future["discount"] = np.nan
future_features = dfe.build_features(pd.concat([serving_history, future], ignore_index=True), rows_from_date=future_dates[0])
future_pred = dm.predict(reloaded, future_features)
check("snapshot alone produces complete features and finite predictions for the next 7 days",
      not future_features[reloaded.feature_name()].isna().any().any() and np.isfinite(future_pred).all(),
      f"{len(future_pred):,} rows, dates {future_dates[0].date()}..{future_dates[-1].date()}, activity_flag=0, holiday_flag=weekend")
""")

md("## 11. Save artifacts")
code("""
for module in ("demand_features.py", "demand_eval.py", "demand_model.py"):
    src = MODULE_DIR / module
    if src.resolve() != (OUTPUT_DIR / module).resolve():
        shutil.copy2(src, OUTPUT_DIR / module)

environment["module_sha256"] = {m: sha256_of(MODULE_DIR / m) for m in ("demand_features.py", "demand_eval.py", "demand_model.py")}
model_card = {
    "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "license": "CC BY 4.0"},
    "served_run": SERVED_RUN,
    "comparison_run": COMPARISON_RUN,
    "objective": served_cfg["objective"],
    "params": dm.model_params(served_cfg["objective"]) | {"num_threads": NUM_THREADS},
    "num_boost_round": served_cfg["best_iteration"],
    "feature_columns": served_booster.feature_name(),
    "categorical_features": [c for c in dm.CATEGORICAL_FEATURES if c in served_booster.feature_name()],
    "same_day_discount": served_cfg["same_day_discount"],
    "weather_mode": served_cfg["weather_mode"],
    "horizon_days": dfe.HORIZON,
    "min_lag_days": dfe.MIN_LAG,
    "required_history_days": dfe.REQUIRED_HISTORY_DAYS,
    "train_target_days": split["final_train_target_days"],
    "prediction_clip_min": 0.0,
    "model_files": {
        SERVED_MODEL_PATH.name: sha256_of(SERVED_MODEL_PATH),
        COMPARISON_MODEL_PATH.name: sha256_of(COMPARISON_MODEL_PATH),
    },
    "eval_metrics_served": eval_results[SERVED_KEY]["overall"],
    "known_limitations": [
        "Actuals on stockout days are censored; recorded-sales bias is reported, not corrected.",
        "Features use history up to t-7 only; the most recent 6 days are unused for near horizons.",
        "Sales values are normalized by the dataset publisher, so errors are not in physical units.",
        "Whether same-day discount is known in advance is unverified; the served model excludes it."
        if not SERVE_SAME_DAY_DISCOUNT else "Served model assumes same-day discount is known in advance (unverified).",
    ],
}
metrics = {"split": split, "validation": validation_results, "eval": eval_results,
           "served_run": SERVED_RUN, "comparison_run": COMPARISON_RUN, "best_baseline_on_validation": BEST_BASELINE}

(OUTPUT_DIR / "nb2_model_card.json").write_text(json.dumps(model_card, indent=2, default=str))
(OUTPUT_DIR / "nb2_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
(OUTPUT_DIR / "nb2_environment.json").write_text(json.dumps(environment, indent=2))
importance.to_csv(OUTPUT_DIR / "nb2_feature_importance.csv", index=False)
parity_sample.to_parquet(OUTPUT_DIR / "nb2_parity_sample.parquet", index=False)
serving_history.to_parquet(OUTPUT_DIR / "nb2_serving_history.parquet", index=False)
eval_out = pd.DataFrame({"store_id": meta["store_id"].to_numpy()[rows_eval], "product_id": meta["product_id"].to_numpy()[rows_eval],
                         "dt": meta["dt"].to_numpy()[rows_eval], "actual": y_all[rows_eval],
                         "stockout_hours": stock_all[rows_eval]})
for name, pred in eval_predictions.items():
    eval_out[f"pred_{name}"] = pred
eval_out.to_parquet(OUTPUT_DIR / "nb2_eval_predictions.parquet", index=False)
(OUTPUT_DIR / "nb2_checks.json").write_text(json.dumps(report, indent=2, default=str))

for p in sorted(list(OUTPUT_DIR.glob("nb2_*")) + list(MODEL_DIR.glob("*.txt"))):
    print(f"{p.relative_to(OUTPUT_DIR)}: {p.stat().st_size:,} bytes")
failed = [k for k, v in report["checks"].items() if not v["passed"]]
print(f"checks: {len(report['checks'])} run, {len(failed)} failed")
""")

nb = nbf.v4.new_notebook()
for i, cell in enumerate(cells):
    cell["id"] = f"cell-{i:02d}"
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = ROOT / "notebooks" / "02_lightgbm_training_evaluation.ipynb"
nbf.write(nb, out)
print(out)
