"""Assemble notebooks/03_chronos2_offline_comparison.ipynb from src/ so the module code inside the
notebook is byte-identical to the source files."""
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
# Notebook 3: Chronos-2 offline comparison

Project 4, retail demand forecasting on FreshRetailNet-50K. Runs on its own: downloads the same
pinned dataset and writes the same evaluation module as notebooks 1 and 2.

Chronos-2 (Amazon, Apache 2.0, 120M parameters) is a pretrained time-series model that forecasts
without training on this dataset. This notebook answers one question with measurements rather than
argument: **is it worth serving instead of the LightGBM model?** That means two numbers, accuracy
on the same evaluation week and inference throughput, not just the first.

The comparison is deliberately offline. The served model stays LightGBM unless Chronos-2 beats it
on both.

Kaggle settings: Internet on. GPU recommended (set Accelerator to a GPU); it runs on CPU, slower.
""")

md("## 1. Configuration")
code("""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODULE_DIR = Path.cwd()
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

HORIZON_DAYS = 7
SEED = 20240626
CONTEXT_DAYS = int(os.environ.get("CONTEXT_DAYS", 90))
SUBSET_SERIES = int(os.environ.get("SUBSET_SERIES", 0))          # 0 uses all 50,000 series
BATCH_SIZE = int(os.environ.get("CHRONOS_BATCH_SIZE", 256))
QUANTILES = [0.1, 0.5, 0.9]
THROUGHPUT_SERIES = int(os.environ.get("THROUGHPUT_SERIES", 512))
CPU_BENCHMARK_SERIES = int(os.environ.get("CPU_BENCHMARK_SERIES", 128))

# measured in notebook 2 on the same eval week, quoted here for comparison
LIGHTGBM_EVAL = {"wape": 0.33168907433084976, "wpe": -0.03410891931355602,
                 "mae": 0.3957166918475688, "rmse": 0.6730458170158612}
# measured on the kind deployment, 10 concurrent users, batching on
SERVICE_MEASURED = {"throughput_rps": 82.2, "p50_ms": 100.0, "p99_ms": 390.0,
                    "model_predict_ms_per_request": 5.4}

report = {"checks": {}}


def check(name, condition, detail=None):
    report["checks"][name] = {"passed": bool(condition), "detail": detail}
    print(("PASS " if condition else "FAIL ") + name + ("" if detail is None else f" | {detail}"))
    assert condition, name
""")
code("""
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "chronos-forecasting>=2.0"], check=True)

import torch
from chronos import Chronos2Pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
environment = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "pyarrow": pa.__version__,
    "torch": torch.__version__,
    "device": DEVICE,
    "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
    "on_kaggle": KAGGLE_WORKING.exists(),
    "context_days": CONTEXT_DAYS,
    "subset_series": SUBSET_SERIES,
}
print(json.dumps(environment, indent=2))
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
    assert sha256_of(dest) == expected, f"{split}: sha256 mismatch"
    print(f"{split}: {dest.stat().st_size:,} bytes, sha256 ok")
    return dest


TRAIN_PATH, EVAL_PATH = fetch("train"), fetch("eval")
""")

md("## 3. Evaluation module")
writefile("demand_features.py", "src/demand_features.py")
writefile("demand_eval.py", "src/demand_eval.py")
code("""
import demand_eval as de
import demand_features as dfe
""")

md("""
## 4. Inputs for Chronos-2

Chronos-2 reads a long frame of context and, optionally, a frame of future values. Columns present
in both frames are treated as known in advance; columns only in the context frame are past-only
covariates. The split matches what the LightGBM model is allowed to see:

- past only: `stock_hour6_22_cnt`, `discount`
- known in advance: `activity_flag`, `holiday_flag`
- target: `sale_amount`

No same-day discount, no weather, exactly as in the served configuration.
""")
code("""
load_cols = list(dict.fromkeys(dfe.KEY_COLS + [dfe.DATE_COL] + dfe.panel_value_cols("none")))
train = pd.read_parquet(TRAIN_PATH, columns=load_cols)
evald = pd.read_parquet(EVAL_PATH, columns=list(dict.fromkeys(load_cols + [dfe.STOCK_COL])))
for frame in (train, evald):
    frame[dfe.DATE_COL] = pd.to_datetime(frame[dfe.DATE_COL])

keys = train[dfe.KEY_COLS].drop_duplicates().sort_values(dfe.KEY_COLS).reset_index(drop=True)
if SUBSET_SERIES:
    keys = keys.sample(SUBSET_SERIES, random_state=SEED).sort_values(dfe.KEY_COLS).reset_index(drop=True)
    train = train.merge(keys, on=dfe.KEY_COLS, how="inner")
    evald = evald.merge(keys, on=dfe.KEY_COLS, how="inner")
    print(f"subset: {len(keys):,} series")

train = train.sort_values(dfe.KEY_COLS + [dfe.DATE_COL]).reset_index(drop=True)
evald = evald.sort_values(dfe.KEY_COLS + [dfe.DATE_COL]).reset_index(drop=True)
train_dates = pd.DatetimeIndex(sorted(train[dfe.DATE_COL].unique()))
eval_dates = pd.DatetimeIndex(sorted(evald[dfe.DATE_COL].unique()))
context_start = train_dates[-CONTEXT_DAYS]
check("eval window is the 7 days after the training window",
      len(eval_dates) == HORIZON_DAYS and eval_dates[0] == train_dates[-1] + pd.Timedelta(days=1),
      f"{eval_dates[0].date()}..{eval_dates[-1].date()}")

series_id = (train["store_id"].astype(str) + "_" + train["product_id"].astype(str))
context_df = pd.DataFrame({
    "item_id": series_id,
    "timestamp": train[dfe.DATE_COL],
    "target": train[dfe.TARGET_COL].astype("float32"),
    "stock_hours": train[dfe.STOCK_COL].astype("float32"),
    "discount": train["discount"].astype("float32"),
    "activity_flag": train["activity_flag"].astype("float32"),
    "holiday_flag": train["holiday_flag"].astype("float32"),
})
context_df = context_df[train[dfe.DATE_COL] >= context_start].reset_index(drop=True)
future_df = pd.DataFrame({
    "item_id": evald["store_id"].astype(str) + "_" + evald["product_id"].astype(str),
    "timestamp": evald[dfe.DATE_COL],
    "activity_flag": evald["activity_flag"].astype("float32"),
    "holiday_flag": evald["holiday_flag"].astype("float32"),
})
check("context and future frames cover the same series",
      context_df["item_id"].nunique() == future_df["item_id"].nunique() == len(keys),
      f"{len(keys):,} series, context {len(context_df):,} rows, future {len(future_df):,} rows")
check("context holds the expected number of days per series",
      int(context_df.groupby("item_id").size().max()) == CONTEXT_DAYS, CONTEXT_DAYS)
""")

md("## 5. Zero-shot forecasts")
code("""
pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEVICE)
t0 = time.perf_counter()
forecast = pipeline.predict_df(
    context_df,
    future_df=future_df,
    prediction_length=HORIZON_DAYS,
    quantile_levels=QUANTILES,
    batch_size=BATCH_SIZE,
    id_column="item_id",
    timestamp_column="timestamp",
    target="target",
)
elapsed = time.perf_counter() - t0
print(f"{len(keys):,} series in {elapsed:.1f}s on {DEVICE} "
      f"({len(keys)/elapsed:.1f} series/s, {len(keys)*HORIZON_DAYS/elapsed:,.0f} forecast rows/s)")

forecast = forecast.rename(columns={"0.5": "chronos_p50", "0.1": "chronos_p10", "0.9": "chronos_p90"})
forecast["chronos"] = np.clip(forecast["chronos_p50"].to_numpy(), 0.0, None)
evald["item_id"] = evald["store_id"].astype(str) + "_" + evald["product_id"].astype(str)
merged = evald.merge(forecast[["item_id", "timestamp", "chronos", "chronos_p10", "chronos_p90"]],
                     left_on=["item_id", dfe.DATE_COL], right_on=["item_id", "timestamp"], how="left")
check("a forecast exists for every eval row", int(merged["chronos"].isna().sum()) == 0, f"{len(merged):,} rows")
""")

md("""
## 6. Accuracy on the same evaluation week

The baselines are recomputed here from the same data, so this notebook does not depend on notebook
2's outputs. The LightGBM figure is quoted from notebook 2's model card, measured on exactly these
rows. If notebook 2's predictions file is available, set `NB2_PREDICTIONS` and the comparison
becomes row-aligned instead of quoted.
""")
code("""
keys_arr, grid, panel = dfe.to_panel(
    pd.concat([train, evald[train.columns]], ignore_index=True), [dfe.TARGET_COL])
origin_idx = len(grid) - HORIZON_DAYS - 1
baseline_pred = {name: de.baseline_forecast(panel[dfe.TARGET_COL], origin_idx, name).reshape(-1)
                 for name in de.BASELINES}
order = (keys_arr["store_id"].astype(str) + "_" + keys_arr["product_id"].astype(str))
baseline_frame = pd.DataFrame({
    "item_id": np.repeat(order.to_numpy(), HORIZON_DAYS),
    dfe.DATE_COL: np.tile(grid[-HORIZON_DAYS:].to_numpy(), len(keys_arr)),
    **{f"pred_{k}": v for k, v in baseline_pred.items()},
})
merged = merged.merge(baseline_frame, on=["item_id", dfe.DATE_COL], how="left", validate="one_to_one")

results = {"chronos2_zeroshot": de.point_metrics(merged[dfe.TARGET_COL], merged["chronos"])}
for name in de.BASELINES:
    results[name] = de.point_metrics(merged[dfe.TARGET_COL], merged[f"pred_{name}"])
results["lightgbm_served_notebook2"] = dict(LIGHTGBM_EVAL, n=len(merged), sum_actual=float(merged[dfe.TARGET_COL].sum()))

nb2_path = os.environ.get("NB2_PREDICTIONS", "")
if nb2_path and Path(nb2_path).exists():
    nb2 = pd.read_parquet(nb2_path)
    nb2["item_id"] = nb2["store_id"].astype(str) + "_" + nb2["product_id"].astype(str)
    nb2[dfe.DATE_COL] = pd.to_datetime(nb2["dt"])
    col = [c for c in nb2.columns if c.startswith("pred_lightgbm") and "no_same_day_discount" in c][0]
    merged = merged.merge(nb2[["item_id", dfe.DATE_COL, col]], on=["item_id", dfe.DATE_COL], how="left")
    merged = merged.rename(columns={col: "lightgbm"})
    results["lightgbm_served_rowaligned"] = de.point_metrics(merged[dfe.TARGET_COL], merged["lightgbm"])
    print("row-aligned LightGBM comparison enabled")

table = pd.DataFrame(results).T[["wape", "wpe", "mae", "rmse"]]
print(table.round(4).sort_values("wape").to_string())
if SUBSET_SERIES:
    print(f"\nNOTE: this run used {len(keys):,} of 50,000 series. Chronos-2 and the baselines are measured on"
          f" that subset; the LightGBM row is notebook 2's full-population figure and is NOT directly"
          f" comparable. Compare against the baselines in this table, or rerun with SUBSET_SERIES=0.")
""")
code("""
merged["horizon_day"] = merged.groupby("item_id").cumcount() + 1
merged["stockout_bucket"] = de.stockout_bucket(merged[dfe.STOCK_COL])
breakdowns = {}
for label, column in (("chronos2_zeroshot", "chronos"), ("moving_average_7", "pred_moving_average_7")):
    frame = merged.rename(columns={dfe.TARGET_COL: "actual", column: "pred"})
    breakdowns[label] = {
        part: de.metrics_by(frame, part).astype({part: str}).to_dict(orient="records")
        for part in ("horizon_day", "stockout_bucket")
    }
    for part in ("horizon_day", "stockout_bucket"):
        print(f"\\n{label} by {part}")
        print(pd.DataFrame(breakdowns[label][part]).drop(columns=["sum_actual"]).round(4).to_string(index=False))

coverage = float(((merged[dfe.TARGET_COL] >= merged["chronos_p10"]) &
                  (merged[dfe.TARGET_COL] <= merged["chronos_p90"])).mean())
print(f"\\nChronos-2 80% interval coverage: {coverage:.3f} (nominal 0.8)")
""")

md("""
## 7. Inference cost

Accuracy is only half the serving question. This measures Chronos-2's throughput on the device it
is running on, and on CPU, against the LightGBM service's measured figures from the kind
deployment.
""")
code("""
bench_ids = context_df["item_id"].drop_duplicates().head(THROUGHPUT_SERIES)
bench_ctx = context_df[context_df["item_id"].isin(bench_ids)]
bench_fut = future_df[future_df["item_id"].isin(bench_ids)]


def benchmark(pipe, ctx, fut, n_series, repeats=2):
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        pipe.predict_df(ctx, future_df=fut, prediction_length=HORIZON_DAYS, quantile_levels=QUANTILES,
                        batch_size=BATCH_SIZE, id_column="item_id", timestamp_column="timestamp", target="target")
        timings.append(time.perf_counter() - t0)
    best = min(timings)
    return {"series": int(n_series), "seconds": round(best, 2),
            "series_per_second": round(n_series / best, 1),
            "ms_per_series": round(best / n_series * 1000, 2)}


throughput = {"device": DEVICE, DEVICE: benchmark(pipeline, bench_ctx, bench_fut, len(bench_ids))}
print(f"{DEVICE}: {throughput[DEVICE]}")

if DEVICE == "cuda":
    cpu_ids = bench_ids.head(CPU_BENCHMARK_SERIES)
    cpu_pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    throughput["cpu"] = benchmark(cpu_pipeline, bench_ctx[bench_ctx["item_id"].isin(cpu_ids)],
                                  bench_fut[bench_fut["item_id"].isin(cpu_ids)], len(cpu_ids), repeats=1)
    print(f"cpu: {throughput['cpu']}")
    del cpu_pipeline

throughput["lightgbm_service_measured"] = SERVICE_MEASURED
print(json.dumps(throughput, indent=2))
""")

md("""
## 8. Reading the result

Two cautions belong with these numbers.

**Contamination cannot be ruled out.** Chronos-2's training corpus is not fully disclosed, and this
dataset was published before the model. If the dataset, or series correlated with it, appeared in
pretraining, a zero-shot score here flatters the model. Published work on time-series foundation
models documents exactly this problem, so the comparison is reported as indicative, not decisive.

**Serving is a separate decision from accuracy.** The LightGBM service answers at 82 requests per
second with a p50 of 100 ms on one CPU pod, with the model itself costing about 5 ms per request.
The throughput figures above are what a foundation model would have to match to replace it, on
hardware that costs more.
""")

md("## 9. Save artifacts")
code("""
environment["chronos_forecasting"] = subprocess.run(
    [sys.executable, "-m", "pip", "show", "chronos-forecasting"], capture_output=True, text=True
).stdout.split("Version:")[1].split()[0]

out_cols = dfe.KEY_COLS + [dfe.DATE_COL, dfe.TARGET_COL, dfe.STOCK_COL, "horizon_day",
                           "chronos", "chronos_p10", "chronos_p90"] + [f"pred_{n}" for n in de.BASELINES]
merged[out_cols].to_parquet(OUTPUT_DIR / "nb3_chronos_predictions.parquet", index=False)
(OUTPUT_DIR / "nb3_chronos_metrics.json").write_text(json.dumps({
    "eval_window": [str(eval_dates[0].date()), str(eval_dates[-1].date())],
    "series": int(len(keys)), "context_days": CONTEXT_DAYS,
    "subset_run": bool(SUBSET_SERIES),
    "lightgbm_reference_is_full_population": bool(SUBSET_SERIES),
    "overall": results, "breakdowns": breakdowns,
    "interval_coverage_80": coverage,
    "lightgbm_reference": LIGHTGBM_EVAL,
}, indent=2, default=str))
(OUTPUT_DIR / "nb3_throughput.json").write_text(json.dumps(throughput, indent=2))
(OUTPUT_DIR / "nb3_environment.json").write_text(json.dumps(environment, indent=2))
(OUTPUT_DIR / "nb3_checks.json").write_text(json.dumps(report, indent=2, default=str))
for p in sorted(OUTPUT_DIR.glob("nb3_*")):
    print(f"{p.name}: {p.stat().st_size:,} bytes")
failed = [k for k, v in report["checks"].items() if not v["passed"]]
print(f"checks: {len(report['checks'])} run, {len(failed)} failed")
""")

nb = nbf.v4.new_notebook()
for i, cell in enumerate(cells):
    cell["id"] = f"cell-{i:02d}"
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = ROOT / "notebooks" / "03_chronos2_offline_comparison.ipynb"
nbf.write(nb, out)
print(out)
