"""Run one locust scenario against the service and record latency percentiles plus server metrics.

Usage:
  python loadtest/run_loadtest.py --host http://127.0.0.1:8000 --label batching-on --users 50 --run-time 60s
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode()


def histogram_stats(metrics_text: str, name: str) -> dict[str, float]:
    total = count = 0.0
    for line in metrics_text.splitlines():
        if line.startswith(f"{name}_sum"):
            total = float(line.split()[-1])
        elif line.startswith(f"{name}_count"):
            count = float(line.split()[-1])
    return {"sum": total, "count": count, "mean": (total / count) if count else float("nan")}


def counter_value(metrics_text: str, pattern: str) -> float:
    total = 0.0
    for line in metrics_text.splitlines():
        if re.match(pattern, line):
            total += float(line.split()[-1])
    return total


def ensure_series_sample(history_path: Path, out: Path, n: int) -> None:
    if out.exists():
        return
    import pandas as pd

    frame = pd.read_parquet(history_path, columns=["store_id", "product_id"]).drop_duplicates()
    frame.sample(min(n, len(frame)), random_state=11).to_csv(out, index=False)
    print(f"wrote {out} with {min(n, len(frame))} series")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--spawn-rate", type=int, default=10)
    parser.add_argument("--run-time", default="60s")
    parser.add_argument("--history", default=None, help="parquet used to generate the series sample once")
    parser.add_argument("--series", type=int, default=5000)
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    series_file = HERE / "series_sample.csv"
    if args.history:
        ensure_series_sample(Path(args.history), series_file, args.series)
    if not series_file.exists():
        print(f"FAIL {series_file} missing; pass --history once to generate it")
        return 1

    model_info = json.loads(fetch(f"{args.host}/model"))
    before = fetch(f"{args.host}/metrics")
    prefix = RESULTS / args.label
    command = [
        sys.executable, "-m", "locust", "-f", str(HERE / "locustfile.py"), "--headless",
        "--host", args.host, "-u", str(args.users), "-r", str(args.spawn_rate),
        "-t", args.run_time, "--csv", str(prefix), "--only-summary",
    ]
    print(" ".join(command))
    result = subprocess.run(command)
    after = fetch(f"{args.host}/metrics")

    stats_file = prefix.with_name(prefix.name + "_stats.csv")
    rows = list(csv.DictReader(open(stats_file)))
    aggregated = next(r for r in rows if r["Name"] == "Aggregated")
    per_endpoint = [r for r in rows if r["Name"] != "Aggregated"]

    def delta(name):
        a, b = histogram_stats(before, name), histogram_stats(after, name)
        count = b["count"] - a["count"]
        total = b["sum"] - a["sum"]
        return {"calls": count, "seconds": total, "mean_seconds": (total / count) if count else float("nan")}

    summary = {
        "label": args.label,
        "host": args.host,
        "users": args.users,
        "run_time": args.run_time,
        "batching": model_info["batching"],
        "model_version": model_info["model_version"],
        "requests": int(aggregated["Request Count"]),
        "failures": int(aggregated["Failure Count"]),
        "requests_per_second": float(aggregated["Requests/s"]),
        "latency_ms": {
            "p50": float(aggregated["50%"]),
            "p95": float(aggregated["95%"]),
            "p99": float(aggregated["99%"]),
            "mean": float(aggregated["Average Response Time"]),
            "max": float(aggregated["Max Response Time"]),
        },
        "per_endpoint": [
            {"name": r["Name"], "requests": int(r["Request Count"]), "p50": float(r["50%"]),
             "p95": float(r["95%"]), "p99": float(r["99%"])}
            for r in per_endpoint
        ],
        "server": {
            "batch_size_mean": (histogram_stats(after, "forecast_batch_size")["sum"] - histogram_stats(before, "forecast_batch_size")["sum"])
            / max(1.0, histogram_stats(after, "forecast_batch_size")["count"] - histogram_stats(before, "forecast_batch_size")["count"]),
            "model_calls": histogram_stats(after, "forecast_model_predict_seconds")["count"] - histogram_stats(before, "forecast_model_predict_seconds")["count"],
            "feature_build": delta("forecast_feature_build_seconds"),
            "model_predict": delta("forecast_model_predict_seconds"),
            "batch_wait": delta("forecast_batch_wait_seconds"),
            "forecast_rows": counter_value(after, r"forecast_items_total ") - counter_value(before, r"forecast_items_total "),
        },
    }
    out = RESULTS / f"{args.label}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("requests", "failures", "requests_per_second", "latency_ms", "server")}, indent=2))
    print(f"\nwritten to {out}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
