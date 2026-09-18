"""Log the Chronos-2 offline comparison into the Project 4 MLflow experiment.

Notebook 3 measured a pretrained model that was evaluated and not served. This records that
evaluation next to the served model's run, so the tracking store holds the comparison rather than
only the winner. No model is logged or registered: nothing here is a serving candidate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow

EXPERIMENT_NAME = "project-4-retail-demand"
RUN_NAME = "chronos2-zeroshot"
REQUIRED = ["nb3_chronos_metrics.json", "nb3_throughput.json", "nb3_environment.json", "nb3_checks.json"]


def flatten(metrics: dict, throughput: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, block in metrics["overall"].items():
        tag = "chronos2" if name == "chronos2_zeroshot" else name
        for key in ("wape", "wpe", "mae", "rmse"):
            if key in block:
                out[f"eval_{tag}_{key}"] = float(block[key])
    for part, prefix in (("horizon_day", "h"), ("stockout_bucket", "stockout")):
        for row in metrics["breakdowns"]["chronos2_zeroshot"][part]:
            key = str(row[part]).replace("-", "_")
            out[f"eval_chronos2_{prefix}{key}_wape"] = float(row["wape"])
            out[f"eval_chronos2_{prefix}{key}_mae"] = float(row["mae"])
    out["interval_coverage_80"] = float(metrics["interval_coverage_80"])
    for device in ("cuda", "cpu"):
        if device in throughput:
            out[f"throughput_{device}_series_per_second"] = float(throughput[device]["series_per_second"])
            out[f"throughput_{device}_ms_per_series"] = float(throughput[device]["ms_per_series"])
    service = throughput.get("lightgbm_service_measured", {})
    for key, value in service.items():
        out[f"served_service_{key}"] = float(value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True, help="folder holding the notebook 3 outputs")
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    parser.add_argument("--run-name", default=RUN_NAME)
    args = parser.parse_args()

    art = Path(args.artifacts_dir)
    missing = [f for f in REQUIRED if not (art / f).exists()]
    if missing:
        print(f"FAIL missing notebook 3 outputs in {art}: {missing}")
        return 1
    metrics = json.loads((art / "nb3_chronos_metrics.json").read_text())
    throughput = json.loads((art / "nb3_throughput.json").read_text())
    environment = json.loads((art / "nb3_environment.json").read_text())
    checks = json.loads((art / "nb3_checks.json").read_text())

    failed = [k for k, v in checks["checks"].items() if not v["passed"]]
    if failed:
        print(f"FAIL notebook 3 recorded failed checks: {failed}")
        return 1
    if metrics.get("subset_run"):
        print(f"FAIL these results are from a {metrics['series']}-series subset, not the full population")
        return 1
    print(f"PASS notebook 3 outputs are a full run: {metrics['series']:,} series, {len(checks['checks'])} checks passed")

    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient()
    existing = {e.name: e for e in client.search_experiments()}
    if args.experiment not in existing:
        print(f"FAIL experiment {args.experiment} does not exist; run register_model.py first")
        return 1
    experiment_id = existing[args.experiment].experiment_id
    print(f"experiment {args.experiment} (id {experiment_id})")

    served_runs = client.search_runs([experiment_id], filter_string="tags.trained_on = 'kaggle'")
    print(f"served-model runs already in this experiment: {[r.info.run_name for r in served_runs]}")

    params = {
        "model": "amazon/chronos-2",
        "model_type": "pretrained_foundation_model",
        "trained_on_this_dataset": False,
        "chronos_forecasting": environment.get("chronos_forecasting"),
        "torch": environment.get("torch"),
        "device": environment.get("device"),
        "gpu": environment.get("gpu"),
        "context_days": metrics["context_days"],
        "series": metrics["series"],
        "eval_window": "..".join(metrics["eval_window"]),
        "horizon_days": 7,
        "same_day_discount": False,
        "weather": False,
    }

    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tags({
            "served": "false",
            "role": "offline_comparison",
            "not_served_reason": (
                "less accurate than the trained model on this data and about 10x slower per series on CPU, "
                "the hardware the service runs on"
            ),
            "contamination_caveat": (
                "Chronos-2's pretraining corpus is not fully disclosed and this dataset predates the model, "
                "so contamination cannot be ruled out; treat as indicative"
            ),
            "notebook": "03_chronos2_offline_comparison.ipynb",
        })
        mlflow.log_params(params)
        logged = flatten(metrics, throughput)
        mlflow.log_metrics(logged)
        for name in REQUIRED:
            mlflow.log_artifact(str(art / name), artifact_path="notebook3_outputs")
        run_id = run.info.run_id

    print(f"\nlogged run {run_id} with {len(logged)} metrics and {len(params)} parameters")
    read_back = client.get_run(run_id)
    for key in ("eval_chronos2_wape", "eval_lightgbm_served_notebook2_wape", "interval_coverage_80",
                "throughput_cpu_series_per_second"):
        if key in read_back.data.metrics:
            print(f"  {key} = {read_back.data.metrics[key]}")
    files = [f.path for f in client.list_artifacts(run_id, "notebook3_outputs")]
    ok = read_back.info.status == "FINISHED" and len(files) == len(REQUIRED)
    print(f"read back: status {read_back.info.status}, artifacts {files}")
    print("PASS comparison run stored next to the served model" if ok else "FAIL run did not store cleanly")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
