"""Log and register the Kaggle-trained demand forecasting model in a local MLflow store.

No training happens here. The model file produced by notebook 2 on Kaggle is checked against
the hash in its model card, logged with its parameters, metrics and artifacts, registered, then
reloaded through mlflow.pyfunc and required to reproduce the parity sample predictions exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

EXPERIMENT_NAME = "project-4-retail-demand"
MODEL_NAME = "retail-demand-lgbm"
REQUIRED_ARTIFACTS = [
    "lgbm_served.txt",
    "nb2_model_card.json",
    "nb2_metrics.json",
    "nb2_environment.json",
    "nb2_feature_importance.csv",
    "nb2_parity_sample.parquet",
]


class DemandForecastModel(mlflow.pyfunc.PythonModel):
    """pyfunc wrapper: column order from the booster, float32 matrix, predictions clipped at zero."""

    def load_context(self, context):
        import lightgbm as lgb

        self.booster = lgb.Booster(model_file=context.artifacts["booster"])
        self.columns = self.booster.feature_name()

    def predict(self, context, model_input, params=None):
        frame = pd.DataFrame(model_input)
        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            raise KeyError(f"feature columns missing: {missing}")
        x = frame[self.columns].to_numpy(dtype=np.float32)
        return np.clip(self.booster.predict(x), 0.0, None)


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def flatten_metrics(metrics: dict) -> dict[str, float]:
    """Both the flattering and the unflattering numbers: overall, per horizon day, per stockout bucket."""
    out: dict[str, float] = {}
    served = metrics["served_run"]
    served_key = f"lightgbm_{served}"
    for name, block in metrics["eval"].items():
        tag = "served" if name == served_key else name
        for m in ("wape", "wpe", "mae", "rmse"):
            out[f"eval_{tag}_{m}"] = float(block["overall"][m])
        for part, prefix in (("by_horizon_day", "h"), ("by_stockout_bucket", "stockout")):
            for row in block.get("breakdowns", {}).get(part, []):
                key = str(row[list(row)[0]]).replace("-", "_")
                out[f"eval_{tag}_{prefix}{key}_wape"] = float(row["wape"])
                out[f"eval_{tag}_{prefix}{key}_mae"] = float(row["mae"])
    for name, block in metrics["validation"].items():
        tag = "served" if name == served else name
        for m in ("wape", "wpe", "mae", "rmse"):
            out[f"validation_{tag}_{m}"] = float(block["overall"][m])
        for row in block.get("breakdowns", {}).get("by_horizon_day", []):
            out[f"validation_{tag}_h{row['horizon_day']}_wape"] = float(row["wape"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True, help="folder holding the notebook 2 Kaggle outputs")
    parser.add_argument("--tracking-uri", required=True, help="e.g. sqlite:///C:/path/to/mlflow.db")
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--parity-rows", type=int, default=0, help="0 uses every row in the parity sample")
    parser.add_argument("--summary-out", default=None, help="defaults to mlflow_registration_summary.json beside the artifacts")
    parser.add_argument("--artifact-root", default=None,
                        help="absolute folder for this experiment's artifact files; used only when the experiment is created")
    args = parser.parse_args()

    art = Path(args.artifacts_dir)
    missing = [f for f in REQUIRED_ARTIFACTS if not (art / f).exists()]
    if missing:
        print(f"FAIL missing artifacts in {art}: {missing}")
        return 1
    card = json.loads((art / "nb2_model_card.json").read_text())
    metrics = json.loads((art / "nb2_metrics.json").read_text())
    kaggle_env = json.loads((art / "nb2_environment.json").read_text())

    model_path = art / "lgbm_served.txt"
    digest = sha256_of(model_path)
    expected = card["model_files"]["lgbm_served.txt"]
    print(f"model file sha256 {digest}")
    if digest != expected:
        print(f"FAIL model file does not match its model card ({expected})")
        return 1
    print("PASS model file matches the hash recorded on Kaggle")

    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient()
    existing = {e.name: e for e in client.search_experiments()}
    if args.experiment in existing:
        experiment_id = existing[args.experiment].experiment_id
        print(f"experiment {args.experiment} already exists (id {experiment_id})")
    else:
        artifact_location = Path(args.artifact_root).resolve().as_uri() if args.artifact_root else None
        experiment_id = client.create_experiment(args.experiment, artifact_location=artifact_location)
        print(f"created experiment {args.experiment} (id {experiment_id})")
    print("\nexperiments in this store:")
    for e in sorted(client.search_experiments(), key=lambda e: int(e.experiment_id)):
        mark = "  <- project 4" if e.experiment_id == experiment_id else ""
        print(f"  id={e.experiment_id:<4} name={e.name}{mark}")
    others = [e for e in client.search_experiments() if e.experiment_id != experiment_id]
    if any(e.name == args.experiment for e in others):
        print("FAIL another experiment carries the same name")
        return 1
    print(f"PASS project 4 experiment id {experiment_id} is distinct from {len(others)} other experiment(s)")
    print(f"artifact location: {client.get_experiment(experiment_id).artifact_location}")

    parity = pd.read_parquet(art / "nb2_parity_sample.parquet")
    if args.parity_rows:
        parity = parity.head(args.parity_rows)
    feature_columns = card["feature_columns"]
    x_parity = parity[feature_columns].astype("float64")
    y_parity = parity["prediction"].to_numpy()

    params = {
        "objective": card["objective"],
        "num_boost_round": card["num_boost_round"],
        "n_features": len(feature_columns),
        "same_day_discount": card["same_day_discount"],
        "weather_mode": card["weather_mode"],
        "horizon_days": card["horizon_days"],
        "min_lag_days": card["min_lag_days"],
        "required_history_days": card["required_history_days"],
        "train_target_days": "..".join(card["train_target_days"]),
        "dataset_repo": card["dataset"]["repo"],
        "dataset_revision": card["dataset"]["revision"],
        "trained_on": "kaggle",
        "lightgbm_version_trained": kaggle_env["lightgbm"],
        "python_version_trained": kaggle_env["python"],
    }
    params.update({f"lgb_{k}": v for k, v in card["params"].items()})

    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run(run_name=f"kaggle-{card['served_run']}") as run:
        mlflow.set_tags({
            "trained_on": "kaggle",
            "served_run": card["served_run"],
            "comparison_run": card["comparison_run"],
            "model_file_sha256": digest,
            "notebook": "02_lightgbm_training_evaluation.ipynb",
            "known_limitations": " | ".join(card["known_limitations"]),
        })
        mlflow.log_params(params)
        logged = flatten_metrics(metrics)
        mlflow.log_metrics(logged)
        for name in REQUIRED_ARTIFACTS:
            mlflow.log_artifact(str(art / name), artifact_path="kaggle_outputs")
        signature = infer_signature(x_parity, y_parity)
        log_kwargs = dict(
            python_model=DemandForecastModel(),
            artifacts={"booster": str(model_path)},
            signature=signature,
            input_example=x_parity.head(5),
            registered_model_name=args.model_name,
            pip_requirements=[
                f"lightgbm=={kaggle_env['lightgbm']}",
                f"pandas=={kaggle_env['pandas']}",
                f"numpy=={kaggle_env['numpy']}",
            ],
        )
        try:
            info = mlflow.pyfunc.log_model(name="model", **log_kwargs)
        except TypeError:
            info = mlflow.pyfunc.log_model(artifact_path="model", **log_kwargs)
        run_id = run.info.run_id
    print(f"\nlogged run {run_id} with {len(logged)} metrics and {len(params)} parameters")
    print(f"eval WAPE served {logged['eval_served_wape']:.4f} | "
          f"horizon day 1 {logged['eval_served_h1_wape']:.4f} vs moving_average_7 {logged['eval_moving_average_7_h1_wape']:.4f}")

    versions = client.search_model_versions(f"name='{args.model_name}'")
    version = max(int(v.version) for v in versions)
    details = client.get_model_version(args.model_name, str(version))
    print(f"registered {args.model_name} version {version}, status {details.status}")

    loaded = mlflow.pyfunc.load_model(f"models:/{args.model_name}/{version}")
    pred = np.asarray(loaded.predict(x_parity)).reshape(-1)
    identical = np.array_equal(pred, y_parity)
    max_diff = float(np.abs(pred - y_parity).max())
    print(f"parity sample rows {len(parity)}, max difference {max_diff:.3g}")
    if not identical:
        print("FAIL reloaded registry model does not reproduce the Kaggle predictions")
        return 1
    print("PASS reloaded registry model reproduces the Kaggle parity predictions exactly")

    summary = {
        "tracking_uri": args.tracking_uri,
        "experiment": {"name": args.experiment, "id": experiment_id},
        "other_experiments": {e.name: e.experiment_id for e in others},
        "run_id": run_id,
        "model_name": args.model_name,
        "model_version": version,
        "model_uri": f"models:/{args.model_name}/{version}",
        "logged_model_uri": info.model_uri,
        "model_file_sha256": digest,
        "parity_rows": int(len(parity)),
        "parity_exact_match": True,
        "eval_wape_served": logged["eval_served_wape"],
    }
    out = Path(args.summary_out) if args.summary_out else art / "mlflow_registration_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
