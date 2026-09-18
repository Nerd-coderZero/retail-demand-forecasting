"""Export a registered model version into a self-contained MLflow store for the container image.

The service must resolve `models:/<name>/<version>` through the registry rather than open a model
file directly, so the image carries a copy of the registry database with its artifact locations
rewritten from host paths to the container path. The user's own store is never modified.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
import re
from urllib.parse import unquote, urlparse

import mlflow
import numpy as np
import pandas as pd

LOCATION_COLUMNS = [
    ("experiments", "artifact_location"),
    ("runs", "artifact_uri"),
    ("logged_models", "artifact_location"),
    ("model_versions", "storage_location"),
]


def uri_to_path(uri: str) -> Path:
    """file:// URI to a local path, Windows drive letters included."""
    path = unquote(urlparse(uri).path)
    if re.match(r"^/[A-Za-z]:", path):
        path = path[1:]
    return Path(path)


def sqlite_path_from_uri(uri: str) -> Path:
    if not uri.startswith("sqlite:///"):
        raise ValueError("only a sqlite tracking uri can be exported")
    return Path(uri[len("sqlite:///"):]).resolve()


def rewrite_locations(db_path: Path, old_root: str, new_root: str) -> int:
    changed = 0
    with sqlite3.connect(db_path) as con:
        for table, column in LOCATION_COLUMNS:
            cur = con.execute(
                f"update {table} set {column} = replace({column}, ?, ?) where {column} like ?",
                (old_root, new_root, f"{old_root}%"),
            )
            changed += cur.rowcount
        con.commit()
    return changed


def artifact_root_of(db_path: Path, model_name: str, version: int) -> tuple[str, str]:
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "select storage_location from model_versions where name = ? and version = ?",
            (model_name, str(version)),
        ).fetchone()
    if not row:
        raise SystemExit(f"model version not found: {model_name} v{version}")
    storage = row[0]
    marker = "/mlartifacts/"
    if marker not in storage:
        raise SystemExit(f"unexpected artifact layout: {storage}")
    return storage[: storage.index(marker) + len(marker) - 1], storage


def verify(db_path: Path, model_uri: str, parity: pd.DataFrame, feature_columns: list[str]) -> bool:
    script = (
        "import json,sys,numpy as np,pandas as pd,mlflow\n"
        "tracking, uri, parquet, cols = sys.argv[1:5]\n"
        "mlflow.set_tracking_uri(tracking)\n"
        "m = mlflow.pyfunc.load_model(uri)\n"
        "frame = pd.read_parquet(parquet)\n"
        "columns = json.loads(cols)\n"
        "pred = np.asarray(m.predict(frame[columns].astype('float64'))).reshape(-1)\n"
        "print(json.dumps({'max_diff': float(np.abs(pred - frame['prediction'].to_numpy()).max()),\n"
        "                  'exact': bool(np.array_equal(pred, frame['prediction'].to_numpy())),\n"
        "                  'rows': int(len(frame))}))\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        parquet = Path(tmp) / "parity.parquet"
        parity.to_parquet(parquet, index=False)
        result = subprocess.run(
            [sys.executable, "-c", script, f"sqlite:///{db_path.as_posix()}", model_uri, str(parquet), json.dumps(feature_columns)],
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr[-2000:])
        return False
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    print(f"verification: {payload}")
    return payload["exact"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="mlflow_registration_summary.json written by register_model.py")
    parser.add_argument("--artifacts-dir", required=True, help="folder with nb2_model_card.json and nb2_parity_sample.parquet")
    parser.add_argument("--out", required=True, help="output folder, copied into the image as /app/model_store")
    parser.add_argument("--container-root", default="/app/model_store")
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    art = Path(args.artifacts_dir)
    card = json.loads((art / "nb2_model_card.json").read_text())
    parity = pd.read_parquet(art / "nb2_parity_sample.parquet")
    model_name, version = summary["model_name"], int(summary["model_version"])
    model_uri = f"models:/{model_name}/{version}"

    source_db = sqlite_path_from_uri(summary["tracking_uri"])
    if not source_db.exists():
        raise SystemExit(f"tracking database not found: {source_db}")
    out = Path(args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    (out / "mlartifacts").mkdir(parents=True)

    host_root_uri, storage_location = artifact_root_of(source_db, model_name, version)
    if not storage_location.startswith(host_root_uri + "/"):
        raise SystemExit(f"model artifacts sit outside the artifact root: {storage_location}")
    model_rel = unquote(storage_location[len(host_root_uri) + 1:])
    src_model_dir = uri_to_path(host_root_uri) / model_rel
    if not src_model_dir.exists():
        raise SystemExit(f"model artifacts not found on disk: {src_model_dir}")
    dst_model_dir = out / "mlartifacts" / model_rel
    dst_model_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_model_dir, dst_model_dir)
    print(f"copied model artifacts: {src_model_dir} -> {dst_model_dir}")

    local_db = out / "mlflow.db"
    shutil.copy2(source_db, local_db)
    local_root_uri = (out / "mlartifacts").as_uri()
    rewrote = rewrite_locations(local_db, host_root_uri, local_root_uri)
    print(f"rewrote {rewrote} artifact locations for local verification")
    if not verify(local_db, model_uri, parity, card["feature_columns"]):
        print("FAIL exported store does not reproduce the parity predictions")
        return 1
    print("PASS exported store reproduces the parity predictions exactly")

    container_root_uri = f"file://{args.container_root}/mlartifacts"
    rewrite_locations(local_db, local_root_uri, container_root_uri)
    manifest = {
        "model_name": model_name,
        "model_version": version,
        "model_uri": model_uri,
        "run_id": summary["run_id"],
        "model_file_sha256": summary["model_file_sha256"],
        "container_tracking_uri": f"sqlite:///{args.container_root}/mlflow.db",
        "container_artifact_root": container_root_uri,
        "feature_columns": card["feature_columns"],
        "required_history_days": card["required_history_days"],
        "horizon_days": card["horizon_days"],
        "eval_wape": card["eval_metrics_served"]["wape"],
        "trained_lightgbm": json.loads((art / "nb2_environment.json").read_text())["lightgbm"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\nexported to {out} ({size / 1e6:.1f} MB)")
    print(f"container tracking uri: {manifest['container_tracking_uri']}")
    print(f"model uri: {model_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
