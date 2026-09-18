"""FastAPI service for 7-day retail demand forecasts.

The model is resolved through the MLflow registry (models:/<name>/<version>) rather than opened as
a file, so the service exercises the registry that register_model.py wrote.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

import demand_features as dfe
from app import metrics
from app.batching import Batcher
from app.config import settings
from app.history import HistoryError, HistoryStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("forecast")

state: dict = {"ready": False}


class ForecastItem(BaseModel):
    store_id: int
    product_id: int
    date: str
    activity_flag: int | None = Field(default=None, ge=0, le=1)
    holiday_flag: int | None = Field(default=None, ge=0, le=1)
    discount: float | None = Field(default=None, gt=0, le=2)

    @field_validator("date")
    @classmethod
    def valid_date(cls, value: str) -> str:
        try:
            pd.Timestamp(value)
        except ValueError as exc:
            raise ValueError(f"date is not a valid date: {value}") from exc
        return value


class ForecastRequest(BaseModel):
    items: list[ForecastItem] = Field(min_length=1)


class ForecastResponse(BaseModel):
    predictions: list[dict]
    model_version: str
    model_uri: str


def load_model():
    import mlflow

    manifest_path = Path(settings.model_store) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tracking_uri = settings.tracking_uri or manifest["container_tracking_uri"]
    model_uri = settings.model_uri or manifest["model_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    name, version = model_uri.removeprefix("models:/").split("/")
    registered = client.get_model_version(name, version)
    if registered.status != "READY":
        raise RuntimeError(f"registry model {model_uri} is not READY: {registered.status}")
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info("loaded %s (run %s) from registry at %s", model_uri, registered.run_id, tracking_uri)
    return model, manifest, model_uri, registered


def predict_batch(payloads: list[list[ForecastItem]]) -> list[list[dict]]:
    """One model call for every request in the batch. Order of results matches order of payloads."""
    history: HistoryStore = state["history"]
    model = state["model"]
    columns = state["feature_columns"]

    rows: list[int] = []
    resolved: list[list[tuple[int, int, ForecastItem]]] = []
    known: dict[tuple[int, int], dict] = {}
    for payload in payloads:
        entries = []
        for item in payload:
            series_row = history.series_row(item.store_id, item.product_id)
            day = history.date_position(pd.Timestamp(item.date).normalize())
            overrides = {}
            if item.activity_flag is not None:
                overrides["activity_flag"] = float(item.activity_flag)
            if item.holiday_flag is not None:
                overrides["holiday_flag"] = float(item.holiday_flag)
            if item.discount is not None:
                overrides["discount"] = float(item.discount)
            if overrides:
                known.setdefault((series_row, day), {}).update(overrides)
            entries.append((series_row, day, item))
            rows.append(series_row)
        resolved.append(entries)

    unique_rows = list(dict.fromkeys(rows))
    started = time.perf_counter()
    features = history.build_features(unique_rows, known)
    metrics.FEATURE_SECONDS.observe(time.perf_counter() - started)

    started = time.perf_counter()
    predictions = np.asarray(model.predict(features[columns].astype("float64"))).reshape(-1)
    metrics.PREDICT_SECONDS.observe(time.perf_counter() - started)

    horizon = history.horizon_days
    if len(features) != len(unique_rows) * horizon:
        raise RuntimeError("feature frame does not hold one block of horizon days per requested series")
    block_keys = features[dfe.KEY_COLS].to_numpy()[::horizon]
    position = {(int(s), int(p)): i for i, (s, p) in enumerate(block_keys)}
    activity = features["activity_flag"].to_numpy()
    holiday = features["holiday_flag"].to_numpy()
    results: list[list[dict]] = []
    for entries in resolved:
        out = []
        for series_row, day, item in entries:
            offset = position[(item.store_id, item.product_id)] * horizon + day
            out.append({
                "store_id": item.store_id,
                "product_id": item.product_id,
                "date": item.date,
                "forecast": float(predictions[offset]),
                "activity_flag": int(activity[offset]),
                "holiday_flag": int(holiday[offset]),
            })
        results.append(out)
    return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, manifest, model_uri, registered = load_model()
    history_frame = pd.read_parquet(settings.history_path)
    history = HistoryStore(history_frame, horizon_days=manifest["horizon_days"],
                           history_days=manifest["required_history_days"])
    state.update({
        "model": model,
        "manifest": manifest,
        "model_uri": model_uri,
        "model_version": registered.version,
        "history": history,
        "feature_columns": manifest["feature_columns"],
    })
    smoke = predict_batch([[ForecastItem(store_id=int(history.keys["store_id"][0]),
                                         product_id=int(history.keys["product_id"][0]),
                                         date=str(history.horizon_dates[0].date()))]])
    if not np.isfinite(smoke[0][0]["forecast"]):
        raise RuntimeError("startup smoke prediction did not return a finite forecast")
    logger.info("startup smoke forecast %s", smoke[0][0])

    batcher = Batcher(predict_batch, settings.max_batch_size, settings.max_batch_wait_ms)
    batcher.on_batch = lambda size, waited, seconds: (
        metrics.BATCH_SIZE.observe(size), metrics.BATCH_WAIT.observe(waited)
    )
    if settings.batching_enabled:
        await batcher.start()
    state["batcher"] = batcher
    state["ready"] = True
    metrics.READY.set(1)
    metrics.MODEL_INFO.labels(model_uri=model_uri, version=str(registered.version),
                              sha256=manifest["model_file_sha256"][:16]).set(1)
    logger.info("service ready: batching=%s max_batch_size=%s max_wait_ms=%s series=%s horizon=%s..%s",
                settings.batching_enabled, settings.max_batch_size, settings.max_batch_wait_ms,
                history.n_series, history.horizon_dates[0].date(), history.horizon_dates[-1].date())
    try:
        yield
    finally:
        state["ready"] = False
        metrics.READY.set(0)
        await batcher.stop()


app = FastAPI(title="Retail demand forecast", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    endpoint = request.url.path
    started = time.perf_counter()
    status = 500
    metrics.IN_FLIGHT.inc()
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        metrics.IN_FLIGHT.dec()
        metrics.REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - started)
        metrics.REQUESTS.labels(endpoint=endpoint, status=str(status)).inc()


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    if not state.get("ready"):
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ready"}


@app.get("/model")
async def model_info():
    if not state.get("ready"):
        raise HTTPException(status_code=503, detail="model not loaded")
    manifest = state["manifest"]
    history: HistoryStore = state["history"]
    return {
        "model_uri": state["model_uri"],
        "model_version": state["model_version"],
        "run_id": manifest["run_id"],
        "model_file_sha256": manifest["model_file_sha256"],
        "trained_lightgbm": manifest["trained_lightgbm"],
        "eval_wape": manifest["eval_wape"],
        "feature_count": len(manifest["feature_columns"]),
        "horizon_days": history.horizon_days,
        "history_days": history.history_days,
        "series": history.n_series,
        "forecast_window": [str(history.horizon_dates[0].date()), str(history.horizon_dates[-1].date())],
        "batching": {
            "enabled": settings.batching_enabled,
            "max_batch_size": settings.max_batch_size,
            "max_batch_wait_ms": settings.max_batch_wait_ms,
        },
    }


@app.get("/metrics")
async def prometheus_metrics():
    return Response(content=metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8")


@app.post("/predict", response_model=ForecastResponse)
async def predict(request: ForecastRequest):
    if not state.get("ready"):
        raise HTTPException(status_code=503, detail="model not loaded")
    if len(request.items) > settings.max_items_per_request:
        raise HTTPException(status_code=413, detail=f"at most {settings.max_items_per_request} items per request")
    try:
        if settings.batching_enabled:
            predictions = await state["batcher"].submit(request.items)
        else:
            import asyncio

            predictions = (await asyncio.to_thread(predict_batch, [request.items]))[0]
            metrics.BATCH_SIZE.observe(1)
            metrics.BATCH_WAIT.observe(0.0)
    except HistoryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    metrics.ITEMS.inc(len(predictions))
    return ForecastResponse(predictions=predictions, model_version=str(state["model_version"]),
                            model_uri=state["model_uri"])
