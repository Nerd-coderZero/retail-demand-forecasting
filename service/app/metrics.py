"""Prometheus metrics exposed on /metrics."""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRY = CollectorRegistry(auto_describe=True)

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0)

REQUESTS = Counter("forecast_requests_total", "HTTP requests", ["endpoint", "status"], registry=REGISTRY)
REQUEST_LATENCY = Histogram(
    "forecast_request_latency_seconds", "End to end request latency", ["endpoint"],
    buckets=LATENCY_BUCKETS, registry=REGISTRY,
)
IN_FLIGHT = Gauge("forecast_requests_in_flight", "Requests being served", registry=REGISTRY)
ITEMS = Counter("forecast_items_total", "Forecast rows returned", registry=REGISTRY)
BATCH_SIZE = Histogram(
    "forecast_batch_size", "Requests coalesced into one model call",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128), registry=REGISTRY,
)
BATCH_WAIT = Histogram(
    "forecast_batch_wait_seconds", "Time a request waited to join a batch",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25), registry=REGISTRY,
)
FEATURE_SECONDS = Histogram(
    "forecast_feature_build_seconds", "Feature construction time per model call",
    buckets=LATENCY_BUCKETS, registry=REGISTRY,
)
PREDICT_SECONDS = Histogram(
    "forecast_model_predict_seconds", "Model prediction time per model call",
    buckets=LATENCY_BUCKETS, registry=REGISTRY,
)
MODEL_INFO = Gauge("forecast_model_info", "Loaded model", ["model_uri", "version", "sha256"], registry=REGISTRY)
READY = Gauge("forecast_ready", "1 when the model and history snapshot are loaded", registry=REGISTRY)


def render() -> bytes:
    return generate_latest(REGISTRY)
