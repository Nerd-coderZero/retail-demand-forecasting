"""Service configuration, read once from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model_store: str = field(default_factory=lambda: os.environ.get("MODEL_STORE", "/app/model_store"))
    history_path: str = field(default_factory=lambda: os.environ.get("HISTORY_PATH", "/app/data/serving_history.parquet"))
    model_uri: str = field(default_factory=lambda: os.environ.get("MODEL_URI", ""))
    tracking_uri: str = field(default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI", ""))
    batching_enabled: bool = field(default_factory=lambda: _bool("BATCHING_ENABLED", True))
    max_batch_size: int = field(default_factory=lambda: _int("MAX_BATCH_SIZE", 32))
    max_batch_wait_ms: int = field(default_factory=lambda: _int("MAX_BATCH_WAIT_MS", 10))
    max_items_per_request: int = field(default_factory=lambda: _int("MAX_ITEMS_PER_REQUEST", 500))
    predict_threads: int = field(default_factory=lambda: _int("PREDICT_THREADS", 1))


settings = Settings()
