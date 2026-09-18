"""Load profile for the forecast service.

Two request shapes, mixed: single-item requests (the case micro-batching is meant to help) and
bulk requests of many items (the case where one call already carries work).
"""
from __future__ import annotations

import csv
import os
import random
from pathlib import Path

from locust import FastHttpUser, between, events, task

SERIES_FILE = Path(os.environ.get("SERIES_FILE", Path(__file__).parent / "series_sample.csv"))
BULK_ITEMS = int(os.environ.get("BULK_ITEMS", 50))
BULK_WEIGHT = int(os.environ.get("BULK_WEIGHT", 1))
SINGLE_WEIGHT = int(os.environ.get("SINGLE_WEIGHT", 9))

SERIES: list[tuple[int, int]] = []
DATES: list[str] = []


@events.test_start.add_listener
def load_series(environment, **_):
    if not SERIES_FILE.exists():
        raise SystemExit(f"series sample not found: {SERIES_FILE} (run run_loadtest.py, which generates it)")
    with open(SERIES_FILE) as fh:
        for row in csv.DictReader(fh):
            SERIES.append((int(row["store_id"]), int(row["product_id"])))
    print(f"loaded {len(SERIES)} series for the load test")


class ForecastUser(FastHttpUser):
    wait_time = between(0, 0)

    def on_start(self):
        if not DATES:
            info = self.client.get("/model").json()
            import datetime as dt

            start = dt.date.fromisoformat(info["forecast_window"][0])
            end = dt.date.fromisoformat(info["forecast_window"][1])
            DATES.extend(str(start + dt.timedelta(days=i)) for i in range((end - start).days + 1))

    def _item(self):
        store_id, product_id = random.choice(SERIES)
        return {"store_id": store_id, "product_id": product_id, "date": random.choice(DATES)}

    @task(SINGLE_WEIGHT)
    def single(self):
        with self.client.post("/predict", json={"items": [self._item()]}, name="/predict single", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"status {r.status_code}: {r.text[:200]}")

    @task(BULK_WEIGHT)
    def bulk(self):
        items = [self._item() for _ in range(BULK_ITEMS)]
        with self.client.post("/predict", json={"items": items}, name=f"/predict bulk x{BULK_ITEMS}", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"status {r.status_code}: {r.text[:200]}")
