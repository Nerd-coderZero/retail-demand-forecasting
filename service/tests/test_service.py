"""Service tests: registry load, prediction correctness against an independent code path,
batching equivalence, input validation and metrics exposure."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT.parent / "src", ROOT / "src"):
    if str(candidate) not in sys.path and candidate.exists():
        sys.path.insert(0, str(candidate))

import demand_features as dfe


def make_client(batching: bool):
    os.environ["BATCHING_ENABLED"] = "true" if batching else "false"
    for module in [m for m in list(sys.modules) if m.startswith("app")]:
        del sys.modules[module]
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


@pytest.fixture(scope="module")
def client():
    with make_client(True) as c:
        yield c


def reference_predictions(items, history_path, model_store):
    """Independent path: features from demand_features, booster loaded straight from the model file."""
    import lightgbm as lgb

    manifest = json.loads((Path(model_store) / "manifest.json").read_text())
    artifacts = list((Path(model_store) / "mlartifacts" / "models").glob("*/artifacts"))
    booster_file = next(artifacts[0].rglob("lgbm_served.txt"))
    booster = lgb.Booster(model_file=str(booster_file))

    history = pd.read_parquet(history_path)
    history[dfe.DATE_COL] = pd.to_datetime(history[dfe.DATE_COL]).dt.normalize()
    last_day = history[dfe.DATE_COL].max()
    horizon = pd.date_range(last_day + pd.Timedelta(days=1), periods=manifest["horizon_days"], freq="D")
    keys = {(int(i.store_id), int(i.product_id)) for i in items}
    subset = history[history.set_index(dfe.KEY_COLS).index.isin(keys)].copy()
    future = subset.drop_duplicates(dfe.KEY_COLS)[dfe.STATIC_COLS].merge(
        pd.DataFrame({dfe.DATE_COL: horizon}), how="cross")
    for column in dfe.panel_value_cols("none"):
        if column not in future:
            future[column] = np.nan
    future["activity_flag"] = 0.0
    future["holiday_flag"] = (future[dfe.DATE_COL].dt.dayofweek >= 5).astype(float)
    features = dfe.build_features(pd.concat([subset, future], ignore_index=True), rows_from_date=horizon[0])
    x = features[booster.feature_name()].to_numpy(np.float32)
    features["prediction"] = np.clip(booster.predict(x), 0.0, None)
    lookup = features.set_index(dfe.KEY_COLS + [dfe.DATE_COL])["prediction"]
    return [float(lookup.loc[(i.store_id, i.product_id, pd.Timestamp(i.date))]) for i in items]


class Item:
    def __init__(self, store_id, product_id, date):
        self.store_id, self.product_id, self.date = store_id, product_id, date

    def payload(self):
        return {"store_id": self.store_id, "product_id": self.product_id, "date": self.date}


def sample_items(client, n=5):
    info = client.get("/model").json()
    first_day = info["forecast_window"][0]
    history = pd.read_parquet(os.environ["HISTORY_PATH"], columns=dfe.KEY_COLS).drop_duplicates()
    picks = history.sample(n, random_state=7)
    return [Item(int(r.store_id), int(r.product_id), first_day) for r in picks.itertuples()]


def test_health_and_model_metadata(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}
    info = client.get("/model").json()
    assert info["model_uri"].startswith("models:/")
    assert info["history_days"] == dfe.REQUIRED_HISTORY_DAYS
    assert info["horizon_days"] == dfe.HORIZON
    assert info["feature_count"] == len(info["feature_columns"]) if "feature_columns" in info else True
    assert info["series"] > 0


def test_predictions_match_an_independent_feature_and_model_path(client):
    items = sample_items(client, 5)
    response = client.post("/predict", json={"items": [i.payload() for i in items]})
    assert response.status_code == 200, response.text
    got = [p["forecast"] for p in response.json()["predictions"]]
    expected = reference_predictions(items, os.environ["HISTORY_PATH"], os.environ["MODEL_STORE"])
    assert np.allclose(got, expected, rtol=0, atol=1e-12), list(zip(got, expected))
    assert all(v >= 0 for v in got)


def test_every_horizon_day_is_servable(client):
    info = client.get("/model").json()
    days = pd.date_range(info["forecast_window"][0], info["forecast_window"][1], freq="D")
    item = sample_items(client, 1)[0]
    payload = [{"store_id": item.store_id, "product_id": item.product_id, "date": str(d.date())} for d in days]
    response = client.post("/predict", json={"items": payload})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == len(days)


def test_batching_off_returns_identical_predictions(client):
    items = sample_items(client, 4)
    batched = client.post("/predict", json={"items": [i.payload() for i in items]}).json()["predictions"]
    with make_client(False) as unbatched_client:
        unbatched = unbatched_client.post("/predict", json={"items": [i.payload() for i in items]}).json()["predictions"]
    assert [p["forecast"] for p in batched] == [p["forecast"] for p in unbatched]


def test_known_covariates_change_the_forecast(client):
    item = sample_items(client, 1)[0]
    base = client.post("/predict", json={"items": [item.payload()]}).json()["predictions"][0]
    promo = client.post("/predict", json={"items": [{**item.payload(), "activity_flag": 1, "holiday_flag": 1}]}).json()["predictions"][0]
    assert promo["activity_flag"] == 1 and promo["holiday_flag"] == 1
    assert base["forecast"] != promo["forecast"]


def test_unknown_series_and_out_of_window_dates_are_rejected(client):
    info = client.get("/model").json()
    bad_series = client.post("/predict", json={"items": [{"store_id": 10**9, "product_id": 1, "date": info["forecast_window"][0]}]})
    assert bad_series.status_code == 422
    item = sample_items(client, 1)[0]
    outside = str(pd.Timestamp(info["forecast_window"][1]) + pd.Timedelta(days=1))[:10]
    bad_date = client.post("/predict", json={"items": [{**item.payload(), "date": outside}]})
    assert bad_date.status_code == 422
    empty = client.post("/predict", json={"items": []})
    assert empty.status_code == 422


def test_metrics_endpoint_exposes_the_expected_series(client):
    client.post("/predict", json={"items": [i.payload() for i in sample_items(client, 2)]})
    body = client.get("/metrics").text
    for name in ("forecast_requests_total", "forecast_request_latency_seconds_bucket", "forecast_batch_size_bucket",
                 "forecast_feature_build_seconds_bucket", "forecast_model_predict_seconds_bucket",
                 "forecast_model_info", "forecast_ready"):
        assert name in body, name


def test_request_order_does_not_change_which_series_each_forecast_belongs_to(client):
    items = sample_items(client, 6)
    ascending = sorted(items, key=lambda i: (i.store_id, i.product_id))
    descending = list(reversed(ascending))
    def mapping(order):
        rows = client.post("/predict", json={"items": [i.payload() for i in order]}).json()["predictions"]
        return {(r["store_id"], r["product_id"], r["date"]): r["forecast"] for r in rows}
    assert mapping(ascending) == mapping(descending)
