import sys
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path(__file__).resolve().parent
for _candidate in (_here, _here.parent / "src"):
    if (_candidate / "demand_model.py").exists():
        sys.path.insert(0, str(_candidate))
        break

import demand_features as dfe
import demand_model as dm


def _synthetic(n_rows=4000, seed=0):
    rng = np.random.default_rng(seed)
    cols = dm.model_feature_columns("none", True)
    frame = pd.DataFrame({c: rng.uniform(0, 5, n_rows) for c in cols})
    for c in dm.CATEGORICAL_FEATURES:
        frame[c] = rng.integers(0, 6, n_rows)
    y = np.maximum(frame["sales_lag_7"].to_numpy() * 0.8 + rng.normal(0, 0.3, n_rows), 0)
    return frame, y, cols


def test_discount_variant_removes_only_same_day_discount():
    with_d = dm.model_feature_columns("none", True)
    without_d = dm.model_feature_columns("none", False)
    assert set(with_d) - set(without_d) == set(dm.SAME_DAY_DISCOUNT_FEATURES)
    assert "discount_mean_7_off7" in without_d
    assert not any(c.startswith("oracle_") for c in with_d + without_d)


def test_categorical_features_are_model_inputs():
    assert set(dm.CATEGORICAL_FEATURES) <= set(dfe.feature_columns("none"))


def test_train_predict_roundtrip_and_clipping(tmp_path=None):
    frame, y, cols = _synthetic()
    x = dm.to_matrix(frame, cols)
    booster = dm.train(x[:3000], y[:3000], cols, "l2", 50, x[3000:], y[3000:], early_stopping_rounds=10, num_threads=1)
    pred = dm.predict(booster, frame.iloc[3000:])
    assert pred.shape == (1000,) and (pred >= 0).all()
    shuffled = frame.iloc[3000:][list(reversed(cols))]
    assert np.array_equal(pred, dm.predict(booster, shuffled))
    text = booster.model_to_string(num_iteration=-1)
    import lightgbm as lgb

    reloaded = lgb.Booster(model_str=text)
    assert np.array_equal(pred, dm.predict(reloaded, frame.iloc[3000:], num_iteration=booster.best_iteration))


def test_matrix_predict_rejects_wrong_column_order():
    frame, y, cols = _synthetic(600)
    x = dm.to_matrix(frame, cols)
    booster = dm.train(x, y, cols, "l2", 5, num_threads=1)
    try:
        dm.predict(booster, x, columns=list(reversed(cols)))
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong column order")


def test_missing_feature_column_raises():
    frame, y, cols = _synthetic(600)
    x = dm.to_matrix(frame, cols)
    booster = dm.train(x, y, cols, "l2", 5, num_threads=1)
    try:
        dm.predict(booster, frame.drop(columns=["sales_lag_7"]))
    except KeyError:
        return
    raise AssertionError("expected KeyError for missing column")


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print(f"PASS {n}")
    print(f"{len(names)} passed")
