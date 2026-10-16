"""Time-based split, weather features, SHAP and experiment tracking."""

import numpy as np
import pandas as pd
import pytest

from src import etl, explain, features, train
from src.tracking import Tracker


@pytest.fixture
def feature_data(raw_sample, cfg, con):
    clean, _ = etl.clean_sql(raw_sample, cfg, con)
    return features.build_features(clean, cfg)


def test_time_split_never_mixes_days(feature_data, cfg):
    sample_cfg = {**cfg, "model": {**cfg["model"], "sample_frac": 1.0}}
    tr, te = train.make_time_split(feature_data, sample_cfg)
    cutoff = pd.Timestamp(cfg["evaluation"]["time_split_test_start"])
    assert (tr["pickup_datetime"] < cutoff).all()
    assert (te["pickup_datetime"] >= cutoff).all()
    assert len(tr) + len(te) == len(feature_data)
    assert not set(tr["id"]) & set(te["id"])


def test_time_validation_uses_latest_training_trips(feature_data, cfg):
    """Early stopping on the time split must not peek at later trips: validation = latest 10%."""
    sample_cfg = {**cfg, "model": {**cfg["model"], "sample_frac": 1.0}}
    tr, _ = train.make_time_split(feature_data, sample_cfg)
    model = train.train_improved(tr, cfg, validation="time")
    assert model.best_iteration > 0


def test_day_features_keep_row_order(feature_data):
    weather = pd.DataFrame({"date": pd.date_range("2016-01-01", "2016-06-30"), "precip_mm": 1.0,
                            "snow_cm": 0.0, "snow_depth_cm": 0.0, "temp_mean_c": 10.0})
    out = train.add_day_features(feature_data, weather)
    assert list(out["id"]) == list(feature_data["id"])
    assert out["pickup_dayofyear"].between(1, 182).all()
    assert (out["precip_mm"] == 1.0).all()


def test_shap_values_add_up_to_prediction(feature_data, cfg):
    """SHAP values plus the base value must equal the model's prediction for every trip."""
    sample_cfg = {**cfg, "model": {**cfg["model"], "sample_frac": 1.0}}
    tr, te = train.make_split(feature_data, sample_cfg)
    model = train.train_improved(tr, cfg)
    X = te[features.FEATURES]
    values = explain.shap_values(model, X)
    reconstructed = values.values.sum(axis=1) + values.base_values
    np.testing.assert_allclose(reconstructed, model.predict(X, num_iteration=model.best_iteration), atol=1e-6)


def test_tracker_disabled_does_nothing(cfg):
    tracker = Tracker({**cfg, "mlflow": {"enabled": False}})
    tracker.log_run("x", split="random", model=None, kind="lightgbm", features=[], params={}, metrics={})
