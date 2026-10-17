"""The prediction API: health, predictions, validation, and training/serving parity."""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from make_demo_model import train_demo_model  # noqa: E402

from api import model as api_model  # noqa: E402
from api.main import app  # noqa: E402
from src import etl, features  # noqa: E402

TRIP = {
    "pickup_latitude": 40.7580, "pickup_longitude": -73.9855,      # Times Square
    "dropoff_latitude": 40.7484, "dropoff_longitude": -73.9857,    # Empire State Building
    "pickup_datetime": "2016-06-15T17:30:00", "passenger_count": 1,
}


@pytest.fixture(scope="module")
def model_file(tmp_path_factory):
    return train_demo_model(tmp_path_factory.mktemp("model") / "model.txt")


@pytest.fixture
def client(model_file, monkeypatch):
    monkeypatch.delenv("MODEL_S3_URI", raising=False)
    monkeypatch.setenv("MODEL_PATH", str(model_file))
    monkeypatch.setenv("MODEL_VERSION", "test-version")
    with TestClient(app) as c:        # "with" runs the startup code that loads the model
        yield c


def errors(response):
    return {e["field"]: e["message"] for e in response.json()["details"]}


# ---------- health and predictions ----------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"]
    assert body["model_version"] == "test-version"
    assert body["n_features"] == len(features.FEATURES)
    assert len(body["model_sha256"]) == 12


def test_predict_returns_minutes(client):
    r = client.post("/predict", json=TRIP)
    assert r.status_code == 200
    body = r.json()
    assert 0 < body["predicted_duration_minutes"] < 360
    assert body["predicted_duration_seconds"] == pytest.approx(body["predicted_duration_minutes"] * 60, abs=3)
    assert body["straight_line_distance_km"] == pytest.approx(1.07, abs=0.01)
    assert body["warnings"] == []
    assert r.headers["x-request-id"]


def test_month_outside_training_period_warns(client):
    r = client.post("/predict", json={**TRIP, "pickup_datetime": "2016-11-15T17:30:00"})
    assert r.status_code == 200
    assert "outside the training period" in r.json()["warnings"][0]


def test_timezone_is_converted_to_new_york_time(client):
    local = client.post("/predict", json=TRIP).json()
    utc = client.post("/predict", json={**TRIP, "pickup_datetime": "2016-06-15T21:30:00Z"}).json()  # = 17:30 EDT
    assert utc["predicted_duration_seconds"] == local["predicted_duration_seconds"]


def test_demo_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "NYC Taxi Trip Duration" in r.text and 'fetch("predict"' in r.text


# ---------- validation ----------

@pytest.mark.parametrize("field, value", [
    ("pickup_latitude", 42.5), ("dropoff_latitude", 40.0),
    ("pickup_longitude", -75.0), ("dropoff_longitude", -73.5),
])
def test_coordinates_outside_nyc_rejected(client, field, value):
    r = client.post("/predict", json={**TRIP, field: value})
    assert r.status_code == 422
    assert "outside New York City" in errors(r)[field]


@pytest.mark.parametrize("passengers", [0, 7, -1])
def test_impossible_passenger_count_rejected(client, passengers):
    r = client.post("/predict", json={**TRIP, "passenger_count": passengers})
    assert r.status_code == 422
    assert "passenger_count" in errors(r)


def test_same_pickup_and_dropoff_rejected(client):
    same = {**TRIP, "dropoff_latitude": TRIP["pickup_latitude"], "dropoff_longitude": TRIP["pickup_longitude"]}
    r = client.post("/predict", json=same)
    assert r.status_code == 422
    assert "same point" in str(r.json())


def test_missing_and_unknown_fields_rejected(client):
    missing = {k: v for k, v in TRIP.items() if k != "pickup_datetime"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**TRIP, "trip_duration": 600}).status_code == 422   # no leakage via API


def test_all_errors_reported_together(client):
    r = client.post("/predict", json={**TRIP, "pickup_latitude": 42.5, "passenger_count": 9})
    assert set(errors(r)) == {"pickup_latitude", "passenger_count"}


# ---------- training / serving parity ----------

def test_api_features_identical_to_training_features(raw_sample, cfg, con, model_file):
    """The API must compute exactly the same features as the training pipeline for the same trip."""
    clean, _ = etl.clean_sql(raw_sample, cfg, con)
    training = features.build_features(clean, cfg)[features.FEATURES].reset_index(drop=True)

    service = api_model.ModelService.__new__(api_model.ModelService)
    service.cfg = cfg
    request_fields = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude",
                      "pickup_datetime", "passenger_count", "vendor_id", "store_and_fwd_flag"]
    serving = service.to_features(clean[request_fields])[features.FEATURES].reset_index(drop=True)

    assert serving.dtypes.equals(training.dtypes)
    assert serving.equals(training)


def test_api_prediction_matches_model(client, model_file, raw_sample, cfg, con):
    """End to end: the HTTP response equals the model's own prediction on training-pipeline features."""
    clean, _ = etl.clean_sql(raw_sample, cfg, con)
    trip = clean.iloc[0]
    expected = np.expm1(lgb.Booster(model_file=str(model_file)).predict(
        features.build_features(clean.iloc[[0]], cfg)[features.FEATURES]))[0]
    def plain(value):                       # numpy / pandas values -> plain JSON values
        return value.isoformat() if hasattr(value, "isoformat") else getattr(value, "item", lambda: value)()

    payload = {f: plain(trip[f])
               for f in ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude",
                         "pickup_datetime", "passenger_count", "vendor_id", "store_and_fwd_flag"]}
    r = client.post("/predict", json=payload)
    assert r.json()["predicted_duration_seconds"] == pytest.approx(expected, abs=0.1)


# ---------- model loading ----------

def test_missing_model_reports_unavailable(monkeypatch, tmp_path):
    monkeypatch.delenv("MODEL_S3_URI", raising=False)
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "does_not_exist.txt"))
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json=TRIP).status_code == 503


def test_model_downloaded_from_s3(monkeypatch, tmp_path, model_file):
    """MODEL_S3_URI: the model is downloaded at startup (boto3 is replaced by a fake here)."""
    downloads = []

    class FakeS3:
        def download_file(self, bucket, key, target):
            downloads.append((bucket, key))
            Path(target).write_bytes(model_file.read_bytes())

    import boto3
    monkeypatch.setattr(boto3, "client", lambda service: FakeS3())
    monkeypatch.setenv("MODEL_S3_URI", "s3://my-bucket/models/model.txt")
    monkeypatch.setenv("MODEL_CACHE_PATH", str(tmp_path / "cached.txt"))
    with TestClient(app) as c:
        body = c.get("/health").json()
    assert downloads == [("my-bucket", "models/model.txt")]
    assert body["model_source"] == "s3://my-bucket/models/model.txt"
