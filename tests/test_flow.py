"""The Prefect flow runs every task in order on the synthetic sample."""

import shutil

import pytest
import yaml
from prefect.testing.utilities import prefect_test_harness

from flows.pipeline_flow import nyc_taxi_pipeline
from tests.conftest import FIXTURES


@pytest.fixture(scope="module", autouse=True)
def prefect_api():
    with prefect_test_harness():          # temporary Prefect database, nothing written to ~/.prefect
        yield


def test_flow_runs_end_to_end(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / "trips_sample.csv", raw_dir / "train.csv")
    shutil.copy(FIXTURES / "weather_sample.csv", raw_dir / "weather.csv")
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {
        "raw_trips": str(raw_dir / "train.csv"), "raw_weather": str(raw_dir / "weather.csv"),
        "processed_dir": str(tmp_path / "processed"), "models_dir": str(tmp_path / "models"),
        "reports_dir": str(tmp_path / "reports"), "figures_dir": str(tmp_path / "reports" / "figures"),
        "mlflow_db": str(tmp_path / "mlflow.db"), "mlflow_artifacts": str(tmp_path / "mlruns"),
    }
    cfg["model"]["sample_frac"] = 1.0
    cfg["speed_map"]["min_trips"] = 1
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    metrics = nyc_taxi_pipeline(config_path=str(config_path), fast=True, mlflow=False)

    assert metrics["rows_removed"] == 39
    assert metrics["improved_model"]["mae_s"] > 0
    for output in ["processed/clean.parquet", "processed/features.parquet", "models/lgbm_improved.txt",
                   "reports/metrics.json", "reports/model_card.md", "reports/speed_map.html"]:
        assert (tmp_path / output).exists(), output
