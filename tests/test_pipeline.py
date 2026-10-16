"""End-to-end: the whole pipeline runs on the sample and writes its outputs."""

import json
import shutil
import sys

import yaml

from src import pipeline
from tests.conftest import FIXTURES


def test_pipeline_end_to_end(tmp_path, monkeypatch):
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
    }
    cfg["model"]["sample_frac"] = 1.0              # the sample is small: use all of it
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["pipeline", "--config", str(config_path)])
    pipeline.main()

    metrics = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert metrics["rows_removed"] == 39
    assert metrics["rows_clean"] == 961
    assert metrics["improved_model"]["mae_s"] > 0
    assert (tmp_path / "models" / "lgbm_improved.txt").exists()
    assert (tmp_path / "reports" / "sql" / "05_slowest_days.csv").exists()
    assert len(list((tmp_path / "reports" / "figures").glob("*.png"))) == 3
