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
        "mlflow_db": str(tmp_path / "mlflow.db"), "mlflow_artifacts": str(tmp_path / "mlruns"),
    }
    cfg["model"]["sample_frac"] = 1.0              # the sample is small: use all of it
    cfg["mlflow"]["log_models"] = False            # keep the test fast
    cfg["speed_map"]["min_trips"] = 1
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
    figures = {f.name for f in (tmp_path / "reports" / "figures").glob("*.png")}
    assert {"shap_summary.png", "shap_pickup_hour.png", "speed_map.png", "feature_importance.png"} <= figures
    assert (tmp_path / "reports" / "speed_map.html").exists()

    # time-based split was evaluated, and every model run was logged to MLflow
    assert metrics["time_split"]["mae_s"] > 0
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    runs = mlflow.search_runs(experiment_names=[cfg["mlflow"]["experiment"]])
    assert len(runs) == len(metrics["comparison"]) == 10
    assert set(runs["tags.split"]) == {"random", "time"}

    card = (tmp_path / "reports" / "model_card.md").read_text()
    for section in ["## Evaluation", "Time-based split", "## Known weaknesses", "blizzard"]:
        assert section in card
