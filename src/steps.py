"""The pipeline as separate steps. Each step reads its inputs from disk and writes its outputs to disk,
so any step can be re-run on its own, and an orchestrator (Prefect, Airflow) can run them as tasks.

    validate -> clean -> build_features -> [spark_crosscheck] -> train -> analytics -> report

src/pipeline.py runs them in order; flows/pipeline_flow.py runs them as a scheduled Prefect flow.
"""

import json
import logging
import time
from datetime import datetime, timezone

import pandas as pd

from src import db, etl, explain, features, plots, schemas, train
from src.model_card import write_model_card
from src.tracking import Tracker

log = logging.getLogger("steps")

ANALYSIS_QUERIES = ["03_daily_kpis.sql", "04_rolling_7day.sql", "05_slowest_days.sql",
                    "06_weather_impact.sql", "07_hourly_by_weekday.sql"]


def prepare(cfg: dict) -> None:
    for key in ["processed_dir", "models_dir", "reports_dir", "figures_dir"]:
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)


def _paths(cfg: dict) -> dict:
    p = cfg["paths"]
    return {"clean": p["processed_dir"] / "clean.parquet", "features": p["processed_dir"] / "features.parquet",
            "summary": p["reports_dir"] / "cleaning_summary.csv", "training": p["reports_dir"] / "training.json"}


def validate(cfg: dict) -> dict:
    """Raw file -> data contract, row count and timestamp checks. Stops the pipeline if the data is bad."""
    return etl.validate_raw(etl.load_raw(cfg["paths"]["raw_trips"]), cfg)


def clean(cfg: dict) -> dict:
    """Raw file -> clean.parquet (SQL cleaning, verified against pandas and the CLEAN contract)."""
    p, out = cfg["paths"], _paths(cfg)
    raw = etl.load_raw(p["raw_trips"])
    con = db.connect()
    clean_df, summary = etl.clean_sql(raw, cfg, con)
    con.close()
    if cfg["cleaning"].get("verify_with_pandas", True):
        etl.verify_sql_matches_pandas(raw, clean_df, summary, cfg)
    schemas.validate(schemas.clean_schema(cfg), clean_df)
    log.info("Clean data passed the CLEAN data contract")
    summary.to_csv(out["summary"], index=False)
    plots.duration_before_after(raw["trip_duration"], clean_df["trip_duration"],
                                p["figures_dir"] / "duration_before_after.png")
    clean_df.to_parquet(out["clean"], index=False)
    return {"rows_raw": len(raw), "rows_removed": len(raw) - len(clean_df), "rows_clean": len(clean_df)}


def build_features(cfg: dict) -> dict:
    """clean.parquet -> features.parquet (FEATURES contract checked)."""
    out = _paths(cfg)
    feats = features.build_features(pd.read_parquet(out["clean"]), cfg)
    schemas.validate(schemas.features_schema(), feats)
    log.info("Features passed the FEATURES data contract")
    feats.to_parquet(out["features"], index=False)
    log.info("Features: %s rows x %d features", f"{len(feats):,}", len(features.FEATURES))
    return {"rows": len(feats)}


def spark_crosscheck(cfg: dict) -> dict:
    """Re-run cleaning + features in PySpark and check they match the pandas/DuckDB outputs exactly."""
    from src import spark_etl

    out = _paths(cfg)
    spark = spark_etl.get_spark()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        clean_df, summary = spark_etl.clean(spark_etl.load_raw(spark, cfg["paths"]["raw_trips"]), cfg)
        result = spark_etl.compare_with_pandas(spark_etl.build_features(clean_df, cfg), summary,
                                               pd.read_parquet(out["features"]), pd.read_csv(out["summary"]))
    finally:
        spark.stop()
    if not result["match"]:
        raise RuntimeError(f"Spark and pandas/DuckDB results differ: {result}")
    log.info("Spark cross-check passed: %s rows, identical rule counts and features", f"{result['rows']:,}")
    return {k: bool(v) if k != "rows" and not isinstance(v, dict) else v for k, v in result.items()}


def _weather(cfg: dict, con):
    """Create the SQL views; return the daily weather table (None if the weather file is missing)."""
    p = cfg["paths"]
    if not p["raw_weather"].exists():
        log.warning("Weather file not found (%s) - skipping weather experiments and SQL analytics",
                    p["raw_weather"])
        return None
    db.execute(con, "02_views.sql", {"features_path": _paths(cfg)["features"], "weather_path": p["raw_weather"]})
    return con.sql("SELECT * FROM weather_daily").df()


def train_models(cfg: dict) -> dict:
    """features.parquet -> all experiments (MLflow), saved model, evaluation tables, figures, SHAP."""
    p, out = cfg["paths"], _paths(cfg)
    feats = pd.read_parquet(out["features"])
    con = db.connect()
    weather = _weather(cfg, con)
    con.close()

    results = train.run_experiments(feats, cfg, weather=weather, tracker=Tracker(cfg))
    model, comparison = results["model"], results["comparison"]
    model.save_model(str(p["models_dir"] / "lgbm_improved.txt"), num_iteration=model.best_iteration)
    comparison.to_csv(p["reports_dir"] / "model_comparison.csv", index=False)
    test, y_test = results["test"], results["test"]["trip_duration"].values
    errors = train.error_by_duration(y_test, results["test_pred_sec"]).reset_index()
    errors.to_csv(p["reports_dir"] / "error_by_duration.csv", index=False)
    plots.feature_importance(model, p["figures_dir"] / "feature_importance.png").to_csv(
        p["reports_dir"] / "feature_importance.csv", index=False)
    plots.predicted_vs_actual(y_test, results["test_pred_sec"], p["figures_dir"] / "predicted_vs_actual.png")

    n = min(cfg["explain"]["shap_sample"], len(test))
    shap_values = explain.shap_values(model, test[features.FEATURES].sample(n, random_state=0))
    mean_shap = explain.summary_plot(shap_values, p["figures_dir"] / "shap_summary.png")
    mean_shap.to_csv(p["reports_dir"] / "shap_importance.csv", index=False)
    explain.hour_dependence_plot(shap_values, p["figures_dir"] / "shap_pickup_hour.png")
    log.info("SHAP: top features %s", ", ".join(mean_shap["feature"].head(4)))

    training = {"improved_model": {**results["improved_metrics"], "best_iteration": model.best_iteration},
                "june_only": results["june_only_metrics"], "time_split": results["time_split_metrics"]}
    out["training"].write_text(json.dumps(training, indent=2))
    return training


def analytics(cfg: dict) -> dict:
    """SQL analysis queries and the speed maps (needs features.parquet and the weather file)."""
    p = cfg["paths"]
    con = db.connect()
    if _weather(cfg, con) is None:
        con.close()
        return {"queries": 0}
    sql_reports = p["reports_dir"] / "sql"
    sql_reports.mkdir(exist_ok=True)
    for name in ANALYSIS_QUERIES:
        result = db.query(con, name)
        result.to_csv(sql_reports / name.replace(".sql", ".csv"), index=False)
        if name == "05_slowest_days.sql":
            log.info("Slowest days:\n%s", result.head(3)[["overall_rank", "date", "median_speed_kmh",
                                                            "snow_depth_cm"]].to_string(index=False))
    sm = cfg["speed_map"]
    grid = db.query(con, "08_speed_by_area.sql", {"cell_deg": sm["cell_deg"], "min_trips": sm["min_trips"]})
    con.close()
    grid.to_csv(sql_reports / "08_speed_by_area.csv", index=False)
    plots.speed_map_png(grid, sm["cell_deg"], p["figures_dir"] / "speed_map.png")
    plots.speed_map_html(grid, sm["cell_deg"], p["reports_dir"] / "speed_map.html")
    log.info("Speed map: %d grid cells", len(grid))
    return {"queries": len(ANALYSIS_QUERIES) + 1, "grid_cells": len(grid)}


def report(cfg: dict, validation: dict, cleaning: dict, training: dict, started: float) -> dict:
    """metrics.json + model card, from the outputs of the earlier steps."""
    p = cfg["paths"]
    comparison = pd.read_csv(p["reports_dir"] / "model_comparison.csv")
    errors = pd.read_csv(p["reports_dir"] / "error_by_duration.csv")
    sql_dir = p["reports_dir"] / "sql"
    slowest = pd.read_csv(sql_dir / "05_slowest_days.csv") if (sql_dir / "05_slowest_days.csv").exists() else None
    impact = pd.read_csv(sql_dir / "06_weather_impact.csv") if (sql_dir / "06_weather_impact.csv").exists() else None
    metrics = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017
        "validation": validation,
        **cleaning,
        "n_features": len(features.FEATURES),
        **training,
        "comparison": comparison.to_dict(orient="records"),
        "runtime_s": round(time.time() - started, 1),
    }
    with open(p["reports_dir"] / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    write_model_card(p["reports_dir"] / "model_card.md", cfg, metrics, comparison, errors, slowest, impact)
    return metrics
