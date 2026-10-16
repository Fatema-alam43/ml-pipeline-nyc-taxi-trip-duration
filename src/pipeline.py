"""Run the full pipeline with one command:

    python -m src.pipeline                   # everything
    python -m src.pipeline --fast            # only the main model (no baselines, time split or weather)
    python -m src.pipeline --no-mlflow       # do not log runs to MLflow

Steps: load -> validate -> clean (SQL) -> features -> train/evaluate (MLflow) -> explain (SHAP)
       -> SQL analytics + speed map -> metrics + model card.
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone

from src import db, etl, explain, features, plots, schemas, train
from src.config import load_config
from src.model_card import write_model_card
from src.tracking import Tracker

log = logging.getLogger("pipeline")

ANALYSIS_QUERIES = ["03_daily_kpis.sql", "04_rolling_7day.sql", "05_slowest_days.sql",
                    "06_weather_impact.sql", "07_hourly_by_weekday.sql"]


def main() -> None:
    parser = argparse.ArgumentParser(description="NYC taxi trip duration pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fast", action="store_true", help="only train the main model")
    parser.add_argument("--no-mlflow", action="store_true", help="do not log runs to MLflow")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)-14s | %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ["mlflow", "alembic", "shap", "matplotlib", "urllib3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config(args.config)
    if args.fast:
        cfg["experiments"].update(run_baselines=False, run_time_split=False, run_weather=False)
    if args.no_mlflow:
        cfg["mlflow"]["enabled"] = False
    p = cfg["paths"]
    for key in ["processed_dir", "models_dir", "reports_dir", "figures_dir"]:
        p[key].mkdir(parents=True, exist_ok=True)
    start = time.time()

    # 1. Extract + validate against the raw data contract
    raw = etl.load_raw(p["raw_trips"])
    validation = etl.validate_raw(raw, cfg)

    # 2. Clean with SQL (DuckDB), verified against the pandas implementation and the clean contract
    con = db.connect()
    clean, cleaning_summary = etl.clean_sql(raw, cfg, con)
    if cfg["cleaning"].get("verify_with_pandas", True):
        etl.verify_sql_matches_pandas(raw, clean, cleaning_summary, cfg)
    schemas.validate(schemas.clean_schema(cfg), clean)
    log.info("Clean data passed the CLEAN data contract")
    cleaning_summary.to_csv(p["reports_dir"] / "cleaning_summary.csv", index=False)
    plots.duration_before_after(raw["trip_duration"], clean["trip_duration"],
                                p["figures_dir"] / "duration_before_after.png")
    clean.to_parquet(p["processed_dir"] / "clean.parquet", index=False)
    rows_removed = len(raw) - len(clean)
    del raw

    # 3. Features
    feats = features.build_features(clean, cfg)
    schemas.validate(schemas.features_schema(), feats)
    log.info("Features passed the FEATURES data contract")
    features_path = p["processed_dir"] / "features.parquet"
    feats.to_parquet(features_path, index=False)
    log.info("Features: %s rows x %d features", f"{len(feats):,}", len(features.FEATURES))

    # SQL views over the features and the weather file (used by the experiments and the analytics)
    has_weather = p["raw_weather"].exists()
    weather = None
    if has_weather:
        db.execute(con, "02_views.sql", {"features_path": features_path, "weather_path": p["raw_weather"]})
        weather = con.sql("SELECT * FROM weather_daily").df()
    else:
        log.warning("Weather file not found (%s) - skipping weather experiments and SQL analytics",
                    p["raw_weather"])

    # 4. Train + evaluate every model, logging each run to MLflow
    tracker = Tracker(cfg)
    results = train.run_experiments(feats, cfg, weather=weather, tracker=tracker)
    model, comparison = results["model"], results["comparison"]
    model.save_model(str(p["models_dir"] / "lgbm_improved.txt"), num_iteration=model.best_iteration)
    comparison.to_csv(p["reports_dir"] / "model_comparison.csv", index=False)
    test, y_test = results["test"], results["test"]["trip_duration"].values
    errors = train.error_by_duration(y_test, results["test_pred_sec"]).reset_index()
    errors.to_csv(p["reports_dir"] / "error_by_duration.csv", index=False)
    importance = plots.feature_importance(model, p["figures_dir"] / "feature_importance.png")
    importance.to_csv(p["reports_dir"] / "feature_importance.csv", index=False)
    plots.predicted_vs_actual(y_test, results["test_pred_sec"], p["figures_dir"] / "predicted_vs_actual.png")

    # 5. Explain the main model with SHAP
    n = min(cfg["explain"]["shap_sample"], len(test))
    shap_values = explain.shap_values(model, test[features.FEATURES].sample(n, random_state=0))
    mean_shap = explain.summary_plot(shap_values, p["figures_dir"] / "shap_summary.png")
    mean_shap.to_csv(p["reports_dir"] / "shap_importance.csv", index=False)
    explain.hour_dependence_plot(shap_values, p["figures_dir"] / "shap_pickup_hour.png")
    log.info("SHAP: top features %s", ", ".join(mean_shap["feature"].head(4)))

    # 6. SQL analytics + speed map
    slowest = weather_impact = None
    if has_weather:
        sql_reports = p["reports_dir"] / "sql"
        sql_reports.mkdir(exist_ok=True)
        for name in ANALYSIS_QUERIES:
            result = db.query(con, name)
            result.to_csv(sql_reports / name.replace(".sql", ".csv"), index=False)
            if name == "05_slowest_days.sql":
                slowest = result
                log.info("Slowest days:\n%s", result.head(3)[["overall_rank", "date", "median_speed_kmh",
                                                                "snow_depth_cm"]].to_string(index=False))
            if name == "06_weather_impact.sql":
                weather_impact = result
        sm = cfg["speed_map"]
        grid = db.query(con, "08_speed_by_area.sql", {"cell_deg": sm["cell_deg"], "min_trips": sm["min_trips"]})
        grid.to_csv(sql_reports / "08_speed_by_area.csv", index=False)
        plots.speed_map_png(grid, sm["cell_deg"], p["figures_dir"] / "speed_map.png")
        plots.speed_map_html(grid, sm["cell_deg"], p["reports_dir"] / "speed_map.html")
        log.info("Speed map: %d grid cells", len(grid))
    con.close()

    # 7. Metrics file + model card
    metrics = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017
        "validation": validation,
        "rows_raw": validation["rows"], "rows_removed": rows_removed, "rows_clean": len(clean),
        "n_features": len(features.FEATURES),
        "improved_model": {**results["improved_metrics"], "best_iteration": model.best_iteration},
        "june_only": results["june_only_metrics"],
        "time_split": results["time_split_metrics"],
        "comparison": comparison.to_dict(orient="records"),
        "runtime_s": round(time.time() - start, 1),
    }
    with open(p["reports_dir"] / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    write_model_card(p["reports_dir"] / "model_card.md", cfg, metrics, comparison, errors, slowest, weather_impact)

    m, t = results["improved_metrics"], results["time_split_metrics"]
    log.info("DONE in %.0f s | rows removed: %s | improved model MAE %.1f s, RMSE %.1f s, R2 %.3f%s",
             metrics["runtime_s"], f"{rows_removed:,}", m["mae_s"], m["rmse_s"], m["r2"],
             f" | time split MAE {t['mae_s']:.1f} s" if t else "")


if __name__ == "__main__":
    main()
