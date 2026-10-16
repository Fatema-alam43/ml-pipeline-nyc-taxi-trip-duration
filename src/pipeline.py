"""Run the full pipeline with one command:

    python -m src.pipeline                   # everything
    python -m src.pipeline --no-baselines    # only the improved model (faster)

Steps: load -> validate -> clean (SQL) -> features -> train/evaluate -> SQL analytics -> save outputs.
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone

from src import db, etl, features, plots, train
from src.config import load_config

log = logging.getLogger("pipeline")

ANALYSIS_QUERIES = ["03_daily_kpis.sql", "04_rolling_7day.sql", "05_slowest_days.sql",
                    "06_weather_impact.sql", "07_hourly_by_weekday.sql"]


def main() -> None:
    parser = argparse.ArgumentParser(description="NYC taxi trip duration pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-baselines", action="store_true", help="skip the comparison models")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)-14s | %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    if args.no_baselines:
        cfg["experiments"]["run_baselines"] = False
    p = cfg["paths"]
    for key in ["processed_dir", "models_dir", "reports_dir", "figures_dir"]:
        p[key].mkdir(parents=True, exist_ok=True)
    start = time.time()

    # 1. Extract + validate
    raw = etl.load_raw(p["raw_trips"])
    validation = etl.validate_raw(raw, cfg)

    # 2. Clean with SQL (DuckDB), optionally verified against the pandas implementation
    con = db.connect()
    clean, cleaning_summary = etl.clean_sql(raw, cfg, con)
    if cfg["cleaning"].get("verify_with_pandas", True):
        etl.verify_sql_matches_pandas(raw, clean, cleaning_summary, cfg)
    cleaning_summary.to_csv(p["reports_dir"] / "cleaning_summary.csv", index=False)
    plots.duration_before_after(raw["trip_duration"], clean["trip_duration"],
                                p["figures_dir"] / "duration_before_after.png")
    clean.to_parquet(p["processed_dir"] / "clean.parquet", index=False)
    rows_removed = len(raw) - len(clean)
    del raw

    # 3. Features
    feats = features.build_features(clean, cfg)
    feats.to_parquet(p["processed_dir"] / "features.parquet", index=False)
    log.info("Features: %s rows x %d features", f"{len(feats):,}", len(features.FEATURES))

    # 4. Train + evaluate
    results = train.run_experiments(feats, cfg)
    model = results["model"]
    model.save_model(str(p["models_dir"] / "lgbm_improved.txt"), num_iteration=model.best_iteration)
    results["comparison"].to_csv(p["reports_dir"] / "model_comparison.csv", index=False)
    y_test = results["test"]["trip_duration"].values
    train.error_by_duration(y_test, results["test_pred_sec"]).to_csv(p["reports_dir"] / "error_by_duration.csv")
    importance = plots.feature_importance(model, p["figures_dir"] / "feature_importance.png")
    importance.to_csv(p["reports_dir"] / "feature_importance.csv", index=False)
    plots.predicted_vs_actual(y_test, results["test_pred_sec"], p["figures_dir"] / "predicted_vs_actual.png")

    # 5. SQL analytics on the Parquet features + weather (not model inputs)
    if p["raw_weather"].exists():
        sql_reports = p["reports_dir"] / "sql"
        sql_reports.mkdir(exist_ok=True)
        db.execute(con, "02_views.sql", {"features_path": p["processed_dir"] / "features.parquet",
                                         "weather_path": p["raw_weather"]})
        for name in ANALYSIS_QUERIES:
            result = db.query(con, name)
            result.to_csv(sql_reports / name.replace(".sql", ".csv"), index=False)
            if name == "05_slowest_days.sql":
                log.info("Slowest days:\n%s", result.head(3)[["overall_rank", "date", "median_speed_kmh",
                                                                "snow_depth_cm"]].to_string(index=False))
            if name == "06_weather_impact.sql":
                log.info("Weather impact:\n%s", result.to_string(index=False))
    else:
        log.warning("Weather file not found (%s) - skipping SQL analytics", p["raw_weather"])
    con.close()

    # 6. Metrics file
    metrics = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "validation": validation,
        "rows_raw": validation["rows"], "rows_removed": rows_removed, "rows_clean": len(clean),
        "n_features": len(features.FEATURES),
        "improved_model": {**results["improved_metrics"], "best_iteration": model.best_iteration},
        "comparison": results["comparison"].to_dict(orient="records"),
        "runtime_s": round(time.time() - start, 1),
    }
    with open(p["reports_dir"] / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    m = results["improved_metrics"]
    log.info("DONE in %.0f s | rows removed: %s | improved model MAE %.1f s, RMSE %.1f s, R2 %.3f",
             metrics["runtime_s"], f"{rows_removed:,}", m["mae_s"], m["rmse_s"], m["r2"])


if __name__ == "__main__":
    main()
