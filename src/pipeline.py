"""Run the full pipeline with one command:

    python -m src.pipeline                   # everything
    python -m src.pipeline --fast            # only the main model (no baselines, time split or weather)
    python -m src.pipeline --no-mlflow       # do not log runs to MLflow
    python -m src.pipeline --spark-check     # also re-run cleaning + features in PySpark and compare

Steps (src/steps.py): validate -> clean (SQL) -> features -> [Spark check] -> train (MLflow, SHAP)
-> SQL analytics + speed map -> metrics + model card.
For a scheduled, orchestrated run see flows/pipeline_flow.py.
"""

import argparse
import logging
import time

from src import steps
from src.config import load_config

log = logging.getLogger("pipeline")


def configure(config_path: str = "config.yaml", fast: bool = False, mlflow: bool = True) -> dict:
    cfg = load_config(config_path)
    if fast:
        cfg["experiments"].update(run_baselines=False, run_time_split=False, run_weather=False)
    if not mlflow:
        cfg["mlflow"]["enabled"] = False
    steps.prepare(cfg)
    return cfg


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)-14s | %(message)s", datefmt="%H:%M:%S")
    for noisy in ["mlflow", "alembic", "shap", "matplotlib", "urllib3", "py4j"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(description="NYC taxi trip duration pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fast", action="store_true", help="only train the main model")
    parser.add_argument("--no-mlflow", action="store_true", help="do not log runs to MLflow")
    parser.add_argument("--spark-check", action="store_true", help="cross-check cleaning + features in PySpark")
    args = parser.parse_args()
    setup_logging()

    start = time.time()
    cfg = configure(args.config, fast=args.fast, mlflow=not args.no_mlflow)
    validation = steps.validate(cfg)
    cleaning = steps.clean(cfg)
    steps.build_features(cfg)
    if args.spark_check:
        steps.spark_crosscheck(cfg)
    training = steps.train_models(cfg)
    steps.analytics(cfg)
    metrics = steps.report(cfg, validation, cleaning, training, start)

    m, t = metrics["improved_model"], metrics["time_split"]
    log.info("DONE in %.0f s | rows removed: %s | improved model MAE %.1f s, RMSE %.1f s, R2 %.3f%s",
             metrics["runtime_s"], f"{metrics['rows_removed']:,}", m["mae_s"], m["rmse_s"], m["r2"],
             f" | time split MAE {t['mae_s']:.1f} s" if t else "")


if __name__ == "__main__":
    main()
