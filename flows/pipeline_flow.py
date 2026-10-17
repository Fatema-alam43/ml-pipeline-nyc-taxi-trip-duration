"""The pipeline as an orchestrated, scheduled Prefect flow (a DAG of tasks).

    download -> validate -> clean -> features -> [spark cross-check] -> train -> analytics -> report

Run once:                 python -m flows.pipeline_flow
Run on a schedule:        prefect server start                      (terminal 1, UI at http://127.0.0.1:4200)
                          python -m flows.pipeline_flow --serve     (terminal 2: weekly, Mondays 06:00 Toronto)

Each task reads its inputs from disk and writes its outputs to disk (src/steps.py), so a failed run can be
retried from the failed task, and the same steps can be moved to Airflow tasks unchanged.
"""

import argparse
import os
import subprocess
import sys
import time

# Show the pipeline's own log messages in the Prefect UI
os.environ.setdefault("PREFECT_LOGGING_EXTRA_LOGGERS", "src,steps,pipeline")

from prefect import flow, task  # noqa: E402
from prefect.artifacts import create_markdown_artifact  # noqa: E402
from prefect.cache_policies import NO_CACHE  # noqa: E402
from prefect.schedules import Cron  # noqa: E402

from src import steps  # noqa: E402
from src.config import PROJECT_ROOT  # noqa: E402
from src.pipeline import configure, setup_logging  # noqa: E402


@task(name="download", retries=2, retry_delay_seconds=60, cache_policy=NO_CACHE)
def download_data(cfg: dict) -> str:
    """Download the raw files from Kaggle, unless they are already there. Retried on network errors."""
    p = cfg["paths"]
    if p["raw_trips"].exists() and p["raw_weather"].exists():
        return "already downloaded"
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "download_data.py")], check=True)
    return "downloaded"


@task(name="validate", cache_policy=NO_CACHE)
def validate_data(cfg: dict) -> dict:
    return steps.validate(cfg)


@task(name="clean", cache_policy=NO_CACHE)
def clean_data(cfg: dict) -> dict:
    return steps.clean(cfg)


@task(name="features", cache_policy=NO_CACHE)
def build_features(cfg: dict) -> dict:
    return steps.build_features(cfg)


@task(name="spark-crosscheck", cache_policy=NO_CACHE)
def spark_crosscheck(cfg: dict) -> dict:
    return steps.spark_crosscheck(cfg)


@task(name="train", cache_policy=NO_CACHE)
def train_models(cfg: dict) -> dict:
    return steps.train_models(cfg)


@task(name="analytics", cache_policy=NO_CACHE)
def run_analytics(cfg: dict) -> dict:
    return steps.analytics(cfg)


@task(name="report", cache_policy=NO_CACHE)
def write_report(cfg: dict, validation: dict, cleaning: dict, training: dict, started: float) -> dict:
    metrics = steps.report(cfg, validation, cleaning, training, started)
    rows = "\n".join(f"| {r['model']} | {r['split']} | {r['mae_s']} | {r['r2']} |" for r in metrics["comparison"])
    create_markdown_artifact(
        key="model-metrics",
        description="Model comparison from the latest pipeline run",
        markdown=(f"### Pipeline run\n\n- Rows removed by cleaning: **{metrics['rows_removed']:,}** "
                  f"of {metrics['rows_raw']:,}\n- Main model MAE: **{metrics['improved_model']['mae_s']} s**\n\n"
                  "| Model | Split | MAE (s) | R² |\n|---|---|---|---|\n" + rows))
    return metrics


@flow(name="nyc-taxi-pipeline", log_prints=True)
def nyc_taxi_pipeline(config_path: str = "config.yaml", fast: bool = False, mlflow: bool = True,
                      spark_check: bool = False, download: bool = True) -> dict:
    """End-to-end pipeline: data download to trained, evaluated, documented model."""
    started = time.time()
    cfg = configure(config_path, fast=fast, mlflow=mlflow)
    if download:
        download_data(cfg)
    validation = validate_data(cfg)
    cleaning = clean_data(cfg)
    build_features(cfg)
    if spark_check:
        spark_crosscheck(cfg)
    training = train_models(cfg)
    run_analytics(cfg)
    metrics = write_report(cfg, validation, cleaning, training, started)
    print(f"Done: {metrics['rows_removed']:,} rows removed, main model MAE {metrics['improved_model']['mae_s']} s")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pipeline with Prefect")
    parser.add_argument("--serve", action="store_true", help="run on a weekly schedule (needs `prefect server start`)")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--spark-check", action="store_true")
    args = parser.parse_args()
    setup_logging()
    if args.serve:
        nyc_taxi_pipeline.serve(
            name="weekly-retrain",
            schedule=Cron("0 6 * * 1", timezone="America/Toronto"),        # every Monday 06:00
            parameters={"fast": args.fast, "spark_check": args.spark_check},
            tags=["nyc-taxi", "ml-pipeline"],
            description="Weekly: download, validate, clean, build features, retrain, evaluate, report.",
        )
    else:
        nyc_taxi_pipeline(fast=args.fast, spark_check=args.spark_check)


if __name__ == "__main__":
    main()
