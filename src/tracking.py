"""Experiment tracking with MLflow: every model run is logged with its settings, scores, features and model.

View the runs with:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import logging
import subprocess

log = logging.getLogger(__name__)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class Tracker:
    """Thin wrapper around MLflow. When disabled in config.yaml, every call does nothing."""

    def __init__(self, cfg: dict):
        self.enabled = cfg.get("mlflow", {}).get("enabled", False)
        if not self.enabled:
            return
        import mlflow

        self.mlflow = mlflow
        self.log_models = cfg["mlflow"].get("log_models", True)
        p = cfg["paths"]
        mlflow.set_tracking_uri(f"sqlite:///{p['mlflow_db']}")
        name = cfg["mlflow"]["experiment"]
        if mlflow.get_experiment_by_name(name) is None:
            p["mlflow_artifacts"].mkdir(parents=True, exist_ok=True)
            mlflow.create_experiment(name, artifact_location=p["mlflow_artifacts"].as_uri())
        mlflow.set_experiment(name)
        self.common_tags = {"git_commit": _git_commit()}
        log.info("MLflow tracking to %s (experiment '%s')", p["mlflow_db"].name, name)

    def log_run(self, name: str, *, split: str, model, kind: str, features: list[str],
                params: dict, metrics: dict, input_example=None) -> None:
        """Log one trained model. kind is 'lightgbm' or 'sklearn'."""
        if not self.enabled:
            return
        mlflow = self.mlflow
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({**self.common_tags, "split": split, "model_type": kind})
            mlflow.log_params({**params, "split": split, "n_features": len(features)})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
            mlflow.log_dict({"features": features}, "features.json")
            if self.log_models:
                signature = None
                if input_example is not None:
                    import warnings

                    from mlflow.models import infer_signature
                    with warnings.catch_warnings():   # MLflow hint about integer columns: not relevant here
                        warnings.simplefilter("ignore", UserWarning)
                        signature = infer_signature(input_example, model.predict(input_example))
                flavor = mlflow.lightgbm if kind == "lightgbm" else mlflow.sklearn
                flavor.log_model(model, name="model", signature=signature)
