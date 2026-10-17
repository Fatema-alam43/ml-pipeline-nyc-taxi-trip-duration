"""Load the trained model (from a local file or S3) and turn trip requests into predictions."""

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import etl, features
from src.config import PROJECT_ROOT

log = logging.getLogger("api.model")
TRAINING_MONTHS = range(1, 7)       # the model was trained on January-June 2016


def resolve_model_file() -> tuple[Path, str]:
    """Where to load the model from.

    MODEL_S3_URI (e.g. s3://bucket/models/model.txt): downloaded to /tmp at startup (used on AWS)
    MODEL_PATH: a local file (default: models/lgbm_improved.txt, used locally and in Docker)
    """
    s3_uri = os.environ.get("MODEL_S3_URI")
    if s3_uri:
        import boto3

        bucket, _, key = s3_uri.removeprefix("s3://").partition("/")
        target = Path(os.environ.get("MODEL_CACHE_PATH", "/tmp/model.txt"))
        if not target.exists():
            log.info("Downloading model from %s", s3_uri)
            boto3.client("s3").download_file(bucket, key, str(target))
        return target, s3_uri
    path = Path(os.environ.get("MODEL_PATH", PROJECT_ROOT / "models" / "lgbm_improved.txt"))
    return path, str(path)


class ModelService:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        path, self.source = resolve_model_file()
        self.booster = lgb.Booster(model_file=str(path))
        names = self.booster.feature_name()
        if names != features.FEATURES:
            raise ValueError(f"Model features {names} do not match the code's FEATURES {features.FEATURES}")
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        self.version = os.environ.get("MODEL_VERSION", f"sha-{self.sha256}")
        self.loaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017
        log.info("Model loaded: version=%s sha256=%s source=%s", self.version, self.sha256, self.source)

    def to_features(self, trips: pd.DataFrame) -> pd.DataFrame:
        """Raw trip fields -> model features, with exactly the same code and dtypes as training."""
        raw = trips.astype({c: t for c, t in etl.RAW_DTYPES.items() if c in trips.columns})
        return features.compute_features(etl.add_distance(raw), self.cfg)

    def predict(self, trips: pd.DataFrame) -> pd.DataFrame:
        feats = self.to_features(trips)
        seconds = np.expm1(self.booster.predict(feats[features.FEATURES]))
        return pd.DataFrame({"seconds": seconds, "distance_km": feats["distance_km"].to_numpy(),
                             "month": feats["pickup_month"].to_numpy()})


def request_to_frame(trips: list) -> pd.DataFrame:
    """Validated requests -> a DataFrame with the raw training columns."""
    return pd.DataFrame([t.model_dump() for t in trips])


def warnings_for(month: int) -> list[str]:
    if month not in TRAINING_MONTHS:
        return [f"Pickup month {month} is outside the training period (January-June); "
                "seasonal effects for this month were never seen, so the prediction is less reliable."]
    return []
