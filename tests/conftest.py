"""Shared test fixtures: config, the synthetic sample files, and a tiny-DataFrame builder."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import db, etl
from src.config import load_config

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def cfg():
    return load_config("config.yaml")


@pytest.fixture(scope="session")
def raw_sample():
    """1,000 synthetic trips with planted problems (see scripts/make_test_fixtures.py)."""
    return etl.load_raw(FIXTURES / "trips_sample.csv")


@pytest.fixture(scope="session")
def sample_cases():
    return pd.read_csv(FIXTURES / "trips_sample_cases.csv")


@pytest.fixture
def con():
    connection = db.connect()
    yield connection
    connection.close()


def make_trips(*rows: dict) -> pd.DataFrame:
    """Build a raw-format DataFrame from a few hand-written trips.

    Every field has a valid default (a 10-minute Midtown trip), so each test only
    states the values it cares about.
    """
    defaults = {
        "vendor_id": 1, "pickup_datetime": "2016-03-14 17:24:55", "passenger_count": 1,
        "pickup_longitude": -73.98, "pickup_latitude": 40.75,
        "dropoff_longitude": -73.96, "dropoff_latitude": 40.77,
        "store_and_fwd_flag": "N", "trip_duration": 600,
    }
    records = [{**defaults, **row} for row in rows]
    df = pd.DataFrame(records)
    df.insert(0, "id", [f"t{i}" for i in range(len(df))])
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = df["pickup_datetime"] + pd.to_timedelta(df["trip_duration"], unit="s")
    df = df[["id", "vendor_id", "pickup_datetime", "dropoff_datetime", "passenger_count",
             "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
             "store_and_fwd_flag", "trip_duration"]]
    return df.astype(etl.RAW_DTYPES)


@pytest.fixture
def trips():
    return make_trips


__all__ = ["make_trips", "np"]
