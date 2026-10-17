"""Feature engineering. Every feature is known when the trip starts, so none leaks the duration."""

import numpy as np
import pandas as pd

from src.etl import haversine_km

# Model inputs, in a fixed order
FEATURES = [
    "distance_km", "manhattan_km", "bearing",
    "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude",
    "pickup_hour", "pickup_dayofweek", "pickup_month", "is_weekend", "is_rush_hour",
    "airport_trip", "passenger_count", "vendor_id", "store_and_fwd_flag",
]

# Columns that must never be model inputs: both reveal the target
LEAKY_COLUMNS = ["speed_kmph", "dropoff_datetime"]


def add_time_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    pickup = df["pickup_datetime"]
    df["pickup_hour"] = pickup.dt.hour.astype("int8")
    df["pickup_dayofweek"] = pickup.dt.dayofweek.astype("int8")      # Monday = 0
    df["pickup_month"] = pickup.dt.month.astype("int8")
    df["is_weekend"] = (df["pickup_dayofweek"] >= 5).astype("int8")
    rush = df["pickup_hour"].isin(cfg["features"]["rush_hours"])
    df["is_rush_hour"] = (rush & (df["is_weekend"] == 0)).astype("int8")  # weekdays only
    return df


def add_geo_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    lat1 = df["pickup_latitude"].astype("float64")
    lon1 = df["pickup_longitude"].astype("float64")
    lat2 = df["dropoff_latitude"].astype("float64")
    lon2 = df["dropoff_longitude"].astype("float64")

    # Manhattan distance: north-south leg + east-west leg (streets follow a grid)
    df["manhattan_km"] = (haversine_km(lat1, lon1, lat2, lon1)
                          + haversine_km(lat1, lon1, lat1, lon2)).astype("float32")

    # Initial compass bearing from pickup to dropoff (0 = north, 90 = east)
    dlon = np.radians(lon2 - lon1)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    x = np.sin(dlon) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlon)
    df["bearing"] = ((np.degrees(np.arctan2(x, y)) + 360) % 360).astype("float32")

    # Pickup or dropoff near an airport
    radius = cfg["features"]["airport_radius_km"]
    near = pd.Series(False, index=df.index)
    for a_lat, a_lon in cfg["features"]["airports"].values():
        near |= haversine_km(lat1, lon1, a_lat, a_lon) < radius
        near |= haversine_km(lat2, lon2, a_lat, a_lon) < radius
    df["airport_trip"] = near.astype("int8")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vendor_id"] = (df["vendor_id"] == 2).astype("int8")                      # 1 -> 0, 2 -> 1
    df["store_and_fwd_flag"] = (df["store_and_fwd_flag"] == "Y").astype("int8")  # N -> 0, Y -> 1
    return df


def compute_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """All engineered columns for trips that already have distance_km.

    Used by BOTH training (build_features) and the prediction API, so the two can never drift apart.
    """
    df = add_time_features(df, cfg)
    df = add_geo_features(df, cfg)
    return encode_categoricals(df)


def build_features(clean_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return id, pickup time, the model features and the target."""
    df = compute_features(clean_df, cfg)
    out = df[["id", "pickup_datetime"] + FEATURES + ["trip_duration"]]
    assert not set(LEAKY_COLUMNS) & set(FEATURES), "Leaky column in FEATURES"
    return out
