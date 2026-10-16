"""Feature engineering, including the leakage guard."""

import numpy as np
import pytest

from src import etl, features, schemas
from src.features import FEATURES, LEAKY_COLUMNS


def build(trips_df, cfg):
    return features.build_features(etl.add_distance(trips_df), cfg)


def test_no_leaky_columns_in_features():
    """speed_kmph and dropoff_datetime reveal the trip duration and must never be model inputs."""
    assert "speed_kmph" in LEAKY_COLUMNS and "dropoff_datetime" in LEAKY_COLUMNS
    assert not set(LEAKY_COLUMNS) & set(FEATURES)


def test_output_has_exactly_the_model_features(trips, cfg):
    out = build(trips({}), cfg)
    assert list(out.columns) == ["id", "pickup_datetime", *FEATURES, "trip_duration"]
    assert not set(LEAKY_COLUMNS) & set(out.columns)


@pytest.mark.parametrize("dlat, dlon, expected", [
    (0.05, 0.0, 0), (0.0, 0.05, 90), (-0.05, 0.0, 180), (0.0, -0.05, 270),
], ids=["north", "east", "south", "west"])
def test_bearing_cardinal_directions(dlat, dlon, expected, trips, cfg):
    out = build(trips({"dropoff_latitude": 40.75 + dlat, "dropoff_longitude": -73.98 + dlon}), cfg)
    bearing = out["bearing"].iloc[0]
    assert min(abs(bearing - expected), 360 - abs(bearing - expected)) < 1


def test_bearing_always_between_0_and_360(raw_sample, cfg):
    out = build(raw_sample, cfg)
    assert out["bearing"].between(0, 360, inclusive="left").all()


def test_rush_hour_on_weekday_morning(trips, cfg):
    out = build(trips({"pickup_datetime": "2016-03-14 08:30:00"}), cfg)   # Monday
    assert out["is_rush_hour"].iloc[0] == 1
    assert out["is_weekend"].iloc[0] == 0


def test_no_rush_hour_on_weekends(trips, cfg):
    weekend_rush = [{"pickup_datetime": f"2016-03-{day} {hour:02d}:15:00"}
                    for day in (12, 13)                       # Saturday, Sunday
                    for hour in cfg["features"]["rush_hours"]]
    out = build(trips(*weekend_rush), cfg)
    assert (out["is_weekend"] == 1).all()
    assert (out["is_rush_hour"] == 0).all()


def test_manhattan_distance_at_least_straight_line(raw_sample, cfg):
    out = build(raw_sample, cfg)
    assert (out["manhattan_km"] >= out["distance_km"] * 0.999).all()


def test_airport_trip_flag(trips, cfg):
    out = build(trips({"dropoff_latitude": 40.6420, "dropoff_longitude": -73.7790},   # JFK
                      {}), cfg)                                                       # Midtown
    assert list(out["airport_trip"]) == [1, 0]


def test_categorical_encoding(trips, cfg):
    out = build(trips({"vendor_id": 1, "store_and_fwd_flag": "N"},
                      {"vendor_id": 2, "store_and_fwd_flag": "Y"}), cfg)
    assert list(out["vendor_id"]) == [0, 1]
    assert list(out["store_and_fwd_flag"]) == [0, 1]


def test_features_pass_the_data_contract(raw_sample, cfg, con):
    clean, _ = etl.clean_sql(raw_sample, cfg, con)
    out = features.build_features(clean, cfg)
    schemas.validate(schemas.features_schema(), out)       # raises if invalid


def test_features_contract_rejects_leaky_column(raw_sample, cfg, con):
    clean, _ = etl.clean_sql(raw_sample, cfg, con)
    out = features.build_features(clean, cfg).assign(speed_kmph=np.float32(10))
    with pytest.raises(schemas.pa.errors.SchemaErrors):
        schemas.validate(schemas.features_schema(), out)
