"""Data contracts (pandera): what the data must look like at each stage of the pipeline.

RAW_SCHEMA      - the downloaded train.csv: right columns, types and plausible values
CLEAN_SCHEMA    - after cleaning: every row satisfies the cleaning rules
FEATURES_SCHEMA - model input: 16 features with valid ranges, no leaky columns

Thresholds come from config.yaml, so the contracts and the cleaning rules can never disagree.
"""

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from src.features import FEATURES, LEAKY_COLUMNS


def raw_schema() -> DataFrameSchema:
    """The raw file may contain invalid trips (cleaning removes them), but it must be well-formed."""
    return DataFrameSchema(
        {
            "id": Column(str, unique=True),
            "vendor_id": Column("int8", Check.isin([1, 2])),
            "pickup_datetime": Column("datetime64[ns]"),
            "dropoff_datetime": Column("datetime64[ns]"),
            "passenger_count": Column("int8", Check.ge(0)),
            "pickup_longitude": Column("float32", Check.in_range(-180, 180)),
            "pickup_latitude": Column("float32", Check.in_range(-90, 90)),
            "dropoff_longitude": Column("float32", Check.in_range(-180, 180)),
            "dropoff_latitude": Column("float32", Check.in_range(-90, 90)),
            "store_and_fwd_flag": Column(str, Check.isin(["Y", "N"])),
            "trip_duration": Column("int32", Check.gt(0)),
        },
        strict=True,          # no unexpected columns
        coerce=False,         # types must already be right
        name="raw_trips",
    )


def clean_schema(cfg: dict) -> DataFrameSchema:
    """After cleaning, every row must satisfy every cleaning rule."""
    c, box = cfg["cleaning"], cfg["cleaning"]["nyc_bbox"]
    lat = Check.in_range(box["lat_min"], box["lat_max"])
    lon = Check.in_range(box["lon_min"], box["lon_max"])
    return DataFrameSchema(
        {
            "id": Column(str, unique=True),
            "trip_duration": Column("int32", Check.in_range(c["min_duration_s"], c["max_duration_s"])),
            "passenger_count": Column("int8", Check.ge(c["min_passengers"])),
            "pickup_latitude": Column("float32", lat),
            "dropoff_latitude": Column("float32", lat),
            "pickup_longitude": Column("float32", lon),
            "dropoff_longitude": Column("float32", lon),
            "distance_km": Column("float32", Check.gt(c["min_distance_km"])),
        },
        strict=False,
        name="clean_trips",
    )


def features_schema() -> DataFrameSchema:
    """Model input: exactly the expected features, valid ranges, no missing values, no leakage."""
    binary = Check.isin([0, 1])
    columns = {
        "distance_km": Column("float32", Check.gt(0)),
        "manhattan_km": Column("float32", Check.gt(0)),
        "bearing": Column("float32", Check.in_range(0, 360, include_max=False)),
        "pickup_latitude": Column("float32"),
        "pickup_longitude": Column("float32"),
        "dropoff_latitude": Column("float32"),
        "dropoff_longitude": Column("float32"),
        "pickup_hour": Column("int8", Check.in_range(0, 23)),
        "pickup_dayofweek": Column("int8", Check.in_range(0, 6)),
        "pickup_month": Column("int8", Check.in_range(1, 12)),
        "is_weekend": Column("int8", binary),
        "is_rush_hour": Column("int8", binary),
        "airport_trip": Column("int8", binary),
        "passenger_count": Column("int8", Check.ge(1)),
        "vendor_id": Column("int8", binary),
        "store_and_fwd_flag": Column("int8", binary),
        "trip_duration": Column("int32", Check.gt(0)),
    }
    assert set(columns) - {"trip_duration"} == set(FEATURES), "features_schema is out of sync with FEATURES"
    return DataFrameSchema(
        columns,
        checks=[
            # Leakage guard: columns derived from the target must never reach the model
            Check(lambda df: not set(LEAKY_COLUMNS) & set(df.columns),
                  error=f"leaky column present (one of {LEAKY_COLUMNS})"),
            # Rush hour only exists on weekdays
            Check(lambda df: ~((df["is_weekend"] == 1) & (df["is_rush_hour"] == 1)),
                  error="is_rush_hour must be 0 on weekends"),
        ],
        strict=False,
        name="features",
    )


def validate(schema: DataFrameSchema, df):
    """Validate and report every failure at once (lazy=True) instead of stopping at the first."""
    return schema.validate(df, lazy=True)


__all__ = ["raw_schema", "clean_schema", "features_schema", "validate", "pa"]
