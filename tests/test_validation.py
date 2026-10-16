"""Data contracts and raw-file validation: bad input must be rejected with a clear error."""

import pandas as pd
import pytest

from src import etl, schemas
from src.etl import DataValidationError


def test_valid_sample_passes(raw_sample, cfg):
    report = etl.validate_raw(raw_sample, cfg)
    assert report["rows"] == 1000
    assert report["duration_mismatch_pct"] == 0


def test_excel_truncated_file_is_rejected(cfg):
    """A file with exactly 1,048,575 rows was cut off by Excel (a bug found in this project)."""
    truncated = pd.DataFrame(index=range(cfg["validation"]["excel_row_limit"]))
    with pytest.raises(DataValidationError, match="Excel"):
        etl.validate_raw(truncated, cfg)


def test_timestamps_without_seconds_are_rejected(raw_sample, cfg):
    """Saving in Excel also dropped the seconds, so dropoff - pickup no longer equals trip_duration."""
    damaged = raw_sample.copy()
    damaged["pickup_datetime"] = damaged["pickup_datetime"].dt.floor("min")
    damaged["dropoff_datetime"] = damaged["dropoff_datetime"].dt.floor("min")
    with pytest.raises(DataValidationError, match="timestamps are damaged"):
        etl.validate_raw(damaged, cfg)


def test_duplicate_ids_are_rejected(raw_sample, cfg):
    dup = raw_sample.copy()
    dup.loc[1, "id"] = dup.loc[0, "id"]
    with pytest.raises(DataValidationError, match="data contract"):
        etl.validate_raw(dup, cfg)


def test_wrong_column_type_is_rejected(raw_sample, cfg):
    wrong = raw_sample.astype({"pickup_latitude": "float64"})
    with pytest.raises(DataValidationError, match="data contract"):
        etl.validate_raw(wrong, cfg)


def test_unexpected_column_is_rejected(raw_sample, cfg):
    with pytest.raises(DataValidationError, match="data contract"):
        etl.validate_raw(raw_sample.assign(extra=1), cfg)


def test_invalid_vendor_is_rejected(raw_sample, cfg):
    bad = raw_sample.copy()
    bad.loc[0, "vendor_id"] = 3
    with pytest.raises(DataValidationError, match="data contract"):
        etl.validate_raw(bad, cfg)


def test_clean_contract_rejects_trip_outside_nyc(trips, cfg):
    df = etl.add_distance(trips({"pickup_latitude": 42.0}))
    with pytest.raises(schemas.pa.errors.SchemaErrors):
        schemas.validate(schemas.clean_schema(cfg), df)


def test_clean_contract_rejects_short_trip(trips, cfg):
    df = etl.add_distance(trips({"trip_duration": 30}))
    with pytest.raises(schemas.pa.errors.SchemaErrors):
        schemas.validate(schemas.clean_schema(cfg), df)
