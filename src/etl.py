"""Extract, validate and clean the raw taxi trips."""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Compact types: about half the memory of pandas' 64-bit defaults
RAW_DTYPES = {
    "id": "object", "vendor_id": "int8", "passenger_count": "int8",
    "pickup_longitude": "float32", "pickup_latitude": "float32",
    "dropoff_longitude": "float32", "dropoff_latitude": "float32",
    "store_and_fwd_flag": "object", "trip_duration": "int32",
}
DATETIME_COLUMNS = ["pickup_datetime", "dropoff_datetime"]


class DataValidationError(Exception):
    """Raised when the raw data fails a validation check."""


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between points given in decimal degrees."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    a = np.clip(a, 0, 1)  # guard against floating-point rounding outside [0, 1]
    return 2 * R * np.arcsin(np.sqrt(a))


def load_raw(path) -> pd.DataFrame:
    """Read train.csv with compact dtypes and parsed datetimes."""
    df = pd.read_csv(path, dtype=RAW_DTYPES, parse_dates=DATETIME_COLUMNS)
    log.info("Loaded %s rows x %s columns (%.1f MB)", f"{len(df):,}", df.shape[1],
             df.memory_usage(deep=True).sum() / 1e6)
    return df


def validate_raw(df: pd.DataFrame, cfg: dict) -> dict:
    """Check the raw file is complete and correctly parsed. Raises DataValidationError on failure."""
    v = cfg["validation"]
    rows = len(df)

    if rows == v["excel_row_limit"]:
        raise DataValidationError(
            f"File has exactly {rows:,} rows (Excel's limit): it was probably truncated by Excel. "
            "Re-download it with scripts/download_data.py.")
    if rows != v["expected_rows"]:
        log.warning("Expected %s rows but found %s", f"{v['expected_rows']:,}", f"{rows:,}")

    missing = int(df.isna().sum().sum())
    duplicate_ids = int(df["id"].duplicated().sum())

    # dropoff - pickup must equal trip_duration if the timestamps were parsed correctly
    computed = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds()
    mismatch_pct = float(((computed - df["trip_duration"]).abs() > 1).mean() * 100)
    if mismatch_pct > v["max_duration_mismatch_pct"]:
        raise DataValidationError(
            f"{mismatch_pct:.2f}% of rows have dropoff - pickup != trip_duration: "
            "timestamps are damaged (e.g. seconds lost after saving in Excel).")

    report = {"rows": rows, "missing_values": missing, "duplicate_ids": duplicate_ids,
              "duration_mismatch_pct": round(mismatch_pct, 4),
              "pickup_min": str(df["pickup_datetime"].min()),
              "pickup_max": str(df["pickup_datetime"].max())}
    log.info("Validation passed: %s", report)
    return report


def add_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Add the haversine distance between pickup and dropoff (computed in float64, stored as float32)."""
    df = df.copy()
    df["distance_km"] = haversine_km(
        df["pickup_latitude"].astype("float64"), df["pickup_longitude"].astype("float64"),
        df["dropoff_latitude"].astype("float64"), df["dropoff_longitude"].astype("float64"),
    ).astype("float32")
    return df


def cleaning_rules(df: pd.DataFrame, cfg: dict) -> dict:
    """Return {rule name: boolean mask where True = bad row}, in the order they are applied."""
    c = cfg["cleaning"]
    box = c["nyc_bbox"]
    inside_nyc = (
        df["pickup_latitude"].between(box["lat_min"], box["lat_max"])
        & df["dropoff_latitude"].between(box["lat_min"], box["lat_max"])
        & df["pickup_longitude"].between(box["lon_min"], box["lon_max"])
        & df["dropoff_longitude"].between(box["lon_min"], box["lon_max"])
    )
    return {
        "Duration < 60 s or > 6 h": (df["trip_duration"] < c["min_duration_s"])
                                    | (df["trip_duration"] > c["max_duration_s"]),
        "Zero distance (0 km)": df["distance_km"].fillna(0) <= c["min_distance_km"],
        "Coordinates outside NYC bounding box": ~inside_nyc,
        "Zero passengers": df["passenger_count"] < c["min_passengers"],
    }


def clean(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the cleaning rules in order.

    Returns the cleaned data and a summary with two counts per rule:
    rows flagged by the rule on its own, and rows newly removed after the earlier rules.
    """
    n_raw = len(df)
    masks = cleaning_rules(df, cfg)

    kept = df
    records = []
    for rule, mask in masks.items():
        newly_removed = mask.loc[kept.index]
        kept = kept.loc[~newly_removed]
        records.append({
            "rule": rule,
            "rows_flagged": int(mask.sum()),
            "pct_flagged": round(mask.mean() * 100, 3),
            "newly_removed": int(newly_removed.sum()),
            "rows_remaining": len(kept),
        })

    total_removed = n_raw - len(kept)
    records.append({"rule": "TOTAL removed (any rule)", "rows_flagged": total_removed,
                    "pct_flagged": round(total_removed / n_raw * 100, 3),
                    "newly_removed": total_removed, "rows_remaining": len(kept)})
    summary = pd.DataFrame(records)

    log.info("Cleaning removed %s of %s rows (%.2f%%), %s remain",
             f"{total_removed:,}", f"{n_raw:,}", total_removed / n_raw * 100, f"{len(kept):,}")
    return kept.reset_index(drop=True), summary
