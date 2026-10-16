"""Load the Central Park weather data and relate it to trip speed (analysis only, not a model input)."""

import logging

import pandas as pd

log = logging.getLogger(__name__)


def _round_numeric(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    """Round numeric columns only (leaves date columns untouched)."""
    df = df.copy()
    cols = df.select_dtypes("number").columns
    df[cols] = df[cols].round(decimals)
    return df


def load_weather(path) -> pd.DataFrame:
    """Read the Kaggle weather file and convert it to metric units.

    'T' (a trace of rain/snow, too small to measure) is treated as 0.
    """
    w = pd.read_csv(path)
    w.columns = w.columns.str.strip().str.lower()

    def to_number(col):
        return pd.to_numeric(w[col].astype(str).str.strip().replace("T", "0"), errors="coerce").fillna(0)

    weather = pd.DataFrame({
        "date": pd.to_datetime(w["date"], format="%d-%m-%Y"),       # e.g. 23-1-2016
        "precip_mm": to_number("precipitation") * 25.4,              # inches -> mm
        "snow_cm": to_number("snow fall") * 2.54,                    # inches -> cm
        "snow_depth_cm": to_number("snow depth") * 2.54,             # inches -> cm
        "temp_mean_c": (to_number("average temperature") - 32) * 5 / 9,
    })
    weather = _round_numeric(weather, 2)
    log.info("Weather: %d days (%s to %s)", len(weather),
             weather["date"].min().date(), weather["date"].max().date())
    return weather


def daily_weather_summary(features_df: pd.DataFrame, weather: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-day trip speed joined to that day's weather, plus a rainy vs dry comparison."""
    trips = features_df[["pickup_datetime", "distance_km", "trip_duration"]].copy()
    trips["date"] = trips["pickup_datetime"].dt.normalize()
    trips["speed_kmh"] = trips["distance_km"] / (trips["trip_duration"] / 3600)

    daily = (trips.groupby("date")
             .agg(trips=("trip_duration", "size"),
                  median_duration_min=("trip_duration", lambda s: s.median() / 60),
                  median_speed_kmh=("speed_kmh", "median"))
             .reset_index()
             .merge(weather, on="date", how="left"))
    daily["rainy_day"] = (daily["precip_mm"] >= 1.0).astype("int8")

    rainy_vs_dry = (daily.groupby("rainy_day")
                    .agg(days=("date", "size"), trips=("trips", "sum"),
                         median_speed_kmh=("median_speed_kmh", "median"))
                    .rename(index={0: "dry (< 1 mm)", 1: "rainy (>= 1 mm)"}))
    return _round_numeric(daily, 3), rainy_vs_dry.round(3)
