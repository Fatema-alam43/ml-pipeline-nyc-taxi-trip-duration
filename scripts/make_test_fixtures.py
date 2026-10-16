"""Generate small synthetic test files with known, planted problems.

The tests use synthetic data (not Kaggle data, which may not be redistributed), so every
expected count is known exactly. Re-run only if you change the planted rows:
    python scripts/make_test_fixtures.py

tests/fixtures/trips_sample.csv   1,000 trips in the Kaggle train.csv format
tests/fixtures/weather_sample.csv Jan-Jun 2016 daily weather in the Kaggle weather format
"""

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
rng = np.random.default_rng(42)

# Planted rows. Expected cleaning result (see tests/test_cleaning.py):
#   duration rule      flags 17 (8 short + 6 long + 3 short-and-zero-distance), removes 17
#   zero-distance rule flags 10 (7 + the 3 overlaps),                            removes 7
#   outside-NYC rule   flags 11 (9 + 2 outside-and-zero-passengers),             removes 11
#   zero-passenger     flags 6  (4 + the 2 overlaps),                            removes 4
#   total removed 39 of 1,000 -> 961 clean rows (including 8 boundary rows that must be KEPT)
N_VALID, START = 953, pd.Timestamp("2016-01-01")


def valid_trips(n: int, airport_trips: int = 0) -> pd.DataFrame:
    pickup = START + pd.to_timedelta(rng.integers(0, 181 * 86400, n), unit="s")
    duration = np.clip(np.exp(rng.normal(6.6, 0.6, n)), 120, 3600).astype(int)
    df = pd.DataFrame({
        "vendor_id": rng.integers(1, 3, n),
        "pickup_datetime": pickup,
        "passenger_count": rng.integers(1, 7, n),
        "pickup_longitude": rng.uniform(-74.02, -73.93, n),
        "pickup_latitude": rng.uniform(40.70, 40.80, n),
        "dropoff_longitude": rng.uniform(-74.02, -73.93, n),
        "dropoff_latitude": rng.uniform(40.70, 40.80, n),
        "store_and_fwd_flag": rng.choice(["N", "Y"], n, p=[0.98, 0.02]),
        "trip_duration": duration,
    })
    if airport_trips:  # a few trips ending at JFK airport
        k = airport_trips
        df.loc[: k - 1, ["dropoff_latitude", "dropoff_longitude"]] = [40.6420, -73.7790]
        df.loc[: k - 1, "trip_duration"] = rng.integers(1800, 3600, k)
    return df


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    parts = [valid_trips(N_VALID, airport_trips=20).assign(case="valid")]

    def bad(n, case, **changes):
        d = valid_trips(n)
        for col, value in changes.items():
            d[col] = value(d) if callable(value) else value
        parts.append(d.assign(case=case))

    same_point = {"dropoff_latitude": lambda d: d["pickup_latitude"],
                  "dropoff_longitude": lambda d: d["pickup_longitude"]}
    bad(8, "too_short", trip_duration=lambda d: rng.integers(1, 60, len(d)))
    bad(6, "too_long", trip_duration=lambda d: rng.integers(21601, 3_000_000, len(d)))
    bad(7, "zero_distance", **same_point)
    bad(3, "too_short+zero_distance", trip_duration=lambda d: rng.integers(1, 60, len(d)), **same_point)
    bad(9, "outside_nyc", pickup_latitude=0.0, pickup_longitude=0.0)
    bad(4, "zero_passengers", passenger_count=0)
    bad(2, "outside_nyc+zero_passengers", dropoff_latitude=41.5, passenger_count=0)
    # boundary rows: exactly on a limit, so they must be KEPT
    bad(2, "boundary_min_duration", trip_duration=60)
    bad(2, "boundary_max_duration", trip_duration=21600)
    bad(1, "boundary_lon_max", dropoff_longitude=-73.7)
    bad(1, "boundary_lat_max", pickup_latitude=41.0)
    bad(1, "boundary_lon_min", pickup_longitude=-74.3)
    bad(1, "boundary_lat_min", dropoff_latitude=40.5)

    trips = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=7).reset_index(drop=True)
    trips.insert(0, "id", [f"id{i:07d}" for i in range(len(trips))])
    trips["dropoff_datetime"] = trips["pickup_datetime"] + pd.to_timedelta(trips["trip_duration"], unit="s")
    for col in ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]:
        trips[col] = trips[col].astype("float32")
    cols = ["id", "vendor_id", "pickup_datetime", "dropoff_datetime", "passenger_count",
            "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
            "store_and_fwd_flag", "trip_duration"]
    trips[cols].to_csv(FIXTURES / "trips_sample.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    trips[["id", "case"]].to_csv(FIXTURES / "trips_sample_cases.csv", index=False)

    # Weather in the Kaggle format: day-month-year dates, inches, Fahrenheit, "T" = trace
    days = pd.date_range("2016-01-01", "2016-06-30", freq="D")
    precip = np.where(rng.random(len(days)) < 0.3, rng.exponential(0.4, len(days)).round(2), 0).astype(object)
    precip[rng.random(len(days)) < 0.1] = "T"
    snow_fall = np.zeros(len(days)).astype(object)
    snow_depth = np.zeros(len(days)).astype(object)
    blizzard = days.get_loc(pd.Timestamp("2016-01-23"))
    precip[blizzard], snow_fall[blizzard] = 2.31, 27.3
    for offset, depth in enumerate([6, 22, 19, 17, 9, "T"]):
        snow_depth[blizzard + offset] = depth
    temp = 30 + 50 * (np.arange(len(days)) / len(days)) + rng.normal(0, 5, len(days))
    weather = pd.DataFrame({
        "date": [f"{d.day}-{d.month}-{d.year}" for d in days],
        "maximum temperature": (temp + 8).round().astype(int),
        "minimum temperature": (temp - 8).round().astype(int),
        "average temperature": temp.round(1),
        "precipitation": precip, "snow fall": snow_fall, "snow depth": snow_depth,
    })
    weather.to_csv(FIXTURES / "weather_sample.csv", index=False)
    print(f"Wrote {len(trips):,} trips and {len(weather)} weather days to {FIXTURES}")


if __name__ == "__main__":
    main()
