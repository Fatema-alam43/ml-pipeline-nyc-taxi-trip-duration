"""Cleaning rules: each rule on hand-made trips, exact counts on the sample, and SQL == pandas."""

import pandas as pd
import pytest

from src import etl

# (description, trip values) -> the trip must be REMOVED
BAD_TRIPS = [
    ("59 seconds", {"trip_duration": 59}),
    ("6 h + 1 s", {"trip_duration": 21601}),
    ("zero distance", {"dropoff_latitude": 40.75, "dropoff_longitude": -73.98}),
    ("pickup north of NYC", {"pickup_latitude": 41.01}),
    ("dropoff east of NYC", {"dropoff_longitude": -73.69}),
    ("pickup at 0,0 (GPS error)", {"pickup_latitude": 0.0, "pickup_longitude": 0.0}),
    ("zero passengers", {"passenger_count": 0}),
]

# Exactly on a limit -> the trip must be KEPT
BOUNDARY_TRIPS = [
    ("exactly 60 s", {"trip_duration": 60}),
    ("exactly 6 h", {"trip_duration": 21600}),
    ("dropoff longitude exactly -73.7", {"dropoff_longitude": -73.7}),
    ("pickup latitude exactly 41.0", {"pickup_latitude": 41.0}),
    ("pickup longitude exactly -74.3", {"pickup_longitude": -74.3}),
    ("dropoff latitude exactly 40.5", {"dropoff_latitude": 40.5}),
]


def run_both(raw, cfg, con):
    """Clean with SQL and with pandas; both results are returned."""
    sql_clean, sql_summary = etl.clean_sql(raw, cfg, con)
    pd_clean, pd_summary = etl.clean_pandas(etl.add_distance(raw), cfg)
    return sql_clean, pd_clean


@pytest.mark.parametrize("description, values", BAD_TRIPS, ids=[b[0] for b in BAD_TRIPS])
def test_bad_trip_is_removed(description, values, trips, cfg, con):
    raw = trips({}, values)                      # one valid trip + one bad trip
    sql_clean, pd_clean = run_both(raw, cfg, con)
    assert list(sql_clean["id"]) == ["t0"], f"SQL kept a trip with {description}"
    assert list(pd_clean["id"]) == ["t0"], f"pandas kept a trip with {description}"


@pytest.mark.parametrize("description, values", BOUNDARY_TRIPS, ids=[b[0] for b in BOUNDARY_TRIPS])
def test_boundary_trip_is_kept(description, values, trips, cfg, con):
    raw = trips(values)
    sql_clean, pd_clean = run_both(raw, cfg, con)
    assert len(sql_clean) == 1, f"SQL removed a valid trip with {description}"
    assert len(pd_clean) == 1, f"pandas removed a valid trip with {description}"


def test_trip_breaking_two_rules_is_removed_once(trips, cfg, con):
    raw = trips({"trip_duration": 30, "passenger_count": 0})
    _, summary = etl.clean_sql(raw, cfg, con)
    s = summary.set_index("rule")
    assert s.loc["Duration < 60 s or > 6 h", "newly_removed"] == 1
    assert s.loc["Zero passengers", "rows_flagged"] == 1
    assert s.loc["Zero passengers", "newly_removed"] == 0      # already removed by the duration rule
    assert s.loc["TOTAL removed (any rule)", "newly_removed"] == 1


def test_exact_counts_on_sample(raw_sample, cfg, con):
    """The synthetic sample has known planted problems (scripts/make_test_fixtures.py)."""
    clean, summary = etl.clean_sql(raw_sample, cfg, con)
    expected = pd.DataFrame({
        "rows_flagged":  [17, 10, 11, 6, 39],
        "newly_removed": [17, 7, 11, 4, 39],
        "rows_remaining": [983, 976, 965, 961, 961],
    })
    assert summary[["rows_flagged", "newly_removed", "rows_remaining"]].reset_index(drop=True).equals(expected)
    assert len(clean) == 961


def test_all_planted_boundary_rows_are_kept(raw_sample, sample_cases, cfg, con):
    clean, _ = etl.clean_sql(raw_sample, cfg, con)
    boundary_ids = sample_cases.loc[sample_cases["case"].str.startswith("boundary"), "id"]
    assert boundary_ids.isin(clean["id"]).all()


def test_sql_matches_pandas_on_sample(raw_sample, cfg, con):
    clean, summary = etl.clean_sql(raw_sample, cfg, con)
    etl.verify_sql_matches_pandas(raw_sample, clean, summary, cfg)   # raises if they differ


def test_cleaning_keeps_original_order(raw_sample, cfg, con):
    clean, _ = etl.clean_sql(raw_sample, cfg, con)
    positions = raw_sample.reset_index().set_index("id").loc[clean["id"], "index"]
    assert positions.is_monotonic_increasing
