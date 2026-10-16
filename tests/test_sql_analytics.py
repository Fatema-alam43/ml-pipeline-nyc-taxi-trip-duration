"""The SQL analysis queries run on the sample and return sensible results."""

import pytest

from src import db, etl, features
from tests.conftest import FIXTURES

QUERIES = ["03_daily_kpis.sql", "04_rolling_7day.sql", "05_slowest_days.sql",
           "06_weather_impact.sql", "07_hourly_by_weekday.sql"]


@pytest.fixture
def analytics_con(raw_sample, cfg, con, tmp_path):
    clean, _ = etl.clean_sql(raw_sample, cfg, con)
    path = tmp_path / "features.parquet"
    features.build_features(clean, cfg).to_parquet(path, index=False)
    db.execute(con, "02_views.sql", {"features_path": path, "weather_path": FIXTURES / "weather_sample.csv"})
    return con


@pytest.mark.parametrize("name", QUERIES)
def test_query_returns_rows(name, analytics_con):
    assert len(db.query(analytics_con, name)) > 0


def test_weather_units_converted(analytics_con):
    w = analytics_con.sql("SELECT * FROM weather_daily WHERE date = DATE '2016-01-23'").df()
    assert w["snow_cm"].iloc[0] == pytest.approx(27.3 * 2.54, abs=0.01)     # inches -> cm
    assert w["precip_mm"].iloc[0] == pytest.approx(2.31 * 25.4, abs=0.01)    # inches -> mm


def test_trace_values_become_zero(analytics_con):
    traces = analytics_con.sql("SELECT COUNT(*) FROM weather_daily WHERE precip_mm IS NULL").fetchone()[0]
    assert traces == 0


def test_rolling_average_first_day_equals_itself(analytics_con):
    r = db.query(analytics_con, "04_rolling_7day.sql")
    assert r["rolling_7day_duration_min"].iloc[0] == pytest.approx(r["median_duration_min"].iloc[0])


def test_slowest_days_ranked_from_one(analytics_con):
    r = db.query(analytics_con, "05_slowest_days.sql")
    assert r["overall_rank"].iloc[0] == 1
    assert r["median_speed_kmh"].is_monotonic_increasing


def test_hourly_query_covers_hours(analytics_con):
    r = db.query(analytics_con, "07_hourly_by_weekday.sql")
    assert r["pickup_hour"].between(0, 23).all()


def test_render_rejects_missing_parameter():
    with pytest.raises(KeyError):
        db.render("SELECT {{ missing }}", {})


def test_render_rejects_text_parameters():
    with pytest.raises(TypeError):
        db.render("SELECT {{ x }}", {"x": "1; DROP TABLE trips"})


def test_speed_by_area_grid(analytics_con):
    grid = db.query(analytics_con, "08_speed_by_area.sql", {"cell_deg": 0.01, "min_trips": 1})
    assert grid["trips"].sum() == 961                       # every clean trip lands in exactly one cell
    assert grid["median_speed_kmh"].gt(0).all()
    assert grid["median_speed_kmh"].is_monotonic_increasing  # slowest areas first
