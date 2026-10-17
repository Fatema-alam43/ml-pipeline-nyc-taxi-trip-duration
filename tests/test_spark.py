"""PySpark version of cleaning + features: must match pandas/DuckDB exactly.

Needs Java. Skipped locally if Java is missing; CI sets REQUIRE_SPARK=1 so it can never be skipped there.
"""

import os
import subprocess

import pandas as pd
import pytest


def java_available() -> bool:
    """True only if Java actually runs (macOS has a placeholder `java` command that just shows an error)."""
    try:
        return subprocess.run(["java", "-version"], capture_output=True).returncode == 0
    except OSError:
        return False


if not java_available():
    if os.environ.get("REQUIRE_SPARK"):
        raise RuntimeError("REQUIRE_SPARK is set but Java is not installed")
    pytest.skip("Java is not installed (needed by PySpark)", allow_module_level=True)

from pyspark.sql import functions as F  # noqa: E402

from src import etl, features, spark_etl  # noqa: E402
from tests.conftest import FIXTURES, make_trips  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    session = spark_etl.get_spark("tests")
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def spark_result(spark, cfg):
    clean_df, summary = spark_etl.clean(spark_etl.load_raw(spark, FIXTURES / "trips_sample.csv"), cfg)
    return spark_etl.build_features(clean_df, cfg), summary


def test_same_rule_counts_as_sql(spark_result):
    _, summary = spark_result
    assert [(r["rows_flagged"], r["newly_removed"], r["rows_remaining"]) for r in summary] == [
        (17, 17, 983), (10, 7, 976), (11, 11, 965), (6, 4, 961), (39, 39, 961)]


def test_matches_pandas_exactly(spark_result, raw_sample, cfg, con):
    """Same trips, and bit-identical values for every feature."""
    clean, summary = etl.clean_sql(raw_sample, cfg, con)
    result = spark_etl.compare_with_pandas(spark_result[0], spark_result[1],
                                           features.build_features(clean, cfg), summary)
    assert result["match"], result


def test_boundary_rows_kept(spark_result, sample_cases):
    ids = set(spark_result[0].select("id").toPandas()["id"])
    boundary = sample_cases.loc[sample_cases["case"].str.startswith("boundary"), "id"]
    assert set(boundary) <= ids


def test_float32_limits_matter(spark):
    """Why the code compares with float32 limits: with a float64 limit, longitude -73.7 (stored as
    float32 -73.69999695) would be treated as outside NYC, unlike pandas."""
    df = spark.createDataFrame(pd.DataFrame({"lon": pd.Series([-73.7], dtype="float32")}))
    as_float32 = df.select(F.col("lon").between(spark_etl.f32(-74.3), spark_etl.f32(-73.7))).first()[0]
    as_float64 = df.select(F.col("lon").between(F.lit(-74.3), F.lit(-73.7))).first()[0]
    assert as_float32 is True and as_float64 is False


def test_day_of_week_matches_pandas(spark, cfg):
    """Spark counts Sunday = 1; pandas counts Monday = 0. The conversion must agree for every weekday."""
    trips = make_trips(*[{"pickup_datetime": f"2016-03-{day} 08:30:00"} for day in range(14, 21)])  # Mon..Sun
    sdf = spark.createDataFrame(etl.add_distance(trips))
    out = spark_etl.build_features(sdf, cfg).toPandas()
    assert list(out["pickup_dayofweek"]) == list(range(7))
    assert list(out["is_weekend"]) == [0, 0, 0, 0, 0, 1, 1]
