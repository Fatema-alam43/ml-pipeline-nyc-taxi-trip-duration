"""The cleaning and feature steps in PySpark, for data too large for one machine.

Produces the same result as the pandas/DuckDB pipeline: identical rows removed per rule and
identical features. Run the comparison on the full data with:
    python -m src.spark_etl
"""

import logging
import time

import numpy as np
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from src.config import load_config
from src.features import FEATURES

log = logging.getLogger(__name__)

RAW_SCHEMA = T.StructType([
    T.StructField("id", T.StringType()),
    T.StructField("vendor_id", T.ByteType()),
    T.StructField("pickup_datetime", T.TimestampNTZType()),     # local NYC time, no timezone shifts
    T.StructField("dropoff_datetime", T.TimestampNTZType()),
    T.StructField("passenger_count", T.ByteType()),
    T.StructField("pickup_longitude", T.FloatType()),           # float32, as in pandas
    T.StructField("pickup_latitude", T.FloatType()),
    T.StructField("dropoff_longitude", T.FloatType()),
    T.StructField("dropoff_latitude", T.FloatType()),
    T.StructField("store_and_fwd_flag", T.StringType()),
    T.StructField("trip_duration", T.IntegerType()),
])


def get_spark(app_name: str = "nyc-taxi") -> SparkSession:
    """Local Spark session (all CPU cores). On a cluster or Databricks, use the provided session instead."""
    return (SparkSession.builder.appName(app_name).master("local[*]")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


def load_raw(spark: SparkSession, path) -> DataFrame:
    return (spark.read.schema(RAW_SCHEMA).option("header", True)
            .option("timestampNTZFormat", "yyyy-MM-dd HH:mm:ss").csv(str(path)))


def f32(value: float) -> Column:
    """A float32 literal. Comparing float32 columns with float32 limits matches pandas/numpy 2 exactly
    (a float64 literal would treat e.g. longitude -73.7 as just outside the box)."""
    return F.lit(value).cast(T.FloatType())


def haversine_km(lat1: Column, lon1: Column, lat2: Column, lon2: Column) -> Column:
    """Great-circle distance in km. Same operation order as src/etl.py (subtract, then radians)."""
    a = (F.pow(F.sin(F.radians(lat2 - lat1) / 2), 2)
         + F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) * F.pow(F.sin(F.radians(lon2 - lon1) / 2), 2))
    return 2 * 6371.0 * F.asin(F.sqrt(F.least(F.greatest(a, F.lit(0.0)), F.lit(1.0))))


def _coords(df: DataFrame) -> tuple[Column, Column, Column, Column]:
    return tuple(F.col(c).cast("double") for c in
                 ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"])


def clean(raw: DataFrame, cfg: dict) -> tuple[DataFrame, list[dict]]:
    """Apply the 4 cleaning rules. Returns the clean trips and the per-rule summary (same as etl.clean_sql)."""
    c, box = cfg["cleaning"], cfg["cleaning"]["nyc_bbox"]
    lat1, lon1, lat2, lon2 = _coords(raw)
    df = raw.withColumn("distance_km", haversine_km(lat1, lon1, lat2, lon2).cast(T.FloatType()))

    inside = (F.col("pickup_latitude").between(f32(box["lat_min"]), f32(box["lat_max"]))
              & F.col("dropoff_latitude").between(f32(box["lat_min"]), f32(box["lat_max"]))
              & F.col("pickup_longitude").between(f32(box["lon_min"]), f32(box["lon_max"]))
              & F.col("dropoff_longitude").between(f32(box["lon_min"]), f32(box["lon_max"])))
    rules = {
        "Duration < 60 s or > 6 h": (F.col("trip_duration") < c["min_duration_s"])
                                    | (F.col("trip_duration") > c["max_duration_s"]),
        "Zero distance (0 km)": F.coalesce(F.col("distance_km"), F.lit(0.0).cast(T.FloatType()))
                                <= f32(c["min_distance_km"]),
        "Coordinates outside NYC bounding box": ~F.coalesce(inside, F.lit(False)),
        "Zero passengers": F.col("passenger_count") < c["min_passengers"],
    }
    flags = [f"f{i}" for i in range(len(rules))]
    df = df.select("*", *[cond.alias(name) for name, cond in zip(flags, rules.values(), strict=True)]).cache()

    # One pass over the data: rows flagged by each rule, and newly removed after the earlier rules
    aggs = [F.count(F.lit(1)).alias("n")]
    for i, name in enumerate(flags):
        earlier = [~F.col(f) for f in flags[:i]]
        newly = F.col(name) if not earlier else F.col(name) & _all(earlier)
        aggs += [F.sum(F.col(name).cast("int")).alias(f"flag_{i}"), F.sum(newly.cast("int")).alias(f"new_{i}")]
    counts = df.agg(*aggs).first().asDict()

    n_raw, remaining, summary = counts["n"], counts["n"], []
    for i, rule in enumerate(rules):
        remaining -= counts[f"new_{i}"]
        summary.append({"rule": rule, "rows_flagged": counts[f"flag_{i}"],
                        "pct_flagged": round(100 * counts[f"flag_{i}"] / n_raw, 3),
                        "newly_removed": counts[f"new_{i}"], "rows_remaining": remaining})
    summary.append({"rule": "TOTAL removed (any rule)", "rows_flagged": n_raw - remaining,
                    "pct_flagged": round(100 * (n_raw - remaining) / n_raw, 3),
                    "newly_removed": n_raw - remaining, "rows_remaining": remaining})

    clean_df = df.where(~_any([F.col(f) for f in flags])).drop(*flags)
    return clean_df, summary


def _all(cols: list[Column]) -> Column:
    out = cols[0]
    for col in cols[1:]:
        out = out & col
    return out


def _any(cols: list[Column]) -> Column:
    out = cols[0]
    for col in cols[1:]:
        out = out | col
    return out


def build_features(clean_df: DataFrame, cfg: dict) -> DataFrame:
    """The 16 model features (same definitions as src/features.py)."""
    lat1, lon1, lat2, lon2 = _coords(clean_df)
    pickup = F.col("pickup_datetime")
    # Spark: dayofweek 1 = Sunday ... 7 = Saturday  ->  pandas: 0 = Monday ... 6 = Sunday
    dow = ((F.dayofweek(pickup) + 5) % 7).cast(T.ByteType())
    weekend = (dow >= 5).cast(T.ByteType())
    rush = F.hour(pickup).isin(cfg["features"]["rush_hours"]) & (weekend == 0)

    dlon = F.radians(lon2 - lon1)
    phi1, phi2 = F.radians(lat1), F.radians(lat2)
    x = F.sin(dlon) * F.cos(phi2)
    y = F.cos(phi1) * F.sin(phi2) - F.sin(phi1) * F.cos(phi2) * F.cos(dlon)
    bearing = F.pmod(F.degrees(F.atan2(x, y)) + 360, F.lit(360.0))

    radius = cfg["features"]["airport_radius_km"]
    near_airport = _any([(haversine_km(lat1, lon1, F.lit(a), F.lit(b)) < radius)
                         | (haversine_km(lat2, lon2, F.lit(a), F.lit(b)) < radius)
                         for a, b in cfg["features"]["airports"].values()])

    out = (clean_df
           .withColumn("manhattan_km", (haversine_km(lat1, lon1, lat2, lon1)
                                        + haversine_km(lat1, lon1, lat1, lon2)).cast(T.FloatType()))
           .withColumn("bearing", bearing.cast(T.FloatType()))
           .withColumn("pickup_hour", F.hour(pickup).cast(T.ByteType()))
           .withColumn("pickup_dayofweek", dow)
           .withColumn("pickup_month", F.month(pickup).cast(T.ByteType()))
           .withColumn("is_weekend", weekend)
           .withColumn("is_rush_hour", rush.cast(T.ByteType()))
           .withColumn("airport_trip", near_airport.cast(T.ByteType()))
           .withColumn("vendor_id", (F.col("vendor_id") == 2).cast(T.ByteType()))
           .withColumn("store_and_fwd_flag", (F.col("store_and_fwd_flag") == "Y").cast(T.ByteType())))
    return out.select("id", "pickup_datetime", *FEATURES, "trip_duration")


def compare_with_pandas(spark_features: DataFrame, spark_summary: list[dict],
                        pandas_features, pandas_summary) -> dict:
    """Check Spark against the main pipeline: same rule counts, same trips, identical features
    (bit-for-bit, except the bearing: within 0.001 degrees)."""
    cols = ["rows_flagged", "newly_removed", "rows_remaining"]
    summary_match = [{k: r[k] for k in cols} for r in spark_summary] == \
        pandas_summary[cols].astype(int).to_dict(orient="records")

    sp = spark_features.toPandas().sort_values("id").reset_index(drop=True)
    pd_ = pandas_features.sort_values("id").reset_index(drop=True)
    same_trips = len(sp) == len(pd_) and (sp["id"].astype(str).to_numpy() == pd_["id"].astype(str).to_numpy()).all()
    mismatched, bearing = {}, {}
    if same_trips:
        for col in FEATURES + ["trip_duration"]:
            a = sp[col].to_numpy().astype(pd_[col].dtype)   # same dtype as pandas
            b = pd_[col].to_numpy()
            if col == "bearing":
                # Java and NumPy trigonometry can differ in the last bit on some processors (e.g. Apple Silicon),
                # so the bearing is compared within 0.001 degrees, with 0 and 360 as the same direction.
                diff = abs(a.astype("float64") - b.astype("float64"))
                diff = np.minimum(diff, 360 - diff)
                bad = int((diff > 1e-3).sum())
                bearing = {"bearing_rows_not_bit_identical": int((a != b).sum()),
                           "bearing_max_diff_degrees": float(diff.max())}
            else:
                bad = int((a != b).sum())                   # every other feature: bit-for-bit
            if bad:
                mismatched[col] = bad
    return {"summary_match": summary_match, "same_trips": same_trips, "rows": len(sp),
            "mismatched_features": mismatched, **bearing,
            "match": summary_match and same_trips and not mismatched}


def main() -> None:
    """Run Spark cleaning + features on the full data and compare with the pandas/DuckDB pipeline output."""
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)-14s | %(message)s", datefmt="%H:%M:%S")
    cfg = load_config()
    p = cfg["paths"]
    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")
    start = time.time()
    clean_df, summary = clean(load_raw(spark, p["raw_trips"]), cfg)
    feats = build_features(clean_df, cfg)
    out_path = p["processed_dir"] / "spark_features.parquet"
    feats.write.mode("overwrite").parquet(str(out_path))
    log.info("Spark: %s rows removed, %s clean trips, %.0f s -> %s",
             f"{summary[-1]['newly_removed']:,}", f"{summary[-1]['rows_remaining']:,}", time.time() - start,
             out_path.name)
    for row in summary:
        log.info("  %-38s flagged %8s  newly removed %8s", row["rule"], f"{row['rows_flagged']:,}",
                 f"{row['newly_removed']:,}")

    pandas_summary = pd.read_csv(p["reports_dir"] / "cleaning_summary.csv")
    pandas_features = pd.read_parquet(p["processed_dir"] / "features.parquet")
    result = compare_with_pandas(spark.read.parquet(str(out_path)), summary, pandas_features, pandas_summary)
    log.info("Comparison with the pandas/DuckDB pipeline: %s", result)
    spark.stop()
    if not result["match"]:
        raise SystemExit("Spark and pandas results differ")
    log.info("Spark matches pandas/DuckDB: same rows removed per rule, same trips, same features "
             "(bearing within 0.001 degrees)")


if __name__ == "__main__":
    main()
