# Databricks notebook source
# MAGIC %md
# MAGIC # NYC taxi: bronze → silver → gold
# MAGIC
# MAGIC Runs on Azure Databricks after the Data Factory pipeline (`pl_ingest_nyctaxi`) has copied
# MAGIC the raw files to `bronze`.
# MAGIC
# MAGIC | Layer | Contents |
# MAGIC |---|---|
# MAGIC | **bronze** (input) | `trips.parquet`, `weather.parquet`: raw files from Data Factory, all columns as text |
# MAGIC | **silver** | `trips_clean`, `features`, `cleaning_summary` (Delta tables) |
# MAGIC | **gold** | Small summary tables for Power BI (Delta, plus a CSV copy of each in `gold/export/`) |
# MAGIC
# MAGIC Runs on **serverless compute** (access to the lake through Unity Catalog external locations), or on a classic
# MAGIC dedicated cluster with `auth_mode = account_key` (storage key from Azure Key Vault).
# MAGIC
# MAGIC The cleaning rules and features are the same as in the local pipeline (`src/spark_etl.py`, `sql/01_clean.sql`).
# MAGIC The notebook **checks its own result**: if the number of removed rows differs from the expected value, it fails,
# MAGIC and so does the Data Factory run that started it.

# COMMAND ----------

# MAGIC %md ## 1. Settings

# COMMAND ----------

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

try:                                   # on Databricks
    dbutils.widgets.text("storage_account", "stnyctaxifatema01")
    # unity_catalog: serverless or any Unity Catalog compute, access through external locations (recommended)
    # account_key:   classic dedicated cluster, storage key read from Key Vault through a secret scope
    dbutils.widgets.dropdown("auth_mode", "unity_catalog", ["unity_catalog", "account_key"])
    dbutils.widgets.text("secret_scope", "kv-nyctaxi")
    dbutils.widgets.text("secret_key", "adls-key")
    dbutils.widgets.text("expected_rows_removed", "15704")
    STORAGE_ACCOUNT = dbutils.widgets.get("storage_account")
    EXPECTED_REMOVED = int(dbutils.widgets.get("expected_rows_removed"))
    if dbutils.widgets.get("auth_mode") == "account_key":
        # Storage key read from Azure Key Vault through a secret scope: never written in the notebook
        spark.conf.set(f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
                       dbutils.secrets.get(dbutils.widgets.get("secret_scope"), dbutils.widgets.get("secret_key")))
    LAKE = {layer: f"abfss://{layer}@{STORAGE_ACCOUNT}.dfs.core.windows.net" for layer in ["bronze", "silver", "gold"]}
    TABLE_FORMAT = "delta"
except NameError:                      # local test run: folders on disk
    import os
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    _root = os.environ.get("LAKE_ROOT", "/tmp/lake")
    EXPECTED_REMOVED = int(os.environ.get("EXPECTED_ROWS_REMOVED", "15704"))
    LAKE = {layer: f"{_root}/{layer}" for layer in ["bronze", "silver", "gold"]}
    TABLE_FORMAT = os.environ.get("TABLE_FORMAT", "parquet")   # Delta needs the delta-spark package locally

spark.conf.set("spark.sql.session.timeZone", "UTC")   # times are NYC local time: no time zone shifts

# Same thresholds as config.yaml
CLEANING = {"min_duration_s": 60, "max_duration_s": 21600, "min_distance_km": 1e-6, "min_passengers": 1,
            "lat_min": 40.5, "lat_max": 41.0, "lon_min": -74.3, "lon_max": -73.7}
RUSH_HOURS = [7, 8, 9, 16, 17, 18, 19]
AIRPORTS = {"JFK": (40.6413, -73.7781), "LGA": (40.7769, -73.8740)}
AIRPORT_RADIUS_KM = 2.0
GRID_CELL_DEG, GRID_MIN_TRIPS = 0.005, 30
print(LAKE, "| expected rows removed:", EXPECTED_REMOVED)

# COMMAND ----------

# MAGIC %md ## 2. Read bronze and set the column types
# MAGIC Data Factory copies the CSV as text. Coordinates are converted text → 64-bit → 32-bit float,
# MAGIC exactly like pandas reads them, so the boundary checks give the same answer in every engine.

# COMMAND ----------

def f32(value: float):
    """A 32-bit float literal: compares 32-bit coordinates with 32-bit limits (as pandas/numpy 2 does)."""
    return F.lit(value).cast(T.FloatType())


def typed_trips(raw: DataFrame) -> DataFrame:
    coord = lambda c: F.col(c).cast("double").cast("float")              # noqa: E731
    return raw.select(
        F.col("id").cast("string").alias("id"),
        F.col("vendor_id").cast("tinyint").alias("vendor_id"),
        F.to_timestamp("pickup_datetime", "yyyy-MM-dd HH:mm:ss").alias("pickup_datetime"),
        F.to_timestamp("dropoff_datetime", "yyyy-MM-dd HH:mm:ss").alias("dropoff_datetime"),
        F.col("passenger_count").cast("tinyint").alias("passenger_count"),
        coord("pickup_longitude").alias("pickup_longitude"),
        coord("pickup_latitude").alias("pickup_latitude"),
        coord("dropoff_longitude").alias("dropoff_longitude"),
        coord("dropoff_latitude").alias("dropoff_latitude"),
        F.col("store_and_fwd_flag").cast("string").alias("store_and_fwd_flag"),
        F.col("trip_duration").cast("int").alias("trip_duration"),
    )


def typed_weather(raw: DataFrame) -> DataFrame:
    """Kaggle weather: day-month-year dates, inches and Fahrenheit, 'T' = trace (0). Converted to metric."""
    num = lambda c: F.coalesce(F.when(F.trim(F.col(c)) == "T", F.lit(0.0))      # noqa: E731
                               .otherwise(F.col(c).cast("double")), F.lit(0.0))
    return raw.select(
        F.to_date("date", "d-M-yyyy").alias("date"),
        F.round(num("precipitation") * 25.4, 2).alias("precip_mm"),
        F.round(num("snow_fall") * 2.54, 2).alias("snow_cm"),
        F.round(num("snow_depth") * 2.54, 2).alias("snow_depth_cm"),
        F.round((F.col("average_temperature").cast("double") - 32) * 5 / 9, 2).alias("temp_mean_c"),
    )


raw_trips = typed_trips(spark.read.parquet(f"{LAKE['bronze']}/trips.parquet"))
weather = typed_weather(spark.read.parquet(f"{LAKE['bronze']}/weather.parquet"))
n_raw = raw_trips.count()
print(f"Bronze: {n_raw:,} trips, {weather.count()} weather days")
assert n_raw != 1_048_575, "Exactly 1,048,575 rows: the file was truncated by Excel"

# COMMAND ----------

# MAGIC %md ## 3. Silver: clean the trips (same 4 rules as the local pipeline)

# COMMAND ----------

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Same operation order as src/etl.py (subtract, then radians)."""
    a = (F.pow(F.sin(F.radians(lat2 - lat1) / 2), 2)
         + F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) * F.pow(F.sin(F.radians(lon2 - lon1) / 2), 2))
    return 2 * 6371.0 * F.asin(F.sqrt(F.least(F.greatest(a, F.lit(0.0)), F.lit(1.0))))


def coords(df):
    return [F.col(c).cast("double") for c in
            ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]]


lat1, lon1, lat2, lon2 = coords(raw_trips)
c = CLEANING
with_distance = raw_trips.withColumn("distance_km", haversine_km(lat1, lon1, lat2, lon2).cast("float"))
inside_nyc = (F.col("pickup_latitude").between(f32(c["lat_min"]), f32(c["lat_max"]))
              & F.col("dropoff_latitude").between(f32(c["lat_min"]), f32(c["lat_max"]))
              & F.col("pickup_longitude").between(f32(c["lon_min"]), f32(c["lon_max"]))
              & F.col("dropoff_longitude").between(f32(c["lon_min"]), f32(c["lon_max"])))
RULES = {
    "Duration < 60 s or > 6 h": (F.col("trip_duration") < c["min_duration_s"])
                                | (F.col("trip_duration") > c["max_duration_s"]),
    "Zero distance (0 km)": F.coalesce(F.col("distance_km"), F.lit(0.0).cast("float")) <= f32(c["min_distance_km"]),
    "Coordinates outside NYC bounding box": ~F.coalesce(inside_nyc, F.lit(False)),
    "Zero passengers": F.col("passenger_count") < c["min_passengers"],
}
flags = [f"f{i}" for i in range(len(RULES))]
flagged = with_distance.select("*", *[cond.alias(f) for f, cond in zip(flags, RULES.values(), strict=True)])

# Rows flagged by each rule on its own, and rows newly removed after the earlier rules
aggs = []
for i, f in enumerate(flags):
    newly = F.col(f)
    for earlier in flags[:i]:
        newly = newly & ~F.col(earlier)
    aggs += [F.sum(F.col(f).cast("int")).alias(f"flag_{i}"), F.sum(newly.cast("int")).alias(f"new_{i}")]
counts = flagged.agg(*aggs).first().asDict()

summary_rows, remaining = [], n_raw
for i, rule in enumerate(RULES):
    remaining -= counts[f"new_{i}"]
    summary_rows.append((rule, counts[f"flag_{i}"], counts[f"new_{i}"], remaining))
removed = n_raw - remaining
summary_rows.append(("TOTAL removed (any rule)", removed, removed, remaining))
cleaning_summary = spark.createDataFrame(
    summary_rows, "rule string, rows_flagged long, newly_removed long, rows_remaining long")
cleaning_summary.show(truncate=False)

any_flag = F.col(flags[0])
for f in flags[1:]:
    any_flag = any_flag | F.col(f)
trips_clean = flagged.where(~any_flag).drop(*flags)

# The consistency check: same result as pandas, DuckDB and local PySpark, or the run fails
assert removed == EXPECTED_REMOVED, f"Removed {removed:,} rows, expected {EXPECTED_REMOVED:,}"
print(f"Cleaning check passed: {removed:,} rows removed, {remaining:,} clean trips")

# COMMAND ----------

# MAGIC %md ## 4. Silver: features (same 16 features as `src/features.py`)

# COMMAND ----------

lat1, lon1, lat2, lon2 = coords(trips_clean)
pickup = F.col("pickup_datetime")
dow = ((F.dayofweek(pickup) + 5) % 7).cast("tinyint")          # Spark: Sunday = 1 -> pandas: Monday = 0
weekend = (dow >= 5).cast("tinyint")
dlon = F.radians(lon2 - lon1)
phi1, phi2 = F.radians(lat1), F.radians(lat2)
bearing = F.pmod(F.degrees(F.atan2(F.sin(dlon) * F.cos(phi2),
                                   F.cos(phi1) * F.sin(phi2) - F.sin(phi1) * F.cos(phi2) * F.cos(dlon))) + 360,
                 F.lit(360.0))
near_airport = F.lit(False)
for a_lat, a_lon in AIRPORTS.values():
    near_airport = (near_airport
                    | (haversine_km(lat1, lon1, F.lit(a_lat), F.lit(a_lon)) < AIRPORT_RADIUS_KM)
                    | (haversine_km(lat2, lon2, F.lit(a_lat), F.lit(a_lon)) < AIRPORT_RADIUS_KM))

FEATURES = ["distance_km", "manhattan_km", "bearing", "pickup_latitude", "pickup_longitude", "dropoff_latitude",
            "dropoff_longitude", "pickup_hour", "pickup_dayofweek", "pickup_month", "is_weekend", "is_rush_hour",
            "airport_trip", "passenger_count", "vendor_id", "store_and_fwd_flag"]
features = (trips_clean
            .withColumn("manhattan_km", (haversine_km(lat1, lon1, lat2, lon1)
                                         + haversine_km(lat1, lon1, lat1, lon2)).cast("float"))
            .withColumn("bearing", bearing.cast("float"))
            .withColumn("pickup_hour", F.hour(pickup).cast("tinyint"))
            .withColumn("pickup_dayofweek", dow)
            .withColumn("pickup_month", F.month(pickup).cast("tinyint"))
            .withColumn("is_weekend", weekend)
            .withColumn("is_rush_hour", (F.hour(pickup).isin(RUSH_HOURS) & (weekend == 0)).cast("tinyint"))
            .withColumn("airport_trip", near_airport.cast("tinyint"))
            .withColumn("vendor_id", (F.col("vendor_id") == 2).cast("tinyint"))
            .withColumn("store_and_fwd_flag", (F.col("store_and_fwd_flag") == "Y").cast("tinyint"))
            .select("id", "pickup_datetime", *FEATURES, "trip_duration"))

for name, df in [("trips_clean", trips_clean), ("features", features), ("cleaning_summary", cleaning_summary)]:
    df.write.format(TABLE_FORMAT).mode("overwrite").option("overwriteSchema", "true").save(f"{LAKE['silver']}/{name}")
    print(f"silver/{name} written")

# COMMAND ----------

# MAGIC %md ## 5. Gold: summary tables for Power BI

# COMMAND ----------

f = spark.read.format(TABLE_FORMAT).load(f"{LAKE['silver']}/features") \
    .withColumn("date", F.to_date("pickup_datetime")) \
    .withColumn("speed_kmh", F.col("distance_km") / (F.col("trip_duration") / 3600.0))

daily_kpis = (f.groupBy("date")
              .agg(F.count("*").alias("trips"),
                   F.round(F.median("trip_duration") / 60, 2).alias("median_duration_min"),
                   F.round(F.avg("trip_duration") / 60, 2).alias("mean_duration_min"),
                   F.round(F.median("speed_kmh"), 2).alias("median_speed_kmh"),
                   F.round(F.avg("distance_km"), 2).alias("mean_distance_km"),
                   F.round(F.avg("airport_trip") * 100, 2).alias("airport_trip_pct"))
              .join(weather, "date", "left")
              .withColumn("weekday", F.date_format("date", "EEEE"))
              .orderBy("date"))

hourly_by_weekday = (f.groupBy("pickup_hour", "is_weekend")
                     .agg(F.count("*").alias("trips"),
                          F.round(F.avg("trip_duration") / 60, 2).alias("avg_duration_min"),
                          F.round(F.median("speed_kmh"), 2).alias("median_speed_kmh"))
                     .withColumn("day_type", F.when(F.col("is_weekend") == 1, "Weekend").otherwise("Weekday"))
                     .orderBy("is_weekend", "pickup_hour"))

cell = GRID_CELL_DEG
speed_by_area = (f.groupBy(F.round(F.floor(F.col("pickup_latitude") / cell) * cell + cell / 2, 5).alias("lat"),
                           F.round(F.floor(F.col("pickup_longitude") / cell) * cell + cell / 2, 5).alias("lon"))
                 .agg(F.count("*").alias("trips"),
                      F.round(F.median("speed_kmh"), 2).alias("median_speed_kmh"),
                      F.round(F.median("trip_duration") / 60, 2).alias("median_duration_min"))
                 .where(F.col("trips") >= GRID_MIN_TRIPS)
                 .orderBy("median_speed_kmh"))

total = f.count()
airport_vs_other = (f.groupBy(F.when(F.col("airport_trip") == 1, "Airport (JFK / LaGuardia)")
                              .otherwise("Other trips").alias("trip_type"))
                    .agg(F.count("*").alias("trips"),
                         F.round(F.count("*") / total * 100, 2).alias("share_pct"),
                         F.round(F.avg("trip_duration") / 60, 2).alias("avg_duration_min"),
                         F.round(F.median("distance_km"), 2).alias("median_distance_km"),
                         F.round(F.median("speed_kmh"), 2).alias("median_speed_kmh")))

weather_impact = (daily_kpis
                  .withColumn("condition", F.when(F.col("snow_depth_cm") > 0, "Snow on the ground")
                              .when(F.col("precip_mm") >= 1, "Rainy (>= 1 mm)").otherwise("Dry"))
                  .groupBy("condition")
                  .agg(F.count("*").alias("days"), F.sum("trips").alias("trips"),
                       F.round(F.avg("trips"), 0).alias("avg_trips_per_day"),
                       F.round(F.median("median_speed_kmh"), 2).alias("median_speed_kmh"),
                       F.round(F.median("median_duration_min"), 2).alias("median_duration_min"))
                  .orderBy("median_speed_kmh"))

GOLD = {"daily_kpis": daily_kpis, "hourly_by_weekday": hourly_by_weekday, "speed_by_area": speed_by_area,
        "airport_vs_other": airport_vs_other, "weather_impact": weather_impact,
        "data_quality": spark.read.format(TABLE_FORMAT).load(f"{LAKE['silver']}/cleaning_summary")}
for name, df in GOLD.items():
    df.write.format(TABLE_FORMAT).mode("overwrite").option("overwriteSchema", "true").save(f"{LAKE['gold']}/{name}")
    # A single CSV file per table, easy to load in Power BI
    (df.coalesce(1).write.mode("overwrite").option("header", True)
     .csv(f"{LAKE['gold']}/export/{name}"))
    print(f"gold/{name}: {df.count():,} rows")

# COMMAND ----------

# MAGIC %md ## 6. Quick look at the results

# COMMAND ----------

airport_vs_other.show(truncate=False)
weather_impact.show(truncate=False)
daily_kpis.orderBy("median_speed_kmh").select("date", "weekday", "trips", "median_speed_kmh", "snow_depth_cm") \
    .show(3, truncate=False)
