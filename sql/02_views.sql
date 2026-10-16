-- Analysis views over the pipeline outputs.
--   trips         : engineered features (Parquet written by the pipeline)
--   weather_daily : Kaggle Central Park weather, converted to metric units
--   daily_kpis    : one row per day - trip statistics joined to that day's weather

CREATE OR REPLACE VIEW trips AS
SELECT * FROM read_parquet('{{ features_path }}');

CREATE OR REPLACE VIEW weather_daily AS
WITH raw AS (
    SELECT * FROM read_csv('{{ weather_path }}', header = true, all_varchar = true)
)
SELECT
    strptime("date", '%d-%m-%Y')::DATE AS date,                     -- file uses day-month-year
    -- 'T' = trace amount (too small to measure) -> 0
    ROUND(COALESCE(TRY_CAST(NULLIF(TRIM(precipitation), 'T') AS DOUBLE), 0) * 25.4, 2) AS precip_mm,
    ROUND(COALESCE(TRY_CAST(NULLIF(TRIM("snow fall"), 'T') AS DOUBLE), 0) * 2.54, 2)   AS snow_cm,
    ROUND(COALESCE(TRY_CAST(NULLIF(TRIM("snow depth"), 'T') AS DOUBLE), 0) * 2.54, 2)  AS snow_depth_cm,
    ROUND((TRY_CAST("average temperature" AS DOUBLE) - 32) * 5 / 9, 2)                 AS temp_mean_c
FROM raw;

CREATE OR REPLACE VIEW daily_kpis AS
SELECT
    t.pickup_datetime::DATE                                          AS date,
    COUNT(*)                                                         AS trips,
    ROUND(MEDIAN(t.trip_duration) / 60, 2)                           AS median_duration_min,
    ROUND(AVG(t.trip_duration) / 60, 2)                              AS mean_duration_min,
    ROUND(MEDIAN(t.distance_km / (t.trip_duration / 3600.0)), 2)     AS median_speed_kmh,
    ROUND(AVG(t.distance_km), 2)                                     AS mean_distance_km,
    ANY_VALUE(w.precip_mm)                                           AS precip_mm,
    ANY_VALUE(w.snow_cm)                                             AS snow_cm,
    ANY_VALUE(w.snow_depth_cm)                                       AS snow_depth_cm,
    ANY_VALUE(w.temp_mean_c)                                         AS temp_mean_c
FROM trips t
LEFT JOIN weather_daily w ON w.date = t.pickup_datetime::DATE
GROUP BY 1;
