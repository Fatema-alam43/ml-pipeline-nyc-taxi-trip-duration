-- Clean the raw trips: flag invalid rows with four rules, keep the valid ones,
-- and summarise how many rows each rule affected.
--
-- Input : trips_raw (registered from Python; row_id preserves the original row order)
-- Output: trips_flagged, trips_clean, cleaning_summary
-- Values in double curly braces are filled in from config.yaml by src/db.py.

-- 1. Distance + one boolean flag per rule (TRUE = invalid row)
CREATE OR REPLACE TABLE trips_flagged AS
WITH with_distance AS (
    SELECT
        *,
        CAST(haversine_km(
            CAST(pickup_latitude AS DOUBLE),  CAST(pickup_longitude AS DOUBLE),
            CAST(dropoff_latitude AS DOUBLE), CAST(dropoff_longitude AS DOUBLE)
        ) AS FLOAT) AS distance_km
    FROM trips_raw
)
SELECT
    *,
    (trip_duration < {{ min_duration_s }} OR trip_duration > {{ max_duration_s }})  AS f_duration,
    (COALESCE(distance_km, 0) <= CAST({{ min_distance_km }} AS FLOAT))              AS f_zero_distance,
    NOT COALESCE(
            pickup_latitude   BETWEEN CAST({{ lat_min }} AS FLOAT) AND CAST({{ lat_max }} AS FLOAT)
        AND dropoff_latitude  BETWEEN CAST({{ lat_min }} AS FLOAT) AND CAST({{ lat_max }} AS FLOAT)
        AND pickup_longitude  BETWEEN CAST({{ lon_min }} AS FLOAT) AND CAST({{ lon_max }} AS FLOAT)
        AND dropoff_longitude BETWEEN CAST({{ lon_min }} AS FLOAT) AND CAST({{ lon_max }} AS FLOAT),
        FALSE)                                                                       AS f_outside_nyc,
    (passenger_count < {{ min_passengers }})                                         AS f_zero_passengers
FROM with_distance;

-- 2. Valid trips, in the original row order
CREATE OR REPLACE TABLE trips_clean AS
SELECT * EXCLUDE (f_duration, f_zero_distance, f_outside_nyc, f_zero_passengers)
FROM trips_flagged
WHERE NOT (f_duration OR f_zero_distance OR f_outside_nyc OR f_zero_passengers)
ORDER BY row_id;

-- 3. Summary: rows flagged by each rule on its own, and rows newly removed
--    after the earlier rules (rules are applied in this order).
CREATE OR REPLACE TABLE cleaning_summary AS
WITH totals AS (SELECT COUNT(*) AS n_raw FROM trips_flagged),
per_rule AS (
    SELECT 1 AS rule_order, 'Duration < 60 s or > 6 h' AS rule,
           SUM(f_duration::INT) AS rows_flagged,
           SUM(f_duration::INT) AS newly_removed
    FROM trips_flagged
    UNION ALL
    SELECT 2, 'Zero distance (0 km)',
           SUM(f_zero_distance::INT),
           SUM((f_zero_distance AND NOT f_duration)::INT)
    FROM trips_flagged
    UNION ALL
    SELECT 3, 'Coordinates outside NYC bounding box',
           SUM(f_outside_nyc::INT),
           SUM((f_outside_nyc AND NOT f_duration AND NOT f_zero_distance)::INT)
    FROM trips_flagged
    UNION ALL
    SELECT 4, 'Zero passengers',
           SUM(f_zero_passengers::INT),
           SUM((f_zero_passengers AND NOT f_duration AND NOT f_zero_distance AND NOT f_outside_nyc)::INT)
    FROM trips_flagged
),
with_remaining AS (
    SELECT
        rule_order, rule, rows_flagged,
        ROUND(100.0 * rows_flagged / n_raw, 3) AS pct_flagged,
        newly_removed,
        -- window function: running total of removed rows
        n_raw - SUM(newly_removed) OVER (ORDER BY rule_order) AS rows_remaining
    FROM per_rule, totals
)
SELECT * FROM with_remaining
UNION ALL
SELECT 5, 'TOTAL removed (any rule)',
       SUM(newly_removed), ROUND(100.0 * SUM(newly_removed) / MAX(n_raw), 3),
       SUM(newly_removed), MAX(n_raw) - SUM(newly_removed)
FROM per_rule, totals;
