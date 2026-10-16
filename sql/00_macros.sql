-- Reusable SQL functions (DuckDB macros).
--
-- haversine_km: great-circle distance in km between two points in decimal degrees.
-- The operation order matches src/etl.py exactly (subtract first, then convert to radians),
-- so SQL and Python produce bit-identical distances.
CREATE OR REPLACE MACRO haversine_km(lat1, lon1, lat2, lon2) AS
    2 * 6371.0 * asin(sqrt(least(greatest(
        pow(sin(radians(lat2 - lat1) / 2), 2)
        + cos(radians(lat1)) * cos(radians(lat2)) * pow(sin(radians(lon2 - lon1) / 2), 2),
    0), 1)));
