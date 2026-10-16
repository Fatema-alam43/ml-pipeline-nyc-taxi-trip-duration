-- How does weather relate to traffic? Days grouped by condition.
-- Snow on the ground takes priority over rain, since it affects roads for days.
WITH labelled AS (
    SELECT
        *,
        CASE
            WHEN snow_depth_cm > 0 THEN 'snow on the ground'
            WHEN precip_mm >= 1    THEN 'rainy (>= 1 mm)'
            ELSE 'dry'
        END AS condition
    FROM daily_kpis
)
SELECT
    condition,
    COUNT(*)                                AS days,
    SUM(trips)                              AS trips,
    ROUND(AVG(trips), 0)                    AS avg_trips_per_day,
    ROUND(MEDIAN(median_speed_kmh), 2)      AS median_speed_kmh,
    ROUND(MEDIAN(median_duration_min), 2)   AS median_duration_min
FROM labelled
GROUP BY condition
ORDER BY median_speed_kmh;
