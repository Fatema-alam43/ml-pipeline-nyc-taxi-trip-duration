-- Average trip duration by pickup hour, weekdays vs weekends (one row per hour).
-- Conditional aggregation with FILTER replaces a pivot table.
SELECT
    pickup_hour,
    ROUND(AVG(trip_duration) FILTER (WHERE is_weekend = 0) / 60, 2) AS weekday_avg_min,
    ROUND(AVG(trip_duration) FILTER (WHERE is_weekend = 1) / 60, 2) AS weekend_avg_min,
    COUNT(*) FILTER (WHERE is_weekend = 0)                          AS weekday_trips,
    COUNT(*) FILTER (WHERE is_weekend = 1)                          AS weekend_trips
FROM trips
GROUP BY pickup_hour
ORDER BY pickup_hour;
