-- Typical trip speed by pickup area: the city is divided into a grid of small cells.
-- Speed = straight-line distance / trip time, so it reflects congestion and detours together.
SELECT
    ROUND(FLOOR(pickup_latitude  / {{ cell_deg }}) * {{ cell_deg }} + {{ cell_deg }} / 2, 5) AS lat,
    ROUND(FLOOR(pickup_longitude / {{ cell_deg }}) * {{ cell_deg }} + {{ cell_deg }} / 2, 5) AS lon,
    COUNT(*)                                                             AS trips,
    ROUND(MEDIAN(distance_km / (trip_duration / 3600.0)), 2)             AS median_speed_kmh,
    ROUND(MEDIAN(trip_duration) / 60, 2)                                 AS median_duration_min
FROM trips
GROUP BY 1, 2
HAVING COUNT(*) >= {{ min_trips }}
ORDER BY median_speed_kmh;
