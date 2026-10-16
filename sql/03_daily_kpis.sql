-- Daily KPIs: trips, typical duration and speed for every day, with that day's weather.
SELECT
    date,
    dayname(date) AS weekday,
    trips,
    median_duration_min,
    median_speed_kmh,
    mean_distance_km,
    precip_mm,
    snow_depth_cm
FROM daily_kpis
ORDER BY date;
