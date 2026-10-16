-- The slowest days of the half-year, ranked by median trip speed, with the weather.
-- Window functions: RANK() over all days, and RANK() within each month (PARTITION BY).
SELECT
    RANK() OVER (ORDER BY median_speed_kmh ASC)                              AS overall_rank,
    date,
    dayname(date)                                                            AS weekday,
    median_speed_kmh,
    trips,
    snow_depth_cm,
    precip_mm,
    RANK() OVER (PARTITION BY month(date) ORDER BY median_speed_kmh ASC)     AS rank_in_month
FROM daily_kpis
QUALIFY overall_rank <= 10
ORDER BY overall_rank;
