-- 7-day rolling averages smooth out day-to-day noise and weekly cycles.
-- Window functions: AVG ... OVER (ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) and LAG.
SELECT
    date,
    trips,
    median_duration_min,
    ROUND(AVG(median_duration_min) OVER w7, 2) AS rolling_7day_duration_min,
    ROUND(AVG(trips)               OVER w7, 0) AS rolling_7day_trips,
    -- change vs the same weekday one week earlier
    trips - LAG(trips, 7) OVER (ORDER BY date) AS trips_vs_last_week
FROM daily_kpis
WINDOW w7 AS (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
ORDER BY date;
