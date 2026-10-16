# ML Pipeline: NYC Taxi Trip Duration

[![CI](https://github.com/Fatema-alam43/ml-pipeline-nyc-taxi-trip-duration/actions/workflows/ci.yml/badge.svg)](https://github.com/Fatema-alam43/ml-pipeline-nyc-taxi-trip-duration/actions/workflows/ci.yml)

End-to-end machine learning pipeline that predicts New York City taxi trip duration from 1.46 million trips: data contracts, SQL-based ETL with DuckDB and Parquet, feature engineering, gradient boosting, automated tests with CI, and (in progress) experiment tracking and cloud deployment.

> **Status:** work in progress.

## Getting the data

The data is not stored in this repository. It is downloaded from Kaggle:

1. Create a Kaggle account, go to **Settings → API → Create New Token**, and save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`.
2. Accept the competition rules: <https://www.kaggle.com/competitions/nyc-taxi-trip-duration/rules>
3. Run:

```bash
pip install -r requirements.txt
python scripts/download_data.py
```

Files are saved to `data/raw/` and checked automatically (row counts and timestamps).

## Running the pipeline

```bash
python -m src.pipeline                  # full run: validation, cleaning, features, all models
python -m src.pipeline --no-baselines   # only the improved model (faster)
```

One command runs every step and writes:

| Output | Location |
|---|---|
| Cleaned data and model features | `data/processed/*.parquet` |
| Trained model | `models/lgbm_improved.txt` |
| Metrics (validation, rows removed, model scores) | `reports/metrics.json` |
| Cleaning summary, model comparison, error by trip length | `reports/*.csv` |
| SQL analytics (daily KPIs, rolling averages, slowest days, weather impact, hourly patterns) | `reports/sql/*.csv` |
| Figures | `reports/figures/` |

All thresholds and model settings are in [`config.yaml`](config.yaml).

## SQL layer (DuckDB)

Cleaning and analytics run as plain SQL files on [DuckDB](https://duckdb.org/), an in-process analytical database (no server needed), reading the pipeline's Parquet files directly.

| File | What it does | SQL techniques |
|---|---|---|
| [`00_macros.sql`](sql/00_macros.sql) | Haversine distance as a reusable SQL function | `CREATE MACRO` |
| [`01_clean.sql`](sql/01_clean.sql) | The 4 cleaning rules, valid trips, and a per-rule summary | CTEs, `CASE`-style flags, running total with a window function |
| [`02_views.sql`](sql/02_views.sql) | Views over the Parquet features and the weather file (unit conversion, `T` = trace) | `read_parquet`, `read_csv`, `strptime`, `LEFT JOIN`, `MEDIAN` |
| [`03_daily_kpis.sql`](sql/03_daily_kpis.sql) | Trips, median duration and speed per day | aggregation |
| [`04_rolling_7day.sql`](sql/04_rolling_7day.sql) | 7-day rolling averages and week-over-week change | `AVG() OVER (ROWS BETWEEN 6 PRECEDING ...)`, `LAG()`, named `WINDOW` |
| [`05_slowest_days.sql`](sql/05_slowest_days.sql) | Slowest days overall and per month | `RANK() OVER (PARTITION BY ...)`, `QUALIFY` |
| [`06_weather_impact.sql`](sql/06_weather_impact.sql) | Speed and trip volume on snowy, rainy and dry days | `CASE`, CTE |
| [`07_hourly_by_weekday.sql`](sql/07_hourly_by_weekday.sql) | Average duration by hour, weekdays vs weekends | `FILTER` (conditional aggregation) |

**Verified against pandas:** every run re-applies the cleaning rules in pandas and checks that SQL keeps exactly the same rows, in the same order, with bit-identical distances.

Query results are written to `reports/sql/`.

### Storage: CSV vs Parquet

```bash
python scripts/benchmark_storage.py
```

Parquet files are 3–4x smaller than CSV, load about 55–60x faster in pandas, and query about 30–40x faster in DuckDB:

| Dataset | CSV | Parquet | Size | pandas load (CSV → Parquet) | DuckDB query (CSV → Parquet) |
|---|---|---|---|---|---|
| Raw trips | 200.6 MB | 48.0 MB | 4.2x smaller | 1.48 s → 0.02 s (62.5x faster) | 0.17 s → 0.006 s (29.0x faster) |
| Engineered features | 175.0 MB | 59.6 MB | 2.9x smaller | 1.41 s → 0.03 s (55.7x faster) | 0.229 s → 0.006 s (39.4x faster) |

## Project structure

```
├── config.yaml              # paths, validation and cleaning thresholds, model settings
├── pyproject.toml           # pytest and ruff settings
├── .github/workflows/ci.yml # lint + tests on every push
├── scripts/
│   ├── download_data.py     # downloads and verifies the raw data from Kaggle
│   ├── benchmark_storage.py # CSV vs Parquet: size, load time, query time
│   └── make_test_fixtures.py# generates the synthetic test data
├── src/
│   ├── config.py            # loads config.yaml
│   ├── db.py                # runs the SQL files with DuckDB
│   ├── etl.py               # load, validate, clean (SQL, verified against pandas)
│   ├── schemas.py           # data contracts (pandera) for raw, clean and feature data
│   ├── features.py          # feature engineering (16 features, no target leakage)
│   ├── train.py             # shared train/test split, models, evaluation
│   ├── plots.py             # figures
│   └── pipeline.py          # runs everything: python -m src.pipeline
├── sql/                     # DuckDB SQL: cleaning, views and 5 analysis queries
├── tests/                   # 60 pytest tests + synthetic fixtures
├── api/                     # (in progress) prediction API
├── notebooks/               # exploration
└── reports/figures/         # generated figures
```

## Data quality and testing

**Data contracts** ([`src/schemas.py`](src/schemas.py), using [pandera](https://pandera.readthedocs.io/)) are checked at every stage of the pipeline:

| Stage | Contract |
|---|---|
| Raw file | Exact columns and types, unique ids, valid vendor and flag values, plausible coordinates, positive durations. Also rejects a file with exactly 1,048,575 rows (Excel's row limit) and timestamps whose dropoff − pickup ≠ trip duration (seconds lost in Excel) |
| Clean data | Every row satisfies every cleaning rule (duration, distance, NYC area, passengers) |
| Features | The 16 expected features with valid ranges, no rush hour on weekends, and **no leaky columns** (`speed_kmph`, `dropoff_datetime`) |

**Automated tests** (60 tests, about 5 seconds, run with `pytest`):

- haversine distance against known distances (Empire State Building → Times Square ≈ 1.07 km)
- each cleaning rule on hand-made trips, including values exactly on the limits (e.g. 60 s, longitude −73.7) that must be kept
- exact cleaning counts on a synthetic sample with planted problems, and SQL vs pandas giving identical results
- feature logic (compass bearing, rush hour, airport trips) and the leakage guard
- every SQL analysis query, and the whole pipeline end to end

Tests use **synthetic data** ([`scripts/make_test_fixtures.py`](scripts/make_test_fixtures.py)), so no Kaggle data is stored in the repository and every expected count is known exactly.

**Continuous integration:** [GitHub Actions](.github/workflows/ci.yml) runs the linter ([ruff](https://docs.astral.sh/ruff/)) and all tests on every push, using the exact package versions in `requirements-lock.txt`. Changing a single cleaning rule from `<` to `<=` makes 5 tests fail.

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

## Data sources

- [NYC Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration) – Kaggle competition data
- [Weather data in New York City – 2016](https://www.kaggle.com/datasets/mathijs/weather-data-in-new-york-city-2016) – Central Park station, US National Weather Service

## Background

This project started as a group assignment for SEP 6DA3 (Data Analytics and Big Data) at McMaster University. This repository is my independent extension of that work, published with the instructor's permission.

## License

Code: MIT. The data is subject to Kaggle's terms and is not redistributed here.
