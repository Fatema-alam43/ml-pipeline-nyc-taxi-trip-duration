"""Compare CSV and Parquet: file size, pandas load time and DuckDB query time.

Run after the pipeline (it needs data/processed/features.parquet):
    python scripts/benchmark_storage.py

Writes reports/storage_benchmark.csv and prints a Markdown table for the README.
"""

import shutil
import statistics
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402
from src.etl import DATETIME_COLUMNS, RAW_DTYPES  # noqa: E402

REPEATS = 3


def timed(fn) -> float:
    """Median wall-clock time of fn() over REPEATS runs, in seconds."""
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def size_mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def main() -> None:
    cfg = load_config()
    p = cfg["paths"]
    features_parquet = p["processed_dir"] / "features.parquet"
    if not features_parquet.exists():
        sys.exit("Run the pipeline first: python -m src.pipeline")

    tmp = p["processed_dir"].parent / "benchmark"
    tmp.mkdir(parents=True, exist_ok=True)

    # Build both formats for each dataset
    raw_csv, raw_parquet = p["raw_trips"], tmp / "raw.parquet"
    pd.read_csv(raw_csv, dtype=RAW_DTYPES, parse_dates=DATETIME_COLUMNS).to_parquet(raw_parquet, index=False)
    features_csv = tmp / "features.csv"
    pd.read_parquet(features_parquet).to_csv(features_csv, index=False)

    datasets = {
        "Raw trips": (raw_csv, raw_parquet,
                      lambda f: pd.read_csv(f, dtype=RAW_DTYPES, parse_dates=DATETIME_COLUMNS)),
        "Engineered features": (features_csv, features_parquet,
                                lambda f: pd.read_csv(f, parse_dates=["pickup_datetime"])),
    }
    # The same aggregation query on each file
    query = "SELECT passenger_count, AVG(trip_duration) FROM {src} GROUP BY 1"

    rows = []
    con = duckdb.connect()
    for name, (csv_path, parquet_path, read_csv) in datasets.items():
        csv_load = timed(lambda: read_csv(csv_path))
        pq_load = timed(lambda: pd.read_parquet(parquet_path))
        csv_q = timed(lambda: con.sql(query.format(src=f"read_csv('{csv_path.as_posix()}')")).fetchall())
        pq_q = timed(lambda: con.sql(query.format(src=f"read_parquet('{parquet_path.as_posix()}')")).fetchall())
        rows.append({
            "dataset": name,
            "csv_mb": round(size_mb(csv_path), 1), "parquet_mb": round(size_mb(parquet_path), 1),
            "size_reduction": f"{size_mb(csv_path) / size_mb(parquet_path):.1f}x smaller",
            "pandas_load_csv_s": round(csv_load, 2), "pandas_load_parquet_s": round(pq_load, 2),
            "load_speedup": f"{csv_load / pq_load:.1f}x faster",
            "duckdb_query_csv_s": round(csv_q, 3), "duckdb_query_parquet_s": round(pq_q, 3),
            "query_speedup": f"{csv_q / pq_q:.1f}x faster",
        })
    con.close()
    shutil.rmtree(tmp)

    result = pd.DataFrame(rows)
    out = p["reports_dir"] / "storage_benchmark.csv"
    result.to_csv(out, index=False)

    print("\n| Dataset | CSV | Parquet | Size | pandas load (CSV → Parquet) | DuckDB query (CSV → Parquet) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['dataset']} | {r['csv_mb']} MB | {r['parquet_mb']} MB | {r['size_reduction']} | "
              f"{r['pandas_load_csv_s']} s → {r['pandas_load_parquet_s']} s ({r['load_speedup']}) | "
              f"{r['duckdb_query_csv_s']} s → {r['duckdb_query_parquet_s']} s ({r['query_speedup']}) |")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
