"""Download the raw data for this project from Kaggle.

Files downloaded into data/raw/:
  - train.csv                         NYC Taxi Trip Duration competition (1,458,644 trips)
  - weather_nyc_centralpark_2016.csv  Central Park daily weather, 2016 (366 days)

Requirements (one-time setup):
  1. A Kaggle account with an API token saved at ~/.kaggle/kaggle.json
  2. The competition rules accepted at
     https://www.kaggle.com/competitions/nyc-taxi-trip-duration/rules

Usage:
  python scripts/download_data.py           # download only missing files
  python scripts/download_data.py --force   # download everything again
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

COMPETITION = "nyc-taxi-trip-duration"
WEATHER_DATASET = "mathijs/weather-data-in-new-york-city-2016"

TRAIN_CSV = RAW_DIR / "train.csv"
WEATHER_CSV = RAW_DIR / "weather_nyc_centralpark_2016.csv"

# Expected sizes, used to catch incomplete or damaged files
# (e.g. a CSV re-saved in Excel is cut to 1,048,575 rows)
EXPECTED_TRAIN_ROWS = 1_458_644
EXPECTED_WEATHER_ROWS = 366


def get_api():
    """Authenticate with Kaggle using ~/.kaggle/kaggle.json (or KAGGLE_USERNAME / KAGGLE_KEY)."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("The 'kaggle' package is missing. Run: pip install -r requirements.txt")
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as err:  # missing or invalid token
        sys.exit(f"Kaggle authentication failed: {err}\n"
                 "Save your API token at ~/.kaggle/kaggle.json (see README).")
    return api


def unzip_all(folder: Path) -> None:
    """Extract every .zip in a folder, then delete the archives."""
    for archive in folder.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(folder)
        archive.unlink()


def download_train(api, force: bool) -> None:
    if TRAIN_CSV.exists() and not force:
        print(f"[skip] {TRAIN_CSV.name} already exists")
        return
    print("[download] NYC Taxi Trip Duration - train.csv ...")
    try:
        api.competition_download_file(COMPETITION, "train.zip", path=str(RAW_DIR), force=True, quiet=False)
    except Exception as err:
        sys.exit(f"Competition download failed: {err}\n"
                 "Have you accepted the rules at "
                 "https://www.kaggle.com/competitions/nyc-taxi-trip-duration/rules ?")
    unzip_all(RAW_DIR)


def download_weather(api, force: bool) -> None:
    if WEATHER_CSV.exists() and not force:
        print(f"[skip] {WEATHER_CSV.name} already exists")
        return
    print("[download] Central Park weather 2016 ...")
    tmp = RAW_DIR / "_weather_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(WEATHER_DATASET, path=str(tmp), unzip=True, quiet=False)
    files = list(tmp.glob("*.csv"))
    if not files:
        sys.exit("Weather download contained no CSV file.")
    # Kaggle's file name contains "(1)"; save it under a stable name instead
    shutil.move(str(files[0]), WEATHER_CSV)
    shutil.rmtree(tmp)


def verify() -> None:
    """Check that both files are complete."""
    checks = [(TRAIN_CSV, EXPECTED_TRAIN_ROWS), (WEATHER_CSV, EXPECTED_WEATHER_ROWS)]
    ok = True
    for path, expected in checks:
        if not path.exists():
            print(f"[missing] {path.name}")
            ok = False
            continue
        rows = sum(1 for _ in open(path, encoding="utf-8")) - 1  # minus header
        status = "ok" if rows == expected else f"MISMATCH (expected {expected:,})"
        print(f"[check] {path.name}: {rows:,} rows - {status}")
        ok &= rows == expected
    if ok:
        first = pd.read_csv(TRAIN_CSV, nrows=1, parse_dates=["pickup_datetime"])
        print(f"[check] first pickup time: {first['pickup_datetime'].iloc[0]} (should include seconds)")
        print("All data downloaded and verified.")
    else:
        sys.exit("Verification failed - re-run with --force.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw data from Kaggle.")
    parser.add_argument("--force", action="store_true", help="download again even if files exist")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api = get_api()
    download_train(api, args.force)
    download_weather(api, args.force)
    verify()


if __name__ == "__main__":
    main()
