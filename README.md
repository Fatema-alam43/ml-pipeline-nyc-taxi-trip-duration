# ML Pipeline: NYC Taxi Trip Duration

End-to-end machine learning pipeline that predicts New York City taxi trip duration from 1.46 million trips: data validation, ETL, feature engineering, gradient boosting, and (in progress) SQL analytics, automated testing, experiment tracking and cloud deployment.

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

## Data sources

- [NYC Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration) – Kaggle competition data
- [Weather data in New York City – 2016](https://www.kaggle.com/datasets/mathijs/weather-data-in-new-york-city-2016) – Central Park station, US National Weather Service

## Background

This project started as a group assignment for SEP 6DA3 (Data Analytics and Big Data) at McMaster University. This repository is my independent extension of that work, published with the instructor's permission.

## License

Code: MIT. The data is subject to Kaggle's terms and is not redistributed here.
