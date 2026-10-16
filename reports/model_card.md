# Model card: NYC taxi trip duration

*Generated automatically by `python -m src.pipeline` on 2026-10-16.*

## Model details

- **Task:** predict how long a New York City taxi trip will take (in seconds), at the moment it starts.
- **Model:** LightGBM gradient-boosted trees, trained on `log(1 + trip_duration)`; predictions are converted
  back to seconds. Learning rate 0.05, 64 leaves,
  1566 boosting rounds (chosen by early stopping on a validation set).
- **Inputs (16 features, all known when the trip starts):** `distance_km`, `manhattan_km`, `bearing`, `pickup_latitude`, `pickup_longitude`, `dropoff_latitude`, `dropoff_longitude`, `pickup_hour`, `pickup_dayofweek`, `pickup_month`, `is_weekend`, `is_rush_hour`, `airport_trip`, `passenger_count`, `vendor_id`, `store_and_fwd_flag`.
- **Not used (data leakage):** `dropoff_datetime` and any speed computed from the trip duration — both reveal
  the answer. A data contract and a unit test enforce this.

## Intended use

- Estimating typical trip times for planning, dispatch analysis and dashboards.
- A portfolio demonstration of an end-to-end, tested ML pipeline.

**Out of scope:** pricing or other decisions about individual customers or drivers, real-time routing,
and any period or city other than New York City, January–June 2016.

## Training data

- Kaggle *NYC Taxi Trip Duration* competition data: 1,458,644 trips, January–June 2016.
- 15,704 invalid trips removed (1.08%):
  durations under 1 minute or over 6 hours, zero distance, coordinates outside NYC, zero passengers.
- Models are trained on a 25% random sample of the 1,442,940
  clean trips.

## Evaluation

| Evaluation | MAE (s) | RMSE (s) | R² | Meaning |
|---|---|---|---|---|
| Random split (all months) | 171.0 | 296.9 | 0.799 | Same days appear in training and test, as in the course version |
| Random split, June trips only | 181.6 | 311.7 | 0.809 | Same model, scored only on June: shows how hard June is |
| **Time-based split** (train Jan–May, test June) | **186.6** | 338.8 | 0.779 | **Realistic estimate for future trips**: June days never seen in training |

The time-based split is the honest estimate: in real use the model always predicts trips from days it has
never seen. Its error is higher for two reasons: **June trips are harder** (trips got longer every month of
the period), and the model **cannot learn June-specific conditions** from June itself.

Comparison of all models (test set of the random split unless noted):

| model | split | mae_s | rmse_s | rmsle | r2 |
|---|---|---|---|---|---|
| Template LightGBM (with speed_kmph leak) | random | 17.3 | 70.9 | 0.0603 | 0.989 |
| LightGBM, no leak | random | 223.9 | 357.9 | 0.3934 | 0.708 |
| LightGBM + engineered features (template settings) | random | 181.4 | 311.6 | 0.323 | 0.778 |
| Improved LightGBM (tuned) | random | 171.0 | 296.9 | 0.3103 | 0.799 |
| Linear Regression | random | 262.9 | 425.7 | 0.4332 | 0.587 |
| Improved + real weather | random | 167.2 | 290.3 | 0.3051 | 0.808 |
| Improved + day of year (control) | random | 165.8 | 288.0 | 0.3036 | 0.811 |
| Improved LightGBM (tuned) | time | 186.6 | 338.8 | 0.3211 | 0.779 |
| Improved + real weather | time | 189.8 | 342.4 | 0.3238 | 0.775 |
| Improved + day of year (control) | time | 195.2 | 344.6 | 0.3286 | 0.772 |

## Known weaknesses

**Long trips are underestimated.** Error by actual trip length (random-split test set):

| bucket | trips | mean_actual_min | mean_pred_min | bias_min | mae_min | pct_of_trips |
|---|---|---|---|---|---|---|
| 1-5 min | 10503 | 3.5 | 4.6 | 1.0 | 1.2 | 14.6 |
| 5-10 min | 21717 | 7.5 | 8.0 | 0.5 | 1.6 | 30.1 |
| 10-20 min | 25269 | 14.2 | 13.9 | -0.3 | 2.6 | 35.0 |
| 20-30 min | 9134 | 24.1 | 22.0 | -2.1 | 4.4 | 12.7 |
| 30-60 min | 5009 | 39.1 | 33.9 | -5.2 | 7.9 | 6.9 |
| 1-6 h | 515 | 74.2 | 53.6 | -20.6 | 21.6 | 0.7 |

- The 80% of trips under 20 minutes are predicted within 1.2–2.6 minutes on average.
- Trips over 1 hour are predicted about **20.6 minutes too short** on average. Such trips are
  rare (under 1% of the data), and their causes (accidents, detours, waiting) are not in the data.
- Distances are straight-line (haversine) distances, not road distances.

## Do not use in unusual conditions

The model has **no information about weather, events, road closures or live traffic**. It should not be
relied on during conditions like the January 2016 blizzard. Days with snow on the ground had about **16% fewer trips** than dry days, so the training data contains very few snowy trips.

The slowest days in the data:

| overall_rank | date | median_speed_kmh | snow_depth_cm |
|---|---|---|---|
| 1 | 2016-01-26 | 10.55 | 43.18 |
| 2 | 2016-01-25 | 10.82 | 48.26 |
| 3 | 2016-01-27 | 10.91 | 22.86 |

Daily weather was tested as extra features, alongside a *day-of-year* control (weather is identical for all trips on a day, so it also tells the model *which* day it is):

| Evaluation | No weather | + real weather | + day of year only |
|---|---|---|---|
| Random split | 171.0 | 167.2 | 165.8 |
| Time split | 186.6 | 189.8 | 195.2 |

Weather is therefore **not** used by the model. On the time split, which tests days the model has never seen, weather made predictions worse (186.6 → 189.8 s). Its apparent gain on the random split comes from the model recognising individual days, which the day-of-year control reproduces without any weather.

## Reproducing

```bash
python scripts/download_data.py
python -m src.pipeline
mlflow ui --backend-store-uri sqlite:///mlflow.db     # browse every logged run
```
