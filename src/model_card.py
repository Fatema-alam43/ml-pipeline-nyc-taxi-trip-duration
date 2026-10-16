"""Write reports/model_card.md from the latest pipeline results, so its numbers always match the model."""

from datetime import date

import pandas as pd

from src.features import FEATURES


def _table(df: pd.DataFrame) -> str:
    """DataFrame -> Markdown table (without needing the tabulate package)."""
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def write_model_card(path, cfg: dict, metrics: dict, comparison: pd.DataFrame, errors: pd.DataFrame,
                     slowest_days: pd.DataFrame | None, weather_impact: pd.DataFrame | None) -> None:
    imp, june, time_m = metrics["improved_model"], metrics.get("june_only"), metrics.get("time_split")
    params = cfg["model"]["improved"]["params"]

    eval_rows = [["Random split (all months)", f"{imp['mae_s']:.1f}", f"{imp['rmse_s']:.1f}", imp["r2"],
                  "Same days appear in training and test, as in the course version"]]
    if june:
        eval_rows.append(["Random split, June trips only", f"{june['mae_s']:.1f}", f"{june['rmse_s']:.1f}",
                          june["r2"], "Same model, scored only on June: shows how hard June is"])
    if time_m:
        eval_rows.append(["**Time-based split** (train Jan–May, test June)", f"**{time_m['mae_s']:.1f}**",
                          f"{time_m['rmse_s']:.1f}", time_m["r2"],
                          "**Realistic estimate for future trips**: June days never seen in training"])
    eval_table = _table(pd.DataFrame(eval_rows, columns=["Evaluation", "MAE (s)", "RMSE (s)", "R²", "Meaning"]))

    long_trips = errors.loc[errors["bucket"] == "1-6 h"] if "bucket" in errors else pd.DataFrame()
    long_bias = float(long_trips["bias_min"].iloc[0]) if not long_trips.empty else None
    long_bias_text = f"{abs(long_bias):.1f}" if long_bias is not None else "?"
    short = errors.loc[errors["bucket"].isin(["1-5 min", "5-10 min", "10-20 min"])]
    short_share = short["pct_of_trips"].sum()
    short_mae = f"{short['mae_min'].min():.1f}–{short['mae_min'].max():.1f}"

    weather_text = ""
    if not comparison.empty and "split" in comparison:
        def mae(name, split):
            row = comparison[(comparison["model"] == name) & (comparison["split"] == split)]
            return None if row.empty else float(row["mae_s"].iloc[0])
        lines, verdict = [], ""
        for split, label in [("random", "Random split"), ("time", "Time split")]:
            base, wx, doy = (mae("Improved LightGBM (tuned)", split), mae("Improved + real weather", split),
                             mae("Improved + day of year (control)", split))
            if None not in (base, wx, doy):
                lines.append(f"| {label} | {base:.1f} | {wx:.1f} | {doy:.1f} |")
                if split == "time":
                    change = "made predictions worse" if wx > base else f"improved MAE by only {base - wx:.1f} s"
                    verdict = (f" On the time split, which tests days the model has never seen, weather {change} "
                               f"({base:.1f} → {wx:.1f} s). Its apparent gain on the random split comes from the "
                               "model recognising individual days, which the day-of-year control reproduces "
                               "without any weather.")
        if lines:
            weather_text = ("Daily weather was tested as extra features, alongside a *day-of-year* control "
                            "(weather is identical for all trips on a day, so it also tells the model *which* day "
                            "it is):\n\n| Evaluation | No weather | + real weather | + day of year only |\n"
                            "|---|---|---|---|\n" + "\n".join(lines) + "\n\n"
                            "Weather is therefore **not** used by the model." + verdict + "\n")

    slow_text = ""
    if slowest_days is not None and not slowest_days.empty:
        top = slowest_days.head(3)
        slow_text = "The slowest days in the data:\n\n" + _table(
            top[["overall_rank", "date", "median_speed_kmh", "snow_depth_cm"]].astype(str)) + "\n"
    snow_text = ""
    if weather_impact is not None and not weather_impact.empty:
        w = weather_impact.set_index("condition")
        if {"snow on the ground", "dry"} <= set(w.index):
            drop = 1 - w.loc["snow on the ground", "avg_trips_per_day"] / w.loc["dry", "avg_trips_per_day"]
            snow_text = (f"Days with snow on the ground had about **{drop:.0%} fewer trips** than dry days, "
                         "so the training data contains very few snowy trips.")

    text = f"""# Model card: NYC taxi trip duration

*Generated automatically by `python -m src.pipeline` on {date.today().isoformat()}.*

## Model details

- **Task:** predict how long a New York City taxi trip will take (in seconds), at the moment it starts.
- **Model:** LightGBM gradient-boosted trees, trained on `log(1 + trip_duration)`; predictions are converted
  back to seconds. Learning rate {params['learning_rate']}, {params['num_leaves']} leaves,
  {imp.get('best_iteration', '?')} boosting rounds (chosen by early stopping on a validation set).
- **Inputs ({len(FEATURES)} features, all known when the trip starts):** {', '.join(f'`{f}`' for f in FEATURES)}.
- **Not used (data leakage):** `dropoff_datetime` and any speed computed from the trip duration — both reveal
  the answer. A data contract and a unit test enforce this.

## Intended use

- Estimating typical trip times for planning, dispatch analysis and dashboards.
- A portfolio demonstration of an end-to-end, tested ML pipeline.

**Out of scope:** pricing or other decisions about individual customers or drivers, real-time routing,
and any period or city other than New York City, January–June 2016.

## Training data

- Kaggle *NYC Taxi Trip Duration* competition data: {metrics['rows_raw']:,} trips, January–June 2016.
- {metrics['rows_removed']:,} invalid trips removed ({metrics['rows_removed'] / metrics['rows_raw']:.2%}):
  durations under 1 minute or over 6 hours, zero distance, coordinates outside NYC, zero passengers.
- Models are trained on a {cfg['model']['sample_frac']:.0%} random sample of the {metrics['rows_clean']:,}
  clean trips.

## Evaluation

{eval_table}

The time-based split is the honest estimate: in real use the model always predicts trips from days it has
never seen. Its error is higher for two reasons: **June trips are harder** (trips got longer every month of
the period), and the model **cannot learn June-specific conditions** from June itself.

Comparison of all models (test set of the random split unless noted):

{_table(comparison[['model', 'split', 'mae_s', 'rmse_s', 'rmsle', 'r2']])}

## Known weaknesses

**Long trips are underestimated.** Error by actual trip length (random-split test set):

{_table(errors)}

- The {short_share:.0f}% of trips under 20 minutes are predicted within {short_mae} minutes on average.
- Trips over 1 hour are predicted about **{long_bias_text} minutes too short** on average. Such trips are
  rare (under 1% of the data), and their causes (accidents, detours, waiting) are not in the data.
- Distances are straight-line (haversine) distances, not road distances.

## Do not use in unusual conditions

The model has **no information about weather, events, road closures or live traffic**. It should not be
relied on during conditions like the January 2016 blizzard. {snow_text}

{slow_text}
{weather_text}
## Reproducing

```bash
python scripts/download_data.py
python -m src.pipeline
mlflow ui --backend-store-uri sqlite:///mlflow.db     # browse every logged run
```
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
