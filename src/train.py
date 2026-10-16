"""Train and evaluate the models. All models share one train/test split, so results are comparable."""

import logging
import math

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURES

log = logging.getLogger(__name__)

# The starter template's original feature set (with the leaky speed feature)
TEMPLATE_FEATURES = ["pickup_hour", "pickup_dayofweek", "pickup_month",
                     "distance_km", "speed_kmph", "passenger_count"]
NO_LEAK_FEATURES = [f for f in TEMPLATE_FEATURES if f != "speed_kmph"]


def make_split(features_df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample the cleaned data, then split 80/20 into train and test."""
    m = cfg["model"]
    sample = features_df.sample(frac=m["sample_frac"], random_state=m["random_state"]).reset_index(drop=True)
    train_idx, test_idx = train_test_split(sample.index, test_size=m["test_size"],
                                           random_state=m["random_state"])
    train, test = sample.loc[train_idx], sample.loc[test_idx]
    log.info("Sample %s -> train %s | test %s", f"{len(sample):,}", f"{len(train):,}", f"{len(test):,}")
    return train, test


def make_time_split(features_df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same sample as make_split, but split by date: train before the cut-off, test from it.

    This mimics real use: the model only ever predicts trips from days it has never seen.
    """
    m = cfg["model"]
    cutoff = pd.Timestamp(cfg["evaluation"]["time_split_test_start"])
    sample = features_df.sample(frac=m["sample_frac"], random_state=m["random_state"]).reset_index(drop=True)
    is_test = sample["pickup_datetime"] >= cutoff
    train, test = sample.loc[~is_test], sample.loc[is_test]
    log.info("Time split at %s -> train %s | test %s", cutoff.date(), f"{len(train):,}", f"{len(test):,}")
    return train, test


WEATHER_FEATURES = ["precip_mm", "snow_cm", "snow_depth_cm", "temp_mean_c"]


def add_day_features(features_df: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    """Add daily weather and day-of-year columns, matched on the pickup date (row order unchanged)."""
    df = features_df.copy()
    day = df["pickup_datetime"].dt.normalize()
    df["pickup_dayofyear"] = df["pickup_datetime"].dt.dayofyear.astype("int16")
    if weather is not None:
        w = weather.assign(date=pd.to_datetime(weather["date"])).set_index("date")
        for col in WEATHER_FEATURES:
            df[col] = day.map(w[col]).astype("float32")
    return df


def score(y_true_sec: np.ndarray, pred_log: np.ndarray) -> dict:
    """Metrics from log-space predictions: MAE/RMSE in seconds, RMSLE, R2."""
    pred = np.expm1(pred_log)
    return {
        "mae_s": round(mean_absolute_error(y_true_sec, pred), 1),
        "rmse_s": round(math.sqrt(mean_squared_error(y_true_sec, pred)), 1),
        "rmsle": round(math.sqrt(mean_squared_error(np.log1p(y_true_sec), pred_log)), 4),
        "r2": round(r2_score(y_true_sec, pred), 3),
    }


def _params(section: dict, seed: int) -> dict:
    return {**section["params"], "seed": seed}


def train_template(train, test, features, cfg):
    """Starter-template LightGBM. Like the template, it early-stops on the test set (baselines only)."""
    t = cfg["model"]["template"]
    y_train, y_test = np.log1p(train["trip_duration"]), np.log1p(test["trip_duration"])
    train_set = lgb.Dataset(train[features], label=y_train)
    valid_set = lgb.Dataset(test[features], label=y_test, reference=train_set)
    model = lgb.train(_params(t, cfg["model"]["random_state"]), train_set,
                      num_boost_round=t["num_boost_round"], valid_sets=[valid_set],
                      callbacks=[lgb.early_stopping(t["early_stopping_rounds"], verbose=False)])
    return model, model.predict(test[features], num_iteration=model.best_iteration)


def train_improved(train, cfg, features=None, validation="random"):
    """Tuned LightGBM, early-stopping on a validation split of the TRAINING data (test set untouched).

    validation="random": a random 10% of the training trips (random-split experiments)
    validation="time":   the latest 10% of the training trips (time-split experiments, no peeking ahead)
    """
    features = features or FEATURES
    m, imp = cfg["model"], cfg["model"]["improved"]
    y = np.log1p(train["trip_duration"])
    if validation == "time":
        order = train["pickup_datetime"].argsort(kind="stable").values
        cut = int(len(train) * (1 - m["validation_size"]))
        fit_idx, val_idx = order[:cut], order[cut:]
        X_fit, X_val = train[features].iloc[fit_idx], train[features].iloc[val_idx]
        y_fit, y_val = y.iloc[fit_idx], y.iloc[val_idx]
    else:
        X_fit, X_val, y_fit, y_val = train_test_split(
            train[features], y, test_size=m["validation_size"], random_state=m["random_state"])
    fit_set = lgb.Dataset(X_fit, label=y_fit)
    val_set = lgb.Dataset(X_val, label=y_val, reference=fit_set)
    model = lgb.train(_params(imp, m["random_state"]), fit_set,
                      num_boost_round=imp["num_boost_round"], valid_sets=[val_set],
                      callbacks=[lgb.early_stopping(imp["early_stopping_rounds"], verbose=False)])
    log.info("Improved model (%d features, %s validation): best iteration %d",
             len(features), validation, model.best_iteration)
    return model


def _linear_inputs(df):
    """Log-transform the distances so a linear model fits the log target sensibly."""
    X = df[FEATURES].astype("float64").copy()
    X["distance_km"] = np.log1p(X["distance_km"])
    X["manhattan_km"] = np.log1p(X["manhattan_km"])
    return X


def train_linear(train, test):
    model = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    model.fit(_linear_inputs(train), np.log1p(train["trip_duration"]))
    return model, model.predict(_linear_inputs(test))


def add_leaky_speed(df):
    """The starter template's leaky feature - used ONLY to demonstrate leakage."""
    df = df.copy()
    df["speed_kmph"] = (df["distance_km"] / (df["trip_duration"] / 3600.0 + 1e-9)).clip(upper=200)
    return df


def run_experiments(features_df: pd.DataFrame, cfg: dict, weather: pd.DataFrame | None = None,
                    tracker=None) -> dict:
    """Train and evaluate every model, log each run to MLflow, and return the results.

    Random split (as in the course): baselines, the improved model, Linear Regression, weather variants.
    Time split (train Jan-May, test June): the improved model and weather variants.
    """
    exp = cfg["experiments"]
    data = add_day_features(features_df, weather) if exp.get("run_weather") else features_df
    rows = []

    def record(name, split, feats, model, kind, y_true, pred_log, params, test=None, extra=None):
        metrics = score(y_true, pred_log)
        rows.append({"model": name, "split": split, "n_features": len(feats), **metrics, **(extra or {})})
        log.info("[%s] %s: MAE %.1f s, R2 %.3f", split, name, metrics["mae_s"], metrics["r2"])
        if tracker is not None:
            example = test[feats].head(5) if (test is not None and kind == "lightgbm") else None
            tracker.log_run(name, split=split, model=model, kind=kind, features=feats,
                            params=params, metrics={**metrics, **(extra or {})}, input_example=example)
        return metrics

    template_params = {**cfg["model"]["template"]["params"],
                       "num_boost_round": cfg["model"]["template"]["num_boost_round"]}
    improved_params = {**cfg["model"]["improved"]["params"],
                       "num_boost_round": cfg["model"]["improved"]["num_boost_round"]}

    # ---------------- Random split ----------------
    train, test = make_split(data, cfg)
    y_test = test["trip_duration"].values

    if exp["run_baselines"]:
        train_l, test_l = add_leaky_speed(train), add_leaky_speed(test)
        for name, feats in [("Template LightGBM (with speed_kmph leak)", TEMPLATE_FEATURES),
                            ("LightGBM, no leak", NO_LEAK_FEATURES),
                            ("LightGBM + engineered features (template settings)", FEATURES)]:
            model, pred = train_template(train_l, test_l, feats, cfg)
            record(name, "random", feats, model, "lightgbm", y_test, pred, template_params, test_l)

    model = train_improved(train, cfg)
    pred_log = model.predict(test[FEATURES], num_iteration=model.best_iteration)
    june = (test["pickup_datetime"] >= pd.Timestamp(cfg["evaluation"]["time_split_test_start"])).values
    june_metrics = score(y_test[june], pred_log[june]) if june.any() else {}
    improved = record("Improved LightGBM (tuned)", "random", FEATURES, model, "lightgbm", y_test, pred_log,
                      {**improved_params, "best_iteration": model.best_iteration}, test,
                      extra={f"june_only_{k}": v for k, v in june_metrics.items()})

    if exp["run_baselines"]:
        linear, pred = train_linear(train, test)
        record("Linear Regression", "random", FEATURES, linear, "sklearn", y_test, pred, {"model": "linear"})

    if exp.get("run_weather") and weather is not None:
        for name, extra_cols in [("Improved + real weather", WEATHER_FEATURES),
                                 ("Improved + day of year (control)", ["pickup_dayofyear"])]:
            feats = FEATURES + extra_cols
            m_w = train_improved(train, cfg, feats)
            record(name, "random", feats, m_w, "lightgbm", y_test,
                   m_w.predict(test[feats], num_iteration=m_w.best_iteration),
                   {**improved_params, "best_iteration": m_w.best_iteration}, test)

    # ---------------- Time split ----------------
    time_metrics = None
    if exp.get("run_time_split"):
        t_train, t_test = make_time_split(data, cfg)
        t_y = t_test["trip_duration"].values
        variants = [("Improved LightGBM (tuned)", FEATURES)]
        if exp.get("run_weather") and weather is not None:
            variants += [("Improved + real weather", FEATURES + WEATHER_FEATURES),
                         ("Improved + day of year (control)", FEATURES + ["pickup_dayofyear"])]
        for name, feats in variants:
            m_t = train_improved(t_train, cfg, feats, validation="time")
            metrics = record(name, "time", feats, m_t, "lightgbm", t_y,
                             m_t.predict(t_test[feats], num_iteration=m_t.best_iteration),
                             {**improved_params, "best_iteration": m_t.best_iteration}, t_test)
            if feats == FEATURES:
                time_metrics = metrics

    return {"model": model, "comparison": pd.DataFrame(rows), "improved_metrics": improved,
            "june_only_metrics": june_metrics, "time_split_metrics": time_metrics,
            "test": test, "test_pred_sec": np.expm1(pred_log)}


def error_by_duration(y_true_sec, pred_sec) -> pd.DataFrame:
    """Mean error by actual trip length (negative bias = underestimation)."""
    bins = [60, 300, 600, 1200, 1800, 3600, 6 * 3600]
    labels = ["1-5 min", "5-10 min", "10-20 min", "20-30 min", "30-60 min", "1-6 h"]
    err = pd.DataFrame({"actual": y_true_sec, "pred": pred_sec})
    err["bucket"] = pd.cut(err["actual"], bins=bins, labels=labels, include_lowest=True)
    err["error"] = err["pred"] - err["actual"]
    table = err.groupby("bucket", observed=True).agg(
        trips=("actual", "size"),
        mean_actual_min=("actual", lambda s: s.mean() / 60),
        mean_pred_min=("pred", lambda s: s.mean() / 60),
        bias_min=("error", lambda s: s.mean() / 60),
        mae_min=("error", lambda s: s.abs().mean() / 60),
    )
    table["pct_of_trips"] = table["trips"] / len(err) * 100
    return table.round(1)
