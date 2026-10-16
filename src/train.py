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


def train_improved(train, cfg):
    """Tuned LightGBM, early-stopping on a validation split of the TRAINING data (test set untouched)."""
    m, imp = cfg["model"], cfg["model"]["improved"]
    X_fit, X_val, y_fit, y_val = train_test_split(
        train[FEATURES], np.log1p(train["trip_duration"]),
        test_size=m["validation_size"], random_state=m["random_state"])
    fit_set = lgb.Dataset(X_fit, label=y_fit)
    val_set = lgb.Dataset(X_val, label=y_val, reference=fit_set)
    model = lgb.train(_params(imp, m["random_state"]), fit_set,
                      num_boost_round=imp["num_boost_round"], valid_sets=[val_set],
                      callbacks=[lgb.early_stopping(imp["early_stopping_rounds"], verbose=False)])
    log.info("Improved model: best iteration %d", model.best_iteration)
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


def run_experiments(features_df: pd.DataFrame, cfg: dict) -> dict:
    """Train the improved model (and, optionally, the baselines) and evaluate on the shared test set."""
    train, test = make_split(features_df, cfg)
    y_test = test["trip_duration"].values
    rows = []

    if cfg["experiments"]["run_baselines"]:
        train_l, test_l = add_leaky_speed(train), add_leaky_speed(test)
        for name, feats in [("Template LightGBM (with speed_kmph leak)", TEMPLATE_FEATURES),
                            ("LightGBM, no leak", NO_LEAK_FEATURES),
                            ("LightGBM + engineered features (template settings)", FEATURES)]:
            _, pred = train_template(train_l, test_l, feats, cfg)
            rows.append({"model": name, "n_features": len(feats), **score(y_test, pred)})
            log.info("%s: %s", name, rows[-1])

    model = train_improved(train, cfg)
    pred_log = model.predict(test[FEATURES], num_iteration=model.best_iteration)
    improved = {"model": "Improved LightGBM (tuned)", "n_features": len(FEATURES), **score(y_test, pred_log)}
    rows.append(improved)
    log.info("Improved LightGBM: %s", improved)

    if cfg["experiments"]["run_baselines"]:
        _, pred = train_linear(train, test)
        rows.append({"model": "Linear Regression", "n_features": len(FEATURES), **score(y_test, pred)})
        log.info("Linear Regression: %s", rows[-1])

    return {"model": model, "comparison": pd.DataFrame(rows), "improved_metrics": improved,
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
