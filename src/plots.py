"""Figures saved to reports/figures/."""

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def duration_before_after(raw_duration: pd.Series, clean_duration: pd.Series, path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.logspace(0, np.log10(raw_duration.max()), 80)
    ax[0].hist(raw_duration, bins=bins, color="indianred")
    ax[0].set_xscale("log")
    for x, label in [(60, "1 min"), (6 * 3600, "6 h")]:
        ax[0].axvline(x, color="black", linestyle="--", linewidth=1)
        ax[0].text(x, ax[0].get_ylim()[1] * 0.9, " " + label, fontsize=9)
    ax[0].set(title=f"Before cleaning (n = {len(raw_duration):,})",
              xlabel="Trip duration (seconds, log scale)", ylabel="Number of trips")
    ax[1].hist(clean_duration / 60, bins=80, color="seagreen")
    ax[1].set(title=f"After cleaning (n = {len(clean_duration):,})",
              xlabel="Trip duration (minutes)", ylabel="Number of trips")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def feature_importance(model, path) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": model.feature_name(), "gain": model.feature_importance("gain")})
    imp["share_pct"] = imp["gain"] / imp["gain"].sum() * 100
    imp = imp.sort_values("gain")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(imp["feature"], imp["share_pct"], color="#4c78a8")
    ax.set(xlabel="Share of total importance (gain, %)", title="Feature importance - improved LightGBM")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return imp.sort_values("gain", ascending=False).round(2)


def predicted_vs_actual(y_true_sec, pred_sec, path, seed=42, n=10_000, limit_min=120) -> None:
    idx = np.random.default_rng(seed).choice(len(y_true_sec), size=min(n, len(y_true_sec)), replace=False)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(np.asarray(y_true_sec)[idx] / 60, np.asarray(pred_sec)[idx] / 60, s=6, alpha=0.25, color="#4c78a8")
    ax.plot([0, limit_min], [0, limit_min], "--", color="#e45756", label="Perfect prediction")
    ax.set(xlim=(0, limit_min), ylim=(0, limit_min), xlabel="Actual trip duration (minutes)",
           ylabel="Predicted trip duration (minutes)",
           title=f"Predicted vs actual - improved LightGBM ({len(idx):,} test trips)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
