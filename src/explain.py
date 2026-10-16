"""Model explanations with SHAP: how much each feature pushes a prediction up or down."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402


def shap_values(model, X: pd.DataFrame) -> shap.Explanation:
    """Exact TreeSHAP values, computed by LightGBM itself (multithreaded; identical to shap.TreeExplainer).

    The model predicts log(1 + duration), so values are on the log scale:
    a SHAP value of +0.10 makes the predicted duration about 10.5% longer (e^0.10 = 1.105).
    """
    contrib = model.predict(X, pred_contrib=True, num_iteration=model.best_iteration or None)
    return shap.Explanation(values=contrib[:, :-1], base_values=contrib[:, -1],
                            data=X.to_numpy(), feature_names=list(X.columns))


def summary_plot(values: shap.Explanation, path, max_display: int = 12) -> pd.DataFrame:
    """Beeswarm plot: one dot per trip; position = effect on the prediction, colour = feature value."""
    plt.figure()
    shap.plots.beeswarm(values, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 6)
    plt.title("SHAP: how each feature pushes the predicted trip duration up or down")
    plt.xlabel("Effect on log(trip duration)   (+0.1 ≈ 10% longer)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close("all")
    mean_abs = np.abs(values.values).mean(axis=0)
    return (pd.DataFrame({"feature": values.feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False).round(4))


def hour_dependence_plot(values: shap.Explanation, path) -> None:
    """Effect of the pickup hour, coloured by weekend: shows the daily traffic cycle the model learned."""
    plt.figure()
    shap.plots.scatter(values[:, "pickup_hour"], color=values[:, "is_weekend"], show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 5)
    plt.title("SHAP: effect of pickup hour (colour = weekend)")
    plt.ylabel("Effect on log(trip duration)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close("all")
