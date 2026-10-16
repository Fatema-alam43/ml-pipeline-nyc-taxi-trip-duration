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


# Landmarks drawn on the speed map: name -> (latitude, longitude)
LANDMARKS = {"Midtown": (40.754, -73.984), "Lower Manhattan": (40.709, -74.010),
             "JFK Airport": (40.641, -73.778), "LaGuardia": (40.777, -73.874)}


def speed_map_png(grid: pd.DataFrame, cell_deg: float, path) -> None:
    """Static map: each grid cell coloured by the median straight-line speed of trips starting there."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    if grid.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 9))
    half = cell_deg / 2
    cells = [Rectangle((lon - half, lat - half), cell_deg, cell_deg)
             for lat, lon in zip(grid["lat"], grid["lon"], strict=True)]
    vmin, vmax = grid["median_speed_kmh"].quantile([0.05, 0.95])
    coll = PatchCollection(cells, cmap="RdYlGn", linewidth=0)
    coll.set_array(grid["median_speed_kmh"].to_numpy())
    coll.set_clim(vmin, vmax)
    ax.add_collection(coll)
    ax.set_xlim(grid["lon"].min() - cell_deg, grid["lon"].max() + cell_deg)
    ax.set_ylim(grid["lat"].min() - cell_deg, grid["lat"].max() + cell_deg)
    ax.set_aspect(1 / np.cos(np.radians(40.75)))     # correct east-west stretching at NYC's latitude
    for name, (lat, lon) in LANDMARKS.items():
        ax.annotate(name, (lon, lat), fontsize=9, ha="left", va="bottom",
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.8})
    fig.colorbar(coll, ax=ax, shrink=0.7, label="Median trip speed (km/h, straight-line)")
    ax.set(title="Where NYC taxi trips are slowest: median speed by pickup area",
           xlabel="Longitude", ylabel="Latitude")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def speed_map_html(grid: pd.DataFrame, cell_deg: float, path) -> None:
    """Interactive map (folium): hover a cell to see its speed and number of trips.

    Background tiles: Esri World Light Gray (free, no API key, and they load from a local file -
    OpenStreetMap's servers refuse requests from files opened directly on disk).
    """
    import branca.colormap as cm
    import folium

    if grid.empty:
        return
    vmin, vmax = grid["median_speed_kmh"].quantile([0.05, 0.95])
    colors = cm.LinearColormap(["#d73027", "#fee08b", "#1a9850"], vmin=vmin, vmax=vmax,
                               caption="Median trip speed (km/h)")
    m = folium.Map(tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
        name="Esri World Light Gray",
        max_zoom=16,
    ).add_to(m)
    half = cell_deg / 2
    for row in grid.itertuples():
        folium.Rectangle(
            bounds=[[row.lat - half, row.lon - half], [row.lat + half, row.lon + half]],
            stroke=False, fill=True, fill_opacity=0.7,
            fill_color=colors(min(max(row.median_speed_kmh, vmin), vmax))[:7],   # #rrggbb
            tooltip=f"{row.median_speed_kmh:.1f} km/h · {row.trips:,} trips · {row.median_duration_min:.1f} min",
        ).add_to(m)
    colors.add_to(m)
    # zoom to the data, whatever the window size
    m.fit_bounds([[grid["lat"].min() - half, grid["lon"].min() - half],
                  [grid["lat"].max() + half, grid["lon"].max() + half]])
    m.save(str(path))
