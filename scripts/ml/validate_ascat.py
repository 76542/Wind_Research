"""
Independent Validation Against ASCAT
======================================
Tests all three models against ASCAT scatterometer wind speeds —
a completely independent satellite measurement (not ERA5, not SAR).

This is the strongest validation in the thesis because:
  - ASCAT is an independent instrument (MetOp satellite, C-band scatterometer)
  - Neither the models nor ERA5 used ASCAT data during training
  - It's a real measurement, not a reanalysis product

Pipeline:
  1. Load ASCAT-SAR collocated data
  2. Engineer missing features (VH/VV ratio, time encodings, etc.)
  3. Apply the same scaler used during training
  4. Run CMOD5.N, MLP v3, and RF predictions
  5. Compare all against ASCAT 100m wind speeds

Usage:
  python scripts/ml/validate_ascat.py

Inputs:
  data/ascat/ascat_sar_collocated.csv
  models/mlp_v3_wind_model.pth
  models/rf_wind_model.pkl
  models/feature_scaler.pkl

Outputs:
  outputs/ml/ascat_validation_results.txt
  outputs/ml/ascat_scatter_comparison.png
  outputs/ml/ascat_error_distribution.png
  outputs/ml/ascat_stratified_windspeed.png
  outputs/ml/ascat_stratified_season.png
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.ml.config_ml import (
    MODELS_DIR, OUTPUTS_DIR, ALL_FEATURES, TARGET,
    WIND_SPEED_BINS, WIND_SPEED_LABELS, SEASON_MAP,
    SAR_FEATURES, SPATIAL_FEATURES, TIME_FEATURES, ensure_dirs
)
from scripts.ml.train_mlp_v3 import WindSpeedMLPv3

# Paths
ASCAT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "ascat", "ascat_sar_collocated.csv"
)
V3_MODEL_PATH = os.path.join(MODELS_DIR, "mlp_v3_wind_model.pth")
SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.pkl")

# ASCAT 100m wind speed is the independent truth
ASCAT_TARGET = "wind_speed_100m"


# =========================================================================
# CMOD5.N baseline (same as evaluate.py)
# =========================================================================
def cmod5n_predict_100m(vv_db, incidence_angle_deg):
    sigma0 = 10.0 ** (vv_db / 10.0)
    theta = np.radians(incidence_angle_deg)
    c0 = 0.0015
    alpha_inc = 0.6
    gamma = 1.6
    sigma0_corrected = sigma0 / (np.cos(theta) ** alpha_inc)
    ratio = np.clip(sigma0_corrected / c0, 1e-10, None)
    wind_10m = np.clip(ratio ** (1.0 / gamma), 0.5, 40.0)
    wind_100m = wind_10m * (100.0 / 10.0) ** 0.11
    return wind_100m


# =========================================================================
# METRICS
# =========================================================================
def compute_metrics(y_true, y_pred):
    residuals = y_pred - y_true
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    bias = np.mean(residuals)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Bias": bias}


# =========================================================================
# PREPARE DATA
# =========================================================================
def prepare_ascat_data():
    """
    Load ASCAT-SAR collocated data and engineer the features
    needed by the ML models.
    """
    print("Loading ASCAT-SAR collocated data...")
    df = pd.read_csv(ASCAT_DATA_PATH)
    print(f"  {len(df):,} observations, {df.point_id.nunique()} points")
    print(f"  ASCAT 100m wind: mean={df[ASCAT_TARGET].mean():.2f}, "
          f"range=[{df[ASCAT_TARGET].min():.2f}, {df[ASCAT_TARGET].max():.2f}]")
    print()

    # --- Engineer missing features ---
    print("Engineering features...")

    # VH/VV ratio
    df["VH_VV_ratio"] = df["VH"] / df["VV"]

    # Parse timestamp for time features
    df["sar_dt"] = pd.to_datetime(df["sar_timestamp"])
    df["month"] = df["sar_dt"].dt.month
    df["day_of_year"] = df["sar_dt"].dt.dayofyear

    # Cyclical time encodings
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Offshore distance — approximate from grid data
    # Match by point_id to get offshore_distance_km
    grid_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "raw", "gujarat_sampling_grid.csv"
    )
    if os.path.exists(grid_path):
        grid = pd.read_csv(grid_path)
        # Try different possible column names
        dist_col = None
        for col in ["offshore_distance_km", "distance_km", "offshore_dist"]:
            if col in grid.columns:
                dist_col = col
                break
        if dist_col and "point_id" in grid.columns:
            dist_map = grid.set_index("point_id")[dist_col].to_dict()
            df["offshore_distance_km"] = df["point_id"].map(dist_map)
            print(f"  Mapped offshore_distance_km from grid ({df['offshore_distance_km'].notna().sum()} matched)")
        else:
            print(f"  Grid columns: {grid.columns.tolist()}")
            print("  Could not find distance column, using default 20km")
            df["offshore_distance_km"] = 20.0
    else:
        print(f"  Grid file not found at {grid_path}, using default 20km")
        df["offshore_distance_km"] = 20.0

    # Use SAR lat/lon as the spatial features (these are the actual
    # observation locations the model was trained on)
    df["latitude"] = df["sar_lat"]
    df["longitude"] = df["sar_lon"]

    # Season for stratification
    df["season"] = df["month"].map(SEASON_MAP)

    # Drop rows with NaN in any required feature
    before = len(df)
    df = df.dropna(subset=ALL_FEATURES + [ASCAT_TARGET])
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} rows with missing features")

    print(f"  Final dataset: {len(df):,} observations")
    print()

    return df


def scale_and_predict(df):
    """Apply scaler and run all three models."""

    # Keep raw values for CMOD5.N
    vv_raw = df["VV"].values.copy()
    inc_raw = df["incidence_angle"].values.copy()
    y_ascat = df[ASCAT_TARGET].values.copy()

    # --- Scale features for ML models ---
    print("Scaling features...")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    X_scaled = scaler.transform(df[ALL_FEATURES].values)
    print()

    # --- 1. CMOD5.N ---
    print("Running CMOD5.N...")
    cmod_pred = cmod5n_predict_100m(vv_raw, inc_raw)

    # --- 2. MLP v3 ---
    print("Running MLP v3...")
    checkpoint = torch.load(V3_MODEL_PATH, map_location="cpu", weights_only=False)
    arch = checkpoint["architecture"]
    model = WindSpeedMLPv3(
        input_dim=arch["input_dim"],
        hidden_layers=arch["hidden_layers"],
        dropout_rate=arch["dropout"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        mlp_pred = model(torch.FloatTensor(X_scaled)).numpy()

    # --- 3. RF ---
    print("Running Random Forest...")
    with open(os.path.join(MODELS_DIR, "rf_wind_model.pkl"), "rb") as f:
        rf = pickle.load(f)
    rf_pred = rf.predict(X_scaled)

    print()
    return y_ascat, cmod_pred, mlp_pred, rf_pred


# =========================================================================
# PRINT RESULTS
# =========================================================================
def print_overall(y, cmod, mlp, rf, f_out):
    cm = compute_metrics(y, cmod)
    mm = compute_metrics(y, mlp)
    rm = compute_metrics(y, rf)

    # Also compute ERA5 baseline if available
    table = f"""
{'=' * 70}
INDEPENDENT VALIDATION: MODELS vs ASCAT SCATTEROMETER (100m)
{'=' * 70}
{'Metric':<10} {'CMOD5.N':>12} {'MLP v3':>12} {'Random Forest':>14}
{'-' * 52}
{'RMSE':<10} {cm['RMSE']:>10.3f} ms {mm['RMSE']:>10.3f} ms {rm['RMSE']:>10.3f} ms
{'MAE':<10} {cm['MAE']:>10.3f} ms {mm['MAE']:>10.3f} ms {rm['MAE']:>10.3f} ms
{'R2':<10} {cm['R2']:>10.4f}    {mm['R2']:>10.4f}    {rm['R2']:>10.4f}
{'Bias':<10} {cm['Bias']:>+10.3f} ms {mm['Bias']:>+10.3f} ms {rm['Bias']:>+10.3f} ms
{'=' * 70}
Note: Ground truth = ASCAT scatterometer wind speed at 100m
      (independent satellite measurement, NOT used in training)
"""
    print(table)
    f_out.write(table)
    return cm, mm, rm


def stratified_by_windspeed(y, cmod, mlp, rf, f_out):
    bins = WIND_SPEED_BINS
    labels = WIND_SPEED_LABELS
    bin_idx = np.clip(np.digitize(y, bins) - 1, 0, len(labels) - 1)

    header = "\nSTRATIFIED BY ASCAT WIND SPEED BIN (m/s)"
    print(header)
    f_out.write(header + "\n")

    row_fmt = "{:<8} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}"
    hdr = row_fmt.format("Bin", "N", "CMOD RMSE", "MLP RMSE", "RF RMSE",
                         "CMOD Bias", "MLP Bias", "RF Bias")
    sep = "-" * 78
    print(hdr)
    print(sep)
    f_out.write(hdr + "\n" + sep + "\n")

    results = []
    for i, label in enumerate(labels):
        mask = bin_idx == i
        n = mask.sum()
        if n == 0:
            continue
        cm = compute_metrics(y[mask], cmod[mask])
        mm = compute_metrics(y[mask], mlp[mask])
        rm = compute_metrics(y[mask], rf[mask])

        row = row_fmt.format(label, n,
            f"{cm['RMSE']:.3f}", f"{mm['RMSE']:.3f}", f"{rm['RMSE']:.3f}",
            f"{cm['Bias']:+.3f}", f"{mm['Bias']:+.3f}", f"{rm['Bias']:+.3f}")
        print(row)
        f_out.write(row + "\n")
        results.append({"bin": label, "n": n,
                        "cmod_rmse": cm["RMSE"], "mlp_rmse": mm["RMSE"],
                        "rf_rmse": rm["RMSE"]})

    print()
    f_out.write("\n")
    return results


def stratified_by_season(df, y, cmod, mlp, rf, f_out):
    seasons = df["season"].values
    season_order = ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]

    header = "\nSTRATIFIED BY SEASON"
    print(header)
    f_out.write(header + "\n")

    row_fmt = "{:<14} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}"
    hdr = row_fmt.format("Season", "N", "CMOD RMSE", "MLP RMSE", "RF RMSE",
                         "CMOD Bias", "MLP Bias", "RF Bias")
    sep = "-" * 82
    print(hdr)
    print(sep)
    f_out.write(hdr + "\n" + sep + "\n")

    results = []
    for season in season_order:
        mask = seasons == season
        n = mask.sum()
        if n == 0:
            continue
        cm = compute_metrics(y[mask], cmod[mask])
        mm = compute_metrics(y[mask], mlp[mask])
        rm = compute_metrics(y[mask], rf[mask])

        row = row_fmt.format(season, n,
            f"{cm['RMSE']:.3f}", f"{mm['RMSE']:.3f}", f"{rm['RMSE']:.3f}",
            f"{cm['Bias']:+.3f}", f"{mm['Bias']:+.3f}", f"{rm['Bias']:+.3f}")
        print(row)
        f_out.write(row + "\n")
        results.append({"season": season, "n": n,
                        "cmod_rmse": cm["RMSE"], "mlp_rmse": mm["RMSE"],
                        "rf_rmse": rm["RMSE"]})

    print()
    f_out.write("\n")
    return results


# =========================================================================
# PLOTS
# =========================================================================
def plot_scatter(y, cmod, mlp, rf):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, pred, name, color in [
        (axes[0], cmod, "CMOD5.N", "gray"),
        (axes[1], mlp, "MLP v3", "steelblue"),
        (axes[2], rf, "Random Forest", "darkorange")
    ]:
        m = compute_metrics(y, pred)
        ax.scatter(y, pred, alpha=0.05, s=5, color=color, rasterized=True)
        lims = [0, max(y.max(), pred.max()) + 1]
        ax.plot(lims, lims, "r--", linewidth=1, label="1:1 line")
        ax.set_xlabel("ASCAT Wind Speed 100m (m/s)")
        ax.set_ylabel("Predicted Wind Speed (m/s)")
        ax.set_title(f"{name}\nRMSE={m['RMSE']:.3f} m/s | R²={m['R2']:.4f}")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Independent Validation Against ASCAT Scatterometer",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "ascat_scatter_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_error_dist(y, cmod, mlp, rf):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(-15, 15, 120)
    ax.hist(cmod - y, bins=bins, alpha=0.5, label="CMOD5.N", color="gray",
            density=True)
    ax.hist(mlp - y, bins=bins, alpha=0.5, label="MLP v3", color="steelblue",
            density=True)
    ax.hist(rf - y, bins=bins, alpha=0.5, label="Random Forest",
            color="darkorange", density=True)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction Error vs ASCAT (m/s)")
    ax.set_ylabel("Density")
    ax.set_title("Error Distribution — Independent ASCAT Validation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "ascat_error_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_stratified_ws(ws_results):
    bins_list = [r["bin"] for r in ws_results]
    x = np.arange(len(bins_list))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, [r["cmod_rmse"] for r in ws_results], width,
           label="CMOD5.N", color="gray")
    ax.bar(x, [r["mlp_rmse"] for r in ws_results], width,
           label="MLP v3", color="steelblue")
    ax.bar(x + width, [r["rf_rmse"] for r in ws_results], width,
           label="Random Forest", color="darkorange")
    ax.set_xlabel("ASCAT Wind Speed Bin (m/s)")
    ax.set_ylabel("RMSE (m/s)")
    ax.set_title("Model Performance by Wind Speed Range (ASCAT Validation)")
    ax.set_xticks(x)
    ax.set_xticklabels(bins_list)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "ascat_stratified_windspeed.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_stratified_season(season_results):
    seasons = [r["season"] for r in season_results]
    x = np.arange(len(seasons))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, [r["cmod_rmse"] for r in season_results], width,
           label="CMOD5.N", color="gray")
    ax.bar(x, [r["mlp_rmse"] for r in season_results], width,
           label="MLP v3", color="steelblue")
    ax.bar(x + width, [r["rf_rmse"] for r in season_results], width,
           label="Random Forest", color="darkorange")
    ax.set_xlabel("Season")
    ax.set_ylabel("RMSE (m/s)")
    ax.set_title("Model Performance by Season (ASCAT Validation)")
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "ascat_stratified_season.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =========================================================================
# MAIN
# =========================================================================
def main():
    # Prepare
    df = prepare_ascat_data()

    # Predict
    y_ascat, cmod_pred, mlp_pred, rf_pred = scale_and_predict(df)

    # Results
    results_path = os.path.join(OUTPUTS_DIR, "ascat_validation_results.txt")
    with open(results_path, "w") as f_out:
        f_out.write(f"ASCAT Independent Validation\n")
        f_out.write(f"Observations: {len(y_ascat):,}\n")
        f_out.write(f"Points: {df['point_id'].nunique()}\n\n")

        cm, mm, rm = print_overall(y_ascat, cmod_pred, mlp_pred, rf_pred, f_out)

        ws_results = stratified_by_windspeed(
            y_ascat, cmod_pred, mlp_pred, rf_pred, f_out)

        season_results = stratified_by_season(
            df, y_ascat, cmod_pred, mlp_pred, rf_pred, f_out)

    print(f"Results saved: {results_path}\n")

    # Plots
    print("Generating plots...")
    plot_scatter(y_ascat, cmod_pred, mlp_pred, rf_pred)
    plot_error_dist(y_ascat, cmod_pred, mlp_pred, rf_pred)
    plot_stratified_ws(ws_results)
    plot_stratified_season(season_results)

    print()
    print("=" * 70)
    print("ASCAT VALIDATION COMPLETE — All plots in outputs/ml/")
    print("=" * 70)


if __name__ == "__main__":
    main()