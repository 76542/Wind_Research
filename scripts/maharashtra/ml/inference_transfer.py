"""
scripts/maharashtra/ml/inference_transfer.py
=============================================
Zero-shot transfer: Apply Gujarat-trained MLP v3 to Maharashtra data.

Loads Gujarat model + scaler (no retraining), runs on Maharashtra
collocated dataset, and evaluates transfer performance.

Usage:
  cd Wind_Research
  python -m scripts.maharashtra.ml.inference_transfer
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import torch

# ---------------------------------------------------------------------------
# Path setup — PROJECT_ROOT = Wind_Research/
# (4 levels up from scripts/maharashtra/ml/inference_transfer.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.ml.config_ml import (
    SAR_FEATURES, SPATIAL_FEATURES, TIME_FEATURES, ALL_FEATURES, TARGET,
    VV_MAX, VV_MIN, VH_VV_RATIO_MIN, ERA5_MIN, ERA5_MAX,
    SEASON_MAP, WIND_SPEED_BINS, WIND_SPEED_LABELS
)
from scripts.ml.train_mlp_v3 import WindSpeedMLPv3

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MH_COLLOCATED = os.path.join(PROJECT_ROOT, "data", "processed",
                              "maharashtra", "maharashtra_era5_collocated.csv")
GJ_SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "feature_scaler.pkl")
GJ_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mlp_v3_wind_model.pth")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "maharashtra")
PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "data", "processed",
                                 "maharashtra", "maharashtra_predictions.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================================
# STEP 1: LOAD & CLEAN
# =========================================================================
def load_and_clean(path):
    print("=" * 60)
    print("STEP 1: LOAD & CLEAN MAHARASHTRA DATA")
    print("=" * 60)

    df = pd.read_csv(path)
    n_before = len(df)
    print(f"Loaded {n_before:,} rows from {os.path.basename(path)}")

    remove_mask = (
        (df["VV"] > VV_MAX) |
        (df["VV"] < VV_MIN) |
        (df["VH_VV_ratio"] < VH_VV_RATIO_MIN) |
        (df[TARGET] < ERA5_MIN) |
        (df[TARGET] > ERA5_MAX)
    )

    df = df[~remove_mask].copy()
    print(f"After cleaning: {len(df):,} rows ({n_before - len(df)} removed)")
    print()
    return df


# =========================================================================
# STEP 2: FEATURE ENGINEERING
# =========================================================================
def engineer_features(df):
    print("=" * 60)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 60)

    if 'month' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear

    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["season"] = df["month"].map(SEASON_MAP)

    print(f"Added: {TIME_FEATURES}")
    print(f"Total features: {len(ALL_FEATURES)}")
    print()
    return df


# =========================================================================
# STEP 3: SCALE WITH GUJARAT'S SCALER
# =========================================================================
def scale_with_gujarat_scaler(df):
    print("=" * 60)
    print("STEP 3: SCALE WITH GUJARAT SCALER")
    print("=" * 60)

    with open(GJ_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    print(f"Loaded Gujarat scaler from: {os.path.basename(GJ_SCALER_PATH)}")

    # Distribution comparison (unscaled)
    print("\nMaharashtra feature stats (before scaling):")
    for feat in SAR_FEATURES + SPATIAL_FEATURES[:1]:
        vals = df[feat]
        print(f"  {feat:<20} mean={vals.mean():>8.3f}  std={vals.std():>7.3f}")

    # Scale using Gujarat scaler (transform only, NOT fit)
    df_scaled = df.copy()
    df_scaled[ALL_FEATURES] = scaler.transform(df[ALL_FEATURES])

    print("\nAfter Gujarat scaling:")
    for feat in ALL_FEATURES[:5]:
        vals = df_scaled[feat]
        print(f"  {feat:<20} mean={vals.mean():>8.3f}  std={vals.std():>7.3f}")

    print("\n  If means ≠ 0 or stds ≠ 1, Maharashtra's distribution differs")
    print("  from Gujarat — model may struggle with transfer.")
    print()

    return df, df_scaled


# =========================================================================
# STEP 4: RUN GUJARAT MODEL
# =========================================================================
def predict(df_scaled):
    print("=" * 60)
    print("STEP 4: RUN GUJARAT MODEL ON MAHARASHTRA DATA")
    print("=" * 60)

    device = torch.device("cpu")
    checkpoint = torch.load(GJ_MODEL_PATH, map_location=device,
                            weights_only=False)
    arch = checkpoint["architecture"]

    print(f"Loaded: {os.path.basename(GJ_MODEL_PATH)}")
    print(f"  Trained at epoch: {checkpoint['epoch']}")
    print(f"  Gujarat val RMSE: {checkpoint['val_rmse']:.3f} m/s")
    print(f"  Architecture: {arch['hidden_layers']}")
    print()

    model = WindSpeedMLPv3(
        input_dim=arch["input_dim"],
        hidden_layers=arch["hidden_layers"],
        dropout_rate=arch["dropout"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X = torch.FloatTensor(df_scaled[ALL_FEATURES].values)

    with torch.no_grad():
        predictions = model(X).numpy()

    print(f"Predictions: {len(predictions):,} values")
    print(f"  Range: [{predictions.min():.2f}, {predictions.max():.2f}] m/s")
    print(f"  Mean:  {predictions.mean():.2f} m/s")
    print()

    return predictions


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
# STEP 5: EVALUATE
# =========================================================================
def evaluate(df_raw, predictions):
    print("=" * 60)
    print("STEP 5: EVALUATE TRANSFER PERFORMANCE")
    print("=" * 60)

    y_true = df_raw[TARGET].values
    m = compute_metrics(y_true, predictions)

    results_path = os.path.join(OUTPUT_DIR, "transfer_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        header = f"""
{'=' * 60}
ZERO-SHOT TRANSFER: Gujarat Model → Maharashtra Data
{'=' * 60}
Training data:    Gujarat coast (37,997 obs, 2020-2024)
Test data:        Maharashtra coast ({len(y_true):,} obs, 2020-2024)
Model:            MLP v3 (256-128-64-32, BatchNorm, LeakyReLU, Huber)
Scaler:           Gujarat-fitted StandardScaler
{'=' * 60}

OVERALL METRICS
{'-' * 40}
  RMSE:  {m['RMSE']:.3f} m/s
  MAE:   {m['MAE']:.3f} m/s
  R²:    {m['R2']:.4f}
  Bias:  {m['Bias']:+.3f} m/s

Gujarat model's own val RMSE: 1.285 m/s (for comparison)
"""
        print(header)
        f.write(header)

        # Stratified by wind speed
        bin_indices = np.clip(
            np.digitize(y_true, WIND_SPEED_BINS) - 1,
            0, len(WIND_SPEED_LABELS) - 1
        )

        ws_header = "\nSTRATIFIED BY WIND SPEED BIN\n" + "-" * 50
        print(ws_header)
        f.write(ws_header + "\n")

        for i, label in enumerate(WIND_SPEED_LABELS):
            mask = bin_indices == i
            n = mask.sum()
            if n == 0:
                continue
            bm = compute_metrics(y_true[mask], predictions[mask])
            row = (f"  {label:>6} m/s: N={n:>5}  RMSE={bm['RMSE']:.3f}  "
                   f"Bias={bm['Bias']:+.3f}")
            print(row)
            f.write(row + "\n")

        # Stratified by season
        seasons = df_raw["season"].values if "season" in df_raw.columns \
            else df_raw["month"].map(SEASON_MAP).values

        s_header = "\nSTRATIFIED BY SEASON\n" + "-" * 50
        print(s_header)
        f.write(s_header + "\n")

        for season in ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]:
            mask = seasons == season
            n = mask.sum()
            if n == 0:
                continue
            sm = compute_metrics(y_true[mask], predictions[mask])
            row = (f"  {season:<14} N={n:>5}  RMSE={sm['RMSE']:.3f}  "
                   f"Bias={sm['Bias']:+.3f}")
            print(row)
            f.write(row + "\n")

    print(f"\nResults saved: {results_path}")
    print()
    return m


# =========================================================================
# STEP 6: PLOTS
# =========================================================================
def plot_scatter(y_true, predictions, metrics):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, predictions, alpha=0.1, s=5, color="steelblue",
               rasterized=True)
    lims = [0, max(y_true.max(), predictions.max()) + 1]
    ax.plot(lims, lims, "r--", linewidth=1, label="1:1 line")
    ax.set_xlabel("ERA5 100m Wind Speed (m/s)", fontsize=12)
    ax.set_ylabel("Gujarat Model Prediction (m/s)", fontsize=12)
    ax.set_title(
        f"Zero-Shot Transfer: Gujarat → Maharashtra\n"
        f"RMSE={metrics['RMSE']:.3f} m/s | R²={metrics['R2']:.4f} | "
        f"Bias={metrics['Bias']:+.3f} m/s", fontsize=12
    )
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "transfer_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_error_distribution(y_true, predictions):
    fig, ax = plt.subplots(figsize=(9, 5))
    errors = predictions - y_true
    ax.hist(errors, bins=100, alpha=0.7, color="steelblue", density=True)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.axvline(errors.mean(), color="orange", linestyle="--", linewidth=1,
               label=f"Mean bias: {errors.mean():.3f} m/s")
    ax.set_xlabel("Prediction Error (m/s)")
    ax.set_ylabel("Density")
    ax.set_title("Error Distribution — Gujarat Model on Maharashtra Data")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "transfer_error_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_spatial_pattern(df_raw, predictions):
    df = df_raw.copy()
    df["predicted"] = predictions

    per_point = df.groupby(["point_id", "latitude", "longitude"]).agg(
        mean_era5=("ERA5_WindSpeed_100m_ms", "mean"),
        mean_pred=("predicted", "mean"),
    ).reset_index()

    grid_lon = np.linspace(df.longitude.min(), df.longitude.max(), 200)
    grid_lat = np.linspace(df.latitude.min(), df.latitude.max(), 200)
    grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)
    points = per_point[["longitude", "latitude"]].values

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    vmin = min(per_point["mean_era5"].min(), per_point["mean_pred"].min())
    vmax = max(per_point["mean_era5"].max(), per_point["mean_pred"].max())

    # ERA5 truth
    grid_z = griddata(points, per_point["mean_era5"].values,
                      (grid_lon2d, grid_lat2d), method="cubic")
    im1 = axes[0].pcolormesh(grid_lon, grid_lat, grid_z, cmap="jet",
                              shading="auto", vmin=vmin, vmax=vmax)
    axes[0].set_title("ERA5 Mean 100m Wind (m/s)", fontweight="bold")
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # Model prediction
    grid_z = griddata(points, per_point["mean_pred"].values,
                      (grid_lon2d, grid_lat2d), method="cubic")
    im2 = axes[1].pcolormesh(grid_lon, grid_lat, grid_z, cmap="jet",
                              shading="auto", vmin=vmin, vmax=vmax)
    axes[1].set_title("Gujarat Model Prediction (m/s)", fontweight="bold")
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    # Difference
    diff = per_point["mean_pred"].values - per_point["mean_era5"].values
    grid_z = griddata(points, diff, (grid_lon2d, grid_lat2d), method="cubic")
    max_abs = max(abs(diff.min()), abs(diff.max()), 2.0)
    im3 = axes[2].pcolormesh(grid_lon, grid_lat, grid_z, cmap="RdBu_r",
                              shading="auto", vmin=-max_abs, vmax=max_abs)
    axes[2].set_title("Difference (Pred - ERA5) (m/s)", fontweight="bold")
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    for ax in axes:
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_aspect("equal")

    fig.suptitle(
        "Zero-Shot Transfer: Gujarat-Trained Model → Maharashtra\n"
        "Spatial Pattern Comparison (2020-2024 Mean)",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "transfer_spatial_pattern.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =========================================================================
# MAIN
# =========================================================================
def main():
    print("\n" + "=" * 60)
    print("ZERO-SHOT TRANSFER: Gujarat Model → Maharashtra")
    print("=" * 60 + "\n")

    df = load_and_clean(MH_COLLOCATED)
    df = engineer_features(df)
    df_raw, df_scaled = scale_with_gujarat_scaler(df)

    predictions = predict(df_scaled)

    # Save predictions
    df_raw["predicted_wind_100m"] = predictions
    os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)
    df_raw.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predictions saved: {PREDICTIONS_PATH}")

    # Evaluate
    metrics = evaluate(df_raw, predictions)

    # Plots
    print("Generating plots...")
    y_true = df_raw[TARGET].values
    plot_scatter(y_true, predictions, metrics)
    plot_error_distribution(y_true, predictions)
    plot_spatial_pattern(df_raw, predictions)

    print()
    print("=" * 60)
    print("TRANSFER EVALUATION COMPLETE")
    print(f"All outputs in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()