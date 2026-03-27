"""
Step 7 (Updated): Evaluate All Models on Test Set
===================================================
Three-way comparison:
  1. CMOD5.N — traditional empirical GMF (physics baseline)
  2. Random Forest — ML baseline
  3. MLP v3 — neural network (your contribution)

CMOD5.N is applied to the raw (unscaled) SAR data from the test
points, then extrapolated from 10m to 100m using the power law
with alpha=0.11 (offshore, neutral stability).

Usage:
  cd /path/to/wind-research
  python scripts/ml/evaluate.py

Outputs (all in outputs/ml/):
  evaluation_results.txt
  scatter_comparison.png
  error_distribution.png
  stratified_by_windspeed.png
  stratified_by_season.png
  residual_analysis.png
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.ml.config_ml import (
    TEST_DATA_PATH, CLEAN_DATA_PATH, RF_MODEL_PATH,
    MODELS_DIR, OUTPUTS_DIR,
    ALL_FEATURES, TARGET,
    WIND_SPEED_BINS, WIND_SPEED_LABELS, SEASON_MAP, ensure_dirs
)
from scripts.ml.train_mlp_v3 import WindSpeedMLPv3

# Path to v3 model
V3_MODEL_PATH = os.path.join(MODELS_DIR, "mlp_v3_wind_model.pth")


# =========================================================================
# CMOD5.N — Physics Baseline
# =========================================================================
def cmod5n_wind_speed_10m(vv_db, incidence_angle_deg):
    """
    Simplified CMOD5.N inversion: VV backscatter (dB) -> 10m wind speed.

    This is the same log-space power law inversion used in
    calculate_wind_speed_10m.py, applied here to the test set
    for a fair comparison.
    """
    # Convert VV from dB to linear (sigma0)
    sigma0 = 10.0 ** (vv_db / 10.0)

    # Incidence angle in radians
    theta = np.radians(incidence_angle_deg)

    # CMOD5.N-style coefficients (simplified, averaged over wind direction)
    c0 = 0.0015
    alpha_inc = 0.6
    gamma = 1.6

    # Incidence angle correction
    sigma0_corrected = sigma0 / (np.cos(theta) ** alpha_inc)

    # Invert power law in log space
    ratio = sigma0_corrected / c0
    ratio = np.clip(ratio, 1e-10, None)

    wind_10m = ratio ** (1.0 / gamma)
    wind_10m = np.clip(wind_10m, 0.5, 40.0)

    return wind_10m


def extrapolate_10m_to_100m(wind_10m, alpha=0.11):
    """
    Power law wind profile extrapolation from 10m to 100m hub height.
    U(100) = U(10) * (100/10)^alpha
    alpha = 0.11 for offshore, neutral stability (IEC 61400-3).
    """
    return wind_10m * (100.0 / 10.0) ** alpha


# =========================================================================
# METRICS
# =========================================================================
def compute_metrics(y_true, y_pred):
    """Compute RMSE, MAE, R2, and Bias."""
    residuals = y_pred - y_true
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    bias = np.mean(residuals)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Bias": bias}


# =========================================================================
# LOAD MODELS AND PREDICT
# =========================================================================
def load_and_predict():
    """Load all three models and generate predictions on test set."""
    ensure_dirs()

    # --- Load scaled test data (for MLP and RF) ---
    df_test_scaled = pd.read_csv(TEST_DATA_PATH)
    X_test_scaled = df_test_scaled[ALL_FEATURES].values
    y_test = df_test_scaled[TARGET].values

    # --- Load unscaled clean data for CMOD5.N ---
    df_clean = pd.read_csv(CLEAN_DATA_PATH)
    # Filter to test points only using point_id + timestamp
    test_keys = set(zip(df_test_scaled["point_id"], df_test_scaled["timestamp"]))
    mask = df_clean.apply(
        lambda r: (r["point_id"], r["timestamp"]) in test_keys, axis=1)
    df_test_raw = df_clean[mask].copy()

    # Sort both to ensure alignment
    df_test_scaled = df_test_scaled.sort_values(
        ["point_id", "timestamp"]).reset_index(drop=True)
    df_test_raw = df_test_raw.sort_values(
        ["point_id", "timestamp"]).reset_index(drop=True)

    # Verify alignment
    assert len(df_test_raw) == len(df_test_scaled), \
        f"Row mismatch: {len(df_test_raw)} vs {len(df_test_scaled)}"
    assert (df_test_raw["point_id"].values ==
            df_test_scaled["point_id"].values).all(), "Point ID mismatch"

    y_test = df_test_scaled[TARGET].values
    X_test_scaled = df_test_scaled[ALL_FEATURES].values

    print(f"Test set: {len(y_test):,} samples, "
          f"{df_test_scaled['point_id'].nunique()} points")
    print()

    # --- 1. CMOD5.N predictions ---
    print("Running CMOD5.N...")
    vv_raw = df_test_raw["VV"].values
    inc_raw = df_test_raw["incidence_angle"].values
    cmod_10m = cmod5n_wind_speed_10m(vv_raw, inc_raw)
    cmod_100m = extrapolate_10m_to_100m(cmod_10m)
    print(f"  CMOD5.N 100m: mean={cmod_100m.mean():.2f}, "
          f"range=[{cmod_100m.min():.2f}, {cmod_100m.max():.2f}]")

    # --- 2. MLP v3 predictions ---
    print("Running MLP v3...")
    device = torch.device("cpu")
    checkpoint = torch.load(V3_MODEL_PATH, map_location=device,
                            weights_only=False)
    arch = checkpoint["architecture"]
    model = WindSpeedMLPv3(
        input_dim=arch["input_dim"],
        hidden_layers=arch["hidden_layers"],
        dropout_rate=arch["dropout"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        mlp_pred = model(torch.FloatTensor(X_test_scaled)).numpy()

    # --- 3. RF predictions ---
    print("Running Random Forest...")
    with open(RF_MODEL_PATH, "rb") as f:
        rf = pickle.load(f)
    rf_pred = rf.predict(X_test_scaled)

    print()
    return df_test_scaled, y_test, cmod_100m, mlp_pred, rf_pred


# =========================================================================
# OVERALL METRICS
# =========================================================================
def print_overall_metrics(y_test, cmod_pred, mlp_pred, rf_pred, f_out):
    cmod_m = compute_metrics(y_test, cmod_pred)
    mlp_m = compute_metrics(y_test, mlp_pred)
    rf_m = compute_metrics(y_test, rf_pred)

    table = f"""
{'=' * 65}
OVERALL TEST SET METRICS
{'=' * 65}
{'Metric':<10} {'CMOD5.N':>12} {'MLP v3':>12} {'Random Forest':>14}
{'-' * 52}
{'RMSE':<10} {cmod_m['RMSE']:>10.3f} ms {mlp_m['RMSE']:>10.3f} ms {rf_m['RMSE']:>10.3f} ms
{'MAE':<10} {cmod_m['MAE']:>10.3f} ms {mlp_m['MAE']:>10.3f} ms {rf_m['MAE']:>10.3f} ms
{'R2':<10} {cmod_m['R2']:>10.4f}    {mlp_m['R2']:>10.4f}    {rf_m['R2']:>10.4f}
{'Bias':<10} {cmod_m['Bias']:>+10.3f} ms {mlp_m['Bias']:>+10.3f} ms {rf_m['Bias']:>+10.3f} ms
{'=' * 65}
"""
    print(table)
    f_out.write(table)
    return cmod_m, mlp_m, rf_m


# =========================================================================
# STRATIFIED BY WIND SPEED
# =========================================================================
def stratified_by_windspeed(y_test, cmod_pred, mlp_pred, rf_pred, f_out):
    bins = WIND_SPEED_BINS
    labels = WIND_SPEED_LABELS
    bin_indices = np.clip(np.digitize(y_test, bins) - 1, 0, len(labels) - 1)

    header = "\nSTRATIFIED BY WIND SPEED BIN (m/s)"
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
        mask = bin_indices == i
        n = mask.sum()
        if n == 0:
            continue

        cm = compute_metrics(y_test[mask], cmod_pred[mask])
        mm = compute_metrics(y_test[mask], mlp_pred[mask])
        rm = compute_metrics(y_test[mask], rf_pred[mask])

        row = row_fmt.format(
            label, n,
            f"{cm['RMSE']:.3f}", f"{mm['RMSE']:.3f}", f"{rm['RMSE']:.3f}",
            f"{cm['Bias']:+.3f}", f"{mm['Bias']:+.3f}", f"{rm['Bias']:+.3f}"
        )
        print(row)
        f_out.write(row + "\n")
        results.append({
            "bin": label, "n": n,
            "cmod_rmse": cm["RMSE"], "mlp_rmse": mm["RMSE"],
            "rf_rmse": rm["RMSE"],
            "cmod_bias": cm["Bias"], "mlp_bias": mm["Bias"],
            "rf_bias": rm["Bias"]
        })

    print()
    f_out.write("\n")
    return results


# =========================================================================
# STRATIFIED BY SEASON
# =========================================================================
def stratified_by_season(df_test, y_test, cmod_pred, mlp_pred, rf_pred, f_out):
    seasons = df_test["month"].map(SEASON_MAP).values
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

        cm = compute_metrics(y_test[mask], cmod_pred[mask])
        mm = compute_metrics(y_test[mask], mlp_pred[mask])
        rm = compute_metrics(y_test[mask], rf_pred[mask])

        row = row_fmt.format(
            season, n,
            f"{cm['RMSE']:.3f}", f"{mm['RMSE']:.3f}", f"{rm['RMSE']:.3f}",
            f"{cm['Bias']:+.3f}", f"{mm['Bias']:+.3f}", f"{rm['Bias']:+.3f}"
        )
        print(row)
        f_out.write(row + "\n")
        results.append({
            "season": season, "n": n,
            "cmod_rmse": cm["RMSE"], "mlp_rmse": mm["RMSE"],
            "rf_rmse": rm["RMSE"]
        })

    print()
    f_out.write("\n")
    return results


# =========================================================================
# PLOT 1: SCATTER — 3 panels
# =========================================================================
def plot_scatter(y_test, cmod_pred, mlp_pred, rf_pred):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, pred, name, color in [
        (axes[0], cmod_pred, "CMOD5.N", "gray"),
        (axes[1], mlp_pred, "MLP v3", "steelblue"),
        (axes[2], rf_pred, "Random Forest", "darkorange")
    ]:
        m = compute_metrics(y_test, pred)
        ax.scatter(y_test, pred, alpha=0.1, s=5, color=color, rasterized=True)
        lims = [0, max(y_test.max(), pred.max()) + 1]
        ax.plot(lims, lims, "r--", linewidth=1, label="1:1 line")
        ax.set_xlabel("ERA5 Wind Speed (m/s)")
        ax.set_ylabel("Predicted Wind Speed (m/s)")
        ax.set_title(f"{name}\nRMSE={m['RMSE']:.3f} m/s | R²={m['R2']:.4f}")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "scatter_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =========================================================================
# PLOT 2: ERROR DISTRIBUTION — 3 models
# =========================================================================
def plot_error_distribution(y_test, cmod_pred, mlp_pred, rf_pred):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(-12, 12, 120)
    ax.hist(cmod_pred - y_test, bins=bins, alpha=0.5, label="CMOD5.N",
            color="gray", density=True)
    ax.hist(mlp_pred - y_test, bins=bins, alpha=0.5, label="MLP v3",
            color="steelblue", density=True)
    ax.hist(rf_pred - y_test, bins=bins, alpha=0.5, label="Random Forest",
            color="darkorange", density=True)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction Error (m/s)")
    ax.set_ylabel("Density")
    ax.set_title("Error Distribution — All Models")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "error_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =========================================================================
# PLOT 3: STRATIFIED BAR CHART — WIND SPEED BINS
# =========================================================================
def plot_stratified_windspeed(ws_results):
    bins_list = [r["bin"] for r in ws_results]
    cmod_rmse = [r["cmod_rmse"] for r in ws_results]
    mlp_rmse = [r["mlp_rmse"] for r in ws_results]
    rf_rmse = [r["rf_rmse"] for r in ws_results]

    x = np.arange(len(bins_list))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, cmod_rmse, width, label="CMOD5.N", color="gray")
    ax.bar(x, mlp_rmse, width, label="MLP v3", color="steelblue")
    ax.bar(x + width, rf_rmse, width, label="Random Forest",
           color="darkorange")
    ax.set_xlabel("ERA5 Wind Speed Bin (m/s)")
    ax.set_ylabel("RMSE (m/s)")
    ax.set_title("Model Performance by Wind Speed Range")
    ax.set_xticks(x)
    ax.set_xticklabels(bins_list)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "stratified_by_windspeed.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =========================================================================
# PLOT 4: STRATIFIED BAR CHART — SEASONS
# =========================================================================
def plot_stratified_season(season_results):
    seasons = [r["season"] for r in season_results]
    cmod_rmse = [r["cmod_rmse"] for r in season_results]
    mlp_rmse = [r["mlp_rmse"] for r in season_results]
    rf_rmse = [r["rf_rmse"] for r in season_results]

    x = np.arange(len(seasons))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, cmod_rmse, width, label="CMOD5.N", color="gray")
    ax.bar(x, mlp_rmse, width, label="MLP v3", color="steelblue")
    ax.bar(x + width, rf_rmse, width, label="Random Forest",
           color="darkorange")
    ax.set_xlabel("Season")
    ax.set_ylabel("RMSE (m/s)")
    ax.set_title("Model Performance by Season")
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "stratified_by_season.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =========================================================================
# PLOT 5: RESIDUALS — 3 panels
# =========================================================================
def plot_residuals(y_test, cmod_pred, mlp_pred, rf_pred):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, pred, name, color in [
        (axes[0], cmod_pred, "CMOD5.N", "gray"),
        (axes[1], mlp_pred, "MLP v3", "steelblue"),
        (axes[2], rf_pred, "Random Forest", "darkorange")
    ]:
        residuals = pred - y_test
        ax.scatter(y_test, residuals, alpha=0.1, s=5, color=color,
                   rasterized=True)
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("ERA5 Wind Speed (m/s)")
        ax.set_ylabel("Residual (Pred - Actual) (m/s)")
        ax.set_title(f"{name}")
        ax.set_ylim(-15, 15)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "residual_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =========================================================================
# MAIN
# =========================================================================
def main():
    df_test, y_test, cmod_pred, mlp_pred, rf_pred = load_and_predict()

    results_path = os.path.join(OUTPUTS_DIR, "evaluation_results.txt")
    with open(results_path, "w") as f_out:
        cmod_m, mlp_m, rf_m = print_overall_metrics(
            y_test, cmod_pred, mlp_pred, rf_pred, f_out)

        ws_results = stratified_by_windspeed(
            y_test, cmod_pred, mlp_pred, rf_pred, f_out)

        season_results = stratified_by_season(
            df_test, y_test, cmod_pred, mlp_pred, rf_pred, f_out)

    print(f"Full results saved: {results_path}\n")

    print("Generating plots...")
    plot_scatter(y_test, cmod_pred, mlp_pred, rf_pred)
    plot_error_distribution(y_test, cmod_pred, mlp_pred, rf_pred)
    plot_stratified_windspeed(ws_results)
    plot_stratified_season(season_results)
    plot_residuals(y_test, cmod_pred, mlp_pred, rf_pred)

    print()
    print("=" * 65)
    print("EVALUATION COMPLETE — All plots in outputs/ml/")
    print("=" * 65)


if __name__ == "__main__":
    main()