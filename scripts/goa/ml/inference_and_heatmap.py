"""
scripts/goa/ml/inference_and_heatmap.py
========================================
Runs the Maharashtra fine-tuned model on Goa data (zero-shot from
Maharashtra's perspective), evaluates performance, and generates
the Gujarat-style threshold heatmap.

Why Maharashtra model and not Gujarat?
  Goa is immediately south of Maharashtra — same Konkan coast,
  same latitude range (~15°N), same Arabian Sea exposure. The
  Maharashtra fine-tuned model has already adapted its spatial
  features to this region and should outperform the Gujarat model.

Usage:
  cd Wind_Research
  python -m scripts.goa.ml.inference_and_heatmap
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.ml.config_ml import (
    ALL_FEATURES, TARGET,
    VV_MAX, VV_MIN, VH_VV_RATIO_MIN, ERA5_MIN, ERA5_MAX,
    SEASON_MAP, WIND_SPEED_BINS, WIND_SPEED_LABELS
)
from scripts.ml.train_mlp_v3 import WindSpeedMLPv3

# Paths
GOA_COLLOCATED = os.path.join(PROJECT_ROOT, "data", "processed", "goa",
                               "goa_era5_collocated.csv")
GJ_SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "feature_scaler.pkl")
MH_MODEL_PATH = os.path.join(PROJECT_ROOT, "models",
                              "mlp_v3_maharashtra_finetuned.pth")
GJ_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mlp_v3_wind_model.pth")
COAST_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "goa",
                           "goa_coastline_cache.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "goa")
PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "goa",
                                 "goa_predictions.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred):
    residuals = y_pred - y_true
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    bias = np.mean(residuals)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Bias": bias}


def load_coastline():
    coast_df = pd.read_csv(COAST_PATH)
    segments = []
    current_seg = []
    for _, row in coast_df.iterrows():
        if pd.isna(row['latitude']):
            if len(current_seg) > 1:
                segments.append(current_seg)
            current_seg = []
        else:
            current_seg.append((row['longitude'], row['latitude']))
    if len(current_seg) > 1:
        segments.append(current_seg)
    return segments


def build_land_mask(coast_pts, grid_lon2d, grid_lat2d):
    coast_lats = np.array([p[1] for p in coast_pts])
    coast_lons = np.array([p[0] for p in coast_pts])

    lat_min, lat_max = coast_lats.min(), coast_lats.max()
    bin_size = 0.01
    bin_edges = np.arange(lat_min - bin_size, lat_max + 2 * bin_size, bin_size)
    bin_centers = []
    bin_max_lons = []

    for i in range(len(bin_edges) - 1):
        mask = (coast_lats >= bin_edges[i]) & (coast_lats < bin_edges[i + 1])
        if mask.any():
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_max_lons.append(coast_lons[mask].max())

    bin_centers = np.array(bin_centers)
    bin_max_lons = np.array(bin_max_lons)

    grid_coast_lon = np.interp(grid_lat2d.ravel(), bin_centers, bin_max_lons,
                                left=bin_max_lons[0], right=bin_max_lons[-1])
    grid_coast_lon = grid_coast_lon.reshape(grid_lon2d.shape)

    land_mask = grid_lon2d > (grid_coast_lon - 0.02)

    lat_buffer = 0.1
    lat_mask = ((grid_lat2d < (lat_min - lat_buffer)) |
                (grid_lat2d > (lat_max + lat_buffer)))

    return land_mask | lat_mask


def main():
    print("\n" + "=" * 60)
    print("GOA: Maharashtra Model Transfer + Heatmap")
    print("=" * 60 + "\n")

    # ── Load and prepare data ─────────────────────────────────────
    print("Loading Goa data...")
    df = pd.read_csv(GOA_COLLOCATED)
    n_before = len(df)

    remove_mask = (
        (df["VV"] > VV_MAX) | (df["VV"] < VV_MIN) |
        (df["VH_VV_ratio"] < VH_VV_RATIO_MIN) |
        (df[TARGET] < ERA5_MIN) | (df[TARGET] > ERA5_MAX)
    )
    df = df[~remove_mask].copy()
    print(f"  {n_before} -> {len(df)} after cleaning")

    if 'month' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear

    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["season"] = df["month"].map(SEASON_MAP)

    print(f"  {len(df):,} observations, {df.point_id.nunique()} points")

    # Scale with Gujarat scaler
    with open(GJ_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    df_raw = df.copy()
    df[ALL_FEATURES] = scaler.transform(df[ALL_FEATURES])

    X = torch.FloatTensor(df[ALL_FEATURES].values)
    device = torch.device("cpu")

    # ── Run both models ───────────────────────────────────────────
    # Gujarat original (for comparison)
    print("\nRunning Gujarat model...")
    gj_ckpt = torch.load(GJ_MODEL_PATH, map_location=device, weights_only=False)
    gj_arch = gj_ckpt["architecture"]
    gj_model = WindSpeedMLPv3(gj_arch["input_dim"], gj_arch["hidden_layers"],
                               gj_arch["dropout"])
    gj_model.load_state_dict(gj_ckpt["model_state_dict"])
    gj_model.eval()
    with torch.no_grad():
        gj_pred = gj_model(X).numpy()

    # Maharashtra fine-tuned
    print("Running Maharashtra fine-tuned model...")
    mh_ckpt = torch.load(MH_MODEL_PATH, map_location=device, weights_only=False)
    mh_arch = mh_ckpt["architecture"]
    mh_model = WindSpeedMLPv3(mh_arch["input_dim"], mh_arch["hidden_layers"],
                               mh_arch["dropout"])
    mh_model.load_state_dict(mh_ckpt["model_state_dict"])
    mh_model.eval()
    with torch.no_grad():
        mh_pred = mh_model(X).numpy()

    df_raw["gj_pred"] = gj_pred
    df_raw["mh_pred"] = mh_pred

    # ── Evaluate ──────────────────────────────────────────────────
    y_true = df_raw[TARGET].values
    gj_m = compute_metrics(y_true, gj_pred)
    mh_m = compute_metrics(y_true, mh_pred)

    results_path = os.path.join(OUTPUT_DIR, "goa_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        report = f"""
{'=' * 60}
GOA TRANSFER RESULTS
{'=' * 60}
Observations: {len(y_true):,} ({df_raw.point_id.nunique()} points)
ERA5 mean wind: {y_true.mean():.2f} m/s

                    Gujarat Model    Maharashtra FT
{'=' * 60}
  RMSE (m/s):       {gj_m['RMSE']:>8.3f}          {mh_m['RMSE']:>8.3f}
  MAE (m/s):        {gj_m['MAE']:>8.3f}          {mh_m['MAE']:>8.3f}
  R2:               {gj_m['R2']:>8.4f}          {mh_m['R2']:>8.4f}
  Bias (m/s):       {gj_m['Bias']:>+8.3f}          {mh_m['Bias']:>+8.3f}
{'=' * 60}

Best model for Goa: {'Maharashtra FT' if mh_m['RMSE'] < gj_m['RMSE'] else 'Gujarat'}
"""
        print(report)
        f.write(report)

        # Stratified by wind speed
        bin_indices = np.clip(
            np.digitize(y_true, WIND_SPEED_BINS) - 1,
            0, len(WIND_SPEED_LABELS) - 1)

        f.write("\nSTRATIFIED BY WIND SPEED BIN\n" + "-" * 60 + "\n")
        fmt = "{:<8} {:>5}  {:>10} {:>10} {:>10} {:>10}"
        hdr = fmt.format("Bin", "N", "GJ RMSE", "MH RMSE", "GJ Bias", "MH Bias")
        print(hdr)
        f.write(hdr + "\n")
        for i, label in enumerate(WIND_SPEED_LABELS):
            mask = bin_indices == i
            n = mask.sum()
            if n == 0:
                continue
            gj_bm = compute_metrics(y_true[mask], gj_pred[mask])
            mh_bm = compute_metrics(y_true[mask], mh_pred[mask])
            row = fmt.format(label, n,
                f"{gj_bm['RMSE']:.3f}", f"{mh_bm['RMSE']:.3f}",
                f"{gj_bm['Bias']:+.3f}", f"{mh_bm['Bias']:+.3f}")
            print(row)
            f.write(row + "\n")

    print(f"\nResults saved: {results_path}")

    # Save predictions
    df_raw.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predictions saved: {PREDICTIONS_PATH}")

    # ── Decide which model to use for heatmap ─────────────────────
    best_model = "mh" if mh_m['RMSE'] < gj_m['RMSE'] else "gj"
    best_pred_col = f"{best_model}_pred"
    best_name = "Maharashtra FT" if best_model == "mh" else "Gujarat"
    print(f"\nUsing {best_name} model for heatmap (lower RMSE)")

    # ── Generate heatmap ──────────────────────────────────────────
    print("\nGenerating heatmap...")

    per_point = df_raw.groupby(['point_id', 'latitude', 'longitude']).agg(
        days_gt4=(best_pred_col, lambda x: (x > 4).sum()),
        days_gt6=(best_pred_col, lambda x: (x > 6).sum()),
        days_gt8=(best_pred_col, lambda x: (x > 8).sum()),
        days_gt4_era5=('ERA5_WindSpeed_100m_ms', lambda x: (x > 4).sum()),
        days_gt6_era5=('ERA5_WindSpeed_100m_ms', lambda x: (x > 6).sum()),
        days_gt8_era5=('ERA5_WindSpeed_100m_ms', lambda x: (x > 8).sum()),
    ).reset_index()

    points = per_point[['longitude', 'latitude']].values

    grid_lon = np.linspace(72.4, 74.2, 300)
    grid_lat = np.linspace(14.7, 15.9, 300)
    grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)

    tree = cKDTree(points)
    grid_pts = np.column_stack([grid_lon2d.ravel(), grid_lat2d.ravel()])
    dists, _ = tree.query(grid_pts)
    dist_mask = dists.reshape(grid_lon2d.shape) > 0.40

    # Load coastline and build land mask
    coast_segments = load_coastline()
    coast_all_pts = []
    for seg in coast_segments:
        coast_all_pts.extend(seg)
    land_mask = build_land_mask(coast_all_pts, grid_lon2d, grid_lat2d)
    combined_mask = land_mask | dist_mask

    def make_map(cols, title_prefix, filename, subtitle):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for ax, (col, label) in zip(axes, cols):
            values = per_point[col].values
            grid_z = griddata(points, values, (grid_lon2d, grid_lat2d),
                              method='cubic')
            grid_z[combined_mask] = np.nan
            grid_z = np.clip(grid_z, 0, None)
            im = ax.pcolormesh(grid_lon, grid_lat, grid_z, cmap='jet',
                                shading='auto')
            for seg in coast_segments:
                lons, lats = zip(*seg)
                ax.plot(lons, lats, color='gray', linewidth=0.8, alpha=0.7)
            ax.set_xlim(72.4, 74.2)
            ax.set_ylim(14.7, 15.9)
            ax.set_xlabel('Longitude (E)')
            ax.set_ylabel('Latitude (N)')
            ax.set_title(f'{title_prefix}  |  Days {label}',
                         fontsize=12, fontweight='bold')
            ax.set_aspect('equal')
            ax.set_facecolor('lightgray')
            ax.grid(True, alpha=0.2)
            ax.text(0.03, 0.97, f'{label}', transform=ax.transAxes,
                    fontsize=14, fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='black', alpha=0.8))
            plt.colorbar(im, ax=ax, label='Number of Days', shrink=0.75)
        fig.suptitle(
            f'Offshore Wind Resource Potential - Goa Coast (2020-2024)\n'
            f'{subtitle}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    make_map(
        [('days_gt4', '> 4 m/s'), ('days_gt6', '> 6 m/s'),
         ('days_gt8', '> 8 m/s')],
        f'MLP v3 ({best_name})', 'goa_model_heatmap.png',
        f'MLP v3 SAR-based Prediction ({best_name} Transfer)  |  '
        f'Sentinel-1 100m Hub-Height')

    make_map(
        [('days_gt4_era5', '> 4 m/s'), ('days_gt6_era5', '> 6 m/s'),
         ('days_gt8_era5', '> 8 m/s')],
        'ERA5 (Truth)', 'goa_era5_heatmap.png',
        'ERA5 100m Hub-Height Wind Speed (Ground Truth Reference)')

    print("\n" + "=" * 60)
    print("GOA COMPLETE")
    print(f"All outputs in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()