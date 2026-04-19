"""
scripts/tamilnadu/ml/generate_finetuned_heatmap.py
====================================================
Tamil Nadu east coast heatmap.

Key difference from west coast: land mask is REVERSED.
West coast: land = east of coastline (lon > coast_lon)
East coast: land = west of coastline (lon < coast_lon)

Usage: python -m scripts.tamilnadu.ml.generate_finetuned_heatmap
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
TN_COLLOCATED = os.path.join(PROJECT_ROOT, "data", "processed",
                              "tamilnadu", "tamilnadu_era5_collocated.csv")
GJ_SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "feature_scaler.pkl")
FT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mlp_v3_tamilnadu_finetuned.pth")
GJ_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mlp_v3_wind_model.pth")
COAST_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "tamilnadu",
                           "tamilnadu_coastline_cache.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "tamilnadu")
PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "tamilnadu",
                                 "tamilnadu_predictions.csv")
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
    all_pts = []

    for _, row in coast_df.iterrows():
        if pd.isna(row['latitude']):
            if len(current_seg) > 1:
                segments.append(current_seg)
            current_seg = []
        else:
            current_seg.append((row['longitude'], row['latitude']))
            all_pts.append((row['longitude'], row['latitude']))
    if len(current_seg) > 1:
        segments.append(current_seg)

    print(f"  Loaded coastline: {len(segments)} segments, "
          f"{len(all_pts)} total points")
    return segments, all_pts


def build_land_mask(coast_pts, grid_lon2d, grid_lat2d):
    """
    EAST COAST land mask: land is WEST of coastline.
    For each grid latitude, find the WESTERNMOST coastline longitude
    (minimum lon), then mask everything with lon < coast_lon.
    """
    coast_pts_sorted = sorted(coast_pts, key=lambda p: p[1])
    coast_lats = np.array([p[1] for p in coast_pts_sorted])
    coast_lons = np.array([p[0] for p in coast_pts_sorted])

    lat_min, lat_max = coast_lats.min(), coast_lats.max()
    bin_size = 0.01
    bin_edges = np.arange(lat_min - bin_size, lat_max + 2 * bin_size, bin_size)
    bin_centers = []
    bin_min_lons = []

    for i in range(len(bin_edges) - 1):
        mask = (coast_lats >= bin_edges[i]) & (coast_lats < bin_edges[i + 1])
        if mask.any():
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            # EAST COAST: take MINIMUM lon (westernmost = land boundary)
            bin_min_lons.append(coast_lons[mask].min())

    bin_centers = np.array(bin_centers)
    bin_min_lons = np.array(bin_min_lons)

    grid_coast_lon = np.interp(grid_lat2d.ravel(), bin_centers, bin_min_lons,
                                left=bin_min_lons[0], right=bin_min_lons[-1])
    grid_coast_lon = grid_coast_lon.reshape(grid_lon2d.shape)

    # EAST COAST: land = west of coastline (lon < coast_lon + buffer)
    land_mask = grid_lon2d < (grid_coast_lon + 0.02)

    lat_buffer = 0.1
    lat_mask = ((grid_lat2d < (lat_min - lat_buffer)) |
                (grid_lat2d > (lat_max + lat_buffer)))

    return land_mask | lat_mask


def main():
    print("Loading all Tamil Nadu data...")
    df = pd.read_csv(TN_COLLOCATED)

    remove_mask = (
        (df["VV"] > VV_MAX) | (df["VV"] < VV_MIN) |
        (df["VH_VV_ratio"] < VH_VV_RATIO_MIN) |
        (df[TARGET] < ERA5_MIN) | (df[TARGET] > ERA5_MAX)
    )
    df = df[~remove_mask].copy()

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

    with open(GJ_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    df_raw = df.copy()
    df[ALL_FEATURES] = scaler.transform(df[ALL_FEATURES])

    X = torch.FloatTensor(df[ALL_FEATURES].values)
    device = torch.device("cpu")

    # ── Run both models ───────────────────────────────────────────
    print("Running Gujarat (zero-shot) model...")
    gj_ckpt = torch.load(GJ_MODEL_PATH, map_location=device, weights_only=False)
    gj_arch = gj_ckpt["architecture"]
    gj_model = WindSpeedMLPv3(gj_arch["input_dim"], gj_arch["hidden_layers"],
                               gj_arch["dropout"])
    gj_model.load_state_dict(gj_ckpt["model_state_dict"])
    gj_model.eval()
    with torch.no_grad():
        gj_pred = gj_model(X).numpy()

    print("Running Tamil Nadu FT model...")
    ft_ckpt = torch.load(FT_MODEL_PATH, map_location=device, weights_only=False)
    ft_arch = ft_ckpt["architecture"]
    ft_model = WindSpeedMLPv3(ft_arch["input_dim"], ft_arch["hidden_layers"],
                               ft_arch["dropout"])
    ft_model.load_state_dict(ft_ckpt["model_state_dict"])
    ft_model.eval()
    with torch.no_grad():
        ft_pred = ft_model(X).numpy()

    df_raw["gj_pred"] = gj_pred
    df_raw["ft_pred"] = ft_pred

    # ── Evaluate ──────────────────────────────────────────────────
    y_true = df_raw[TARGET].values
    gj_m = compute_metrics(y_true, gj_pred)
    tn_m = compute_metrics(y_true, ft_pred)

    results_path = os.path.join(OUTPUT_DIR, "tamilnadu_finetuned_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        report = f"""
{'='*70}
TAMIL NADU RESULTS — CROSS-OCEAN TRANSFER
{'='*70}
Observations: {len(y_true):,} ({df_raw.point_id.nunique()} points)
ERA5 mean wind: {y_true.mean():.2f} m/s

                    Gujarat (zero-shot)    Tamil Nadu FT
{'='*70}
  RMSE (m/s):       {gj_m['RMSE']:>8.3f}              {tn_m['RMSE']:>8.3f}
  MAE (m/s):        {gj_m['MAE']:>8.3f}              {tn_m['MAE']:>8.3f}
  R2:               {gj_m['R2']:>8.4f}              {tn_m['R2']:>8.4f}
  Bias (m/s):       {gj_m['Bias']:>+8.3f}              {tn_m['Bias']:>+8.3f}
{'='*70}

Cross-ocean fine-tuning {'SUCCEEDED' if tn_m['R2'] > 0.5 else 'NEEDS INVESTIGATION'}
(Gujarat Arabian Sea -> Tamil Nadu Bay of Bengal)
"""
        print(report); f.write(report)

        bi = np.clip(np.digitize(y_true, WIND_SPEED_BINS) - 1,
                     0, len(WIND_SPEED_LABELS) - 1)
        fmt = "{:<8} {:>5}  {:>10} {:>10} {:>10} {:>10}"
        hdr = fmt.format("Bin", "N", "GJ RMSE", "TN RMSE", "GJ Bias", "TN Bias")
        print(hdr); f.write(hdr + "\n")
        for i, label in enumerate(WIND_SPEED_LABELS):
            mask = bi == i
            n = mask.sum()
            if n == 0: continue
            gm = compute_metrics(y_true[mask], gj_pred[mask])
            tm = compute_metrics(y_true[mask], ft_pred[mask])
            row = fmt.format(label, n,
                f"{gm['RMSE']:.3f}", f"{tm['RMSE']:.3f}",
                f"{gm['Bias']:+.3f}", f"{tm['Bias']:+.3f}")
            print(row); f.write(row + "\n")

    df_raw.to_csv(PREDICTIONS_PATH, index=False)

    # ── Heatmap ───────────────────────────────────────────────────
    print("\nGenerating heatmaps...")

    per_point = df_raw.groupby(['point_id', 'latitude', 'longitude']).agg(
        days_gt4_ft=('ft_pred', lambda x: (x > 4).sum()),
        days_gt6_ft=('ft_pred', lambda x: (x > 6).sum()),
        days_gt8_ft=('ft_pred', lambda x: (x > 8).sum()),
        days_gt4_era5=('ERA5_WindSpeed_100m_ms', lambda x: (x > 4).sum()),
        days_gt6_era5=('ERA5_WindSpeed_100m_ms', lambda x: (x > 6).sum()),
        days_gt8_era5=('ERA5_WindSpeed_100m_ms', lambda x: (x > 8).sum()),
    ).reset_index()

    points = per_point[['longitude', 'latitude']].values

    grid_lon = np.linspace(77.0, 82.0, 400)
    grid_lat = np.linspace(7.9, 13.7, 400)
    grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)

    tree = cKDTree(points)
    grid_pts = np.column_stack([grid_lon2d.ravel(), grid_lat2d.ravel()])
    dists, _ = tree.query(grid_pts)
    dist_mask = dists.reshape(grid_lon2d.shape) > 0.40

    print("Loading coastline and building land mask...")
    coast_segments, coast_all_pts = load_coastline()
    land_mask = build_land_mask(coast_all_pts, grid_lon2d, grid_lat2d)
    combined_mask = land_mask | dist_mask

    # Coastal anchor points
    coast_lons = np.array([p[0] for p in coast_all_pts])
    coast_lats = np.array([p[1] for p in coast_all_pts])
    coast_sub = np.column_stack([coast_lons[::20], coast_lats[::20]])
    _, coast_nn_idx = tree.query(coast_sub)

    def make_map(cols, title_prefix, filename, subtitle):
        fig, axes = plt.subplots(1, 3, figsize=(20, 10))

        for ax, (col, label) in zip(axes, cols):
            values = per_point[col].values
            coast_vals = values[coast_nn_idx]
            aug_pts = np.vstack([points, coast_sub])
            aug_vals = np.concatenate([values, coast_vals])
            grid_z = griddata(aug_pts, aug_vals, (grid_lon2d, grid_lat2d),
                              method='cubic')
            grid_z[combined_mask] = np.nan
            grid_z = np.clip(grid_z, 0, None)

            im = ax.pcolormesh(grid_lon, grid_lat, grid_z, cmap='jet',
                                shading='auto')

            for seg in coast_segments:
                lons, lats = zip(*seg)
                ax.plot(lons, lats, color='gray', linewidth=0.8, alpha=0.7)

            ax.set_xlim(77.0, 82.0)
            ax.set_ylim(7.9, 13.7)
            ax.set_xlabel('Longitude (°E)', fontsize=11)
            ax.set_ylabel('Latitude (°N)', fontsize=11)
            ax.set_title(f'{title_prefix}  |  Days {label}',
                         fontsize=12, fontweight='bold')
            ax.set_aspect('equal')
            ax.set_facecolor('lightgray')
            ax.grid(True, alpha=0.2, linewidth=0.3)

            ax.text(0.03, 0.97, f'{label}', transform=ax.transAxes,
                    fontsize=14, fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='black', alpha=0.8))
            ax.text(0.97, 0.02, f'{title_prefix}', transform=ax.transAxes,
                    fontsize=8, ha='right', va='bottom', alpha=0.4,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            plt.colorbar(im, ax=ax, label='Number of Days', shrink=0.75,
                         pad=0.02)

        fig.suptitle(
            f'Offshore Wind Resource Potential — Tamil Nadu East Coast (2020-2024)\n'
            f'{subtitle}',
            fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    make_map(
        [('days_gt4_ft', '> 4 m/s'), ('days_gt6_ft', '> 6 m/s'),
         ('days_gt8_ft', '> 8 m/s')],
        'MLP v3 (Tamil Nadu FT)', 'tamilnadu_finetuned_heatmap.png',
        'MLP v3 SAR-based Prediction (Fine-Tuned from Gujarat)  |  '
        'Sentinel-1 100m Hub-Height')

    make_map(
        [('days_gt4_era5', '> 4 m/s'), ('days_gt6_era5', '> 6 m/s'),
         ('days_gt8_era5', '> 8 m/s')],
        'ERA5 (Truth)', 'tamilnadu_era5_heatmap.png',
        'ERA5 100m Hub-Height Wind Speed (Ground Truth Reference)')

    print("\nAll heatmaps generated!")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()