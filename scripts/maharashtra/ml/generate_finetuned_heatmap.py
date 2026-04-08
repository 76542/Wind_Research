"""
scripts/maharashtra/ml/generate_finetuned_heatmap.py
=====================================================
Runs both models on ALL points and generates Gujarat-style
threshold heatmaps with proper ocean-only masking.

The land mask is built from the actual GEE LSIB coastline:
for each grid latitude, any point east of the coastline
is masked as land. This prevents interpolation from bleeding
onto land and ensures a tight fit to the coastline.

Usage:
  cd Wind_Research
  python -m scripts.maharashtra.ml.generate_finetuned_heatmap
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
    SEASON_MAP
)
from scripts.ml.train_mlp_v3 import WindSpeedMLPv3

# Paths
MH_COLLOCATED = os.path.join(PROJECT_ROOT, "data", "processed",
                              "maharashtra", "maharashtra_era5_collocated.csv")
GJ_SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "feature_scaler.pkl")
FT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mlp_v3_maharashtra_finetuned.pth")
ZS_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mlp_v3_wind_model.pth")
COAST_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "maharashtra",
                           "maharashtra_coastline_cache.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "maharashtra")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_coastline():
    """Load coastline and return segments + a lat-sorted array for masking."""
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
    Build a land mask: True where grid point is on land (east of coastline).

    For Maharashtra's Konkan coast, land is EAST of the coastline.
    For each grid latitude, we find the coastline longitude and mask
    everything with longitude > coastline_lon.

    To handle the jagged coastline, we bin coast points by latitude
    and take the EASTERNMOST longitude in each bin as the boundary.
    This ensures no data leaks onto land even at inlets/headlands.
    """
    # Sort coast points by latitude
    coast_pts_sorted = sorted(coast_pts, key=lambda p: p[1])
    coast_lats = np.array([p[1] for p in coast_pts_sorted])
    coast_lons = np.array([p[0] for p in coast_pts_sorted])

    # Bin coastline by latitude (0.05° bins) and take EASTERNMOST lon per bin
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

    # For each grid point, interpolate the coastline longitude at that latitude
    # and check if the grid point is east of it
    grid_coast_lon = np.interp(grid_lat2d.ravel(), bin_centers, bin_max_lons,
                                left=bin_max_lons[0], right=bin_max_lons[-1])
    grid_coast_lon = grid_coast_lon.reshape(grid_lon2d.shape)

    # Land = grid longitude > coastline longitude (east of coast)
    land_mask = grid_lon2d > (grid_coast_lon - 0.02)

    # Also mask points outside the latitude range of the coastline (with buffer)
    lat_buffer = 0.1
    lat_mask = ((grid_lat2d < (lat_min - lat_buffer)) |
                (grid_lat2d > (lat_max + lat_buffer)))

    return land_mask | lat_mask


def main():
    # ── Load and prepare ALL data ─────────────────────────────────────
    print("Loading all Maharashtra data...")
    df = pd.read_csv(MH_COLLOCATED)

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

    print(f"  {len(df):,} observations, {df.point_id.nunique()} points")

    with open(GJ_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    df_raw = df.copy()
    df[ALL_FEATURES] = scaler.transform(df[ALL_FEATURES])

    X = torch.FloatTensor(df[ALL_FEATURES].values)
    device = torch.device("cpu")

    # ── Run BOTH models ───────────────────────────────────────────────
    print("Running zero-shot model...")
    zs_ckpt = torch.load(ZS_MODEL_PATH, map_location=device, weights_only=False)
    zs_arch = zs_ckpt["architecture"]
    zs_model = WindSpeedMLPv3(zs_arch["input_dim"], zs_arch["hidden_layers"],
                               zs_arch["dropout"])
    zs_model.load_state_dict(zs_ckpt["model_state_dict"])
    zs_model.eval()
    with torch.no_grad():
        zs_pred = zs_model(X).numpy()

    print("Running fine-tuned model...")
    ft_ckpt = torch.load(FT_MODEL_PATH, map_location=device, weights_only=False)
    ft_arch = ft_ckpt["architecture"]
    ft_model = WindSpeedMLPv3(ft_arch["input_dim"], ft_arch["hidden_layers"],
                               ft_arch["dropout"])
    ft_model.load_state_dict(ft_ckpt["model_state_dict"])
    ft_model.eval()
    with torch.no_grad():
        ft_pred = ft_model(X).numpy()

    df_raw["zs_pred"] = zs_pred
    df_raw["ft_pred"] = ft_pred

    print(f"  Zero-shot range: [{zs_pred.min():.2f}, {zs_pred.max():.2f}] m/s")
    print(f"  Fine-tuned range: [{ft_pred.min():.2f}, {ft_pred.max():.2f}] m/s")

    # ── Per-point threshold stats ─────────────────────────────────────
    per_point = df_raw.groupby(['point_id', 'latitude', 'longitude']).agg(
        days_gt4_zs=('zs_pred', lambda x: (x > 4).sum()),
        days_gt6_zs=('zs_pred', lambda x: (x > 6).sum()),
        days_gt8_zs=('zs_pred', lambda x: (x > 8).sum()),
        days_gt4_ft=('ft_pred', lambda x: (x > 4).sum()),
        days_gt6_ft=('ft_pred', lambda x: (x > 6).sum()),
        days_gt8_ft=('ft_pred', lambda x: (x > 8).sum()),
        days_gt4_era5=('ERA5_WindSpeed_100m_ms', lambda x: (x > 4).sum()),
        days_gt6_era5=('ERA5_WindSpeed_100m_ms', lambda x: (x > 6).sum()),
        days_gt8_era5=('ERA5_WindSpeed_100m_ms', lambda x: (x > 8).sum()),
    ).reset_index()

    points = per_point[['longitude', 'latitude']].values

    # ── Interpolation grid ────────────────────────────────────────────
    grid_lon = np.linspace(72.2, 73.7, 400)
    grid_lat = np.linspace(15.4, 20.3, 400)
    grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)

    # Distance mask — allow interpolation up to 0.4° from nearest point
    # (larger than before so data extends closer to coast)
    tree = cKDTree(points)
    grid_pts = np.column_stack([grid_lon2d.ravel(), grid_lat2d.ravel()])
    dists, _ = tree.query(grid_pts)
    dist_mask = dists.reshape(grid_lon2d.shape) > 0.40

    # ── Load coastline and build land mask ────────────────────────────
    print("Loading coastline and building land mask...")
    coast_segments, coast_all_pts = load_coastline()
    land_mask = build_land_mask(coast_all_pts, grid_lon2d, grid_lat2d)
    print(f"  Land mask: {land_mask.sum():,} / {land_mask.size:,} grid points masked")

    # Combined mask: on land OR too far from any data point
    combined_mask = land_mask | dist_mask

    # ── Plot function ─────────────────────────────────────────────────
    def make_map(cols, title_prefix, filename, subtitle):
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))

        for ax, (col, label) in zip(axes, cols):
            values = per_point[col].values
            grid_z = griddata(points, values, (grid_lon2d, grid_lat2d),
                              method='cubic')
            grid_z[combined_mask] = np.nan
            grid_z = np.clip(grid_z, 0, None)

            im = ax.pcolormesh(grid_lon, grid_lat, grid_z, cmap='jet',
                                shading='auto')

            # Draw actual GEE LSIB coastline
            for seg in coast_segments:
                lons, lats = zip(*seg)
                ax.plot(lons, lats, color='gray', linewidth=0.8, alpha=0.7)

            ax.set_xlim(72.2, 73.7)
            ax.set_ylim(15.4, 20.2)
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
            f'Offshore Wind Resource Potential — Maharashtra Coast (2020-2024)\n'
            f'{subtitle}',
            fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    # ── Generate all three maps ───────────────────────────────────────
    print("\nGenerating heatmaps...")

    make_map(
        [('days_gt4_zs', '> 4 m/s'), ('days_gt6_zs', '> 6 m/s'),
         ('days_gt8_zs', '> 8 m/s')],
        'MLP v3 (SAR)', 'maharashtra_zeroshot_heatmap.png',
        'MLP v3 SAR-based Prediction (Zero-Shot Transfer)  |  '
        'Sentinel-1 100m Hub-Height'
    )

    make_map(
        [('days_gt4_ft', '> 4 m/s'), ('days_gt6_ft', '> 6 m/s'),
         ('days_gt8_ft', '> 8 m/s')],
        'MLP v3 Fine-Tuned', 'maharashtra_finetuned_heatmap.png',
        'MLP v3 SAR-based Prediction (Fine-Tuned)  |  '
        'Sentinel-1 100m Hub-Height'
    )

    make_map(
        [('days_gt4_era5', '> 4 m/s'), ('days_gt6_era5', '> 6 m/s'),
         ('days_gt8_era5', '> 8 m/s')],
        'ERA5 (Truth)', 'maharashtra_era5_heatmap.png',
        'ERA5 100m Hub-Height Wind Speed (Ground Truth Reference)'
    )

    print("\nAll three heatmaps generated!")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()