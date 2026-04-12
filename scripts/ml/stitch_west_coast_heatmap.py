"""
scripts/ml/stitch_west_coast_heatmap.py
========================================
Stitches Gujarat + Maharashtra + Goa + Karnataka + Kerala into
one continuous west coast offshore wind resource heatmap.

For each state, loads predictions CSV, picks the best model column,
computes per-point threshold days, combines all points, and generates
a single ~8°N to ~24.5°N heatmap.

Usage: python -m scripts.ml.stitch_west_coast_heatmap
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

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
    print("Cartopy found")
except ImportError:
    HAS_CARTOPY = False
    print("Cartopy not found — land may bleed onto Gujarat")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.ml.config_ml import (
    ALL_FEATURES, TARGET, VV_MAX, VV_MIN, VH_VV_RATIO_MIN,
    ERA5_MIN, ERA5_MAX, SEASON_MAP
)
from scripts.ml.train_mlp_v3 import WindSpeedMLPv3

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

GJ_SCALER = os.path.join(PROJECT_ROOT, "models", "feature_scaler.pkl")

# ── State configurations ──────────────────────────────────────────
STATES = {
    'Gujarat': {
        'data': os.path.join(PROJECT_ROOT, "data", "processed", "era5_collocated.csv"),
        'model': os.path.join(PROJECT_ROOT, "models", "mlp_v3_wind_model.pth"),
        'coast': os.path.join(PROJECT_ROOT, "data", "raw", "gujarat_coastline_cache.csv"),
        'pred_col': None,  # will compute from model
    },
    'Maharashtra': {
        'data': os.path.join(PROJECT_ROOT, "data", "processed", "maharashtra",
                             "maharashtra_era5_collocated.csv"),
        'model': os.path.join(PROJECT_ROOT, "models", "mlp_v3_maharashtra_finetuned.pth"),
        'coast': os.path.join(PROJECT_ROOT, "data", "raw", "maharashtra",
                              "maharashtra_coastline_cache.csv"),
        'pred_col': None,
    },
    'Goa': {
        'data': os.path.join(PROJECT_ROOT, "data", "processed", "goa",
                             "goa_era5_collocated.csv"),
        'model': os.path.join(PROJECT_ROOT, "models", "mlp_v3_maharashtra_finetuned.pth"),
        'coast': os.path.join(PROJECT_ROOT, "data", "raw", "goa",
                              "goa_coastline_cache.csv"),
        'pred_col': None,
    },
    'Karnataka': {
        'data': os.path.join(PROJECT_ROOT, "data", "processed", "karnataka",
                             "karnataka_era5_collocated.csv"),
        'model': os.path.join(PROJECT_ROOT, "models", "mlp_v3_karnataka_finetuned.pth"),
        'coast': os.path.join(PROJECT_ROOT, "data", "raw", "karnataka",
                              "karnataka_coastline_cache.csv"),
        'pred_col': None,
    },
    'Kerala': {
        'data': os.path.join(PROJECT_ROOT, "data", "processed", "kerala",
                             "kerala_era5_collocated.csv"),
        'model': os.path.join(PROJECT_ROOT, "models", "mlp_v3_kerala_finetuned.pth"),
        'coast': os.path.join(PROJECT_ROOT, "data", "raw", "kerala",
                              "kerala_coastline_cache.csv"),
        'pred_col': None,
    },
}


def load_and_prepare(data_path):
    """Load collocated CSV and add cyclical features."""
    df = pd.read_csv(data_path)
    df = df[~((df["VV"] > VV_MAX) | (df["VV"] < VV_MIN) |
              (df["VH_VV_ratio"] < VH_VV_RATIO_MIN) |
              (df[TARGET] < ERA5_MIN) | (df[TARGET] > ERA5_MAX))].copy()

    if 'month' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear

    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    return df


def run_model(model_path, X, device):
    """Run a model and return predictions."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    arch = ckpt["architecture"]
    model = WindSpeedMLPv3(arch["input_dim"], arch["hidden_layers"], arch["dropout"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        return model(X).numpy()


def load_coastline(coast_path):
    """Load coastline segments from CSV."""
    coast_df = pd.read_csv(coast_path)
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
    """Build land mask from coastline points."""
    coast_pts_sorted = sorted(coast_pts, key=lambda p: p[1])
    coast_lats = np.array([p[1] for p in coast_pts_sorted])
    coast_lons = np.array([p[0] for p in coast_pts_sorted])

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
    return land_mask


def main():
    print("\n" + "=" * 70)
    print("WEST COAST STITCHED HEATMAP")
    print("Gujarat + Maharashtra + Goa + Karnataka + Kerala")
    print("=" * 70)

    device = torch.device("cpu")

    with open(GJ_SCALER, "rb") as f:
        scaler = pickle.load(f)

    # ── Process each state ────────────────────────────────────────
    all_per_point = []

    for state_name, cfg in STATES.items():
        print(f"\n--- {state_name} ---")

        # Load and prepare data
        df = load_and_prepare(cfg['data'])
        print(f"  {len(df):,} observations, {df.point_id.nunique()} points")

        # Scale and predict
        df_raw = df.copy()
        df[ALL_FEATURES] = scaler.transform(df[ALL_FEATURES])
        X = torch.FloatTensor(df[ALL_FEATURES].values)

        pred = run_model(cfg['model'], X, device)
        df_raw['best_pred'] = np.clip(pred, 0, None)

        # Per-point threshold counts
        pp = df_raw.groupby(['point_id', 'latitude', 'longitude']).agg(
            days_gt4=('best_pred', lambda x: (x > 4).sum()),
            days_gt6=('best_pred', lambda x: (x > 6).sum()),
            days_gt8=('best_pred', lambda x: (x > 8).sum()),
            days_gt4_era5=(TARGET, lambda x: (x > 4).sum()),
            days_gt6_era5=(TARGET, lambda x: (x > 6).sum()),
            days_gt8_era5=(TARGET, lambda x: (x > 8).sum()),
        ).reset_index()
        pp['state'] = state_name

        print(f"  {len(pp)} points | Days>4: {pp.days_gt4.mean():.0f} avg | "
              f"Days>6: {pp.days_gt6.mean():.0f} avg | "
              f"Days>8: {pp.days_gt8.mean():.0f} avg")

        all_per_point.append(pp)

    # ── Combine all states ────────────────────────────────────────
    combined = pd.concat(all_per_point, ignore_index=True)
    print(f"\n{'='*70}")
    print(f"COMBINED: {len(combined)} total points across 5 states")
    print(f"Latitude range: {combined.latitude.min():.2f} - {combined.latitude.max():.2f}")
    print(f"Longitude range: {combined.longitude.min():.2f} - {combined.longitude.max():.2f}")

    points = combined[['longitude', 'latitude']].values

    # ── Interpolation grid (full west coast) ──────────────────────
    grid_lon = np.linspace(67.0, 78.0, 800)
    grid_lat = np.linspace(7.8, 24.8, 800)
    grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)

    # Distance mask
    tree = cKDTree(points)
    grid_pts = np.column_stack([grid_lon2d.ravel(), grid_lat2d.ravel()])
    dists, _ = tree.query(grid_pts)
    dist_mask = dists.reshape(grid_lon2d.shape) > 0.40

    # ── Load ALL coastlines and build combined land mask ──────────
    print("\nLoading all coastlines...")
    all_coast_segments = []
    all_coast_pts = []

    for state_name, cfg in STATES.items():
        if os.path.exists(cfg['coast']):
            segments = load_coastline(cfg['coast'])
            all_coast_segments.extend(segments)
            for seg in segments:
                all_coast_pts.extend(seg)
            print(f"  {state_name}: {len(segments)} segments")

    land_mask = build_land_mask(all_coast_pts, grid_lon2d, grid_lat2d)
    combined_mask = land_mask | dist_mask

    print(f"  Total coastline points: {len(all_coast_pts):,}")

    # ── Plot function ─────────────────────────────────────────────
    def make_map(cols, title_prefix, filename, subtitle):
        if HAS_CARTOPY:
            proj = ccrs.PlateCarree()
            fig, axes = plt.subplots(1, 3, figsize=(24, 14),
                                     subplot_kw={"projection": proj})
        else:
            fig, axes = plt.subplots(1, 3, figsize=(24, 14))

        for ax, (col, label) in zip(axes, cols):
            values = combined[col].values

            grid_z = griddata(points, values, (grid_lon2d, grid_lat2d),
                              method='linear')
            grid_z[combined_mask] = np.nan
            grid_z = np.clip(grid_z, 0, None)

            if HAS_CARTOPY:
                im = ax.pcolormesh(grid_lon, grid_lat, grid_z, cmap='jet',
                                    shading='auto', transform=proj, zorder=1)
                ax.add_feature(cfeature.LAND.with_scale("10m"),
                               facecolor="#cccccc", edgecolor="black",
                               linewidth=0.7, zorder=2)
                ax.add_feature(cfeature.COASTLINE.with_scale("10m"),
                               edgecolor="black", linewidth=0.8, zorder=3)
                ax.add_feature(cfeature.BORDERS.with_scale("10m"),
                               edgecolor="#555555", linewidth=0.5,
                               linestyle="--", zorder=3)
                ax.set_extent([66.5, 78.0, 7.8, 24.8], crs=proj)
                gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                                  color="gray", alpha=0.5, linestyle="--")
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {"color": "black", "size": 8}
                gl.ylabel_style = {"color": "black", "size": 8}
            else:
                im = ax.pcolormesh(grid_lon, grid_lat, grid_z, cmap='jet',
                                    shading='auto')
                for seg in all_coast_segments:
                    lons, lats = zip(*seg)
                    ax.plot(lons, lats, color='gray', linewidth=0.5, alpha=0.6)
                ax.set_xlim(66.5, 78.0)
                ax.set_ylim(7.8, 24.8)

            ax.set_xlabel('Longitude (°E)', fontsize=11)
            ax.set_ylabel('Latitude (°N)', fontsize=11)
            ax.set_title(f'{title_prefix}  |  Days {label}',
                         fontsize=13, fontweight='bold')
            ax.set_facecolor('white')

            ax.text(0.03, 0.97, f'{label}', transform=ax.transAxes,
                    fontsize=16, fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='black', alpha=0.8))

            # State labels
            state_positions = [
                ('Gujarat', 69.0, 22.5),
                ('Maharashtra', 72.0, 18.0),
                ('Goa', 72.5, 15.3),
                ('Karnataka', 73.5, 13.0),
                ('Kerala', 74.5, 10.0),
            ]
            for sname, slon, slat in state_positions:
                ax.text(slon, slat, sname, fontsize=7, color='black',
                        alpha=0.5, fontstyle='italic')

            plt.colorbar(im, ax=ax, label='Number of Days', shrink=0.6,
                         pad=0.02)

        fig.suptitle(
            f'Offshore Wind Resource Potential — Indian West Coast (2020-2024)\n'
            f'{subtitle}',
            fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(path, dpi=180, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {path}")

    # ── Generate maps ─────────────────────────────────────────────
    print("\nGenerating stitched heatmaps...")

    make_map(
        [('days_gt4', '> 4 m/s'), ('days_gt6', '> 6 m/s'),
         ('days_gt8', '> 8 m/s')],
        'MLP v3 (SAR)', 'west_coast_model_heatmap.png',
        'MLP v3 SAR-based Prediction  |  Sentinel-1 100m Hub-Height\n'
        'Gujarat (Home) + Maharashtra (FT) + Goa (MH Transfer) + '
        'Karnataka (FT) + Kerala (FT)')

    make_map(
        [('days_gt4_era5', '> 4 m/s'), ('days_gt6_era5', '> 6 m/s'),
         ('days_gt8_era5', '> 8 m/s')],
        'ERA5 (Truth)', 'west_coast_era5_heatmap.png',
        'ERA5 100m Hub-Height Wind Speed (Ground Truth Reference)')

    # ── Summary table ─────────────────────────────────────────────
    summary_path = os.path.join(OUTPUT_DIR, "west_coast_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        header = f"""
{'='*70}
INDIAN WEST COAST — OFFSHORE WIND RESOURCE SUMMARY
{'='*70}
Total sampling points: {len(combined)}
Latitude coverage: {combined.latitude.min():.2f}°N to {combined.latitude.max():.2f}°N
Coastline length: ~1,800 km

Per-state breakdown:
"""
        f.write(header)
        print(header)

        fmt = "{:<15} {:>6} {:>8} {:>8} {:>8}  {:>10}"
        hdr = fmt.format("State", "Points", ">4 avg", ">6 avg", ">8 avg", "Model")
        f.write(hdr + "\n")
        print(hdr)
        f.write("-" * 65 + "\n")

        model_names = {
            'Gujarat': 'Home (v3)',
            'Maharashtra': 'Fine-tuned',
            'Goa': 'MH FT xfer',
            'Karnataka': 'Fine-tuned',
            'Kerala': 'Fine-tuned',
        }

        for state in ['Gujarat', 'Maharashtra', 'Goa', 'Karnataka', 'Kerala']:
            s = combined[combined.state == state]
            row = fmt.format(state, len(s),
                f"{s.days_gt4.mean():.0f}", f"{s.days_gt6.mean():.0f}",
                f"{s.days_gt8.mean():.0f}", model_names[state])
            f.write(row + "\n")
            print(row)

    print(f"\nSummary: {summary_path}")
    print(f"\n{'='*70}")
    print("WEST COAST STITCHING COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()