"""
scripts/ml/stitch_full_coastline_heatmap.py
=============================================
Stitches ALL 8 states across both coasts into one map.

West coast: Gujarat + Maharashtra + Goa + Karnataka + Kerala
East coast: Tamil Nadu + Andhra Pradesh + Odisha

701 total points, ~110K observations, 8°N to 24.5°N.

Usage: python -m scripts.ml.stitch_full_coastline_heatmap
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
    print("Cartopy not found")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.ml.config_ml import (
    ALL_FEATURES, TARGET, VV_MAX, VV_MIN, VH_VV_RATIO_MIN,
    ERA5_MIN, ERA5_MAX, SEASON_MAP
)
from scripts.ml.train_mlp_v3 import WindSpeedMLPv3

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

GJ_SCALER = os.path.join(PROJECT_ROOT, "models", "feature_scaler.pkl")

# ── All 8 states ─────────────────────────────────────────────────
STATES = {
    # West coast
    'Gujarat': {
        'coast': 'west',
        'data': os.path.join(PROJECT_ROOT, "data", "processed", "era5_collocated.csv"),
        'model': os.path.join(PROJECT_ROOT, "models", "mlp_v3_wind_model.pth"),
        'model_label': 'Home (v3)',
    },
    'Maharashtra': {
        'coast': 'west',
        'data': os.path.join(PROJECT_ROOT, "data", "processed", "maharashtra",
                             "maharashtra_era5_collocated.csv"),
        'model': os.path.join(PROJECT_ROOT, "models", "mlp_v3_maharashtra_finetuned.pth"),
        'model_label': 'FT from Gujarat',
    },
    'Goa': {
        'coast': 'west',
        'data': os.path.join(PROJECT_ROOT, "data", "processed", "goa",
                             "goa_era5_collocated.csv"),
        'model': os.path.join(PROJECT_ROOT, "models", "mlp_v3_maharashtra_finetuned.pth"),
        'model_label': 'MH FT transfer',
    },
    'Karnataka': {
        'coast': 'west',
        'data': os.path.join(PROJECT_ROOT, "data", "processed", "karnataka",
                             "karnataka_era5_collocated.csv"),
        'model': os.path.join(PROJECT_ROOT, "models", "mlp_v3_karnataka_finetuned.pth"),
        'model_label': 'FT from MH FT',
    },
    'Kerala': {
        'coast': 'west',
        'data': os.path.join(PROJECT_ROOT, "data", "processed", "kerala",
                             "kerala_era5_collocated.csv"),
        'model': os.path.join(PROJECT_ROOT, "models", "mlp_v3_kerala_finetuned.pth"),
        'model_label': 'FT from KA FT',
    },
    # East coast
    'Tamil Nadu': {
        'coast': 'east',
        'data': os.path.join(PROJECT_ROOT, "data", "processed", "tamilnadu",
                             "tamilnadu_era5_collocated.csv"),
        'model': os.path.join(PROJECT_ROOT, "models", "mlp_v3_tamilnadu_finetuned.pth"),
        'model_label': 'FT from Gujarat',
    },
    'Andhra Pradesh': {
        'coast': 'east',
        'data': os.path.join(PROJECT_ROOT, "data", "processed", "andhrapradesh",
                             "andhrapradesh_era5_collocated.csv"),
        'model': os.path.join(PROJECT_ROOT, "models", "mlp_v3_andhrapradesh_finetuned.pth"),
        'model_label': 'FT from TN FT',
    },
    'Odisha': {
        'coast': 'east',
        'data': os.path.join(PROJECT_ROOT, "data", "processed", "odisha",
                             "odisha_era5_collocated.csv"),
        'model': os.path.join(PROJECT_ROOT, "models", "mlp_v3_odisha_finetuned.pth"),
        'model_label': 'FT from AP FT',
    },
}


def load_and_prepare(data_path):
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
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    arch = ckpt["architecture"]
    model = WindSpeedMLPv3(arch["input_dim"], arch["hidden_layers"], arch["dropout"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        return model(X).numpy()


def main():
    print("\n" + "=" * 70)
    print("FULL INDIAN COASTLINE STITCHED HEATMAP")
    print("8 States | Both Coasts | 701 Points | ~110K Observations")
    print("=" * 70)

    device = torch.device("cpu")

    with open(GJ_SCALER, "rb") as f:
        scaler = pickle.load(f)

    # ── Process each state ────────────────────────────────────────
    all_per_point = []
    total_obs = 0

    for state_name, cfg in STATES.items():
        print(f"\n--- {state_name} ({cfg['coast']} coast) ---")

        df = load_and_prepare(cfg['data'])
        total_obs += len(df)
        print(f"  {len(df):,} observations, {df.point_id.nunique()} points")

        df_raw = df.copy()
        df[ALL_FEATURES] = scaler.transform(df[ALL_FEATURES])
        X = torch.FloatTensor(df[ALL_FEATURES].values)

        pred = run_model(cfg['model'], X, device)
        df_raw['best_pred'] = np.clip(pred, 0, None)

        pp = df_raw.groupby(['point_id', 'latitude', 'longitude']).agg(
            days_gt4=('best_pred', lambda x: (x > 4).sum()),
            days_gt6=('best_pred', lambda x: (x > 6).sum()),
            days_gt8=('best_pred', lambda x: (x > 8).sum()),
            days_gt4_era5=(TARGET, lambda x: (x > 4).sum()),
            days_gt6_era5=(TARGET, lambda x: (x > 6).sum()),
            days_gt8_era5=(TARGET, lambda x: (x > 8).sum()),
        ).reset_index()
        pp['state'] = state_name
        pp['coast'] = cfg['coast']

        print(f"  {len(pp)} points | Days>4: {pp.days_gt4.mean():.0f} avg | "
              f"Days>6: {pp.days_gt6.mean():.0f} avg | "
              f"Days>8: {pp.days_gt8.mean():.0f} avg")

        all_per_point.append(pp)

    # ── Combine ───────────────────────────────────────────────────
    combined = pd.concat(all_per_point, ignore_index=True)
    print(f"\n{'='*70}")
    print(f"COMBINED: {len(combined)} total points across 8 states")
    print(f"Total observations: {total_obs:,}")
    print(f"Latitude: {combined.latitude.min():.2f} - {combined.latitude.max():.2f}")
    print(f"Longitude: {combined.longitude.min():.2f} - {combined.longitude.max():.2f}")

    points = combined[['longitude', 'latitude']].values

    # ── Grid covering full India coastline ────────────────────────
    grid_lon = np.linspace(66.5, 89.0, 1000)
    grid_lat = np.linspace(7.5, 25.0, 1000)
    grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)

    # Distance mask
    tree = cKDTree(points)
    grid_pts = np.column_stack([grid_lon2d.ravel(), grid_lat2d.ravel()])
    dists, _ = tree.query(grid_pts)
    dist_mask = dists.reshape(grid_lon2d.shape) > 0.40

    # ── Plot ──────────────────────────────────────────────────────
    def make_map(cols, title_prefix, filename, subtitle):
        if HAS_CARTOPY:
            proj = ccrs.PlateCarree()
            fig, axes = plt.subplots(1, 3, figsize=(30, 14),
                                     subplot_kw={"projection": proj})
        else:
            fig, axes = plt.subplots(1, 3, figsize=(30, 14))

        fig.patch.set_facecolor("white")

        for ax, (col, label) in zip(axes, cols):
            values = combined[col].values

            grid_z = griddata(points, values, (grid_lon2d, grid_lat2d),
                              method='linear')
            grid_z[dist_mask] = np.nan
            grid_z = np.clip(grid_z, 0, None)

            if HAS_CARTOPY:
                im = ax.pcolormesh(grid_lon, grid_lat, grid_z, cmap='jet',
                                    shading='auto', transform=proj, zorder=1)
                ax.add_feature(cfeature.LAND.with_scale("10m"),
                               facecolor="#cccccc", edgecolor="black",
                               linewidth=0.5, zorder=2)
                ax.add_feature(cfeature.COASTLINE.with_scale("10m"),
                               edgecolor="black", linewidth=0.6, zorder=3)
                ax.add_feature(cfeature.BORDERS.with_scale("10m"),
                               edgecolor="#555555", linewidth=0.4,
                               linestyle="--", zorder=3)
                ax.set_extent([66.5, 89.0, 7.5, 25.0], crs=proj)
                gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                                  color="gray", alpha=0.5, linestyle="--")
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {"color": "black", "size": 8}
                gl.ylabel_style = {"color": "black", "size": 8}
            else:
                im = ax.pcolormesh(grid_lon, grid_lat, grid_z, cmap='jet',
                                    shading='auto')
                ax.set_xlim(66.5, 89.0)
                ax.set_ylim(7.5, 25.0)

            ax.set_title(f'{title_prefix}  |  Days {label}',
                         fontsize=13, fontweight='bold')
            ax.set_facecolor('white')

            ax.text(0.02, 0.97, f'{label}', transform=ax.transAxes,
                    fontsize=16, fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='black', alpha=0.8))

            # State labels
            state_positions = [
                ('Gujarat', 68.5, 22.5),
                ('Maharashtra', 72.0, 18.0),
                ('Goa', 72.8, 15.5),
                ('Karnataka', 73.8, 13.0),
                ('Kerala', 75.0, 10.0),
                ('Tamil Nadu', 79.5, 10.0),
                ('Andhra\nPradesh', 82.0, 16.0),
                ('Odisha', 85.5, 20.5),
            ]
            for sname, slon, slat in state_positions:
                ax.text(slon, slat, sname, fontsize=6, color='black',
                        alpha=0.5, fontstyle='italic',
                        ha='center', va='center')

            plt.colorbar(im, ax=ax, label='Number of Days', shrink=0.5,
                         pad=0.02)

        fig.suptitle(
            f'Offshore Wind Resource Potential — Indian Coastline (2020-2024)\n'
            f'{subtitle}',
            fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"\nSaved: {path}")

    print("\nGenerating full coastline heatmaps...")

    make_map(
        [('days_gt4', '> 4 m/s'), ('days_gt6', '> 6 m/s'),
         ('days_gt8', '> 8 m/s')],
        'MLP v3 (SAR)', 'full_coastline_model_heatmap.png',
        'MLP v3 SAR-based Prediction  |  Sentinel-1 100m Hub-Height\n'
        'West: Gujarat + Maharashtra + Goa + Karnataka + Kerala  |  '
        'East: Tamil Nadu + AP + Odisha')

    make_map(
        [('days_gt4_era5', '> 4 m/s'), ('days_gt6_era5', '> 6 m/s'),
         ('days_gt8_era5', '> 8 m/s')],
        'ERA5 (Truth)', 'full_coastline_era5_heatmap.png',
        'ERA5 100m Hub-Height Wind Speed (Ground Truth Reference)')

    # ── Summary ───────────────────────────────────────────────────
    summary_path = os.path.join(OUTPUT_DIR, "full_coastline_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        header = f"""
{'='*75}
INDIAN COASTLINE — FULL OFFSHORE WIND RESOURCE SUMMARY
{'='*75}
Total sampling points: {len(combined)}
Total observations: {total_obs:,}
States covered: 8 (5 west coast + 3 east coast)
Latitude coverage: {combined.latitude.min():.2f}°N to {combined.latitude.max():.2f}°N

Transfer learning chains:
  West: Gujarat (home) → Maharashtra FT → Goa (transfer) → Karnataka FT → Kerala FT
  East: Gujarat (home) → Tamil Nadu FT → Andhra Pradesh FT → Odisha FT

Per-state breakdown:
"""
        f.write(header)
        print(header)

        fmt = "{:<20} {:>5} {:>6} {:>8} {:>8} {:>8}  {:>16}"
        hdr = fmt.format("State", "Coast", "Pts", ">4 avg", ">6 avg", ">8 avg", "Model")
        f.write(hdr + "\n")
        print(hdr)
        f.write("-" * 80 + "\n")

        state_order = ['Gujarat', 'Maharashtra', 'Goa', 'Karnataka', 'Kerala',
                        'Tamil Nadu', 'Andhra Pradesh', 'Odisha']

        for state in state_order:
            s = combined[combined.state == state]
            coast = STATES[state]['coast'][0].upper()
            row = fmt.format(state, coast, len(s),
                f"{s.days_gt4.mean():.0f}", f"{s.days_gt6.mean():.0f}",
                f"{s.days_gt8.mean():.0f}", STATES[state]['model_label'])
            f.write(row + "\n")
            print(row)

        f.write(f"\n{'='*75}\n")
        f.write(f"West coast total: {len(combined[combined.coast=='west'])} points\n")
        f.write(f"East coast total: {len(combined[combined.coast=='east'])} points\n")
        f.write(f"Grand total: {len(combined)} points, {total_obs:,} observations\n")

    print(f"\nSummary: {summary_path}")
    print(f"\n{'='*70}")
    print("FULL COASTLINE STITCHING COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()