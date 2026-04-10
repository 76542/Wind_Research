"""
scripts/karnataka/ml/inference_and_heatmap.py
==============================================
Runs Gujarat + Maharashtra FT models on Karnataka, picks best, generates heatmap.
Usage: python -m scripts.karnataka.ml.inference_and_heatmap
"""
import os, sys, pickle, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.ml.config_ml import (
    ALL_FEATURES, TARGET, VV_MAX, VV_MIN, VH_VV_RATIO_MIN,
    ERA5_MIN, ERA5_MAX, SEASON_MAP, WIND_SPEED_BINS, WIND_SPEED_LABELS)
from scripts.ml.train_mlp_v3 import WindSpeedMLPv3

KA_COLLOCATED = os.path.join(PROJECT_ROOT, "data", "processed", "karnataka", "karnataka_era5_collocated.csv")
GJ_SCALER = os.path.join(PROJECT_ROOT, "models", "feature_scaler.pkl")
MH_MODEL = os.path.join(PROJECT_ROOT, "models", "mlp_v3_maharashtra_finetuned.pth")
GJ_MODEL = os.path.join(PROJECT_ROOT, "models", "mlp_v3_wind_model.pth")
COAST_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "karnataka", "karnataka_coastline_cache.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "karnataka")
PREDICTIONS = os.path.join(PROJECT_ROOT, "data", "processed", "karnataka", "karnataka_predictions.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred):
    r = y_pred - y_true
    rmse = np.sqrt(np.mean(r**2))
    mae = np.mean(np.abs(r))
    ss_res, ss_tot = np.sum(r**2), np.sum((y_true - y_true.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Bias": np.mean(r)}


def load_coastline():
    coast_df = pd.read_csv(COAST_PATH)
    segments, seg = [], []
    for _, row in coast_df.iterrows():
        if pd.isna(row['latitude']):
            if len(seg) > 1: segments.append(seg)
            seg = []
        else: seg.append((row['longitude'], row['latitude']))
    if len(seg) > 1: segments.append(seg)
    return segments


def build_land_mask(coast_pts, grid_lon2d, grid_lat2d):
    clats = np.array([p[1] for p in coast_pts])
    clons = np.array([p[0] for p in coast_pts])
    lat_min, lat_max = clats.min(), clats.max()
    bin_size = 0.01
    bins = np.arange(lat_min - bin_size, lat_max + 2*bin_size, bin_size)
    centers, max_lons = [], []
    for i in range(len(bins)-1):
        mask = (clats >= bins[i]) & (clats < bins[i+1])
        if mask.any():
            centers.append((bins[i]+bins[i+1])/2)
            max_lons.append(clons[mask].max())
    centers, max_lons = np.array(centers), np.array(max_lons)
    gcl = np.interp(grid_lat2d.ravel(), centers, max_lons,
                     left=max_lons[0], right=max_lons[-1]).reshape(grid_lon2d.shape)
    land = grid_lon2d > (gcl - 0.02)
    lat_out = (grid_lat2d < lat_min - 0.1) | (grid_lat2d > lat_max + 0.1)
    return land | lat_out


def main():
    print("\n" + "="*60 + "\nKARNATAKA: Model Transfer + Heatmap\n" + "="*60)

    # Load & prepare
    df = pd.read_csv(KA_COLLOCATED)
    n0 = len(df)
    df = df[~((df["VV"]>VV_MAX)|(df["VV"]<VV_MIN)|(df["VH_VV_ratio"]<VH_VV_RATIO_MIN)|
              (df[TARGET]<ERA5_MIN)|(df[TARGET]>ERA5_MAX))].copy()
    print(f"  {n0} -> {len(df)} after cleaning")

    if 'month' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
    df["sin_month"] = np.sin(2*np.pi*df["month"]/12)
    df["cos_month"] = np.cos(2*np.pi*df["month"]/12)
    df["sin_doy"] = np.sin(2*np.pi*df["day_of_year"]/365)
    df["cos_doy"] = np.cos(2*np.pi*df["day_of_year"]/365)
    df["season"] = df["month"].map(SEASON_MAP)
    print(f"  {len(df):,} obs, {df.point_id.nunique()} points")

    with open(GJ_SCALER, "rb") as f: scaler = pickle.load(f)
    df_raw = df.copy()
    df[ALL_FEATURES] = scaler.transform(df[ALL_FEATURES])
    X = torch.FloatTensor(df[ALL_FEATURES].values)
    device = torch.device("cpu")

    # Run both models
    def run_model(path, name):
        print(f"Running {name}...")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        a = ckpt["architecture"]
        m = WindSpeedMLPv3(a["input_dim"], a["hidden_layers"], a["dropout"])
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        with torch.no_grad(): return m(X).numpy()

    gj_pred = run_model(GJ_MODEL, "Gujarat model")
    mh_pred = run_model(MH_MODEL, "Maharashtra FT model")
    df_raw["gj_pred"], df_raw["mh_pred"] = gj_pred, mh_pred

    # Evaluate
    y = df_raw[TARGET].values
    gj_m, mh_m = compute_metrics(y, gj_pred), compute_metrics(y, mh_pred)

    results_path = os.path.join(OUTPUT_DIR, "karnataka_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        report = f"""
{'='*60}
KARNATAKA TRANSFER RESULTS
{'='*60}
Observations: {len(y):,} ({df_raw.point_id.nunique()} points)
ERA5 mean wind: {y.mean():.2f} m/s

                    Gujarat Model    Maharashtra FT
{'='*60}
  RMSE (m/s):       {gj_m['RMSE']:>8.3f}          {mh_m['RMSE']:>8.3f}
  MAE (m/s):        {gj_m['MAE']:>8.3f}          {mh_m['MAE']:>8.3f}
  R2:               {gj_m['R2']:>8.4f}          {mh_m['R2']:>8.4f}
  Bias (m/s):       {gj_m['Bias']:>+8.3f}          {mh_m['Bias']:>+8.3f}
{'='*60}
Best: {'Maharashtra FT' if mh_m['RMSE'] < gj_m['RMSE'] else 'Gujarat'}
"""
        print(report); f.write(report)

        # Stratified
        bi = np.clip(np.digitize(y, WIND_SPEED_BINS)-1, 0, len(WIND_SPEED_LABELS)-1)
        fmt = "{:<8} {:>5}  {:>10} {:>10} {:>10} {:>10}"
        hdr = fmt.format("Bin","N","GJ RMSE","MH RMSE","GJ Bias","MH Bias")
        print(hdr); f.write("\n" + hdr + "\n")
        for i, label in enumerate(WIND_SPEED_LABELS):
            mask = bi == i; n = mask.sum()
            if n == 0: continue
            gm, mm = compute_metrics(y[mask], gj_pred[mask]), compute_metrics(y[mask], mh_pred[mask])
            row = fmt.format(label, n, f"{gm['RMSE']:.3f}", f"{mm['RMSE']:.3f}",
                           f"{gm['Bias']:+.3f}", f"{mm['Bias']:+.3f}")
            print(row); f.write(row + "\n")

    print(f"\nResults: {results_path}")
    df_raw.to_csv(PREDICTIONS, index=False)
    print(f"Predictions: {PREDICTIONS}")

    # Heatmap
    best = "mh" if mh_m['RMSE'] < gj_m['RMSE'] else "gj"
    best_col = f"{best}_pred"
    best_name = "Maharashtra FT" if best == "mh" else "Gujarat"
    print(f"\nUsing {best_name} for heatmap")

    pp = df_raw.groupby(['point_id','latitude','longitude']).agg(
        days_gt4=(best_col, lambda x: (x>4).sum()),
        days_gt6=(best_col, lambda x: (x>6).sum()),
        days_gt8=(best_col, lambda x: (x>8).sum()),
        days_gt4_era5=('ERA5_WindSpeed_100m_ms', lambda x: (x>4).sum()),
        days_gt6_era5=('ERA5_WindSpeed_100m_ms', lambda x: (x>6).sum()),
        days_gt8_era5=('ERA5_WindSpeed_100m_ms', lambda x: (x>8).sum()),
    ).reset_index()

    points = pp[['longitude','latitude']].values
    glon = np.linspace(73.0, 75.5, 400)
    glat = np.linspace(11.7, 15.0, 400)
    glon2d, glat2d = np.meshgrid(glon, glat)

    tree = cKDTree(points)
    dists, _ = tree.query(np.column_stack([glon2d.ravel(), glat2d.ravel()]))
    dist_mask = dists.reshape(glon2d.shape) > 0.40

    coast_segs = load_coastline()
    coast_all = [p for s in coast_segs for p in s]
    land_mask = build_land_mask(coast_all, glon2d, glat2d)
    combined = land_mask | dist_mask

    def make_map(cols, prefix, fname, subtitle):
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        for ax, (col, label) in zip(axes, cols):
            gz = griddata(points, pp[col].values, (glon2d, glat2d), method='cubic')
            gz[combined] = np.nan
            gz = np.clip(gz, 0, None)
            im = ax.pcolormesh(glon, glat, gz, cmap='jet', shading='auto')
            for seg in coast_segs:
                lons, lats = zip(*seg)
                ax.plot(lons, lats, color='gray', linewidth=0.8, alpha=0.7)
            ax.set_xlim(73.0, 75.5); ax.set_ylim(11.7, 15.0)
            ax.set_xlabel('Longitude (E)'); ax.set_ylabel('Latitude (N)')
            ax.set_title(f'{prefix}  |  Days {label}', fontsize=12, fontweight='bold')
            ax.set_aspect('equal'); ax.set_facecolor('lightgray')
            ax.grid(True, alpha=0.2)
            ax.text(0.03, 0.97, f'{label}', transform=ax.transAxes, fontsize=14,
                    fontweight='bold', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.8))
            plt.colorbar(im, ax=ax, label='Number of Days', shrink=0.75)
        fig.suptitle(f'Offshore Wind Resource Potential - Karnataka Coast (2020-2024)\n{subtitle}',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, fname)
        plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
        print(f"Saved: {path}")

    make_map([('days_gt4','> 4 m/s'),('days_gt6','> 6 m/s'),('days_gt8','> 8 m/s')],
             f'MLP v3 ({best_name})', 'karnataka_model_heatmap.png',
             f'MLP v3 SAR-based Prediction ({best_name} Transfer)  |  Sentinel-1 100m Hub-Height')
    make_map([('days_gt4_era5','> 4 m/s'),('days_gt6_era5','> 6 m/s'),('days_gt8_era5','> 8 m/s')],
             'ERA5 (Truth)', 'karnataka_era5_heatmap.png',
             'ERA5 100m Hub-Height Wind Speed (Ground Truth Reference)')

    print(f"\n{'='*60}\nKARNATAKA COMPLETE\n{'='*60}")

if __name__ == "__main__":
    main()