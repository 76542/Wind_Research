"""
scripts/odisha/collocate_era5_sar.py
Usage: python -m scripts.odisha.collocate_era5_sar
"""
import os, xarray as xr, pandas as pd, numpy as np
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAR_FILE = os.path.join(BASE_DIR, "data", "processed", "odisha", "odisha_sar_timeseries.csv")
GRID_FILE = os.path.join(BASE_DIR, "data", "raw", "odisha", "odisha_sampling_grid.csv")
ERA5_DIR = Path(os.path.join(BASE_DIR, "data", "raw", "odisha", "era5_downloads"))
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "odisha", "odisha_era5_collocated.csv")

print("Loading SAR timeseries...")
sar = pd.read_csv(SAR_FILE, parse_dates=["timestamp"])
print("Loading sampling grid...")
grid = pd.read_csv(GRID_FILE)[["point_id", "latitude", "longitude"]]
sar = sar.merge(grid, on="point_id", how="left")
print(f"SAR data: {len(sar):,} rows, {sar['point_id'].nunique()} points")

era5_files = sorted(ERA5_DIR.glob("era5_odisha_*.nc"))
if not era5_files: era5_files = sorted(ERA5_DIR.glob("era5_*.nc"))
print(f"\nFound {len(era5_files)} ERA5 files")

ds = xr.open_mfdataset(era5_files, combine="by_coords")
u_var = "u100" if "u100" in ds else "u100m"
v_var = "v100" if "v100" in ds else "v100m"
ds["wind_speed_100m"] = np.sqrt(ds[u_var]**2 + ds[v_var]**2)
era5_lats, era5_lons = ds["latitude"].values, ds["longitude"].values
time_dim = "valid_time" if "valid_time" in ds.dims else "time"
era5_times = pd.to_datetime(ds[time_dim].values)
print(f"  ERA5: {len(era5_lats)} lats x {len(era5_lons)} lons, {len(era5_times)} timesteps")

print("\nCo-locating...")
era5_dates = {t.date(): i for i, t in enumerate(era5_times)}
def nearest_idx(arr, val): return int(np.argmin(np.abs(arr - val)))

era5_ws, matched, unmatched = [], 0, 0
for _, row in sar.iterrows():
    obs_date = row["timestamp"].date()
    if obs_date not in era5_dates:
        era5_ws.append(np.nan); unmatched += 1; continue
    t_idx = era5_dates[obs_date]
    lat_idx = nearest_idx(era5_lats, row["latitude"])
    lon_idx = nearest_idx(era5_lons, row["longitude"])
    ws = float(ds["wind_speed_100m"].isel(**{time_dim: t_idx, "latitude": lat_idx, "longitude": lon_idx}).values)
    era5_ws.append(ws); matched += 1
    if matched % 5000 == 0: print(f"  ...{matched:,} rows")

sar["ERA5_WindSpeed_100m_ms"] = era5_ws
print(f"\n  Matched: {matched:,} | Unmatched: {unmatched:,}")

sar_out = sar.dropna(subset=["ERA5_WindSpeed_100m_ms"]).copy()
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
sar_out.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {len(sar_out):,} rows -> {OUTPUT_FILE}")
print(sar_out["ERA5_WindSpeed_100m_ms"].describe().round(3))