"""
ascat_collocation.py
--------------------
Collocates ASCAT 100m extrapolated wind speed with Sentinel-1 SAR
acquisitions from era5_collocated.csv.

For each SAR observation, finds the nearest ASCAT observation within:
  - Time window : ±12 hours
  - Spatial threshold : < 0.5 degrees (~50 km)

Output: data/ascat/ascat_sar_collocated.csv
Columns include SAR features, ERA5 100m wind, and ASCAT 100m wind
for three-way validation comparison.
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# ── Config ───────────────────────────────────────────────────────────────────
SAR_CSV        = "data/processed/era5_collocated.csv"
ASCAT_CSV      = "data/ascat/ascat_combined_100m.csv"
OUTPUT_CSV     = "data/ascat/ascat_sar_collocated.csv"
TIME_WINDOW_HR = 12    # hours either side of SAR acquisition
SPATIAL_THRESH = 0.5   # degrees (~50 km)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading SAR dataset...")
df_sar = pd.read_csv(SAR_CSV)
df_sar["timestamp"] = pd.to_datetime(df_sar["timestamp"])
print(f"  SAR observations: {len(df_sar)}")

print("Loading ASCAT dataset...")
df_ascat = pd.read_csv(ASCAT_CSV)
df_ascat["time"] = pd.to_datetime(df_ascat["time"])
print(f"  ASCAT observations: {len(df_ascat)}")

# ── Collocation ───────────────────────────────────────────────────────────────
print("\nRunning collocation (this may take a few minutes)...")
matched_rows = []

for i, sar_row in df_sar.iterrows():
    if i % 1000 == 0:
        print(f"  Processing SAR row {i}/{len(df_sar)}...")

    # Time filter
    window = df_ascat[
        (df_ascat["time"] >= sar_row["timestamp"] - pd.Timedelta(hours=TIME_WINDOW_HR)) &
        (df_ascat["time"] <= sar_row["timestamp"] + pd.Timedelta(hours=TIME_WINDOW_HR))
    ]

    if window.empty:
        continue

    # Spatial nearest neighbour
    coords = window[["latitude", "longitude"]].values
    tree = cKDTree(coords)
    dist, idx = tree.query([sar_row["latitude"], sar_row["longitude"]])

    if dist < SPATIAL_THRESH:
        match = window.iloc[idx].copy()
        match["sar_timestamp"]    = sar_row["timestamp"]
        match["sar_lat"]          = sar_row["latitude"]
        match["sar_lon"]          = sar_row["longitude"]
        match["era5_wind_100m"]   = sar_row["ERA5_WindSpeed_100m_ms"]
        match["VV"]               = sar_row["VV"]
        match["VH"]               = sar_row["VH"]
        match["incidence_angle"]  = sar_row["incidence_angle"]
        match["point_id"]         = sar_row["point_id"]
        match["spatial_dist_deg"] = dist
        matched_rows.append(match)

# ── Save results ──────────────────────────────────────────────────────────────
df_matched = pd.DataFrame(matched_rows).reset_index(drop=True)

print(f"\nMatched observations: {len(df_matched)}")
print(df_matched[["sar_timestamp", "sar_lat", "sar_lon",
                   "wind_speed", "wind_speed_100m", "era5_wind_100m"]].head(10))

df_matched.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved to {OUTPUT_CSV}")