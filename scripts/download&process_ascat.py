"""
Downloads MetOp-B ASCAT 10m wind data (ascending + descending passes)
from CMEMS for the Gujarat offshore region (2020-2024),
merges both passes, applies power law extrapolation to 100m,
and saves the result as a CSV.

Region: 19.5-24.5N, 67.5-74.5E (Gujarat offshore bounding box)
Dataset: WIND_GLO_PHY_L3_MY_012_005 (MetOp-B ASCAT, 0.25 degree, reprocessed)
"""

import os
import copernicusmarine
import xarray as xr
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = "data/ascat"
MIN_LON, MAX_LON = 67.5, 74.5
MIN_LAT, MAX_LAT = 19.5, 24.5
START_DATE = "2020-01-01T00:00:00"
END_DATE   = "2024-12-31T23:59:59"
ALPHA      = 0.11  # Hellmann exponent for open ocean (power law)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Step 1: Download ascending pass ─────────────────────────────────────────
print("Downloading ascending pass...")
copernicusmarine.subset(
    dataset_id="cmems_obs-wind_glo_phy_my_l3-metopb-ascat-asc-0.25deg_P1D-i",
    variables=["wind_speed", "eastward_wind", "northward_wind"],
    minimum_longitude=MIN_LON,
    maximum_longitude=MAX_LON,
    minimum_latitude=MIN_LAT,
    maximum_latitude=MAX_LAT,
    start_datetime=START_DATE,
    end_datetime=END_DATE,
    output_filename="ascat_metopb_asc_gujarat_2020_2024.nc",
    output_directory=OUTPUT_DIR,
)
print("Ascending pass downloaded.")

# ── Step 2: Download descending pass ────────────────────────────────────────
print("Downloading descending pass...")
copernicusmarine.subset(
    dataset_id="cmems_obs-wind_glo_phy_my_l3-metopb-ascat-des-0.25deg_P1D-i",
    variables=["wind_speed", "eastward_wind", "northward_wind"],
    minimum_longitude=MIN_LON,
    maximum_longitude=MAX_LON,
    minimum_latitude=MIN_LAT,
    maximum_latitude=MAX_LAT,
    start_datetime=START_DATE,
    end_datetime=END_DATE,
    output_filename="ascat_metopb_des_gujarat_2020_2024.nc",
    output_directory=OUTPUT_DIR,
)
print("Descending pass downloaded.")

# ── Step 3: Load, merge, clean ───────────────────────────────────────────────
print("Loading and merging passes...")
asc = xr.open_dataset(f"{OUTPUT_DIR}/ascat_metopb_asc_gujarat_2020_2024.nc")
des = xr.open_dataset(f"{OUTPUT_DIR}/ascat_metopb_des_gujarat_2020_2024.nc")

df_asc = asc[["wind_speed"]].to_dataframe().reset_index().dropna()
df_asc["pass"] = "ascending"

df_des = des[["wind_speed"]].to_dataframe().reset_index().dropna()
df_des["pass"] = "descending"

df_ascat = pd.concat([df_asc, df_des], ignore_index=True)
df_ascat = df_ascat[df_ascat["wind_speed"] > 0].reset_index(drop=True)

# ── Step 4: Power law extrapolation 10m → 100m ──────────────────────────────
df_ascat["wind_speed_100m"] = df_ascat["wind_speed"] * (100 / 10) ** ALPHA

# ── Step 5: Save ─────────────────────────────────────────────────────────────
output_path = f"{OUTPUT_DIR}/ascat_combined_100m.csv"
df_ascat.to_csv(output_path, index=False)

print(f"\nTotal valid ASCAT observations: {len(df_ascat)}")
print(f"Saved to {output_path}")
print(df_ascat.head())