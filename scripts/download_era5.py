"""
download_era5.py
----------------
Downloads ERA5 100m wind components (U + V) matched to
Sentinel-1 overpass dates (2020–2024).

- Sentinel-1 overpasses are at ~06:56 UTC → we use ERA5 07:00 UTC
- Downloads one NetCDF file per year to keep file sizes manageable
- Covers Gujarat bounding box: lon 67.5–74.5°E, lat 19.5–24.5°N

Requirements:
    pip install cdsapi xarray netCDF4 scipy
    
    ~/.cdsapirc must contain:
        url: https://cds.climate.copernicus.eu/api
        key: <your-api-key>
"""

import os
import cdsapi
import pandas as pd
from pathlib import Path

# ── Paths (relative to this script's location inside scripts/) ────────────────

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAR_FILE   = os.path.join(BASE_DIR, "data", "processed", "gujarat_sar_timeseries.csv")
OUTPUT_DIR = Path(os.path.join(BASE_DIR, "data", "raw", "era5_downloads"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Gujarat bounding box [North, West, South, East]
BBOX = [24.5, 67.5, 19.5, 74.5]

# Nearest ERA5 hour to Sentinel-1 overpass time (~06:56 UTC)
OVERPASS_HOUR = "07:00"

# ── Load SAR data and extract unique (year, month, day) combos ───────────────

print("Reading SAR timeseries...")
df = pd.read_csv(SAR_FILE, parse_dates=["timestamp"])

# Get unique overpass dates
overpass_dates = (
    df[["year", "month", "day"]]
    .drop_duplicates()
    .sort_values(["year", "month", "day"])
    .reset_index(drop=True)
)
print(f"Found {len(overpass_dates)} unique overpass dates across "
      f"{overpass_dates['year'].nunique()} years")

# ── Download one NetCDF per year ──────────────────────────────────────────────

c = cdsapi.Client()

for year in sorted(overpass_dates["year"].unique()):
    out_file = OUTPUT_DIR / f"era5_gujarat_100m_wind_{year}.nc"
    
    if out_file.exists():
        print(f"[{year}] Already downloaded → skipping ({out_file})")
        continue

    year_dates = overpass_dates[overpass_dates["year"] == year]
    
    # Unique months in this year
    months = sorted(year_dates["month"].unique())
    
    # All overpass days in this year (ERA5 will only give data for requested days)
    days = sorted(year_dates["day"].unique())
    
    print(f"\n[{year}] Requesting ERA5 for {len(year_dates)} overpass dates, "
          f"{len(months)} months...")
    
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "100m_u_component_of_wind",
                "100m_v_component_of_wind",
            ],
            "year":  str(year),
            "month": [f"{m:02d}" for m in months],
            "day":   [f"{d:02d}" for d in days],
            "time":  OVERPASS_HOUR,
            "area":  BBOX,
            "format": "netcdf",
        },
        str(out_file),
    )
    print(f"[{year}] Saved → {out_file}")

print("\nAll downloads complete.")
print(f"Files saved in: {OUTPUT_DIR.resolve()}")