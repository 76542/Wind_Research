"""
scripts/goa/download_era5.py

Usage:
  cd Wind_Research
  python -m scripts.goa.download_era5
"""

import os
import cdsapi
import pandas as pd
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAR_FILE = os.path.join(BASE_DIR, "data", "processed", "goa",
                        "goa_sar_timeseries.csv")
OUTPUT_DIR = Path(os.path.join(BASE_DIR, "data", "raw", "goa", "era5_downloads"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Goa bounding box [North, West, South, East]
BBOX = [15.8, 72.5, 14.8, 74.1]
OVERPASS_HOUR = "07:00"

print("Reading Goa SAR timeseries...")
df = pd.read_csv(SAR_FILE, parse_dates=["timestamp"])

overpass_dates = (
    df[["year", "month", "day"]]
    .drop_duplicates()
    .sort_values(["year", "month", "day"])
    .reset_index(drop=True)
)
print(f"Found {len(overpass_dates)} unique overpass dates across "
      f"{overpass_dates['year'].nunique()} years")

c = cdsapi.Client()

for year in sorted(overpass_dates["year"].unique()):
    out_file = OUTPUT_DIR / f"era5_goa_100m_wind_{year}.nc"

    if out_file.exists():
        print(f"[{year}] Already downloaded -> skipping ({out_file})")
        continue

    year_dates = overpass_dates[overpass_dates["year"] == year]
    months = sorted(year_dates["month"].unique())
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
            "year": str(year),
            "month": [f"{m:02d}" for m in months],
            "day": [f"{d:02d}" for d in days],
            "time": OVERPASS_HOUR,
            "area": BBOX,
            "format": "netcdf",
        },
        str(out_file),
    )
    print(f"[{year}] Saved -> {out_file}")

print("\nAll downloads complete.")
print(f"Files saved in: {OUTPUT_DIR.resolve()}")