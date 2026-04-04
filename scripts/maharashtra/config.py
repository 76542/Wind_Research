"""
scripts/maharashtra/config.py
==============================
Configuration for Maharashtra coast — mirrors scripts/config.py structure
so all downstream scripts work with minimal changes.

Bounding box:
  - Lat: 15.5°N (Goa border) to 20.0°N (Gujarat border)
  - Lon: 71.0°E to 73.5°E (extends ~150-200 km offshore)
"""

import ee
import os

# Project Configuration (same GEE project)
PROJECT_ID = 'wind-research'

# BASE_DIR = Wind_Research/ (three levels up from scripts/maharashtra/config.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Data Directories ──────────────────────────────────────────────────────
# Separate subdirectories so Gujarat data is untouched
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'maharashtra')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'maharashtra')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'maharashtra')
MODELS_DIR = os.path.join(BASE_DIR, 'models')  # shared — Gujarat model lives here
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ── Study Area: Maharashtra Coast ─────────────────────────────────────────
MAHARASHTRA_BOUNDS = {
    'min_lon': 71.0,
    'max_lon': 73.5,
    'min_lat': 15.5,
    'max_lat': 20.0
}

STATE_BOUNDS = MAHARASHTRA_BOUNDS
STATE_NAME = 'Maharashtra'
STATE_PREFIX = 'MH'

# ── Time Period (same as Gujarat) ─────────────────────────────────────────
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'

# ── SAR Parameters (identical to Gujarat) ─────────────────────────────────
SAR_CONFIG = {
    'instrument_mode': 'IW',
    'polarization': ['VV', 'VH'],
    'orbit': 'DESCENDING',
    'resolution': 10,
}

PROCESSING_CONFIG = {
    'apply_border_noise_correction': True,
    'apply_thermal_noise_removal': True,
    'scale_factor': 10000,
}

# ── Sampling Grid Parameters (same spacing as Gujarat) ────────────────────
GRID_CONFIG = {
    'coastal_interval_km': 15,
    'offshore_distances_km': [20, 40, 60, 80],
}

# ── Export Settings ───────────────────────────────────────────────────────
EXPORT_CONFIG = {
    'scale': 100,
    'crs': 'EPSG:4326',
    'file_format': 'GeoTIFF',
    'folder': 'SAR_Maharashtra',
}

HUB_HEIGHT = 100

NN_CONFIG = {
    'train_test_split': 0.8,
    'validation_split': 0.2,
    'random_seed': 42,
}


def get_coastline():
    """
    Fetch Maharashtra's sea-facing coastline from GEE's LSIB dataset.
    Same approach as Gujarat's get_gujarat_coastline().
    """
    import pandas as pd

    cache_path = os.path.join(RAW_DATA_DIR, 'maharashtra_coastline_cache.csv')

    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    print("  [Coastline] Fetching Maharashtra coastline from GEE LSIB dataset...")

    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception:
        pass

    # Maharashtra coastal bounding box (slightly wider to capture full coast shape)
    LON_MIN, LON_MAX = 72.0, 74.0
    LAT_MIN, LAT_MAX = 15.5, 20.0

    india_geom = (
        ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
        .filter(ee.Filter.eq("country_na", "India"))
        .geometry()
        .getInfo()
    )

    all_coords = []

    def collect_coords(geometry):
        gtype = geometry.get('type')
        raw = geometry.get('coordinates', [])

        if gtype == 'Polygon':
            ring = raw[0]
            for i in range(len(ring) - 1):
                all_coords.append(('seg', ring[i], ring[i + 1]))
        elif gtype == 'MultiPolygon':
            for polygon in raw:
                ring = polygon[0]
                for i in range(len(ring) - 1):
                    all_coords.append(('seg', ring[i], ring[i + 1]))
        elif gtype == 'GeometryCollection':
            for geom in geometry.get('geometries', []):
                collect_coords(geom)

    collect_coords(india_geom)

    coords = []
    for _, (lon1, lat1), (lon2, lat2) in all_coords:
        p1_in = LON_MIN <= lon1 <= LON_MAX and LAT_MIN <= lat1 <= LAT_MAX
        p2_in = LON_MIN <= lon2 <= LON_MAX and LAT_MIN <= lat2 <= LAT_MAX
        if p1_in and p2_in:
            coords.append({'longitude': lon1, 'latitude': lat1})
            coords.append({'longitude': lon2, 'latitude': lat2})
            coords.append({'longitude': float('nan'), 'latitude': float('nan')})

    df = pd.DataFrame(coords)
    df.to_csv(cache_path, index=False)
    print(f"  [Coastline] Cached {len(df):,} coastline points to {cache_path}")

    return df