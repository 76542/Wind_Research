"""
scripts/goa/config.py
======================
Configuration for Goa coast.

Bounding box:
  - Lat: 14.8°N (Karnataka border) to 15.8°N (Maharashtra border)
  - Lon: 72.5°E to 74.1°E (extends ~100km offshore)

Goa has ~105km of coastline, the smallest west coast state.
"""

import ee
import os

PROJECT_ID = 'wind-research'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'goa')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'goa')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'goa')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

STATE_BOUNDS = {
    'min_lon': 72.5,
    'max_lon': 74.1,
    'min_lat': 14.8,
    'max_lat': 15.8
}

STATE_NAME = 'Goa'
STATE_PREFIX = 'GA'

START_DATE = '2020-01-01'
END_DATE = '2024-12-31'

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

GRID_CONFIG = {
    'coastal_interval_km': 15,
    'offshore_distances_km': [20, 40, 60, 80],
}

EXPORT_CONFIG = {
    'scale': 100,
    'crs': 'EPSG:4326',
    'file_format': 'GeoTIFF',
    'folder': 'SAR_Goa',
}

HUB_HEIGHT = 100

NN_CONFIG = {
    'train_test_split': 0.8,
    'validation_split': 0.2,
    'random_seed': 42,
}


def get_coastline():
    """Fetch Goa's sea-facing coastline from GEE LSIB dataset."""
    import pandas as pd

    cache_path = os.path.join(RAW_DATA_DIR, 'goa_coastline_cache.csv')

    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    print("  [Coastline] Fetching Goa coastline from GEE LSIB dataset...")

    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception:
        pass

    LON_MIN, LON_MAX = 73.5, 74.2
    LAT_MIN, LAT_MAX = 14.8, 15.8

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