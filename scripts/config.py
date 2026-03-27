import ee
import os

# Project Configuration
PROJECT_ID = 'wind-research'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Directories
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
WIND_DATA_DIR = os.path.join(BASE_DIR, 'data', 'wind_measurements')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, WIND_DATA_DIR, 
                  OUTPUT_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Study Area: Gujarat Coast
# Approximate bounding box for Gujarat coastal region
GUJARAT_BOUNDS = {
    'min_lon': 68.0,   # Western boundary
    'max_lon': 73.0,   # Eastern boundary
    'min_lat': 20.0,   # Southern boundary
    'max_lat': 23.5    # Northern boundary
}

# Offshore extent (in km from coast)
OFFSHORE_DISTANCE_KM = 100  # How far offshore to include

# Time Period for Analysis - 5 YEARS
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'

# Sentinel-1 SAR Parameters
SAR_CONFIG = {
    'instrument_mode': 'IW',  # Interferometric Wide swath
    'polarization': ['VV', 'VH'],  # Dual polarization
    'orbit': 'DESCENDING',  # or 'ASCENDING' or None for both
    'resolution': 10,  # meters
}

# Processing Parameters
PROCESSING_CONFIG = {
    'apply_border_noise_correction': True,
    'apply_thermal_noise_removal': True,
    'scale_factor': 10000,  # For integer conversion
}

# Sampling Grid Parameters
GRID_CONFIG = {
    'coastal_interval_km': 15,  # Spacing along coastline
    'offshore_distances_km': [20, 40, 60, 80],  # Distances from coast
}

# Export Settings
EXPORT_CONFIG = {
    'scale': 100,  # Export resolution in meters
    'crs': 'EPSG:4326',  # WGS84
    'file_format': 'GeoTIFF',
    'folder': 'SAR_Gujarat',  # Google Drive folder name
}

# Wind Speed Parameters (for later modeling)
HUB_HEIGHT = 100  # meters (typical offshore wind turbine hub height)

# Neural Network Parameters (for future use)
NN_CONFIG = {
    'train_test_split': 0.8,  # 80% training, 20% testing
    'validation_split': 0.2,  # 20% of training for validation
    'random_seed': 42,
}

def get_gujarat_coastline():
    """
    Fetch Gujarat's sea-facing coastline from GEE's LSIB dataset.

    Gets India's boundary as a LINE (perimeter), then filters coordinates
    to Gujarat's coastal bounding box. This avoids the artificial straight
    lines that appear when intersecting polygons with a bounding box.

    Returns:
        pd.DataFrame with columns ['latitude', 'longitude']
    """
    import pandas as pd
    import numpy as np

    cache_path = os.path.join(RAW_DATA_DIR, 'gujarat_coastline_cache.csv')

    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    print("  [Coastline] Fetching Gujarat coastline from GEE LSIB dataset...")

    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception:
        pass

    # Gujarat coastal bounding box
    LON_MIN, LON_MAX = 67.5, 74.5
    LAT_MIN, LAT_MAX = 19.8, 24.5

    # Get India's full boundary geometry as GeoJSON
    india_geom = (
        ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
        .filter(ee.Filter.eq("country_na", "India"))
        .geometry()
        .getInfo()
    )

    # ── Extract ALL India boundary coordinates ────────────────────────────────
    all_coords = []

    def collect_coords(geometry):
        gtype = geometry.get('type')
        raw = geometry.get('coordinates', [])

        if gtype == 'Polygon':
            # Each polygon is a closed ring — walk it as a sequence of segments
            ring = raw[0]
            for i in range(len(ring) - 1):
                all_coords.append(('seg', ring[i], ring[i+1]))

        elif gtype == 'MultiPolygon':
            for polygon in raw:
                ring = polygon[0]
                for i in range(len(ring) - 1):
                    all_coords.append(('seg', ring[i], ring[i+1]))

        elif gtype == 'GeometryCollection':
            for geom in geometry.get('geometries', []):
                collect_coords(geom)

    collect_coords(india_geom)

    # ── Keep only segments where BOTH endpoints are inside Gujarat bbox ───────
    # This filters out inland borders and bbox-edge artifacts
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