"""
scripts/andhrapradesh/create_sampling_grid.py
==============================================
East coast grid — sea is EAST, land is WEST.

Usage: python -m scripts.andhrapradesh.create_sampling_grid
"""

import ee
import numpy as np
import pandas as pd
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from scripts.andhrapradesh.config import (
    PROJECT_ID, RAW_DATA_DIR, GRID_CONFIG,
    STATE_PREFIX, STATE_BOUNDS, STATE_NAME, get_coastline
)


class SamplingGridGenerator:
    def __init__(self):
        ee.Initialize(project=PROJECT_ID)
        self.offshore_distances = GRID_CONFIG['offshore_distances_km']
        self.coastal_interval_km = GRID_CONFIG['coastal_interval_km']

    def haversine_distance(self, lon1, lat1, lon2, lat2):
        R = 6371.0
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    def _build_continuous_coastline(self, coastline_df):
        all_pts, current_segment = [], []
        for _, row in coastline_df.iterrows():
            if pd.isna(row['latitude']):
                if len(current_segment) > 1: all_pts.extend(current_segment)
                current_segment = []
            else:
                current_segment.append((row['longitude'], row['latitude']))
        if len(current_segment) > 1: all_pts.extend(current_segment)
        deduped = [all_pts[0]]
        for p in all_pts[1:]:
            if p != deduped[-1]: deduped.append(p)
        print(f"  Continuous coastline: {len(deduped):,} unique vertices")
        return deduped

    def _resample_coastline(self, coast_pts):
        interval = self.coastal_interval_km
        resampled = [coast_pts[0]]
        accumulated = 0.0
        for i in range(1, len(coast_pts)):
            lon_prev, lat_prev = coast_pts[i-1]
            lon_curr, lat_curr = coast_pts[i]
            seg_dist = self.haversine_distance(lon_prev, lat_prev, lon_curr, lat_curr)
            if seg_dist < 1e-6: continue
            remaining = seg_dist
            while accumulated + remaining >= interval:
                need = interval - accumulated
                frac = need / seg_dist
                emit_lon = lon_prev + frac * (lon_curr - lon_prev)
                emit_lat = lat_prev + frac * (lat_curr - lat_prev)
                resampled.append((emit_lon, emit_lat))
                lon_prev, lat_prev = emit_lon, emit_lat
                seg_dist = self.haversine_distance(lon_prev, lat_prev, lon_curr, lat_curr)
                remaining = seg_dist
                accumulated = 0.0
            accumulated += remaining
        print(f"  Resampled coastline -> {len(resampled)} points (~{interval} km)")
        return resampled

    def _seaward_direction(self, pts, idx):
        n = len(pts)
        i_prev, i_next = max(0, idx-1), min(n-1, idx+1)
        dx = pts[i_next][0] - pts[i_prev][0]
        dy = pts[i_next][1] - pts[i_prev][1]
        mag = np.sqrt(dx**2 + dy**2)
        if mag < 1e-10: dx, dy = 1.0, 0.0
        else: dx, dy = dx/mag, dy/mag
        perp_a, perp_b = (-dy, dx), (dy, -dx)
        # Land reference: interior AP (Western Ghats / Deccan side)
        land_lon, land_lat = 77.0, 16.0
        lon0, lat0 = pts[idx]
        vtl = (land_lon - lon0, land_lat - lat0)
        mag_l = np.sqrt(vtl[0]**2 + vtl[1]**2)
        if mag_l < 1e-10:
            return perp_a
        vtl = (vtl[0]/mag_l, vtl[1]/mag_l)
        dot_a = perp_a[0]*vtl[0] + perp_a[1]*vtl[1]
        dot_b = perp_b[0]*vtl[0] + perp_b[1]*vtl[1]
        return perp_a if dot_a < dot_b else perp_b

    def _project_offshore(self, lon, lat, direction, distance_km):
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(np.radians(lat))
        return (lon + direction[0]*distance_km/km_per_deg_lon,
                lat + direction[1]*distance_km/km_per_deg_lat)

    def _filter_to_ocean(self, grid_df, batch_size=500):
        print(f"\n  Validating {len(grid_df)} points against GEE water + elevation mask...")
        water = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence')
        elev = ee.Image('USGS/SRTMGL1_003').select('elevation')
        ocean_mask = water.gte(50).And(elev.lte(5)).rename('occurrence')
        water_ids = set()
        rows = list(grid_df.itertuples(index=False))
        n_batches = (len(rows) + batch_size - 1) // batch_size
        for b in range(n_batches):
            batch = rows[b*batch_size:(b+1)*batch_size]
            features = [ee.Feature(ee.Geometry.Point([r.longitude, r.latitude]),
                        {'point_id': r.point_id}) for r in batch]
            fc = ee.FeatureCollection(features)
            sampled = ocean_mask.sampleRegions(collection=fc, scale=300, geometries=False)
            for f in sampled.getInfo()['features']:
                if f['properties'].get('occurrence', 0) == 1:
                    water_ids.add(f['properties']['point_id'])
            print(f"    Batch {b+1}/{n_batches} done ({len(water_ids)} ocean points)")
        filtered = grid_df[grid_df['point_id'].isin(water_ids)].copy()
        print(f"  Removed {len(grid_df)-len(filtered)} onshore -> {len(filtered)} remain")
        return filtered

    def generate_grid(self):
        print("=" * 60)
        print(f"Step 1 - Loading {STATE_NAME} coastline...")
        coastline_df = get_coastline()
        print(f"  Raw coastline: {len(coastline_df):,} rows")

        print(f"\nStep 2 - Resampling...")
        coast_pts = self._resample_coastline(
            self._build_continuous_coastline(coastline_df))

        print(f"\nStep 3 - Projecting offshore (EASTWARD into Bay of Bengal)...")
        raw_points = []
        pid = 1
        for idx, (lon, lat) in enumerate(coast_pts):
            d = self._seaward_direction(coast_pts, idx)
            for dist in self.offshore_distances:
                off_lon, off_lat = self._project_offshore(lon, lat, d, dist)
                raw_points.append({
                    'point_id': f'{STATE_PREFIX}_{pid:04d}',
                    'longitude': round(off_lon, 5), 'latitude': round(off_lat, 5),
                    'offshore_distance_km': dist,
                    'coastal_lon': round(lon, 5), 'coastal_lat': round(lat, 5)})
                pid += 1
        grid_df = pd.DataFrame(raw_points)
        print(f"  Generated {len(grid_df)} candidate points")

        print(f"\nStep 4 - Filtering to ocean...")
        grid_df = self._filter_to_ocean(grid_df)

        # Trim to AP boundaries
        grid_df = grid_df[(grid_df['latitude'] >= 13.6) &
                          (grid_df['latitude'] <= 19.4)].copy()
        print(f"  After border trim: {len(grid_df)} points")

        # Exclude Visakhapatnam harbour (~17.65-17.75N, nearshore)
        grid_df = grid_df[~((grid_df['latitude'] > 17.65) &
                             (grid_df['latitude'] < 17.75) &
                             (grid_df['offshore_distance_km'] <= 20))].copy()
        print(f"  After Vizag harbour exclusion: {len(grid_df)} points")

        grid_df = grid_df.reset_index(drop=True)
        grid_df['point_id'] = [f'{STATE_PREFIX}_{i+1:04d}' for i in range(len(grid_df))]
        return grid_df

    def save_grid(self, grid_df, filename='andhrapradesh_sampling_grid.csv'):
        output_path = os.path.join(RAW_DATA_DIR, filename)
        grid_df.to_csv(output_path, index=False)
        print(f"\nGrid saved to: {output_path}")
        geojson_path = output_path.replace('.csv', '.geojson')
        features = [{'type': 'Feature',
                     'geometry': {'type': 'Point',
                                  'coordinates': [row['longitude'], row['latitude']]},
                     'properties': {'point_id': row['point_id'],
                                    'offshore_distance_km': row['offshore_distance_km'],
                                    'coastal_lon': row['coastal_lon'],
                                    'coastal_lat': row['coastal_lat']}}
                    for _, row in grid_df.iterrows()]
        with open(geojson_path, 'w') as f:
            json.dump({'type': 'FeatureCollection', 'features': features}, f, indent=2)
        print(f"GeoJSON saved to: {geojson_path}")

    def print_stats(self, grid_df):
        print(f"\n{'='*60}\nFINAL GRID - {STATE_NAME}\n{'='*60}")
        print(f"Total points: {len(grid_df)}")
        for dist in sorted(grid_df['offshore_distance_km'].unique()):
            print(f"  {dist:.0f} km: {(grid_df['offshore_distance_km']==dist).sum()}")
        print(f"Lat: {grid_df['latitude'].min():.3f} - {grid_df['latitude'].max():.3f}")
        print(f"Lon: {grid_df['longitude'].min():.3f} - {grid_df['longitude'].max():.3f}")


def main():
    print(f"{'='*60}\n{STATE_NAME} East Coast - Sampling Grid Generator\n{'='*60}\n")
    gen = SamplingGridGenerator()
    grid_df = gen.generate_grid()
    gen.print_stats(grid_df)
    gen.save_grid(grid_df)

if __name__ == "__main__":
    main()