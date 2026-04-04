"""
scripts/maharashtra/create_sampling_grid.py
=============================================
Generate systematic offshore sampling grid for Maharashtra coast.

Differences from Gujarat:
  - No Rann of Kutch / Pakistan / Gulf of Kutch exclusions
  - No Saurashtra gap fill points
  - Mumbai harbour / Thane creek exclusion instead
  - MH_ point ID prefix

Usage:
  cd Wind_Research
  python -m scripts.maharashtra.create_sampling_grid
"""

import ee
import numpy as np
import pandas as pd
import json
import os
import sys

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from scripts.maharashtra.config import (
    PROJECT_ID, RAW_DATA_DIR, GRID_CONFIG,
    STATE_PREFIX, STATE_BOUNDS, STATE_NAME,
    get_coastline
)


class SamplingGridGenerator:
    def __init__(self):
        ee.Initialize(project=PROJECT_ID)
        self.offshore_distances = GRID_CONFIG['offshore_distances_km']
        self.coastal_interval_km = GRID_CONFIG['coastal_interval_km']

    # ------------------------------------------------------------------
    # Geometry helpers (identical to Gujarat)
    # ------------------------------------------------------------------

    def haversine_distance(self, lon1, lat1, lon2, lat2):
        R = 6371.0
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    def _build_continuous_coastline(self, coastline_df):
        all_pts = []
        current_segment = []

        for _, row in coastline_df.iterrows():
            if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                if len(current_segment) > 1:
                    all_pts.extend(current_segment)
                current_segment = []
            else:
                current_segment.append((row['longitude'], row['latitude']))

        if len(current_segment) > 1:
            all_pts.extend(current_segment)

        deduped = [all_pts[0]]
        for p in all_pts[1:]:
            if p != deduped[-1]:
                deduped.append(p)

        print(f"  Continuous coastline: {len(deduped):,} unique vertices")
        return deduped

    def _resample_coastline(self, coast_pts):
        interval = self.coastal_interval_km
        resampled = [coast_pts[0]]
        accumulated = 0.0

        for i in range(1, len(coast_pts)):
            lon_prev, lat_prev = coast_pts[i - 1]
            lon_curr, lat_curr = coast_pts[i]

            seg_dist = self.haversine_distance(
                lon_prev, lat_prev, lon_curr, lat_curr
            )
            if seg_dist < 1e-6:
                continue

            remaining = seg_dist

            while accumulated + remaining >= interval:
                need = interval - accumulated
                frac = need / seg_dist
                emit_lon = lon_prev + frac * (lon_curr - lon_prev)
                emit_lat = lat_prev + frac * (lat_curr - lat_prev)
                resampled.append((emit_lon, emit_lat))
                lon_prev = emit_lon
                lat_prev = emit_lat
                seg_dist = self.haversine_distance(
                    lon_prev, lat_prev, lon_curr, lat_curr
                )
                remaining = seg_dist
                accumulated = 0.0

            accumulated += remaining

        print(f"  Resampled coastline → {len(resampled)} points "
              f"(~{interval} km spacing)")
        return resampled

    def _seaward_direction(self, pts, idx):
        n = len(pts)
        i_prev = max(0, idx - 1)
        i_next = min(n - 1, idx + 1)

        lon_prev, lat_prev = pts[i_prev]
        lon_next, lat_next = pts[i_next]

        dx = lon_next - lon_prev
        dy = lat_next - lat_prev
        mag = np.sqrt(dx ** 2 + dy ** 2)
        if mag < 1e-10:
            dx, dy = 1.0, 0.0
        else:
            dx, dy = dx / mag, dy / mag

        perp_a = (-dy, dx)
        perp_b = (dy, -dx)

        land_lon, land_lat = 78.0, 22.0
        lon0, lat0 = pts[idx]
        vec_to_land = (land_lon - lon0, land_lat - lat0)
        mag_land = np.sqrt(vec_to_land[0] ** 2 + vec_to_land[1] ** 2)
        vec_to_land = (vec_to_land[0] / mag_land, vec_to_land[1] / mag_land)

        dot_a = perp_a[0] * vec_to_land[0] + perp_a[1] * vec_to_land[1]
        dot_b = perp_b[0] * vec_to_land[0] + perp_b[1] * vec_to_land[1]

        return perp_a if dot_a < dot_b else perp_b

    def _project_offshore(self, lon, lat, direction, distance_km):
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(np.radians(lat))
        d_lon = direction[0] * distance_km / km_per_deg_lon
        d_lat = direction[1] * distance_km / km_per_deg_lat
        return lon + d_lon, lat + d_lat

    # ------------------------------------------------------------------
    # GEE water-mask + elevation validation
    # ------------------------------------------------------------------

    def _filter_to_ocean(self, grid_df, batch_size=500):
        print(f"\n  Validating {len(grid_df)} points against GEE water + "
              f"elevation mask (batches of {batch_size})...")

        water = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence')
        elev = ee.Image('USGS/SRTMGL1_003').select('elevation')
        ocean_mask = water.gte(50).And(elev.lte(5)).rename('occurrence')

        water_ids = set()
        rows = list(grid_df.itertuples(index=False))
        n_batches = (len(rows) + batch_size - 1) // batch_size

        for b in range(n_batches):
            batch = rows[b * batch_size: (b + 1) * batch_size]
            features = [
                ee.Feature(
                    ee.Geometry.Point([r.longitude, r.latitude]),
                    {'point_id': r.point_id}
                )
                for r in batch
            ]
            fc = ee.FeatureCollection(features)
            sampled = ocean_mask.sampleRegions(
                collection=fc, scale=300, geometries=False
            )
            results = sampled.getInfo()['features']
            for f in results:
                if f['properties'].get('occurrence', 0) == 1:
                    water_ids.add(f['properties']['point_id'])

            print(f"    Batch {b + 1}/{n_batches} done "
                  f"({len(water_ids)} ocean points so far)")

        filtered = grid_df[grid_df['point_id'].isin(water_ids)].copy()
        removed = len(grid_df) - len(filtered)
        print(f"  Removed {removed} onshore/inland points → "
              f"{len(filtered)} valid offshore points remain")
        return filtered

    # ------------------------------------------------------------------
    # Main grid generation
    # ------------------------------------------------------------------

    def generate_grid(self):
        print("=" * 60)
        print(f"Step 1 — Loading {STATE_NAME} coastline from GEE...")
        coastline_df = get_coastline()
        print(f"  Raw coastline: {len(coastline_df):,} rows")

        print(f"\nStep 2 — Building continuous coastline & resampling...")
        coast_pts_raw = self._build_continuous_coastline(coastline_df)
        coast_pts = self._resample_coastline(coast_pts_raw)

        print(f"\nStep 3 — Projecting offshore points...")
        raw_points = []
        point_id = 1

        for idx, (lon, lat) in enumerate(coast_pts):
            direction = self._seaward_direction(coast_pts, idx)
            for dist in self.offshore_distances:
                off_lon, off_lat = self._project_offshore(
                    lon, lat, direction, dist
                )
                raw_points.append({
                    'point_id': f'{STATE_PREFIX}_{point_id:04d}',
                    'longitude': round(off_lon, 5),
                    'latitude': round(off_lat, 5),
                    'offshore_distance_km': dist,
                    'coastal_lon': round(lon, 5),
                    'coastal_lat': round(lat, 5),
                })
                point_id += 1

        grid_df = pd.DataFrame(raw_points)
        print(f"  Generated {len(grid_df)} candidate offshore points")

        # ── Maharashtra-specific exclusions ───────────────────────────
        # (No Saurashtra gap fill, no Rann/Kutch, no Pakistan)

        print(f"\nStep 4 — Filtering to ocean-only points...")
        grid_df = self._filter_to_ocean(grid_df)

        # Trim south of Goa border
        grid_df = grid_df[grid_df['latitude'] >= 15.6].copy()
        print(f"  After Goa border trim: {len(grid_df)} points remain")

        # Exclude Mumbai harbour / Thane creek (narrow inlets ~19.0-19.2°N)
        grid_df = grid_df[~(
            (grid_df['latitude'] > 18.9) &
            (grid_df['latitude'] < 19.3) &
            (grid_df['longitude'] > 72.85)
        )].copy()
        print(f"  After Mumbai harbour exclusion: {len(grid_df)} points remain")

        # Re-index
        grid_df = grid_df.reset_index(drop=True)
        grid_df['point_id'] = [f'{STATE_PREFIX}_{i + 1:04d}'
                               for i in range(len(grid_df))]

        return grid_df

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_grid(self, grid_df, filename='maharashtra_sampling_grid.csv'):
        output_path = os.path.join(RAW_DATA_DIR, filename)
        grid_df.to_csv(output_path, index=False)
        print(f"\nSampling grid saved to: {output_path}")

        geojson_path = output_path.replace('.csv', '.geojson')
        features = [
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [row['longitude'], row['latitude']]
                },
                'properties': {
                    'point_id': row['point_id'],
                    'offshore_distance_km': row['offshore_distance_km'],
                    'coastal_lon': row['coastal_lon'],
                    'coastal_lat': row['coastal_lat'],
                }
            }
            for _, row in grid_df.iterrows()
        ]
        with open(geojson_path, 'w') as f:
            json.dump(
                {'type': 'FeatureCollection', 'features': features},
                f, indent=2
            )
        print(f"GeoJSON saved to: {geojson_path}")
        return output_path

    def print_stats(self, grid_df):
        print("\n" + "=" * 60)
        print(f"FINAL SAMPLING GRID STATISTICS — {STATE_NAME}")
        print("=" * 60)
        print(f"Total offshore points : {len(grid_df)}")
        print(f"\nBreakdown by offshore distance:")
        for dist in sorted(grid_df['offshore_distance_km'].unique()):
            n = (grid_df['offshore_distance_km'] == dist).sum()
            print(f"  {dist:3.0f} km : {n} points")
        print(f"\nLatitude  range : {grid_df['latitude'].min():.3f}° – "
              f"{grid_df['latitude'].max():.3f}°N")
        print(f"Longitude range : {grid_df['longitude'].min():.3f}° – "
              f"{grid_df['longitude'].max():.3f}°E")
        print("=" * 60)


def main():
    print("=" * 60)
    print(f"{STATE_NAME} Coast — Offshore Sampling Grid Generator")
    print("(Using actual GEE coastline + ocean validation)")
    print("=" * 60 + "\n")

    generator = SamplingGridGenerator()
    grid_df = generator.generate_grid()
    generator.print_stats(grid_df)
    generator.save_grid(grid_df)

    print(f"\nNext step: extract SAR timeseries with this grid.")


if __name__ == "__main__":
    main()