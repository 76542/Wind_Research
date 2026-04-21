import ee
ee.Initialize(project='wind-research')

india = (ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    .filter(ee.Filter.eq("country_na", "India")).geometry().getInfo())

# Check what coordinates exist in the 18-19.5N range
all_coords = []
def collect(geom):
    gt = geom.get('type')
    raw = geom.get('coordinates', [])
    if gt == 'Polygon':
        for pt in raw[0]: all_coords.append(pt)
    elif gt == 'MultiPolygon':
        for poly in raw:
            for pt in poly[0]: all_coords.append(pt)
    elif gt == 'GeometryCollection':
        for g in geom.get('geometries', []): collect(g)
collect(india)

import pandas as pd
df = pd.DataFrame(all_coords, columns=['lon','lat'])
gap = df[(df.lat > 18.0) & (df.lat < 19.5) & (df.lon > 82)]
print(gap.sort_values('lat'))
print(f"\nLon range in gap zone: {gap.lon.min():.2f} - {gap.lon.max():.2f}")