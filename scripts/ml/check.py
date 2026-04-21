import pandas as pd

coast = pd.read_csv("data/raw/andhrapradesh/andhrapradesh_coastline_cache.csv")
print(f"AP coastline lat range: {coast.latitude.min():.3f} to {coast.latitude.max():.3f}")
print(f"\nPoints above 18N:")
print(coast[coast.latitude > 18.0].dropna().head(10))