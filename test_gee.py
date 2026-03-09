import ee

# Initialize Earth Engine with your project
ee.Initialize(project='wind-research')

# Test: Get an image count
s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
print(f"Total Sentinel-1 images in catalog: {s1.size().getInfo()}")
print("Earth Engine authentication successful!")