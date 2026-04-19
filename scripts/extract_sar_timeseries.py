import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.tamilnadu import config

class SARTimeSeriesExtractor:
    """
    Extract Sentinel-1 SAR time series at sampling grid points
    """
    
    def __init__(self, grid_file):
        """
        Initialize with sampling grid
        
        Args:
            grid_file: Path to CSV file with sampling points
        """
        ee.Initialize(project=config.PROJECT_ID)
        
        # Load sampling grid
        self.grid_df = pd.read_csv(grid_file)
        print(f"Loaded {len(self.grid_df)} sampling points")
        
        # Convert to Earth Engine FeatureCollection
        self.sampling_points = self._create_ee_points()
        
    def _create_ee_points(self):
        """
        Convert pandas DataFrame to Earth Engine FeatureCollection
        """
        features = []
        for idx, row in self.grid_df.iterrows():
            point = ee.Geometry.Point([row['longitude'], row['latitude']])
            feature = ee.Feature(point, {
                'point_id': row['point_id'],
                'offshore_distance_km': row['offshore_distance_km']            
            })
            features.append(feature)
        
        return ee.FeatureCollection(features)
    
    def get_sentinel1_collection(self, start_date=None, end_date=None):
        """
        Get Sentinel-1 collection for the study period
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Filtered image collection
        """
        start = start_date or config.START_DATE
        end = end_date or config.END_DATE
        
        print(f"\nFetching Sentinel-1 data from {start} to {end}...")
        
        # Create bounding box for state of interest
        bounds = config.STATE_BOUNDS
        aoi = ee.Geometry.Rectangle([
            bounds['min_lon'], bounds['min_lat'],
            bounds['max_lon'], bounds['max_lat']
        ])
        
        # Load and filter Sentinel-1 collection
        collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(aoi) \
            .filterDate(start, end) \
            .filter(ee.Filter.eq('instrumentMode', config.SAR_CONFIG['instrument_mode'])) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        
        # Filter by orbit if specified
        if config.SAR_CONFIG['orbit']:
            collection = collection.filter(
                ee.Filter.eq('orbitProperties_pass', config.SAR_CONFIG['orbit'])
            )
        
        count = collection.size().getInfo()
        print(f"Found {count} SAR images")
        
        return collection
    
    def extract_point_values(self, image, point):
        """
        Extract SAR values at a specific point
        
        Args:
            image: ee.Image
            point: ee.Feature with point geometry
            
        Returns:
            Dictionary with extracted values
        """
        # Sample the image at point location
        sample = image.sample(
            region=point.geometry(),
            scale=10,  # 10m resolution
            geometries=True
        ).first()
        
        return sample
    
    def process_collection_at_points(self, collection):
        """
        Extract time series for all points from collection
        
        This is the main extraction function
        """
        print("\nExtracting SAR time series at all points...")
        print("This may take several minutes...")
        
        # We'll process in batches to avoid timeout
        # Get image list
        image_list = collection.toList(collection.size())
        n_images = collection.size().getInfo()
        
        print(f"Processing {n_images} images across {len(self.grid_df)} points...")
        
        # Extract for each image
        all_data = []
        
        # Process in monthly chunks to avoid timeout
        start_date = datetime.strptime(config.START_DATE, '%Y-%m-%d')
        end_date = datetime.strptime(config.END_DATE, '%Y-%m-%d')
        
        current_date = start_date
        chunk_count = 0
        
        while current_date < end_date:
            chunk_count += 1
            chunk_end = min(current_date + timedelta(days=90), end_date)  # 3-month chunks
            
            chunk_start_str = current_date.strftime('%Y-%m-%d')
            chunk_end_str = chunk_end.strftime('%Y-%m-%d')
            
            print(f"\nProcessing chunk {chunk_count}: {chunk_start_str} to {chunk_end_str}")
            
            # Get collection for this chunk
            chunk_collection = collection.filterDate(chunk_start_str, chunk_end_str)
            chunk_size = chunk_collection.size().getInfo()
            
            if chunk_size == 0:
                print(f"  No images in this period")
                current_date = chunk_end
                continue
            
            print(f"  Found {chunk_size} images")
            
            # Sample the collection at all points
            sampled = chunk_collection.map(lambda img: img.select(['VV', 'VH', 'angle']))
            
            # Extract values - we'll do this point by point to avoid memory issues
            chunk_data = self._extract_chunk_data(sampled, chunk_start_str, chunk_end_str)
            all_data.extend(chunk_data)
            
            print(f"  Extracted {len(chunk_data)} samples")
            
            current_date = chunk_end
        
        print(f"\nTotal samples extracted: {len(all_data)}")
        
        return pd.DataFrame(all_data)
    
    def _extract_chunk_data(self, collection, start_str, end_str):
        """
        Extract data for a time chunk
        
        This uses a more efficient sampling approach
        """
        # Create a function to extract values at all points for one image
        def sample_image(image):
            # Add ratio band
            vv = image.select('VV')
            vh = image.select('VH')
            ratio = vh.divide(vv).rename('VH_VV_ratio')
            
            img_with_ratio = image.addBands(ratio)
            
            # Sample at all points
            samples = img_with_ratio.sampleRegions(
                collection=self.sampling_points,
                scale=10,
                geometries=True
            )
            
            # Add timestamp
            samples = samples.map(lambda f: f.set('timestamp', image.date().millis()))
            
            return samples
        
        # Map over all images
        all_samples = collection.map(sample_image).flatten()
        
        # Convert to list and get info
        try:
            sample_list = all_samples.getInfo()['features']
            
            # Parse into records
            records = []
            for sample in sample_list:
                props = sample['properties']
                
                # Skip if any required value is None
                if props.get('VV') is None or props.get('VH') is None:
                    continue
                
                record = {
                    'timestamp': datetime.fromtimestamp(props['timestamp'] / 1000),
                    'point_id': props['point_id'],
                    'offshore_distance_km': props['offshore_distance_km'],
                    'VV': props['VV'],
                    'VH': props['VH'],
                    'VH_VV_ratio': props.get('VH_VV_ratio'),
                    'incidence_angle': props.get('angle')
                }
                records.append(record)
            
            return records
            
        except Exception as e:
            print(f"  Error extracting chunk: {e}")
            return []
    
    def save_timeseries(self, df, filename='gujarat_sar_timeseries.csv'):
        """
        Save extracted time series to CSV
        """
        # Add date components for easier analysis
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Sort by timestamp and point
        df = df.sort_values(['point_id', 'timestamp'])
        
        # Save
        output_path = os.path.join(config.PROCESSED_DATA_DIR, filename)
        df.to_csv(output_path, index=False)
        
        print(f"\nTime series data saved to: {output_path}")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Number of unique points: {df['point_id'].nunique()}")
        
        # Print statistics
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"Total samples: {len(df)}")
        print(f"Unique points: {df['point_id'].nunique()}")
        print(f"Samples per point (avg): {len(df) / df['point_id'].nunique():.1f}")
        print(f"\nVV backscatter range: {df['VV'].min():.2f} to {df['VV'].max():.2f} dB")
        print(f"VH backscatter range: {df['VH'].min():.2f} to {df['VH'].max():.2f} dB")
        print(f"VH/VV ratio range: {df['VH_VV_ratio'].min():.3f} to {df['VH_VV_ratio'].max():.3f}")
        
        print("\nSamples by year:")
        print(df['year'].value_counts().sort_index())
        print("="*60)
        
        return output_path


def main():
    """
    Main execution
    """
    print("="*60)
    print("Sentinel-1 SAR Time Series Extraction")
    print("Offshore Wind Resource Assessment")
    print("="*60)
    
    # Load grid
    grid_file = os.path.join(config.RAW_DATA_DIR, 'tamilnadu_sampling_grid.csv')
    
    if not os.path.exists(grid_file):
        print(f"ERROR: Grid file not found: {grid_file}")
        print("Please run create_sampling_grid.py first")
        return
    
    # Initialize extractor
    extractor = SARTimeSeriesExtractor(grid_file)
    
    # Get SAR collection
    collection = extractor.get_sentinel1_collection()
    
    # Extract time series
    timeseries_df = extractor.process_collection_at_points(collection)
    
    # Save results
    if len(timeseries_df) > 0:
        extractor.save_timeseries(timeseries_df, filename='tamilnadu_sar_timeseries.csv')
        print("\n" + "="*60)
        print("Time series extraction completed successfully!")
        print("="*60)
    else:
        print("\nWARNING: No data extracted. Check your parameters.")


if __name__ == "__main__":
    main()