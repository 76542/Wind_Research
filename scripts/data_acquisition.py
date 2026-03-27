import ee
import os
import sys
from datetime import datetime
import config

class SARDataAcquisition:
    """
    Class to handle Sentinel-1 SAR data acquisition for Gujarat coast
    """
    
    def __init__(self):
        """Initialize Earth Engine and set up parameters"""
        ee.Initialize(project=config.PROJECT_ID)
        self.aoi = self._define_study_area()
        
    def _define_study_area(self):
        """
        Define the Area of Interest (AOI) for Gujarat coast
        """
        bounds = config.GUJARAT_BOUNDS
        
        # Create rectangle geometry
        aoi = ee.Geometry.Rectangle([
            bounds['min_lon'], 
            bounds['min_lat'],
            bounds['max_lon'], 
            bounds['max_lat']
        ])
        
        return aoi
    
    def get_sentinel1_collection(self, start_date=None, end_date=None):
        """
        Retrieve and filter Sentinel-1 SAR data collection
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Filtered image collection
        """
        start = start_date or config.START_DATE
        end = end_date or config.END_DATE
        
        print(f"Fetching Sentinel-1 data from {start} to {end}")
        
        # Load Sentinel-1 GRD collection
        collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(self.aoi) \
            .filterDate(start, end) \
            .filter(ee.Filter.eq('instrumentMode', config.SAR_CONFIG['instrument_mode']))
        
        # Filter by orbit direction if specified
        if config.SAR_CONFIG['orbit']:
            collection = collection.filter(
                ee.Filter.eq('orbitProperties_pass', config.SAR_CONFIG['orbit'])
            )
        
        # Filter by polarization - keep images that have both VV and VH
        collection = collection.filter(
            ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')
        ).filter(
            ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')
        )
        
        print(f"Total images found: {collection.size().getInfo()}")
        
        return collection
    
    def process_sar_image(self, image):
        """
        Process individual SAR image - extract bands and apply conversions
        
        Args:
            image: ee.Image
            
        Returns:
            Processed image with metadata
        """
        # Select VV and VH bands
        vv = image.select('VV')
        vh = image.select('VH')
        
        # Calculate VH/VV ratio (useful for wind speed estimation)
        ratio = vh.divide(vv).rename('VH_VV_ratio')
        
        # Get incidence angle
        angle = image.select('angle')
        
        # Combine all bands
        processed = ee.Image.cat([vv, vh, ratio, angle]) \
            .set('system:time_start', image.get('system:time_start')) \
            .set('orbitProperties_pass', image.get('orbitProperties_pass'))
        
        return processed
    
    def create_composite(self, collection, period='monthly'):
        """
        Create temporal composite from collection
        
        Args:
            collection: Image collection
            period: 'monthly', 'seasonal', or 'annual'
            
        Returns:
            Composite image
        """
        if period == 'monthly':
            # Calculate median for the entire period
            composite = collection.median()
        elif period == 'annual':
            composite = collection.mean()
        else:
            composite = collection.median()
        
        return composite.clip(self.aoi)
    
    def export_to_drive(self, image, description, folder=None):
        """
        Export image to Google Drive
        
        Args:
            image: ee.Image to export
            description: Export task description
            folder: Google Drive folder name
        """
        folder_name = folder or config.EXPORT_CONFIG['folder']
        
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=folder_name,
            scale=config.EXPORT_CONFIG['scale'],
            region=self.aoi,
            crs=config.EXPORT_CONFIG['crs'],
            maxPixels=1e13
        )
        
        task.start()
        print(f"Export task '{description}' started")
        print(f"Task ID: {task.id}")
        
        return task
    
    def get_collection_info(self, collection):
        """
        Print information about the image collection
        """
        size = collection.size().getInfo()
        print(f"\nCollection Information:")
        print(f"Total images: {size}")
        
        if size > 0:
            first = ee.Image(collection.first())
            date = datetime.fromtimestamp(
                first.get('system:time_start').getInfo() / 1000
            )
            print(f"First image date: {date.strftime('%Y-%m-%d')}")
            
            bands = first.bandNames().getInfo()
            print(f"Available bands: {bands}")
            
            # Get date range
            dates = collection.aggregate_array('system:time_start').getInfo()
            if dates:
                first_date = datetime.fromtimestamp(min(dates) / 1000)
                last_date = datetime.fromtimestamp(max(dates) / 1000)
                print(f"Date range: {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")


def main():
    """
    Main execution function
    """
    print("="*60)
    print("Sentinel-1 SAR Data Acquisition for Gujarat Coast")
    print("="*60)
    
    # Initialize acquisition object
    sar = SARDataAcquisition()
    
    # Get Sentinel-1 collection
    collection = sar.get_sentinel1_collection()
    
    # Get collection info
    sar.get_collection_info(collection)
    
    # Process all images in collection
    processed_collection = collection.map(sar.process_sar_image)
    
    # Create monthly composite
    print("\nCreating composite image...")
    composite = sar.create_composite(processed_collection, period='monthly')
    
    # Export to Google Drive
    print("\nInitiating export to Google Drive...")
    description = f"Gujarat_SAR_Composite_{config.START_DATE}_{config.END_DATE}"
    task = sar.export_to_drive(composite, description)
    
    print("\n" + "="*60)
    print("Data acquisition pipeline completed!")
    print("Check your Google Drive for export progress")
    print("="*60)


if __name__ == "__main__":
    main()