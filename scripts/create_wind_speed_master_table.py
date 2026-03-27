import pandas as pd
import numpy as np
import os
import sys 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import config

class WindSpeedMasterTable:
    """
    Create single comprehensive wind speed table
    """
    
    def __init__(self, data_file):
        """Load data with wind speeds"""
        print(f"Loading data from: {data_file}")
        self.df = pd.read_csv(data_file)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        print(f"Loaded {len(self.df)} samples")
    
    def create_master_table(self):
        """
        Create comprehensive master table with all wind speed data
        """
        print("\nCreating master wind speed table...")
        
        # Create the master table with clean column names
        master_table = self.df[[
            'timestamp',
            'point_id',
            'latitude',
            'longitude',
            'offshore_distance_km',
            'VV',
            'VH',
            'VH_VV_ratio',
            'incidence_angle',
            'wind_speed_10m',
            'year',
            'month',
            'day'
        ]].copy()
        
        # Rename columns for clarity
        master_table.columns = [
            'Timestamp',
            'Location_ID',
            'Latitude_deg',
            'Longitude_deg',
            'Offshore_Distance_km',
            'VV_Backscatter_dB',
            'VH_Backscatter_dB',
            'VH_VV_Ratio',
            'Incidence_Angle_deg',
            'Wind_Speed_10m_ms',
            'Year',
            'Month',
            'Day'
        ]
        
        # Add season column
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Pre-Monsoon'
            elif month in [6, 7, 8, 9]:
                return 'Monsoon'
            else:
                return 'Post-Monsoon'
        
        master_table['Season'] = master_table['Month'].apply(get_season)
        
        # Round numerical values for readability
        master_table['Latitude_deg'] = master_table['Latitude_deg'].round(4)
        master_table['Longitude_deg'] = master_table['Longitude_deg'].round(4)
        master_table['VV_Backscatter_dB'] = master_table['VV_Backscatter_dB'].round(2)
        master_table['VH_Backscatter_dB'] = master_table['VH_Backscatter_dB'].round(2)
        master_table['VH_VV_Ratio'] = master_table['VH_VV_Ratio'].round(3)
        master_table['Incidence_Angle_deg'] = master_table['Incidence_Angle_deg'].round(2)
        master_table['Wind_Speed_10m_ms'] = master_table['Wind_Speed_10m_ms'].round(2)
        
        # Sort by location and date
        master_table = master_table.sort_values(['Location_ID', 'Timestamp'])
        
        # Save
        output_path = os.path.join(config.OUTPUT_DIR, 'Gujarat_10m_WindSpeed_MasterTable.csv')
        master_table.to_csv(output_path, index=False)
        
        print(f"\nMaster table saved to: {output_path}")
        print(f"Total records: {len(master_table):,}")
        
        # Print summary statistics
        print("\n" + "="*80)
        print("MASTER TABLE SUMMARY")
        print("="*80)
        print(f"Time Period: {master_table['Timestamp'].min()} to {master_table['Timestamp'].max()}")
        print(f"Number of Locations: {master_table['Location_ID'].nunique()}")
        print(f"Total Observations: {len(master_table):,}")
        print(f"\nWind Speed Statistics:")
        print(f"  Mean:   {master_table['Wind_Speed_10m_ms'].mean():.2f} m/s")
        print(f"  Median: {master_table['Wind_Speed_10m_ms'].median():.2f} m/s")
        print(f"  Min:    {master_table['Wind_Speed_10m_ms'].min():.2f} m/s")
        print(f"  Max:    {master_table['Wind_Speed_10m_ms'].max():.2f} m/s")
        print(f"  Std:    {master_table['Wind_Speed_10m_ms'].std():.2f} m/s")
        
        print("\n" + "="*80)
        print("SAMPLE DATA (First 10 rows)")
        print("="*80)
        print(master_table.head(10).to_string(index=False))
        print("="*80)
        
        return master_table, output_path
    
    def create_summary_statistics_table(self, master_table):
        """
        Create a single summary statistics table
        """
        print("\nCreating summary statistics table...")
        
        summary_data = []
        
        # Overall statistics
        summary_data.append({
            'Category': 'Overall',
            'Subcategory': 'All Data',
            'Sample_Count': len(master_table),
            'Mean_Wind_Speed_ms': master_table['Wind_Speed_10m_ms'].mean(),
            'Median_Wind_Speed_ms': master_table['Wind_Speed_10m_ms'].median(),
            'Std_Dev_ms': master_table['Wind_Speed_10m_ms'].std(),
            'Min_Wind_Speed_ms': master_table['Wind_Speed_10m_ms'].min(),
            'Max_Wind_Speed_ms': master_table['Wind_Speed_10m_ms'].max()
        })
        
        # By offshore distance
        for dist in sorted(master_table['Offshore_Distance_km'].unique()):
            subset = master_table[master_table['Offshore_Distance_km'] == dist]
            summary_data.append({
                'Category': 'Offshore Distance',
                'Subcategory': f'{int(dist)} km',
                'Sample_Count': len(subset),
                'Mean_Wind_Speed_ms': subset['Wind_Speed_10m_ms'].mean(),
                'Median_Wind_Speed_ms': subset['Wind_Speed_10m_ms'].median(),
                'Std_Dev_ms': subset['Wind_Speed_10m_ms'].std(),
                'Min_Wind_Speed_ms': subset['Wind_Speed_10m_ms'].min(),
                'Max_Wind_Speed_ms': subset['Wind_Speed_10m_ms'].max()
            })
        
        # By season
        for season in ['Winter', 'Pre-Monsoon', 'Monsoon', 'Post-Monsoon']:
            subset = master_table[master_table['Season'] == season]
            summary_data.append({
                'Category': 'Season',
                'Subcategory': season,
                'Sample_Count': len(subset),
                'Mean_Wind_Speed_ms': subset['Wind_Speed_10m_ms'].mean(),
                'Median_Wind_Speed_ms': subset['Wind_Speed_10m_ms'].median(),
                'Std_Dev_ms': subset['Wind_Speed_10m_ms'].std(),
                'Min_Wind_Speed_ms': subset['Wind_Speed_10m_ms'].min(),
                'Max_Wind_Speed_ms': subset['Wind_Speed_10m_ms'].max()
            })
        
        # By year
        for year in sorted(master_table['Year'].unique()):
            subset = master_table[master_table['Year'] == year]
            summary_data.append({
                'Category': 'Year',
                'Subcategory': str(year),
                'Sample_Count': len(subset),
                'Mean_Wind_Speed_ms': subset['Wind_Speed_10m_ms'].mean(),
                'Median_Wind_Speed_ms': subset['Wind_Speed_10m_ms'].median(),
                'Std_Dev_ms': subset['Wind_Speed_10m_ms'].std(),
                'Min_Wind_Speed_ms': subset['Wind_Speed_10m_ms'].min(),
                'Max_Wind_Speed_ms': subset['Wind_Speed_10m_ms'].max()
            })
        
        # Create dataframe
        summary_df = pd.DataFrame(summary_data)
        
        # Round values
        summary_df['Mean_Wind_Speed_ms'] = summary_df['Mean_Wind_Speed_ms'].round(2)
        summary_df['Median_Wind_Speed_ms'] = summary_df['Median_Wind_Speed_ms'].round(2)
        summary_df['Std_Dev_ms'] = summary_df['Std_Dev_ms'].round(2)
        summary_df['Min_Wind_Speed_ms'] = summary_df['Min_Wind_Speed_ms'].round(2)
        summary_df['Max_Wind_Speed_ms'] = summary_df['Max_Wind_Speed_ms'].round(2)
        
        # Save
        output_path = os.path.join(config.OUTPUT_DIR, 'Gujarat_10m_WindSpeed_Summary.csv')
        summary_df.to_csv(output_path, index=False)
        
        print(f"Summary statistics saved to: {output_path}")
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)
        
        return summary_df


def main():
    """
    Main execution
    """
    print("="*80)
    print("Creating Master Wind Speed Table")
    print("Gujarat Offshore Wind Resource Assessment")
    print("="*80)
    
    # Load data with wind speeds
    data_file = os.path.join(config.PROCESSED_DATA_DIR, 'gujarat_sar_with_windspeed.csv')
    
    if not os.path.exists(data_file):
        print(f"ERROR: Wind speed data file not found: {data_file}")
        print("Please run calculate_wind_speed_10m.py first")
        return
    
    # Create master table
    creator = WindSpeedMasterTable(data_file)
    master_table, master_path = creator.create_master_table()
    summary_table = creator.create_summary_statistics_table(master_table)
    
    print("\n" + "="*80)
    print("COMPLETED!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  1. Gujarat_10m_WindSpeed_MasterTable.csv")
    print(f"     - Complete dataset with all {len(master_table):,} observations")
    print(f"     - Clean column names, organized by location and time")
    print(f"  2. Gujarat_10m_WindSpeed_Summary.csv")
    print(f"     - Summary statistics by offshore distance, season, and year")
    print("="*80)


if __name__ == "__main__":
    main()