import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import config

class WindSpeedCalculator:
    """
    Calculate 10m wind speed from SAR backscatter using empirical CMOD-like relationships
    """
    
    def __init__(self, data_file):
        """
        Load SAR time series data
        
        Args:
            data_file: Path to CSV file with SAR time series
        """
        print(f"Loading data from: {data_file}")
        self.df = pd.read_csv(data_file)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Load grid file to get coordinates
        grid_file = os.path.join(config.RAW_DATA_DIR, 'gujarat_sampling_grid.csv')
        grid_df = pd.read_csv(grid_file)
        
        # Merge to add longitude/latitude
        self.df = self.df.merge(
            grid_df[['point_id', 'longitude', 'latitude']], 
            on='point_id', 
            how='left'
        )
        
        print(f"Loaded {len(self.df)} samples")
    
    def cmod5n_simplified(self, sigma0_vv, incidence_angle):
        """
        Empirical C-band VV GMF inversion for 10m wind speed.
        Calibrated so that:
        sigma0 = -20 dB → ~5 m/s
        sigma0 = -15 dB → ~8.5 m/s  
        sigma0 = -10 dB → ~15 m/s
        With incidence angle correction relative to 35° reference.
        """
        # Incidence angle correction (~0.25 dB/degree relative to 35°)
        angle_correction = 0.25 * (incidence_angle - 35)
        sigma0_adj = sigma0_vv - angle_correction

        # Log-space power law inversion: sigma0_dB = -34.68 + 21 * log10(U10)
        wind_speed = np.power(10.0, (sigma0_adj + 34.68) / 21.0)

        # Physical constraints
        wind_speed = np.clip(wind_speed, 0.5, 30.0)

        return wind_speed
    
    def empirical_formula(self, sigma0_vv, incidence_angle):
        """
        Alternative empirical formula based on literature
        
        Based on: Monaldo et al. (2001) and Lehner et al. (1998)
        
        Args:
            sigma0_vv: VV backscatter in dB
            incidence_angle: Incidence angle in degrees
            
        Returns:
            Wind speed at 10m in m/s
        """
        # This formula works well for moderate incidence angles (25-40°)
        # U10 = a * exp(b * sigma0_vv) + c
        
        # Coefficients adjusted for typical ocean conditions
        a = 2.5
        b = 0.11
        c = 2.0
        
        # Incidence angle correction
        angle_factor = 1 + 0.01 * (incidence_angle - 35)
        
        wind_speed = (a * np.exp(b * sigma0_vv) + c) * angle_factor
        
        # Physical constraints
        wind_speed = np.clip(wind_speed, 0, 30)
        
        return wind_speed
    
    def calculate_wind_speeds(self, method='cmod5n'):
        """
        Calculate 10m wind speed for all data points
        
        Args:
            method: 'cmod5n' or 'empirical'
        """
        print(f"\nCalculating 10m wind speed using {method} method...")
        
        if method == 'cmod5n':
            self.df['wind_speed_10m'] = self.cmod5n_simplified(
                self.df['VV'], 
                self.df['incidence_angle']
            )
        elif method == 'empirical':
            self.df['wind_speed_10m'] = self.empirical_formula(
                self.df['VV'], 
                self.df['incidence_angle']
            )
        else:
            raise ValueError("Method must be 'cmod5n' or 'empirical'")
        
        print(f"Wind speed range: {self.df['wind_speed_10m'].min():.2f} to {self.df['wind_speed_10m'].max():.2f} m/s")
        print(f"Mean wind speed: {self.df['wind_speed_10m'].mean():.2f} m/s")
        
        return self.df
    
    def create_summary_table(self):
        """
        Create comprehensive summary tables
        """
        print("\nGenerating summary tables...")
        
        # 1. Overall statistics
        overall_stats = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th %ile', '75th %ile'],
            'Wind Speed (m/s)': [
                self.df['wind_speed_10m'].mean(),
                self.df['wind_speed_10m'].median(),
                self.df['wind_speed_10m'].std(),
                self.df['wind_speed_10m'].min(),
                self.df['wind_speed_10m'].max(),
                self.df['wind_speed_10m'].quantile(0.25),
                self.df['wind_speed_10m'].quantile(0.75)
            ]
        })
        
        # 2. By offshore distance
        offshore_stats = self.df.groupby('offshore_distance_km')['wind_speed_10m'].agg([
            ('Mean (m/s)', 'mean'),
            ('Median (m/s)', 'median'),
            ('Std Dev (m/s)', 'std'),
            ('Min (m/s)', 'min'),
            ('Max (m/s)', 'max'),
            ('Sample Count', 'count')
        ]).round(2)
        
        # 3. By month (seasonal analysis)
        monthly_stats = self.df.groupby('month')['wind_speed_10m'].agg([
            ('Mean (m/s)', 'mean'),
            ('Median (m/s)', 'median'),
            ('Std Dev (m/s)', 'std')
        ]).round(2)
        monthly_stats.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # 4. By year
        yearly_stats = self.df.groupby('year')['wind_speed_10m'].agg([
            ('Mean (m/s)', 'mean'),
            ('Median (m/s)', 'median'),
            ('Std Dev (m/s)', 'std'),
            ('Sample Count', 'count')
        ]).round(2)
        
        # 5. By location (top 10 highest and lowest mean wind speed points)
        location_stats = self.df.groupby(['point_id', 'latitude', 'longitude', 'offshore_distance_km'])['wind_speed_10m'].agg([
            ('Mean (m/s)', 'mean'),
            ('Std Dev (m/s)', 'std'),
            ('Sample Count', 'count')
        ]).round(2).reset_index()
        
        top_10_locations = location_stats.nlargest(10, 'Mean (m/s)')
        bottom_10_locations = location_stats.nsmallest(10, 'Mean (m/s)')
        
        # Save tables
        output_dir = config.OUTPUT_DIR
        
        overall_stats.to_csv(os.path.join(output_dir, 'wind_speed_overall_stats.csv'), index=False)
        offshore_stats.to_csv(os.path.join(output_dir, 'wind_speed_by_offshore_distance.csv'))
        monthly_stats.to_csv(os.path.join(output_dir, 'wind_speed_by_month.csv'))
        yearly_stats.to_csv(os.path.join(output_dir, 'wind_speed_by_year.csv'))
        top_10_locations.to_csv(os.path.join(output_dir, 'top_10_wind_speed_locations.csv'), index=False)
        bottom_10_locations.to_csv(os.path.join(output_dir, 'bottom_10_wind_speed_locations.csv'), index=False)
        
        print("\nTables saved to outputs directory")
        
        # Print to console
        print("\n" + "="*70)
        print("OVERALL WIND SPEED STATISTICS")
        print("="*70)
        print(overall_stats.to_string(index=False))
        
        print("\n" + "="*70)
        print("WIND SPEED BY OFFSHORE DISTANCE")
        print("="*70)
        print(offshore_stats.to_string())
        
        print("\n" + "="*70)
        print("SEASONAL WIND SPEED PATTERNS")
        print("="*70)
        print(monthly_stats.to_string())
        
        print("\n" + "="*70)
        print("WIND SPEED BY YEAR")
        print("="*70)
        print(yearly_stats.to_string())
        
        print("\n" + "="*70)
        print("TOP 10 LOCATIONS (Highest Mean Wind Speed)")
        print("="*70)
        print(top_10_locations.to_string(index=False))
        
        print("\n" + "="*70)
        print("BOTTOM 10 LOCATIONS (Lowest Mean Wind Speed)")
        print("="*70)
        print(bottom_10_locations.to_string(index=False))
        
        return {
            'overall': overall_stats,
            'offshore': offshore_stats,
            'monthly': monthly_stats,
            'yearly': yearly_stats,
            'top_10': top_10_locations,
            'bottom_10': bottom_10_locations
        }
    
    def create_visualizations(self):
        """
        Create visualizations of wind speed estimates
        """
        print("\nGenerating wind speed visualizations...")
        
        fig = plt.figure(figsize=(16, 12))
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle('10m Wind Speed Estimates from SAR Data\nGujarat Offshore Region (2020-2024)',
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Wind speed distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(self.df['wind_speed_10m'], bins=50, color='skyblue', 
                edgecolor='black', alpha=0.7)
        ax1.axvline(self.df['wind_speed_10m'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f"Mean: {self.df['wind_speed_10m'].mean():.2f} m/s")
        ax1.set_xlabel('Wind Speed (m/s)', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('Wind Speed Distribution', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Seasonal pattern
        ax2 = fig.add_subplot(gs[0, 1])
        monthly_mean = self.df.groupby('month')['wind_speed_10m'].mean()
        monthly_std = self.df.groupby('month')['wind_speed_10m'].std()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax2.errorbar(range(1, 13), monthly_mean, yerr=monthly_std, 
                    marker='o', capsize=5, linewidth=2, markersize=8, color='darkgreen')
        ax2.set_xlabel('Month', fontweight='bold')
        ax2.set_ylabel('Wind Speed (m/s)', fontweight='bold')
        ax2.set_title('Seasonal Wind Speed Pattern', fontweight='bold')
        ax2.set_xticks(range(1, 13))
        ax2.set_xticklabels(months, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axvspan(6, 9, alpha=0.2, color='cyan', label='Monsoon')
        ax2.legend()
        
        # 3. Spatial distribution
        ax3 = fig.add_subplot(gs[0, 2])
        point_means = self.df.groupby('point_id').agg({
            'longitude': 'first',
            'latitude': 'first',
            'wind_speed_10m': 'mean'
        })
        scatter = ax3.scatter(point_means['longitude'], point_means['latitude'],
                            c=point_means['wind_speed_10m'], cmap='jet', 
                            s=100, edgecolors='black', linewidth=0.5, vmin=5, vmax=12)
        ax3.set_xlabel('Longitude (°E)', fontweight='bold')
        ax3.set_ylabel('Latitude (°N)', fontweight='bold')
        ax3.set_title('Mean Wind Speed Distribution', fontweight='bold')
        plt.colorbar(scatter, ax=ax3, label='Wind Speed (m/s)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Time series (sample location)
        ax4 = fig.add_subplot(gs[1, :2])
        sample_point = self.df['point_id'].iloc[0]
        sample_data = self.df[self.df['point_id'] == sample_point].sort_values('timestamp')
        ax4.plot(sample_data['timestamp'], sample_data['wind_speed_10m'], 
                alpha=0.6, linewidth=0.8, color='steelblue')
        # Add rolling mean
        if len(sample_data) > 30:
            rolling = sample_data.set_index('timestamp')['wind_speed_10m'].rolling(window=30, center=True).mean()
            ax4.plot(rolling.index, rolling.values, color='red', linewidth=2, label='30-day avg')
        ax4.set_xlabel('Date', fontweight='bold')
        ax4.set_ylabel('Wind Speed (m/s)', fontweight='bold')
        ax4.set_title(f'Wind Speed Time Series - {sample_point}', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Offshore distance comparison
        ax5 = fig.add_subplot(gs[1, 2])
        offshore_data = [self.df[self.df['offshore_distance_km'] == d]['wind_speed_10m'].values 
                        for d in sorted(self.df['offshore_distance_km'].unique())]
        bp = ax5.boxplot(offshore_data, 
                        labels=[f'{int(d)}km' for d in sorted(self.df['offshore_distance_km'].unique())],
                        patch_artist=True, showfliers=False)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax5.set_xlabel('Offshore Distance', fontweight='bold')
        ax5.set_ylabel('Wind Speed (m/s)', fontweight='bold')
        ax5.set_title('Wind Speed vs Offshore Distance', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. VV backscatter vs Wind Speed
        ax6 = fig.add_subplot(gs[2, 0])
        sample_df = self.df.sample(min(5000, len(self.df)))
        scatter = ax6.scatter(sample_df['VV'], sample_df['wind_speed_10m'],
                            c=sample_df['incidence_angle'], cmap='viridis',
                            s=10, alpha=0.5)
        ax6.set_xlabel('VV Backscatter (dB)', fontweight='bold')
        ax6.set_ylabel('Wind Speed (m/s)', fontweight='bold')
        ax6.set_title('Backscatter vs Wind Speed', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax6, label='Incidence Angle (°)')
        
        # 7. Wind speed by year
        ax7 = fig.add_subplot(gs[2, 1])
        yearly_data = [self.df[self.df['year'] == y]['wind_speed_10m'].values 
                      for y in sorted(self.df['year'].unique())]
        ax7.boxplot(yearly_data, labels=sorted(self.df['year'].unique()),
                   patch_artist=True, showfliers=False)
        ax7.set_xlabel('Year', fontweight='bold')
        ax7.set_ylabel('Wind Speed (m/s)', fontweight='bold')
        ax7.set_title('Annual Wind Speed Variation', fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        
        # 8. Wind speed classes
        ax8 = fig.add_subplot(gs[2, 2])
        bins = [0, 3, 6, 9, 12, 15, 30]
        labels = ['<3 m/s', '3-6 m/s', '6-9 m/s', '9-12 m/s', '12-15 m/s', '>15 m/s']
        self.df['wind_class'] = pd.cut(self.df['wind_speed_10m'], bins=bins, labels=labels)
        class_counts = self.df['wind_class'].value_counts().sort_index()
        colors_pie = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b']
        ax8.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%',
               colors=colors_pie, startangle=90)
        ax8.set_title('Wind Speed Classes', fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(config.OUTPUT_DIR, 'wind_speed_analysis.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to: {output_path}")
        
        return fig
    
    def save_wind_speed_data(self, filename='gujarat_sar_with_windspeed.csv'):
        """
        Save the dataframe with calculated wind speeds
        """
        output_path = os.path.join(config.PROCESSED_DATA_DIR, filename)
        self.df.to_csv(output_path, index=False)
        print(f"\nData with wind speeds saved to: {output_path}")
        return output_path


def main():
    """
    Main execution
    """
    print("="*70)
    print("10m Wind Speed Calculation from SAR Backscatter")
    print("Using CMOD5.N-like Empirical Relationship")
    print("="*70)
    
    # Load data
    data_file = os.path.join(config.PROCESSED_DATA_DIR, 'gujarat_sar_timeseries.csv')
    
    if not os.path.exists(data_file):
        print(f"ERROR: Data file not found: {data_file}")
        print("Please run extract_sar_timeseries.py first")
        return
    
    # Initialize calculator
    calc = WindSpeedCalculator(data_file)
    
    # Calculate wind speeds
    calc.calculate_wind_speeds(method='cmod5n')
    
    # Create summary tables
    tables = calc.create_summary_table()
    
    # Create visualizations
    calc.create_visualizations()
    
    # Save data with wind speeds
    calc.save_wind_speed_data()
    
    print("\n" + "="*70)
    print("Wind speed calculation completed!")
    print("="*70)
    print("\nGenerated files:")
    print("  - wind_speed_analysis.png (comprehensive visualization)")
    print("  - wind_speed_overall_stats.csv")
    print("  - wind_speed_by_offshore_distance.csv")
    print("  - wind_speed_by_month.csv")
    print("  - wind_speed_by_year.csv")
    print("  - top_10_wind_speed_locations.csv")
    print("  - bottom_10_wind_speed_locations.csv")
    print("  - gujarat_sar_with_windspeed.csv (full dataset)")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()