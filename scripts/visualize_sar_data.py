import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import config

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

class SARDataVisualizer:
    """
    Create impressive visualizations of SAR time series data
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
        grid_df = pd.read_csv(grid_file)[['point_id', 'longitude', 'latitude', 'offshore_distance_km']]
        
        # Merge coordinates into main dataframe
        self.df = self.df.merge(grid_df, on='point_id', how='left', suffixes=('', '_grid'))
        
        # Use grid offshore_distance if available
        if 'offshore_distance_km_grid' in self.df.columns:
            self.df['offshore_distance_km'] = self.df['offshore_distance_km_grid']
            self.df = self.df.drop('offshore_distance_km_grid', axis=1)
        
        print(f"Loaded {len(self.df)} samples")
        print(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        
    def create_overview_dashboard(self):
        """
        Create comprehensive overview dashboard
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Gujarat Offshore Wind Resource Assessment\nSentinel-1 SAR Data Analysis (2020-2024)', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Time series for sample points (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_sample_timeseries(ax1)
        
        # 2. Spatial distribution map (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_spatial_coverage(ax2)
        
        # 3. Monthly sampling distribution (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_temporal_distribution(ax3)
        
        # 4. VV vs VH scatter (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_vv_vh_relationship(ax4)
        
        # 5. Seasonal patterns (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_seasonal_patterns(ax5)
        
        # 6. Offshore distance comparison (bottom left)
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_offshore_comparison(ax6)
        
        # 7. VH/VV ratio distribution (bottom center)
        ax7 = fig.add_subplot(gs[2, 1])
        self._plot_ratio_distribution(ax7)
        
        # 8. Data quality metrics (bottom right)
        ax8 = fig.add_subplot(gs[2, 2])
        self._plot_data_quality(ax8)
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(config.OUTPUT_DIR, 'sar_analysis_dashboard.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"\nDashboard saved to: {output_path}")
        
        return fig
    
    def _plot_sample_timeseries(self, ax):
        """
        Plot time series for multiple sample points
        """
        # Select 4 representative points at different offshore distances
        sample_points = []
        for dist in [20, 40, 60, 80]:
            point = self.df[self.df['offshore_distance_km'] == dist]['point_id'].iloc[0]
            sample_points.append(point)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        for point, color in zip(sample_points, colors):
            data = self.df[self.df['point_id'] == point].sort_values('timestamp')
            offshore_dist = data['offshore_distance_km'].iloc[0]
            
            ax.plot(data['timestamp'], data['VV'], 
                   label=f'{point} ({offshore_dist}km offshore)', 
                   alpha=0.7, linewidth=1, color=color)
        
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('VV Backscatter (dB)', fontweight='bold')
        ax.set_title('SAR Time Series at Sample Locations', fontweight='bold', pad=10)
        ax.legend(loc='upper right', frameon=True, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _plot_spatial_coverage(self, ax):
        """
        Plot spatial distribution of sampling points
        """
        # Create scatter plot colored by offshore distance
        scatter = ax.scatter(self.df.groupby('point_id')['longitude'].first(),
                        self.df.groupby('point_id')['latitude'].first(),
                        c=self.df.groupby('point_id')['offshore_distance_km'].first(),
                        cmap='viridis', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Longitude (°E)', fontweight='bold')
        ax.set_ylabel('Latitude (°N)', fontweight='bold')
        ax.set_title('Sampling Grid Coverage', fontweight='bold', pad=10)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='Offshore Distance (km)')
        
        # Add accurate Gujarat coastline
        coastline = config.get_gujarat_coastline()
        ax.plot(coastline['longitude'], coastline['latitude'],
            'r-', linewidth=2, alpha=0.7, label='Gujarat Coastline')
        
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    
    def _plot_temporal_distribution(self, ax):
        """
        Plot temporal sampling distribution
        """
        monthly_counts = self.df.groupby(['year', 'month']).size()
        
        ax.bar(range(len(monthly_counts)), monthly_counts.values, 
               color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Time Period', fontweight='bold')
        ax.set_ylabel('Number of Samples', fontweight='bold')
        ax.set_title('Temporal Data Distribution', fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set x-ticks to show years
        year_positions = []
        year_labels = []
        for i, (year, month) in enumerate(monthly_counts.index):
            if month == 1:
                year_positions.append(i)
                year_labels.append(str(year))
        
        ax.set_xticks(year_positions)
        ax.set_xticklabels(year_labels)
    
    def _plot_vv_vh_relationship(self, ax):
        """
        Plot VV vs VH backscatter relationship
        """
        # Sample for performance (plot 5000 random points)
        sample_df = self.df.sample(min(5000, len(self.df)))
        
        scatter = ax.scatter(sample_df['VV'], sample_df['VH'], 
                           c=sample_df['offshore_distance_km'], 
                           cmap='plasma', s=10, alpha=0.5)
        
        ax.set_xlabel('VV Backscatter (dB)', fontweight='bold')
        ax.set_ylabel('VH Backscatter (dB)', fontweight='bold')
        ax.set_title('VV-VH Polarization Relationship', fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        
        # Add 1:1 line
        lims = [max(ax.get_xlim()[0], ax.get_ylim()[0]),
                min(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'r--', alpha=0.5, linewidth=1, label='1:1 line')
        ax.legend(fontsize=7)
        
        plt.colorbar(scatter, ax=ax, label='Offshore Dist. (km)')
    
    def _plot_seasonal_patterns(self, ax):
        """
        Plot seasonal patterns in backscatter
        """
        # Group by month
        monthly_vv = self.df.groupby('month')['VV'].agg(['mean', 'std'])
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        ax.errorbar(range(1, 13), monthly_vv['mean'], yerr=monthly_vv['std'],
                   marker='o', capsize=5, capthick=2, linewidth=2, 
                   markersize=6, color='darkblue', label='VV ± σ')
        
        ax.set_xlabel('Month', fontweight='bold')
        ax.set_ylabel('Mean VV Backscatter (dB)', fontweight='bold')
        ax.set_title('Seasonal Backscatter Pattern', fontweight='bold', pad=10)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(months, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Highlight monsoon season
        ax.axvspan(6, 9, alpha=0.2, color='cyan', label='Monsoon')
        ax.legend(fontsize=7)
    
    def _plot_offshore_comparison(self, ax):
        """
        Compare backscatter at different offshore distances
        """
        offshore_dists = sorted(self.df['offshore_distance_km'].unique())
        
        data_to_plot = [self.df[self.df['offshore_distance_km'] == dist]['VV'].values 
                       for dist in offshore_dists]
        
        bp = ax.boxplot(data_to_plot, labels=[f'{int(d)}km' for d in offshore_dists],
                       patch_artist=True, showfliers=False)
        
        # Color boxes
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Offshore Distance', fontweight='bold')
        ax.set_ylabel('VV Backscatter (dB)', fontweight='bold')
        ax.set_title('Backscatter vs Offshore Distance', fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_ratio_distribution(self, ax):
        """
        Plot VH/VV ratio distribution
        """
        # Filter outliers for better visualization
        ratio_data = self.df['VH_VV_ratio']
        q1, q99 = ratio_data.quantile([0.01, 0.99])
        filtered_ratio = ratio_data[(ratio_data >= q1) & (ratio_data <= q99)]
        
        ax.hist(filtered_ratio, bins=50, color='coral', alpha=0.7, 
               edgecolor='black', linewidth=0.5)
        
        # Add mean line
        mean_ratio = filtered_ratio.mean()
        ax.axvline(mean_ratio, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_ratio:.3f}')
        
        ax.set_xlabel('VH/VV Ratio', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Cross-Polarization Ratio Distribution', fontweight='bold', pad=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_data_quality(self, ax):
        """
        Display data quality metrics
        """
        # Calculate metrics
        total_points = self.df['point_id'].nunique()
        total_samples = len(self.df)
        date_range = (self.df['timestamp'].max() - self.df['timestamp'].min()).days
        avg_samples_per_point = total_samples / total_points
        coverage = (total_samples / (total_points * date_range)) * 12  # Expected: ~every 12 days
        
        metrics = {
            'Total Sampling Points': total_points,
            'Total Observations': f'{total_samples:,}',
            'Time Span (days)': date_range,
            'Avg. Samples/Point': f'{avg_samples_per_point:.1f}',
            'Temporal Coverage': f'{coverage*100:.1f}%',
            'Years Covered': '2020-2024'
        }
        
        # Remove axes
        ax.axis('off')
        
        # Create table
        y_position = 0.9
        ax.text(0.5, 0.95, 'Dataset Summary', ha='center', va='top',
               fontsize=12, fontweight='bold', transform=ax.transAxes)
        
        for key, value in metrics.items():
            ax.text(0.1, y_position, f'{key}:', ha='left', va='top',
                   fontweight='bold', fontsize=9, transform=ax.transAxes)
            ax.text(0.9, y_position, f'{value}', ha='right', va='top',
                   fontsize=9, transform=ax.transAxes)
            y_position -= 0.12
        
        # Add a box around it
        from matplotlib.patches import Rectangle
        rect = Rectangle((0.05, 0.05), 0.9, 0.9, linewidth=2, 
                        edgecolor='black', facecolor='lightgray', 
                        alpha=0.2, transform=ax.transAxes)
        ax.add_patch(rect)
    
    def create_detailed_timeseries_plot(self):
        """
        Create detailed time series plot for selected points
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Detailed SAR Time Series Analysis\nSample Locations at Different Offshore Distances',
                    fontsize=14, fontweight='bold')
        
        # One plot for each offshore distance
        offshore_dists = sorted(self.df['offshore_distance_km'].unique())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        for idx, (dist, color) in enumerate(zip(offshore_dists, colors)):
            ax = axes[idx // 2, idx % 2]
            
            # Get a sample point at this distance
            point_id = self.df[self.df['offshore_distance_km'] == dist]['point_id'].iloc[0]
            data = self.df[self.df['point_id'] == point_id].sort_values('timestamp')
            
            # Plot VV and VH
            ax.plot(data['timestamp'], data['VV'], label='VV', 
                   color=color, linewidth=1.5, alpha=0.8)
            ax.plot(data['timestamp'], data['VH'], label='VH', 
                   color=color, linewidth=1.5, alpha=0.5, linestyle='--')
            
            # Add rolling mean
            if len(data) > 30:
                rolling_vv = data.set_index('timestamp')['VV'].rolling(window=30, center=True).mean()
                ax.plot(rolling_vv.index, rolling_vv.values, 
                       color='black', linewidth=2, alpha=0.7, label='30-day avg')
            
            ax.set_xlabel('Date', fontweight='bold')
            ax.set_ylabel('Backscatter (dB)', fontweight='bold')
            ax.set_title(f'{point_id} - {int(dist)}km Offshore', fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(config.OUTPUT_DIR, 'detailed_timeseries.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Detailed time series saved to: {output_path}")
        
        return fig
    
    def create_spatial_heatmap(self):
        """
        Create spatial heatmap of average backscatter
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Calculate mean VV for each point
        point_stats = self.df.groupby('point_id').agg({
            'longitude': 'first',
            'latitude': 'first',
            'VV': 'mean',
            'VH': 'mean'
        }).reset_index()
        
        # VV heatmap
        scatter1 = axes[0].scatter(point_stats['longitude'], point_stats['latitude'],
                                  c=point_stats['VV'], cmap='RdYlBu_r', 
                                  s=100, edgecolors='black', linewidth=0.5)
        axes[0].set_xlabel('Longitude (°E)', fontweight='bold')
        axes[0].set_ylabel('Latitude (°N)', fontweight='bold')
        axes[0].set_title('Mean VV Backscatter Distribution', fontweight='bold')
        plt.colorbar(scatter1, ax=axes[0], label='VV (dB)')
        axes[0].grid(True, alpha=0.3)
        
        # VH heatmap
        scatter2 = axes[1].scatter(point_stats['longitude'], point_stats['latitude'],
                                  c=point_stats['VH'], cmap='RdYlBu_r', 
                                  s=100, edgecolors='black', linewidth=0.5)
        axes[1].set_xlabel('Longitude (°E)', fontweight='bold')
        axes[1].set_ylabel('Latitude (°N)', fontweight='bold')
        axes[1].set_title('Mean VH Backscatter Distribution', fontweight='bold')
        plt.colorbar(scatter2, ax=axes[1], label='VH (dB)')
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle('Spatial Distribution of Mean SAR Backscatter\nGujarat Offshore Region (2020-2024)',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(config.OUTPUT_DIR, 'spatial_heatmap.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Spatial heatmap saved to: {output_path}")
        
        return fig


def main():
    """
    Main execution - create all visualizations
    """
    print("="*60)
    print("SAR Data Visualization")
    print("Creating publication-quality figures...")
    print("="*60 + "\n")
    
    # Load data
    data_file = os.path.join(config.PROCESSED_DATA_DIR, 'gujarat_sar_timeseries.csv')
    
    if not os.path.exists(data_file):
        print(f"ERROR: Data file not found: {data_file}")
        print("Please run extract_sar_timeseries.py first")
        return
    
    # Create visualizer
    viz = SARDataVisualizer(data_file)
    
    # Create visualizations
    print("\n1. Creating overview dashboard...")
    viz.create_overview_dashboard()
    
    print("\n2. Creating detailed time series plots...")
    viz.create_detailed_timeseries_plot()
    
    print("\n3. Creating spatial heatmaps...")
    viz.create_spatial_heatmap()
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)
    print(f"\nAll figures saved to: {config.OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - sar_analysis_dashboard.png (comprehensive overview)")
    print("  - detailed_timeseries.png (detailed time series)")
    print("  - spatial_heatmap.png (spatial distribution)")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()