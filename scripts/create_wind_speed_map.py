import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium import plugins
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import config


def _coastline_to_polylines(coastline_df):
    """Split coastline df (with NaN separators) into list of segments for folium."""
    import math
    segments, current = [], []
    for _, row in coastline_df.iterrows():
        if math.isnan(row['latitude']) or math.isnan(row['longitude']):
            if current:
                segments.append(current)
                current = []
        else:
            current.append([row['latitude'], row['longitude']])
    if current:
        segments.append(current)
    return segments

class WindSpeedMapper:
    """
    Create map visualizations of wind speed data
    """
    
    def __init__(self, data_file):
        """Load master wind speed data"""
        print(f"Loading data from: {data_file}")
        self.df = pd.read_csv(data_file)
        print(f"Loaded {len(self.df):,} observations")
        
        # Calculate mean wind speed for each location
        self.location_stats = self.df.groupby('Location_ID').agg({
            'Latitude_deg': 'first',
            'Longitude_deg': 'first',
            'Offshore_Distance_km': 'first',
            'Wind_Speed_10m_ms': ['mean', 'std', 'min', 'max', 'count']
        }).reset_index()
        
        # Flatten column names
        self.location_stats.columns = ['Location_ID', 'Latitude', 'Longitude', 
                                       'Offshore_Distance_km', 'Mean_Wind_Speed', 
                                       'Std_Wind_Speed', 'Min_Wind_Speed', 
                                       'Max_Wind_Speed', 'Sample_Count']
        
        print(f"Processing {len(self.location_stats)} unique locations")
    
    def create_static_map(self):
        print("\nCreating static map...")
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Plot wind speed points
        scatter = ax.scatter(
            self.location_stats['Longitude'],
            self.location_stats['Latitude'],
            c=self.location_stats['Mean_Wind_Speed'],
            s=150,
            cmap='RdYlGn',
            vmin=6,
            vmax=11,
            edgecolors='black',
            linewidth=1,
            alpha=0.8,
            zorder=5
        )
        
        # Add accurate Gujarat coastline
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scripts import config as cfg
        coastline = cfg.get_gujarat_coastline()
        
        coastline_clean = coastline.dropna()
        ax.plot(coastline_clean['longitude'], coastline_clean['latitude'],
            'k-', linewidth=2, label='Gujarat Coastline', zorder=3)

        # Colorbar, labels, title
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Mean Wind Speed at 10m (m/s)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude (°E)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Latitude (°N)', fontsize=13, fontweight='bold')
        ax.set_title(
            'Gujarat Offshore Wind Speed Distribution at 10m Height\n'
            'Estimated from Sentinel-1 SAR Data (2020-2024)',
            fontsize=14, fontweight='bold', pad=15
        )
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()

        output_path = os.path.join(config.OUTPUT_DIR, 'Gujarat_WindSpeed_Map.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Static map saved to: {output_path}")
        
        

    
    def create_interactive_map(self):
        """
        Create interactive HTML map using Folium
        """
        print("\nCreating interactive HTML map...")
        
        # Center map on Gujarat
        center_lat = self.location_stats['Latitude'].mean()
        center_lon = self.location_stats['Longitude'].mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=7,
            tiles='OpenStreetMap'
        )

        # Add different tile layers (alternative base maps)
        folium.TileLayer('CartoDB positron', name='Light Mode').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='Dark Mode').add_to(m)
        
        # Create feature groups for different offshore distances
        offshore_groups = {}
        for dist in sorted(self.location_stats['Offshore_Distance_km'].unique()):
            offshore_groups[dist] = folium.FeatureGroup(name=f'{int(dist)} km Offshore')
        
        # Add markers for each location
        for idx, row in self.location_stats.iterrows():
            # Determine color based on wind speed
            wind_speed = row['Mean_Wind_Speed']
            if wind_speed < 7:
                color = 'red'
                category = 'Low'
            elif wind_speed < 8.5:
                color = 'orange'
                category = 'Moderate'
            elif wind_speed < 10:
                color = 'lightgreen'
                category = 'Good'
            else:
                color = 'green'
                category = 'Excellent'
            
            # Create popup with detailed info
            popup_html = f"""
            <div style="font-family: Arial; width: 250px;">
                <h4 style="margin-bottom: 10px; color: #2c3e50;">{row['Location_ID']}</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background-color: #ecf0f1;">
                        <td style="padding: 5px;"><b>Location:</b></td>
                        <td style="padding: 5px;">{row['Latitude']:.4f}°N, {row['Longitude']:.4f}°E</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px;"><b>Offshore:</b></td>
                        <td style="padding: 5px;">{row['Offshore_Distance_km']:.0f} km</td>
                    </tr>
                    <tr style="background-color: #ecf0f1;">
                        <td style="padding: 5px;"><b>Mean Wind Speed:</b></td>
                        <td style="padding: 5px; color: {color}; font-weight: bold;">{wind_speed:.2f} m/s</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px;"><b>Wind Category:</b></td>
                        <td style="padding: 5px;">{category}</td>
                    </tr>
                    <tr style="background-color: #ecf0f1;">
                        <td style="padding: 5px;"><b>Std Dev:</b></td>
                        <td style="padding: 5px;">{row['Std_Wind_Speed']:.2f} m/s</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px;"><b>Range:</b></td>
                        <td style="padding: 5px;">{row['Min_Wind_Speed']:.2f} - {row['Max_Wind_Speed']:.2f} m/s</td>
                    </tr>
                    <tr style="background-color: #ecf0f1;">
                        <td style="padding: 5px;"><b>Samples:</b></td>
                        <td style="padding: 5px;">{row['Sample_Count']:.0f}</td>
                    </tr>
                </table>
                <p style="margin-top: 10px; font-size: 11px; color: #7f8c8d;">
                    Data Period: 2020-2024<br>
                    Source: Sentinel-1 SAR
                </p>
            </div>
            """
            
            # Add marker
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=8,
                popup=folium.Popup(popup_html, max_width=300),
                color='black',
                fillColor=color,
                fillOpacity=0.7,
                weight=2,
                tooltip=f"{row['Location_ID']}: {wind_speed:.2f} m/s"
            ).add_to(offshore_groups[row['Offshore_Distance_km']])
        
        # Add all offshore groups to map
        for group in offshore_groups.values():
            group.add_to(m)
        
        # Add Gujarat coastline (split at NaN separators)
        from scripts import config as cfg
        coastline_df = cfg.get_gujarat_coastline()
        for segment in _coastline_to_polylines(coastline_df):
            folium.PolyLine(
                segment,
                color='red',
                weight=2,
                opacity=0.9,
                tooltip='Gujarat Coastline'
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 200px; height: auto; 
                    background-color: white; z-index:9999; font-size:14px;
                    border:2px solid grey; border-radius: 5px; padding: 10px">
            <p style="margin: 0; font-weight: bold; text-align: center; 
                     background-color: #2c3e50; color: white; padding: 5px;">
                Wind Speed Categories
            </p>
            <p style="margin: 5px 0;"><i class="fa fa-circle" style="color:red"></i> Low (&lt;7 m/s)</p>
            <p style="margin: 5px 0;"><i class="fa fa-circle" style="color:orange"></i> Moderate (7-8.5 m/s)</p>
            <p style="margin: 5px 0;"><i class="fa fa-circle" style="color:lightgreen"></i> Good (8.5-10 m/s)</p>
            <p style="margin: 5px 0;"><i class="fa fa-circle" style="color:green"></i> Excellent (&gt;10 m/s)</p>
            <hr style="margin: 10px 0;">
            <p style="margin: 5px 0; font-size: 12px; color: #7f8c8d;">
                Based on 10m wind speed<br>
                estimates from SAR data<br>
                (2020-2024)
            </p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(m)
        
        # Add measure tool
        plugins.MeasureControl().add_to(m)
        
        # Save
        output_path = os.path.join(config.OUTPUT_DIR, 'Gujarat_WindSpeed_Interactive_Map.html')
        m.save(output_path)
        print(f"Interactive map saved to: {output_path}")
        
        return m
    
    def create_heatmap(self):
        """
        Create heatmap visualization
        """
        print("\nCreating wind speed heatmap...")
        
        # Prepare data for heatmap
        heat_data = [[row['Latitude'], row['Longitude'], row['Mean_Wind_Speed']] 
                     for idx, row in self.location_stats.iterrows()]
        
        # Create base map
        center_lat = self.location_stats['Latitude'].mean()
        center_lon = self.location_stats['Longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=7,
            tiles='CartoDB dark_matter'
        )
        
        # Add heatmap
        plugins.HeatMap(
            heat_data,
            min_opacity=0.3,
            max_val=self.location_stats['Mean_Wind_Speed'].max(),
            radius=25,
            blur=25,
            gradient={0.0: 'blue', 0.5: 'lime', 0.7: 'yellow', 1.0: 'red'}
        ).add_to(m)
        
        # Add coastline (split at NaN separators)
        from scripts import config as cfg
        coastline_df = cfg.get_gujarat_coastline()
        for segment in _coastline_to_polylines(coastline_df):
            folium.PolyLine(
                segment,
                color='white',
                weight=2,
                opacity=0.9
            ).add_to(m)
        
        # Save
        output_path = os.path.join(config.OUTPUT_DIR, 'Gujarat_WindSpeed_Heatmap.html')
        m.save(output_path)
        print(f"Heatmap saved to: {output_path}")
        
        return m


def main():
    """
    Main execution
    """
    print("="*80)
    print("Creating Wind Speed Maps for Gujarat Coast")
    print("="*80)
    
    # Load data
    data_file = os.path.join(config.OUTPUT_DIR, 'Gujarat_10m_WindSpeed_MasterTable.csv')
    
    if not os.path.exists(data_file):
        print(f"ERROR: Data file not found: {data_file}")
        print("Please run create_wind_speed_master_table.py first")
        return
    
    # Create mapper
    mapper = WindSpeedMapper(data_file)
    
    # Create visualizations
    print("\n1. Creating static map...")
    mapper.create_static_map()
    
    print("\n2. Creating interactive HTML map...")
    mapper.create_interactive_map()
    
    print("\n3. Creating wind speed heatmap...")
    mapper.create_heatmap()
    
    print("\n" + "="*80)
    print("MAPPING COMPLETED!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. Gujarat_WindSpeed_Map.png")
    print("  2. Gujarat_WindSpeed_Interactive_Map.html")
    print("     - Interactive map - click on points for details")
    print("  3. Gujarat_WindSpeed_Heatmap.html")
    print("     - Heat map visualization")
    print("="*80)


if __name__ == "__main__":
    main()