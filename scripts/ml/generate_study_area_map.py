"""
scripts/ml/generate_study_area_map.py
======================================
Figure 1 — Study Area Map (publication-ready)

Usage: python -m scripts.ml.generate_study_area_map
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("ERROR: Cartopy is required for this figure.")
    sys.exit(1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRIDS = {
    'Gujarat': {
        'path': os.path.join(PROJECT_ROOT, "data", "raw", "gujarat_sampling_grid.csv"),
        'coast': 'west',
    },
    'Maharashtra': {
        'path': os.path.join(PROJECT_ROOT, "data", "raw", "maharashtra",
                             "maharashtra_sampling_grid.csv"),
        'coast': 'west',
    },
    'Goa': {
        'path': os.path.join(PROJECT_ROOT, "data", "raw", "goa",
                             "goa_sampling_grid.csv"),
        'coast': 'west',
    },
    'Karnataka': {
        'path': os.path.join(PROJECT_ROOT, "data", "raw", "karnataka",
                             "karnataka_sampling_grid.csv"),
        'coast': 'west',
    },
    'Kerala': {
        'path': os.path.join(PROJECT_ROOT, "data", "raw", "kerala",
                             "kerala_sampling_grid.csv"),
        'coast': 'west',
    },
    'Tamil Nadu': {
        'path': os.path.join(PROJECT_ROOT, "data", "raw", "tamilnadu",
                             "tamilnadu_sampling_grid.csv"),
        'coast': 'east',
    },
    'Andhra Pradesh': {
        'path': os.path.join(PROJECT_ROOT, "data", "raw", "andhrapradesh",
                             "andhrapradesh_sampling_grid.csv"),
        'coast': 'east',
    },
    'Odisha': {
        'path': os.path.join(PROJECT_ROOT, "data", "raw", "odisha",
                             "odisha_sampling_grid.csv"),
        'coast': 'east',
    },
}


def main():
    print("=" * 60)
    print("GENERATING STUDY AREA MAP (Figure 1)")
    print("=" * 60)

    # ── Load all grids ────────────────────────────────────────────
    all_points = []
    for state_name, cfg in GRIDS.items():
        if not os.path.exists(cfg['path']):
            print(f"  WARNING: {cfg['path']} not found, skipping {state_name}")
            continue
        df = pd.read_csv(cfg['path'])
        df['state'] = state_name
        df['coast'] = cfg['coast']
        all_points.append(df)
        print(f"  {state_name}: {len(df)} points")

    combined = pd.concat(all_points, ignore_index=True)
    print(f"\n  Total: {len(combined)} sampling points")

    # ── State colors (all distinct, no yellow) ────────────────────
    state_colors = {
        'Gujarat':          '#e41a1c',
        'Maharashtra':      '#ff7f00',
        'Goa':              '#b8860b',
        'Karnataka':        '#4daf4a',
        'Kerala':           '#377eb8',
        'Tamil Nadu':       '#984ea3',
        'Andhra Pradesh':   '#e7298a',
        'Odisha':           '#a65628',
    }

    # ── Figure ────────────────────────────────────────────────────
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(14, 18), subplot_kw={"projection": proj})
    fig.patch.set_facecolor("white")

    ax.set_extent([66.0, 89.5, 6.5, 25.5], crs=proj)

    # Base map
    ax.add_feature(cfeature.OCEAN.with_scale("10m"),
                   facecolor="#e6f2ff", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"),
                   facecolor="#f0efe9", edgecolor="none", zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"),
                   edgecolor="#333333", linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"),
                   edgecolor="#888888", linewidth=0.4,
                   linestyle="--", zorder=3)

    try:
        ax.add_feature(cfeature.NaturalEarthFeature(
            'cultural', 'admin_1_states_provinces_lines', '10m',
            edgecolor='#aaaaaa', facecolor='none', linewidth=0.3),
            zorder=2)
    except Exception:
        pass

    # ── Plot points by state ──────────────────────────────────────
    for state_name in state_colors:
        subset = combined[combined['state'] == state_name]
        if len(subset) == 0:
            continue
        ax.scatter(subset['longitude'], subset['latitude'],
                   c=state_colors[state_name], s=12, alpha=0.85,
                   edgecolors='black', linewidths=0.2,
                   transform=proj, zorder=5)

    # ── State labels (repositioned to avoid overlap) ──────────────
    state_labels = [
        ('Gujarat',          70.5,  21.8),
        ('Maharashtra',      74.0,  18.5),
        ('Goa',              74.3,  15.4),
        ('Karnataka',        75.3,  14.2),
        ('Kerala',           76.7,  10.2),
        ('Tamil Nadu',       78.8,  10.8),
        ('Andhra\nPradesh',  79.5,  15.5),
        ('Odisha',           84.8,  20.0),
    ]

    for label, lon, lat in state_labels:
        ax.text(lon, lat, label, transform=proj,
                fontsize=8, fontweight='bold', color='#333333',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#cccccc', alpha=0.85),
                zorder=6)

    # ── Geographic features (only thesis-relevant ones) ───────────
    geo_features = [
        (70.5,  23.2,  'Gulf of Kutch'),
        (72.3,  22.5,  'Gulf of\nKhambhat'),
        (77.5,  7.8,   'Kanyakumari'),
        (79.5,  9.2,   'Palk Strait'),
        (81.5,  16.5,  'Krishna-Godavari\nDelta'),
        (84.7,  19.8,  'Chilika Lake'),
        (87.6,  20.4,  'Mahanadi\nDelta'),
    ]

    for lon, lat, label in geo_features:
        ax.text(lon, lat, label, transform=proj,
                fontsize=6.5, color='#555555', fontstyle='italic',
                ha='center', va='center', zorder=6)

    # ── Water body labels ─────────────────────────────────────────
    ax.text(67.5, 17.0, 'Arabian\nSea', transform=proj,
            fontsize=12, color='#4477aa', fontstyle='italic',
            ha='center', va='center', alpha=0.5, fontweight='bold',
            zorder=4)
    ax.text(85.5, 12.5, 'Bay of\nBengal', transform=proj,
            fontsize=12, color='#4477aa', fontstyle='italic',
            ha='center', va='center', alpha=0.5, fontweight='bold',
            zorder=4)
    ax.text(73.0, 7.3, 'Indian Ocean', transform=proj,
            fontsize=10, color='#4477aa', fontstyle='italic',
            ha='center', va='center', alpha=0.4,
            zorder=4)

    # ── Gridlines ─────────────────────────────────────────────────
    gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                      color="gray", alpha=0.4, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"color": "black", "size": 9}
    gl.ylabel_style = {"color": "black", "size": 9}

    # ── Legend ────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=state_colors[s], edgecolor='black',
                       linewidth=0.3,
                       label=f'{s} ({len(combined[combined.state == s])})')
        for s in state_colors if len(combined[combined.state == s]) > 0
    ]
    leg = ax.legend(handles=legend_elements, loc='lower left',
                    fontsize=7, framealpha=0.9, edgecolor='#cccccc',
                    title='State (sampling points)', title_fontsize=8,
                    ncol=1)
    leg.set_zorder(7)

    # ── Title (concise — details go in caption) ───────────────────
    ax.set_title(
        'Study Area — Offshore Wind Resource Assessment Along Indian Coastline\n'
        '721 Sampling Points  |  8 States  |  2020–2024',
        fontsize=13, fontweight='bold', pad=15)

    # ── Summary box ───────────────────────────────────────────────
    summary_text = (
        "West Coast (5 states): 463 points\n"
        "  Gujarat \u2192 Maharashtra \u2192 Goa \u2192\n"
        "  Karnataka \u2192 Kerala\n"
        "\n"
        "East Coast (3 states): 258 points\n"
        "  Tamil Nadu \u2192 Andhra Pradesh \u2192\n"
        "  Odisha\n"
        "\n"
        f"Total: {len(combined)} points\n"
        "Observations: 113,405\n"
        "Offshore: 20, 40, 60, 80 km\n"
        "Period: 2020\u20132024"
    )
    ax.text(0.98, 0.02, summary_text, transform=ax.transAxes,
            fontsize=7.5, verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#999999', alpha=0.9),
            fontfamily='monospace', zorder=7)

    # ── Save ──────────────────────────────────────────────────────
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "figure1_study_area_map.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {output_path}")

    print(f"\n{'='*60}")
    print("STUDY AREA MAP COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()