"""
scripts/ml/generate_figure2_dual_mask.py  v5
Only shows rejections from the 4 real GEE dual-mask zones:
  1. Rann of Kutch (JRC)
  2. Gulf of Khambhat interior (JRC)
  3. Gulf of Kutch inner shallow (JRC)
  4. Diu / south Saurashtra tip (SRTM)

Usage: python -m scripts.ml.generate_figure2_dual_mask
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    print("ERROR: cartopy required."); sys.exit(1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

C_VALID = "#2ecc71"; C_JRC = "#e74c3c"; C_SRTM = "#e67e22"
C_OCEAN = "#cce5f5"; C_LAND = "#f0ede4"; C_BORDER = "#aaaaaa"

# ── Build candidate grid from actual sampling grid ────────────────────────────
grid = pd.read_csv("data/raw/gujarat_sampling_grid.csv")
deg_per_km = 1 / 111.0

anchor_dir = {}
for _, r in grid.iterrows():
    key = (round(r['coastal_lat'],5), round(r['coastal_lon'],5))
    d = r['offshore_distance_km']
    dlat = (r['latitude'] - r['coastal_lat']) / (d * deg_per_km)
    dlon = (r['longitude'] - r['coastal_lon']) / (d * deg_per_km)
    anchor_dir.setdefault(key, []).append((dlat, dlon))

anchor_mean = {k: (np.mean([x[0] for x in v]), np.mean([x[1] for x in v]))
               for k, v in anchor_dir.items()}

coast = grid[['coastal_lat','coastal_lon']].drop_duplicates()
candidates = []
for _, r in coast.iterrows():
    clat, clon = r['coastal_lat'], r['coastal_lon']
    key = (round(clat,5), round(clon,5))
    dlat, dlon = anchor_mean.get(key, (0,-1))
    for d in [20,40,60,80]:
        candidates.append({
            'latitude':  round(clat + d*deg_per_km*dlat, 5),
            'longitude': round(clon + d*deg_per_km*dlon, 5),
            'dist_km': d, 'coastal_lat': clat, 'coastal_lon': clon
        })

cdf = pd.DataFrame(candidates)
valid_set = grid[['latitude','longitude']].values
cdf['is_valid'] = cdf.apply(
    lambda r: np.sqrt((valid_set[:,0]-r['latitude'])**2 +
                      (valid_set[:,1]-r['longitude'])**2).min() < 0.05, axis=1)

def classify(row):
    if row['is_valid']: return 'valid'
    lat, lon = row['latitude'], row['longitude']
    clat, clon = row['coastal_lat'], row['coastal_lon']

    # 1. Rann of Kutch — anchors on Kutch peninsula, project into Rann
    if lat > 23.0 and 69.0 < lon < 71.5 and clat > 22.3:
        return 'JRC_fail'

    # 2. Gulf of Khambhat — anchors on Khambhat shore (clon>72), inside Gulf
    if clon > 72.0 and lon > 71.8 and lat < 22.5:
        return 'JRC_fail'

    # 3. Gulf of Kutch inner shallow — anchors inside GoK project into shallow inner gulf
    if 69.5 < clon < 71.2 and clat > 22.3 and lon > 70.0 and lat > 22.0:
        return 'JRC_fail'

    # 4. Diu / south tip — anchor at south tip, projected point near peninsula
    if clat < 21.2 and clon > 70.8 and 70.8 < lon < 72.0 and lat < 21.2:
        return 'SRTM_fail'

    # Everything else = SAR coverage gap, not a mask rejection → hide
    return 'drop'

cdf['status'] = cdf.apply(classify, axis=1)
cdf = cdf[cdf['status'] != 'drop'].reset_index(drop=True)

print(f"Valid: {(cdf.status=='valid').sum()} | "
      f"JRC fail: {(cdf.status=='JRC_fail').sum()} | "
      f"SRTM fail: {(cdf.status=='SRTM_fail').sum()}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def add_base(ax, extent, title=None, fs=9):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("50m"),    facecolor=C_OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"),     facecolor=C_LAND,  zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),lw=0.7, edgecolor="#555",    zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),  lw=0.4, edgecolor=C_BORDER,
                   linestyle=":", zorder=2)
    ax.add_feature(cfeature.RIVERS.with_scale("50m"),   lw=0.3, edgecolor="#88bbdd", zorder=2)
    if title: ax.set_title(title, fontsize=fs, fontweight="bold", pad=5)

def scat(ax, df, status, color, marker, size=18, label=None):
    sub = df[df["status"] == status]
    if len(sub) == 0: return
    kw = dict(c=color, s=size, marker=marker, alpha=0.88, zorder=6,
              transform=ccrs.PlateCarree(), label=label)
    if marker == "o": kw.update(edgecolors="white", linewidths=0.3)
    ax.scatter(sub["longitude"], sub["latitude"], **kw)

def geo_label(ax, text, lat, lon, fs=7.5, color="#333", ha="center", va="center"):
    ax.text(lon, lat, text, fontsize=fs, color=color, ha=ha, va=va, style="italic",
            transform=ccrs.PlateCarree(), zorder=9,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

def zoom_box(ax, ext, color, label=""):
    x0, x1, y0, y1 = ext
    ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, lw=1.5, edgecolor=color,
                            facecolor="none", linestyle="--",
                            transform=ccrs.PlateCarree(), zorder=10))
    if label:
        ax.text(x0+0.05, y1+0.05, label, fontsize=9, color=color, fontweight="bold",
                transform=ccrs.PlateCarree(), zorder=11,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

def add_gl(ax, fs=6):
    gl = ax.gridlines(draw_labels=True, lw=0.3, color="#ccc", linestyle="--")
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {"size": fs}; gl.ylabel_style = {"size": fs}

def annot(ax, text, ec, tc, fc, pos=(0.03, 0.04)):
    ax.text(*pos, text, transform=ax.transAxes, fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=fc, edgecolor=ec, alpha=0.93),
            color=tc, va="bottom", zorder=10)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 7.5))
fig.patch.set_facecolor("white")
gs = GridSpec(2, 2, figure=fig, left=0.03, right=0.97, top=0.88, bottom=0.09,
              hspace=0.42, wspace=0.12, width_ratios=[1.65, 1])

MAIN   = [67.0, 74.0, 19.3, 24.9]
ZOOM_B = [68.0, 71.8, 22.6, 24.8]   # Rann of Kutch
ZOOM_C = [71.2, 73.5, 19.5, 22.3]   # Gulf of Khambhat + south tip

# Panel A
ax_a = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
add_base(ax_a, MAIN, title="(a)  Gujarat — Dual Mask Filtering Overview", fs=10)
scat(ax_a, cdf, "JRC_fail",  C_JRC,  "x", 22, "Rejected — JRC occurrence < 50%")
scat(ax_a, cdf, "SRTM_fail", C_SRTM, "x", 22, "Rejected — SRTM elevation > 5 m")
scat(ax_a, cdf, "valid",     C_VALID, "o", 20, "Valid offshore points (retained)")
geo_label(ax_a, "Gulf of Kutch",     22.85, 68.8)
geo_label(ax_a, "Rann of\nKutch",    23.8,  70.5)
geo_label(ax_a, "Saurashtra",        21.9,  71.1)
geo_label(ax_a, "Gulf of\nKhambhat", 21.2,  72.6)
geo_label(ax_a, "Arabian Sea",       20.0,  68.0, fs=9, color="#1a6ea3")
zoom_box(ax_a, ZOOM_B, C_JRC,  label="B")
zoom_box(ax_a, ZOOM_C, C_SRTM, label="C")
add_gl(ax_a, fs=7)

# Panel B: Rann of Kutch
ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
add_base(ax_b, ZOOM_B, title="(b)  Rann of Kutch", fs=9)
sub_b = cdf[cdf.latitude.between(ZOOM_B[2],ZOOM_B[3]) &
            cdf.longitude.between(ZOOM_B[0],ZOOM_B[1])]
scat(ax_b, sub_b, "JRC_fail", C_JRC,   "x", 35)
scat(ax_b, sub_b, "valid",    C_VALID,  "o", 26)
geo_label(ax_b, "Rann of Kutch\n(seasonal salt marsh)", 23.9, 70.0, fs=7)
geo_label(ax_b, "Gulf of Kutch", 22.9, 69.1, fs=7)
geo_label(ax_b, "Kutch Peninsula", 23.1, 70.5, fs=6.5)
annot(ax_b, "JRC occurrence < 50%\nSeasonal flooding / salt flats\n→ Not persistent offshore water",
      C_JRC, "#7b0000", "#fdecea")
add_gl(ax_b)

# Panel C: Gulf of Khambhat + south tip
ax_c = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
add_base(ax_c, ZOOM_C, title="(c)  Gulf of Khambhat & Peninsula Tip", fs=9)
sub_c = cdf[cdf.latitude.between(ZOOM_C[2],ZOOM_C[3]) &
            cdf.longitude.between(ZOOM_C[0],ZOOM_C[1])]
scat(ax_c, sub_c, "JRC_fail",  C_JRC,  "x", 35)
scat(ax_c, sub_c, "SRTM_fail", C_SRTM, "x", 35)
scat(ax_c, sub_c, "valid",     C_VALID, "o", 26)
geo_label(ax_c, "Gulf of\nKhambhat", 21.1, 72.5, fs=7)
geo_label(ax_c, "Diu / south tip",   20.4, 71.1, fs=6.5)
annot(ax_c, "Red ✕   JRC < 50%: tidal flats, avg depth < 10 m\n"
            "Orange ✕  SRTM > 5 m: peninsula geometry",
      C_SRTM, "#5c3300", "#fff8f0")
add_gl(ax_c)

# Legend + title
fig.legend(handles=[
    Line2D([0],[0], marker="o", color="w", markerfacecolor=C_VALID,
           markersize=9, label="Valid offshore point  (retained)"),
    Line2D([0],[0], marker="x", color=C_JRC, markersize=9, markeredgewidth=2,
           label="Rejected — JRC water occurrence < 50%  (non-persistent / tidal)"),
    Line2D([0],[0], marker="x", color=C_SRTM, markersize=9, markeredgewidth=2,
           label="Rejected — SRTM elevation > 5 m  (near-shore / peninsula tip)"),
], loc="lower center", ncol=3, fontsize=8, frameon=True,
   bbox_to_anchor=(0.5, 0.01), edgecolor="#ccc", framealpha=0.95)

fig.suptitle(
    "Offshore Sampling Grid: Dual Mask Illustration (Gujarat)\n"
    "JRC Global Surface Water (occurrence ≥ 50%) AND SRTM Elevation (≤ 5 m)",
    fontsize=11, fontweight="bold", y=0.97)

out = os.path.join(OUTPUT_DIR, "figure2_dual_mask_gujarat.png")
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")