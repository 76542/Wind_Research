import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
import torch
import torch.nn as nn

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
    print("Cartopy found")
except ImportError:
    HAS_CARTOPY = False
    print("Cartopy not found — using fallback")

# ── Paths (mirrors config_ml.py) ──────────────────────────────────────────────
BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV    = os.path.join(BASE, "data", "processed", "era5_collocated.csv")
SCALER_PATH = os.path.join(BASE, "models", "feature_scaler.pkl")
MODEL_PATH  = os.path.join(BASE, "models", "mlp_v3_wind_model.pth")
OUT_PATH    = os.path.join(BASE, "outputs", "gujarat_final_wind_heatmap.png")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# ── Exact feature list from config_ml.py ──────────────────────────────────────
ALL_FEATURES = [
    "VV", "VH", "VH_VV_ratio", "incidence_angle",       # SAR
    "offshore_distance_km", "latitude", "longitude",      # Spatial
    "sin_month", "cos_month", "sin_doy", "cos_doy"        # Cyclical time
]
TARGET = "ERA5_WindSpeed_100m_ms"

# ── MLP architecture (matches mlp_v3_wind_model.pth) ─────────────────────────────
class WindSpeedMLPv3(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super(WindSpeedMLPv3, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)

# ── Load data & engineer cyclical features ─────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_CSV)

# Engineer cyclical time features exactly as preprocessing.py would have
df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
df["sin_doy"]   = np.sin(2 * np.pi * df["day_of_year"] / 365)
df["cos_doy"]   = np.cos(2 * np.pi * df["day_of_year"] / 365)

df = df.dropna(subset=ALL_FEATURES + [TARGET, "point_id"])
print(f"  {len(df)} rows, {df['point_id'].nunique()} unique points")

# ── ERA5 counts per point ──────────────────────────────────────────────────────
print("Computing ERA5 wind potential...")
meta = df.groupby("point_id")[["latitude","longitude"]].first().reset_index()

def count_days(df, col, thresh):
    grp = df.groupby("point_id")[col].apply(lambda x: (x > thresh).sum())
    return grp.reset_index().rename(columns={col: "days"}).merge(meta, on="point_id")


# ── Load saved scaler & model ──────────────────────────────────────────────────
print("Loading scaler...")
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

print("Loading MLP...")
device = torch.device("cpu")
model = WindSpeedMLPv3(input_dim=11, hidden_layers=[256, 128, 64, 32], dropout_rate=0.2).to(device)
state  = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
model.eval()

print("Running inference...")
X   = scaler.transform(df[ALL_FEATURES].values)
X_t = torch.tensor(X, dtype=torch.float32)
preds = []
with torch.no_grad():
    for i in range(0, len(X_t), 4096):
        preds.append(model(X_t[i:i+4096]).numpy())
df["mlp_pred"] = np.clip(np.concatenate(preds), 0, None)

mlp_6 = count_days(df, "mlp_pred", 6)
mlp_8 = count_days(df, "mlp_pred", 8)
mlp_4 = count_days(df, "mlp_pred", 4)
print(f"  MLP >4 m/s: max={mlp_4['days'].max()}, mean={mlp_4['days'].mean():.0f}")
print(f"  MLP >6 m/s: max={mlp_6['days'].max()}, mean={mlp_6['days'].mean():.0f}")
print(f"  MLP >8 m/s: max={mlp_8['days'].max()}, mean={mlp_8['days'].mean():.0f}")

# ── Interpolation — masked strictly to convex hull of actual points ────────────
lat_min, lat_max = 19.3, 24.8
lon_min, lon_max = 67.3, 74.8
gi_lon, gi_lat = np.meshgrid(
    np.linspace(lon_min, lon_max, 400),
    np.linspace(lat_min, lat_max, 400)
)

# Convex hull of the 218 actual sampling points
from scipy.spatial import cKDTree

# Build KDTree of actual sampling points once
all_pts   = meta[["longitude","latitude"]].values
kd_tree   = cKDTree(all_pts)
flat_grid = np.column_stack([gi_lon.ravel(), gi_lat.ravel()])
dists, _  = kd_tree.query(flat_grid)
too_far   = dists > 0.25

def interp(df_pt):
    pts  = df_pt[["longitude","latitude"]].values
    vals = df_pt["days"].values.astype(float)
    grid = griddata(pts, vals, (gi_lon, gi_lat), method="linear")
    grid.ravel()[too_far] = np.nan
    return grid

print("Interpolating (strictly within sampling point hull)...")

g_mlp_4 = interp(mlp_4)
g_mlp_6  = interp(mlp_6)
g_mlp_8  = interp(mlp_8)

# ── Plot ──────────────────────────────────────────────────────────────────────
print("Plotting...")
cmap = plt.cm.jet

if HAS_CARTOPY:
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 3, figsize=(22, 7),
                         subplot_kw={"projection": proj})
else:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

fig.patch.set_facecolor("white")

panels = [
    (0, 0, "MLP v3 (SAR)", "> 4 m/s", g_mlp_4, mlp_4, mlp_4["days"].max()),
    (0, 1, "MLP v3 (SAR)", "> 6 m/s", g_mlp_6, mlp_6, mlp_6["days"].max()),
    (0, 2, "MLP v3 (SAR)", "> 8 m/s", g_mlp_8, mlp_8, mlp_8["days"].max()),
]
for row, col, src_label, thresh_label, grid, pts_df, vmax in panels:
    ax = axes[col]

    if HAS_CARTOPY:
        im = ax.pcolormesh(gi_lon, gi_lat, grid, cmap=cmap,
                           vmin=0, vmax=vmax, shading="auto",
                           transform=proj, zorder=1)
        ax.add_feature(cfeature.LAND.with_scale("10m"),
                       facecolor="#cccccc", edgecolor="black",
                       linewidth=0.7, zorder=2)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"),
                       edgecolor="black", linewidth=0.8, zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"),
                       edgecolor="#555555", linewidth=0.5,
                       linestyle="--", zorder=3)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                          color="gray", alpha=0.5, linestyle="--")
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlabel_style = {"color": "black", "size": 8}
        gl.ylabel_style = {"color": "black", "size": 8}
        ax.scatter(pts_df["longitude"], pts_df["latitude"],
                   c=pts_df["days"], cmap=cmap, vmin=0, vmax=vmax,
                   s=8, edgecolors="none", zorder=5, transform=proj)
    else:
        im = ax.pcolormesh(gi_lon, gi_lat, grid, cmap=cmap,
                           vmin=0, vmax=vmax, shading="auto", zorder=1)
        ax.scatter(pts_df["longitude"], pts_df["latitude"],
                   c=pts_df["days"], cmap=cmap, vmin=0, vmax=vmax,
                   s=8, edgecolors="none", zorder=5)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel("Longitude (°E)", color="black", fontsize=9)
        ax.set_ylabel("Latitude (°N)", color="black", fontsize=9)
        ax.tick_params(colors="black", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("black")

    cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
    cb.set_label("Number of Days", color="black", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="black")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="black", fontsize=8)

    ax.set_facecolor("white")
    ax.set_title(f"{src_label}  |  Days {thresh_label}",
                 color="black", fontsize=11, pad=7, fontweight="bold")

    ax.text(0.03, 0.95, thresh_label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", color="black", va="top",
            bbox=dict(facecolor="white", edgecolor="black",
                      alpha=0.75, boxstyle="round,pad=0.3"))

    ax.text(0.97, 0.03, src_label, transform=ax.transAxes,
            fontsize=8, color="black", ha="right",
            bbox=dict(facecolor="white", edgecolor="gray",
                      alpha=0.6, boxstyle="round,pad=0.25"))

fig.suptitle(
    "Offshore Wind Resource Potential — Gujarat Coast  (2020–2024)\n"
    "MLP v3 SAR-based Prediction  |  Sentinel-1 100m Hub-Height",
    color="black", fontsize=13, fontweight="bold", y=1.01
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n✅ Saved → {OUT_PATH}")