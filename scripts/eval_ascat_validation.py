"""
eval_ascat_validation.py  — READ-ONLY
=====================================
Reproduces the ASCAT independent-validation metrics for Gujarat by running the
already-trained Gujarat base model on the collocated ASCAT/SAR rows.

It rebuilds the 11 model features on data/ascat/ascat_sar_collocated.csv
(joining offshore_distance_km from the Gujarat sampling grid by point_id,
computing the VH/VV ratio and the sin/cos time encodings), scales them with
the saved Gujarat StandardScaler, runs mlp_v3_model.pth, and prints:
  - MLP   vs ASCAT-100m   (the headline; expect ~3.34 / -2.29 / -0.09)
  - ERA5  vs ASCAT-100m   (sanity; ~3.19 / -2.18 / 0.00)
  - MLP   vs ERA5-100m    (feature-reconstruction check; should be ~1.3 RMSE
                           if the features were rebuilt correctly)

Trains nothing, saves nothing, touches no model or map.
Run from project root:
    python -m scripts.eval_ascat_validation
"""
import os, sys, pickle, numpy as np, pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.ml.config_ml import ALL_FEATURES, TARGET
from scripts.ml.train_mlp_v3 import WindSpeedMLPv3

SCALER   = os.path.join(PROJECT_ROOT, "models", "feature_scaler.pkl")
MODEL    = os.path.join(PROJECT_ROOT, "models", "mlp_v3_wind_model.pth")   # Gujarat base
COLLOC   = os.path.join(PROJECT_ROOT, "data", "ascat", "ascat_sar_collocated.csv")
GRID     = os.path.join(PROJECT_ROOT, "data", "raw", "gujarat_sampling_grid.csv")

ASCAT_100M = "wind_speed_100m"   # ASCAT extrapolated to 100 m
ERA5_100M  = "era5_wind_100m"


def metrics(pred, truth, label):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    m = np.isfinite(pred) & np.isfinite(truth)
    p, t = pred[m], truth[m]
    res = p - t
    rmse = float(np.sqrt(np.mean(res ** 2)))
    bias = float(np.mean(res))
    r2 = float(1 - np.sum(res ** 2) / np.sum((t - t.mean()) ** 2))
    print(f"  {label:<26} N={m.sum():>6}  RMSE={rmse:7.3f}  bias={bias:+7.3f}  R2={r2:8.4f}")


def main():
    df = pd.read_csv(COLLOC)
    print(f"Collocated rows: {len(df)}")

    # offshore_distance_km by point_id from the sampling grid
    grid = pd.read_csv(GRID)[["point_id", "offshore_distance_km"]]
    df = df.merge(grid, on="point_id", how="left")
    missing = df["offshore_distance_km"].isna().sum()
    if missing:
        print(f"  WARNING: {missing} rows had no offshore_distance match — dropping them")
        df = df.dropna(subset=["offshore_distance_km"]).copy()

    # rebuild the remaining features
    df["VH_VV_ratio"] = df["VH"] / df["VV"]
    df["latitude"]    = df["sar_lat"]
    df["longitude"]   = df["sar_lon"]
    ts = pd.to_datetime(df["sar_timestamp"])
    mth, doy = ts.dt.month, ts.dt.dayofyear
    df["sin_month"] = np.sin(2 * np.pi * mth / 12)
    df["cos_month"] = np.cos(2 * np.pi * mth / 12)
    df["sin_doy"]   = np.sin(2 * np.pi * doy / 365)
    df["cos_doy"]   = np.cos(2 * np.pi * doy / 365)

    missing_feats = [c for c in ALL_FEATURES if c not in df.columns]
    if missing_feats:
        print(f"  ERROR: still missing features {missing_feats} — check column names")
        return

    with open(SCALER, "rb") as f:
        scaler = pickle.load(f)
    X = scaler.transform(df[ALL_FEATURES].values)

    ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)
    arch = ckpt["architecture"]
    model = WindSpeedMLPv3(arch["input_dim"], arch["hidden_layers"], arch["dropout"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        mlp = model(torch.FloatTensor(X)).squeeze().numpy()

    print("\nASCAT independent validation (Gujarat):")
    metrics(mlp,              df[ASCAT_100M], "MLP  vs ASCAT-100m")
    metrics(df[ERA5_100M],    df[ASCAT_100M], "ERA5 vs ASCAT-100m")
    metrics(mlp,              df[ERA5_100M],  "MLP  vs ERA5-100m (check)")
    print()


if __name__ == "__main__":
    main()