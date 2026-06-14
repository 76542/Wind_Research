"""
eval_testset_metrics.py  — READ-ONLY
====================================
Re-evaluates each ALREADY-TRAINED fine-tuned model on the SAME held-out
test split (seed=42, identical to finetune.py) and prints test RMSE / R2 / Bias.

It does NOT train, does NOT call torch.save, and does NOT touch any model
or map. Inference is deterministic (model.eval(): dropout off, BatchNorm
uses stored running stats), so these reproduce the original run's test
metrics exactly.

Place in scripts/ and run from project root:
    python -m scripts.eval_testset_metrics
(or just `python scripts/eval_testset_metrics.py`)
"""
import os, sys, pickle, numpy as np, pandas as pd
import torch

# scripts/eval_testset_metrics.py  ->  project root is one level up from scripts/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.ml.config_ml import (
    ALL_FEATURES, TARGET, VV_MAX, VV_MIN, VH_VV_RATIO_MIN,
    ERA5_MIN, ERA5_MAX)
from scripts.ml.train_mlp_v3 import WindSpeedMLPv3

GJ_SCALER = os.path.join(PROJECT_ROOT, "models", "feature_scaler.pkl")

# display name -> (data subdir / csv stem, model filename)
STATES = {
    "Maharashtra":    ("maharashtra",   "mlp_v3_maharashtra_finetuned.pth"),
    "Karnataka":      ("karnataka",     "mlp_v3_karnataka_finetuned.pth"),
    "Kerala":         ("kerala",        "mlp_v3_kerala_finetuned.pth"),
    "Tamil Nadu":     ("tamilnadu",     "mlp_v3_tamilnadu_finetuned.pth"),
    "Andhra Pradesh": ("andhrapradesh", "mlp_v3_andhrapradesh_finetuned.pth"),
    "Odisha":         ("odisha",        "mlp_v3_odisha_finetuned.pth"),
}


def load_clean(csv_path):
    df = pd.read_csv(csv_path)
    df = df[~((df["VV"] > VV_MAX) | (df["VV"] < VV_MIN) |
              (df["VH_VV_ratio"] < VH_VV_RATIO_MIN) |
              (df[TARGET] < ERA5_MIN) | (df[TARGET] > ERA5_MAX))].copy()
    if "month" not in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["month"] = df["timestamp"].dt.month
        df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_doy"]   = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"]   = np.cos(2 * np.pi * df["day_of_year"] / 365)
    return df


def held_out_test(df):
    # identical to finetune.py: seed 42, shuffle point_ids, last 15% = test
    point_ids = df["point_id"].unique()
    np.random.seed(42)
    np.random.shuffle(point_ids)
    n = len(point_ids)
    n_train, n_val = int(0.70 * n), int(0.15 * n)
    test_ids = point_ids[n_train + n_val:]
    return df[df["point_id"].isin(test_ids)].copy()


def main():
    with open(GJ_SCALER, "rb") as f:
        scaler = pickle.load(f)
    device = torch.device("cpu")

    print(f"\n{'State':<16}{'N_test':>8}{'RMSE':>10}{'R2':>10}{'Bias':>10}")
    print("-" * 54)
    for name, (sub, mdl) in STATES.items():
        csv = os.path.join(PROJECT_ROOT, "data", "processed", sub,
                           f"{sub}_era5_collocated.csv")
        mpath = os.path.join(PROJECT_ROOT, "models", mdl)
        if not (os.path.exists(csv) and os.path.exists(mpath)):
            print(f"{name:<16}  (missing csv or model — check path)")
            continue

        test_df = held_out_test(load_clean(csv))
        test_df[ALL_FEATURES] = scaler.transform(test_df[ALL_FEATURES])
        X = torch.FloatTensor(test_df[ALL_FEATURES].values)
        y = test_df[TARGET].values

        ckpt = torch.load(mpath, map_location=device, weights_only=False)
        arch = ckpt["architecture"]
        model = WindSpeedMLPv3(arch["input_dim"], arch["hidden_layers"], arch["dropout"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        with torch.no_grad():
            pred = model(X).squeeze().numpy()

        res = pred - y
        rmse = float(np.sqrt(np.mean(res ** 2)))
        r2 = float(1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2))
        bias = float(np.mean(res))
        print(f"{name:<16}{len(test_df):>8}{rmse:>10.4f}{r2:>10.4f}{bias:>+10.4f}")

    print()


if __name__ == "__main__":
    main()