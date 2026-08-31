"""
eval_testset_binned.py  — READ-ONLY
===================================
Extends eval_testset_metrics.py. Re-evaluates each already-trained fine-tuned
model on the SAME held-out test split (seed=42, identical to finetune.py) and
reports RMSE *and* bias stratified by ERA5 wind-speed bin -- computed on the
TEST SPLIT ONLY, so the binned tables sit on the exact same rows as your
Table VI test metrics.

It does NOT train, does NOT call torch.save, and does NOT touch any model or
map. Inference is deterministic (model.eval()), so the overall metrics
reproduce the original run exactly -- use them to cross-check against Table VI.

Gujarat is intentionally NOT included (base model, different split; already
binned on its test set in evaluation_results.txt). Maharashtra's column here
is on a slightly different split than your finetune run -- use the
maharashtra_finetune_results.txt bins for the paper, not this script's MH.

Run from project root:
    python -m scripts.eval_testset_binned
"""
import os, sys, pickle, numpy as np, pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.ml.config_ml import (
    ALL_FEATURES, TARGET, VV_MAX, VV_MIN, VH_VV_RATIO_MIN,
    ERA5_MIN, ERA5_MAX)
from scripts.ml.train_mlp_v3 import WindSpeedMLPv3

GJ_SCALER = os.path.join(PROJECT_ROOT, "models", "feature_scaler.pkl")

# display name -> (data subdir / csv stem, model filename, short column label)
STATES = {
    "Maharashtra":    ("maharashtra",   "mlp_v3_maharashtra_finetuned.pth",   "MH"),
    "Karnataka":      ("karnataka",     "mlp_v3_karnataka_finetuned.pth",     "KA"),
    "Kerala":         ("kerala",        "mlp_v3_kerala_finetuned.pth",        "KL"),
    "Tamil Nadu":     ("tamilnadu",     "mlp_v3_tamilnadu_finetuned.pth",     "TN"),
    "Andhra Pradesh": ("andhrapradesh", "mlp_v3_andhrapradesh_finetuned.pth", "AP"),
    "Odisha":         ("odisha",        "mlp_v3_odisha_finetuned.pth",        "OD"),
}

BIN_EDGES  = [0, 3, 6, 9, 12, 15, np.inf]
BIN_LABELS = ["0-3", "3-6", "6-9", "9-12", "12-15", "15+"]


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


def predict(test_df, scaler, mpath, device):
    test_df = test_df.copy()
    test_df[ALL_FEATURES] = scaler.transform(test_df[ALL_FEATURES])
    X = torch.FloatTensor(test_df[ALL_FEATURES].values)
    ckpt = torch.load(mpath, map_location=device, weights_only=False)
    arch = ckpt["architecture"]
    model = WindSpeedMLPv3(arch["input_dim"], arch["hidden_layers"], arch["dropout"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        return model(X).squeeze().numpy()


def overall(pred, y):
    res = pred - y
    rmse = float(np.sqrt(np.mean(res ** 2)))
    r2 = float(1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2))
    bias = float(np.mean(res))
    return rmse, r2, bias


def main():
    with open(GJ_SCALER, "rb") as f:
        scaler = pickle.load(f)
    device = torch.device("cpu")

    overall_rows = []                            # (name, N, rmse, r2, bias)
    bin_rmse = {lab: {} for lab in BIN_LABELS}   # bin -> {short: rmse or None}
    bin_bias = {lab: {} for lab in BIN_LABELS}   # bin -> {short: bias or None}
    cols = []

    for name, (sub, mdl, short) in STATES.items():
        csv = os.path.join(PROJECT_ROOT, "data", "processed", sub,
                           f"{sub}_era5_collocated.csv")
        mpath = os.path.join(PROJECT_ROOT, "models", mdl)
        if not (os.path.exists(csv) and os.path.exists(mpath)):
            print(f"{name:<16}  (missing csv or model -- check path)")
            continue

        test_df = held_out_test(load_clean(csv))
        y = test_df[TARGET].values
        pred = predict(test_df, scaler, mpath, device)

        rmse, r2, bias = overall(pred, y)
        overall_rows.append((name, len(test_df), rmse, r2, bias))
        cols.append(short)

        bins = pd.cut(y, bins=BIN_EDGES, labels=BIN_LABELS, right=False)
        for lab in BIN_LABELS:
            m = np.asarray(bins == lab)
            if m.sum() == 0:
                bin_rmse[lab][short] = None
                bin_bias[lab][short] = None
            else:
                d = pred[m] - y[m]
                bin_rmse[lab][short] = float(np.sqrt(np.mean(d ** 2)))
                bin_bias[lab][short] = float(np.mean(d))

    # ---- overall test metrics: cross-check against Table VI ----
    print(f"\nOVERALL TEST METRICS (should match Table VI)")
    print(f"{'State':<16}{'N_test':>8}{'RMSE':>10}{'R2':>10}{'Bias':>10}")
    print("-" * 54)
    for name, n, rmse, r2, bias in overall_rows:
        print(f"{name:<16}{n:>8}{rmse:>10.4f}{r2:>10.4f}{bias:>+10.4f}")

    # ---- RMSE by wind-speed bin (test split) ----
    print(f"\nTEST-SET RMSE BY WIND-SPEED BIN (m/s)")
    header = f"{'Bin':<8}" + "".join(f"{c:>10}" for c in cols)
    print(header); print("-" * len(header))
    for lab in BIN_LABELS:
        row = f"{lab:<8}"
        for c in cols:
            v = bin_rmse[lab].get(c)
            row += f"{'n/a':>10}" if v is None else f"{v:>10.3f}"
        print(row)

    # ---- bias by wind-speed bin (test split) ----
    print(f"\nTEST-SET BIAS BY WIND-SPEED BIN (m/s, pred - true)")
    print(header); print("-" * len(header))
    for lab in BIN_LABELS:
        row = f"{lab:<8}"
        for c in cols:
            v = bin_bias[lab].get(c)
            row += f"{'n/a':>10}" if v is None else f"{v:>+10.3f}"
        print(row)
    print()


if __name__ == "__main__":
    main()