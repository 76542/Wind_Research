"""
scripts/andhrapradesh/ml/finetune.py
=====================================
Fine-tune from Tamil Nadu FT weights.
East coast chain: Gujarat original → Tamil Nadu FT → Andhra Pradesh FT

Usage: python -m scripts.andhrapradesh.ml.finetune
"""
import os, sys, pickle, numpy as np, pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.ml.config_ml import (
    ALL_FEATURES, TARGET, VV_MAX, VV_MIN, VH_VV_RATIO_MIN,
    ERA5_MIN, ERA5_MAX, SEASON_MAP)
from scripts.ml.train_mlp_v3 import WindSpeedMLPv3

AP_COLLOCATED = os.path.join(PROJECT_ROOT, "data", "processed", "andhrapradesh",
                              "andhrapradesh_era5_collocated.csv")
GJ_SCALER = os.path.join(PROJECT_ROOT, "models", "feature_scaler.pkl")
TN_MODEL = os.path.join(PROJECT_ROOT, "models", "mlp_v3_tamilnadu_finetuned.pth")
OUTPUT_MODEL = os.path.join(PROJECT_ROOT, "models", "mlp_v3_andhrapradesh_finetuned.pth")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def main():
    print("\n" + "="*60)
    print("ANDHRA PRADESH FINE-TUNING (from Tamil Nadu FT)")
    print("East coast chain: Gujarat → TN FT → AP FT")
    print("="*60)

    # ── Load data ─────────────────────────────────────────────────
    df = pd.read_csv(AP_COLLOCATED)
    n0 = len(df)
    df = df[~((df["VV"]>VV_MAX)|(df["VV"]<VV_MIN)|
              (df["VH_VV_ratio"]<VH_VV_RATIO_MIN)|
              (df[TARGET]<ERA5_MIN)|(df[TARGET]>ERA5_MAX))].copy()
    print(f"  {n0} -> {len(df)} after cleaning")

    if 'month' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear

    df["sin_month"] = np.sin(2*np.pi*df["month"]/12)
    df["cos_month"] = np.cos(2*np.pi*df["month"]/12)
    df["sin_doy"] = np.sin(2*np.pi*df["day_of_year"]/365)
    df["cos_doy"] = np.cos(2*np.pi*df["day_of_year"]/365)
    df["season"] = df["month"].map(SEASON_MAP)

    # ── Spatial split 70/15/15 ────────────────────────────────────
    point_ids = df['point_id'].unique()
    np.random.seed(42)
    np.random.shuffle(point_ids)
    n = len(point_ids)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train_ids = point_ids[:n_train]
    val_ids = point_ids[n_train:n_train+n_val]
    test_ids = point_ids[n_train+n_val:]

    train_df = df[df['point_id'].isin(train_ids)].copy()
    val_df = df[df['point_id'].isin(val_ids)].copy()
    test_df = df[df['point_id'].isin(test_ids)].copy()

    print(f"\n  Split: {len(train_df)} train ({len(train_ids)} pts), "
          f"{len(val_df)} val ({len(val_ids)} pts), "
          f"{len(test_df)} test ({len(test_ids)} pts)")

    # ── Scale with Gujarat scaler ─────────────────────────────────
    with open(GJ_SCALER, "rb") as f:
        scaler = pickle.load(f)

    for subset in [train_df, val_df, test_df]:
        subset[ALL_FEATURES] = scaler.transform(subset[ALL_FEATURES])

    X_train = torch.FloatTensor(train_df[ALL_FEATURES].values)
    y_train = torch.FloatTensor(train_df[TARGET].values)
    X_val = torch.FloatTensor(val_df[ALL_FEATURES].values)
    y_val = torch.FloatTensor(val_df[TARGET].values)
    X_test = torch.FloatTensor(test_df[ALL_FEATURES].values)
    y_test = torch.FloatTensor(test_df[TARGET].values)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                               batch_size=256, shuffle=True)

    # ── Load Tamil Nadu FT model ──────────────────────────────────
    device = torch.device("cpu")
    ckpt = torch.load(TN_MODEL, map_location=device, weights_only=False)
    arch = ckpt["architecture"]
    model = WindSpeedMLPv3(arch["input_dim"], arch["hidden_layers"], arch["dropout"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"\n  Loaded Tamil Nadu FT model: {arch['hidden_layers']}")

    # ── Fine-tune ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=15, factor=0.5, min_lr=1e-6)

    best_val_rmse = float('inf')
    best_state = None
    patience_counter = 0
    max_patience = 40
    max_epochs = 300

    print(f"\n  Training: max {max_epochs} epochs, patience {max_patience}")
    print(f"  {'Epoch':>6} {'Train RMSE':>12} {'Val RMSE':>12} {'LR':>12}")
    print("-" * 50)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).squeeze()
            val_rmse = torch.sqrt(torch.mean((val_pred - y_val)**2)).item()
            train_pred = model(X_train).squeeze()
            train_rmse = torch.sqrt(torch.mean((train_pred - y_train)**2)).item()

        scheduler.step(val_rmse)
        current_lr = optimizer.param_groups[0]['lr']

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = model.state_dict().copy()
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch <= 5 or patience_counter == 0:
            marker = " *" if patience_counter == 0 else ""
            print(f"  {epoch:>6} {train_rmse:>12.4f} {val_rmse:>12.4f} "
                  f"{current_lr:>12.6f}{marker}")

        if patience_counter >= max_patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # ── Evaluate on test set ──────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).squeeze().numpy()

    y_test_np = y_test.numpy()
    residuals = test_pred - y_test_np
    test_rmse = np.sqrt(np.mean(residuals**2))
    test_r2 = 1 - np.sum(residuals**2) / np.sum((y_test_np - y_test_np.mean())**2)
    test_bias = np.mean(residuals)

    print(f"\n{'='*60}")
    print(f"ANDHRA PRADESH FINE-TUNED RESULTS (test set)")
    print(f"{'='*60}")
    print(f"  Best epoch:  {best_epoch}")
    print(f"  Best val RMSE: {best_val_rmse:.4f} m/s")
    print(f"  Test RMSE:   {test_rmse:.4f} m/s")
    print(f"  Test R2:     {test_r2:.4f}")
    print(f"  Test Bias:   {test_bias:+.4f} m/s")

    # ── Save model ────────────────────────────────────────────────
    torch.save({
        "model_state_dict": best_state,
        "architecture": {
            "input_dim": arch["input_dim"],
            "hidden_layers": arch["hidden_layers"],
            "dropout": arch["dropout"],
        },
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "source": "andhrapradesh_finetuned_from_tamilnadu_ft",
    }, OUTPUT_MODEL)

    print(f"\n  Model saved: {OUTPUT_MODEL}")
    print("="*60)


if __name__ == "__main__":
    main()