"""
scripts/maharashtra/ml/finetune.py
====================================
Approach 2: Fine-tune Gujarat-trained MLP v3 on Maharashtra data.

Takes the Gujarat model weights as initialization and updates them
using a subset of Maharashtra observations. Compares fine-tuned
performance against zero-shot transfer (Approach 1).

Strategy:
  - Spatial split: 70% train / 15% val / 15% test (by point_id)
  - Load Gujarat weights + scaler
  - Fine-tune with LOW learning rate (1e-4) — we want to adapt,
    not destroy what the model already learned
  - Freeze early layers initially, then unfreeze (optional)
  - Early stopping on val RMSE

Usage:
  cd Wind_Research
  python -m scripts.maharashtra.ml.finetune
"""

import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.ml.config_ml import (
    SAR_FEATURES, SPATIAL_FEATURES, TIME_FEATURES, ALL_FEATURES, TARGET,
    VV_MAX, VV_MIN, VH_VV_RATIO_MIN, ERA5_MIN, ERA5_MAX,
    SEASON_MAP, WIND_SPEED_BINS, WIND_SPEED_LABELS
)
from scripts.ml.train_mlp_v3 import WindSpeedMLPv3

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MH_COLLOCATED = os.path.join(PROJECT_ROOT, "data", "processed",
                              "maharashtra", "maharashtra_era5_collocated.csv")
GJ_SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "feature_scaler.pkl")
GJ_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mlp_v3_wind_model.pth")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "maharashtra")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FT_MODEL_PATH = os.path.join(MODELS_DIR, "mlp_v3_maharashtra_finetuned.pth")
FT_PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "data", "processed",
                                    "maharashtra", "maharashtra_finetuned_predictions.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Fine-tuning hyperparameters
# ---------------------------------------------------------------------------
FT_LEARNING_RATE = 1e-4       # 10x lower than Gujarat training (1e-3)
FT_BATCH_SIZE = 128
FT_MAX_EPOCHS = 200
FT_EARLY_STOP_PATIENCE = 20
FT_LR_REDUCE_PATIENCE = 8
FT_LR_REDUCE_FACTOR = 0.5
FT_MIN_LR = 1e-6
FT_WEIGHT_DECAY = 1e-5

# Split ratios
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SPLIT_SEED = 42


# =========================================================================
# DATA PREPARATION
# =========================================================================
def load_and_prepare():
    """Load, clean, engineer features, and split Maharashtra data."""
    print("=" * 60)
    print("STEP 1: LOAD & PREPARE DATA")
    print("=" * 60)

    # Load
    df = pd.read_csv(MH_COLLOCATED)
    n_before = len(df)
    print(f"Loaded {n_before:,} rows")

    # Clean (same thresholds as Gujarat)
    remove_mask = (
        (df["VV"] > VV_MAX) |
        (df["VV"] < VV_MIN) |
        (df["VH_VV_ratio"] < VH_VV_RATIO_MIN) |
        (df[TARGET] < ERA5_MIN) |
        (df[TARGET] > ERA5_MAX)
    )
    df = df[~remove_mask].copy()
    print(f"After cleaning: {len(df):,} rows ({n_before - len(df)} removed)")

    # Feature engineering
    if 'month' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear

    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["season"] = df["month"].map(SEASON_MAP)

    return df


def spatial_split(df):
    """Split by point_id (same strategy as Gujarat preprocessing)."""
    print("\n" + "=" * 60)
    print("STEP 2: SPATIAL TRAIN/VAL/TEST SPLIT")
    print("=" * 60)

    points = df.groupby("point_id").agg(
        offshore_distance_km=("offshore_distance_km", "first"),
        n_obs=("VV", "count")
    ).reset_index()

    rng = np.random.RandomState(SPLIT_SEED)

    train_points, val_points, test_points = [], [], []

    for dist, group in points.groupby("offshore_distance_km"):
        point_ids = group["point_id"].tolist()
        rng.shuffle(point_ids)

        n = len(point_ids)
        n_train = max(1, int(n * TRAIN_FRAC))
        n_val = max(1, int(n * VAL_FRAC))

        train_points.extend(point_ids[:n_train])
        val_points.extend(point_ids[n_train:n_train + n_val])
        test_points.extend(point_ids[n_train + n_val:])

    df_train = df[df["point_id"].isin(train_points)].copy()
    df_val = df[df["point_id"].isin(val_points)].copy()
    df_test = df[df["point_id"].isin(test_points)].copy()

    print(f"Total points: {len(points)}")
    print(f"  Train: {len(train_points):>3} points -> {len(df_train):>6,} observations")
    print(f"  Val:   {len(val_points):>3} points -> {len(df_val):>6,} observations")
    print(f"  Test:  {len(test_points):>3} points -> {len(df_test):>6,} observations")

    # Verify no overlap
    assert len(set(train_points) & set(val_points)) == 0
    assert len(set(train_points) & set(test_points)) == 0
    assert len(set(val_points) & set(test_points)) == 0

    print(f"\nERA5 stats:")
    for name, split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        print(f"  {name}: mean={split[TARGET].mean():.2f}, "
              f"std={split[TARGET].std():.2f}")
    print()

    return df_train, df_val, df_test


def scale_data(df_train, df_val, df_test):
    """Scale using Gujarat's fitted scaler (no re-fitting)."""
    print("=" * 60)
    print("STEP 3: SCALE WITH GUJARAT SCALER")
    print("=" * 60)

    with open(GJ_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Keep raw copies for evaluation
    df_train_raw = df_train.copy()
    df_val_raw = df_val.copy()
    df_test_raw = df_test.copy()

    # Scale
    df_train[ALL_FEATURES] = scaler.transform(df_train[ALL_FEATURES])
    df_val[ALL_FEATURES] = scaler.transform(df_val[ALL_FEATURES])
    df_test[ALL_FEATURES] = scaler.transform(df_test[ALL_FEATURES])

    print(f"Scaled all splits using Gujarat scaler")
    print()

    return df_train, df_val, df_test, df_train_raw, df_val_raw, df_test_raw


# =========================================================================
# FINE-TUNING
# =========================================================================
def finetune():
    """Main fine-tuning pipeline."""

    # ── Prepare data ──────────────────────────────────────────────────
    df = load_and_prepare()
    df_train, df_val, df_test = spatial_split(df)
    df_train, df_val, df_test, df_train_raw, df_val_raw, df_test_raw = \
        scale_data(df_train, df_val, df_test)

    # Tensors
    X_train = torch.FloatTensor(df_train[ALL_FEATURES].values)
    y_train = torch.FloatTensor(df_train[TARGET].values)
    X_val = torch.FloatTensor(df_val[ALL_FEATURES].values)
    y_val = torch.FloatTensor(df_val[TARGET].values)
    X_test = torch.FloatTensor(df_test[ALL_FEATURES].values)
    y_test = torch.FloatTensor(df_test[TARGET].values)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=FT_BATCH_SIZE,
        shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    # ── Load Gujarat model ────────────────────────────────────────────
    print("=" * 60)
    print("STEP 4: LOAD GUJARAT MODEL & FINE-TUNE")
    print("=" * 60)

    checkpoint = torch.load(GJ_MODEL_PATH, map_location=device,
                            weights_only=False)
    arch = checkpoint["architecture"]

    model = WindSpeedMLPv3(
        input_dim=arch["input_dim"],
        hidden_layers=arch["hidden_layers"],
        dropout_rate=arch["dropout"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print(f"Loaded Gujarat model (epoch {checkpoint['epoch']}, "
          f"val RMSE {checkpoint['val_rmse']:.3f} m/s)")

    # ── Zero-shot baseline on this split ──────────────────────────────
    model.eval()
    with torch.no_grad():
        zs_val_pred = model(X_val.to(device)).cpu().numpy()
        zs_test_pred = model(X_test.to(device)).cpu().numpy()

    zs_val_rmse = np.sqrt(np.mean((zs_val_pred - y_val.numpy()) ** 2))
    zs_test_rmse = np.sqrt(np.mean((zs_test_pred - y_test.numpy()) ** 2))
    print(f"\nZero-shot baseline on this split:")
    print(f"  Val RMSE:  {zs_val_rmse:.3f} m/s")
    print(f"  Test RMSE: {zs_test_rmse:.3f} m/s")
    print()

    # ── Fine-tune ─────────────────────────────────────────────────────
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    criterion = nn.HuberLoss(delta=2.0)  # Same as Gujarat v3 training
    mse_criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=FT_LEARNING_RATE,
                                 weight_decay=FT_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=FT_LR_REDUCE_PATIENCE,
        factor=FT_LR_REDUCE_FACTOR,
        min_lr=FT_MIN_LR
    )

    print(f"Fine-tuning hyperparameters:")
    print(f"  Learning rate: {FT_LEARNING_RATE} (Gujarat was 1e-3)")
    print(f"  Batch size:    {FT_BATCH_SIZE}")
    print(f"  Max epochs:    {FT_MAX_EPOCHS}")
    print(f"  Early stop:    {FT_EARLY_STOP_PATIENCE} epochs")
    print(f"  Loss:          HuberLoss(delta=2.0)")
    print()
    print(f"Training on {len(X_train):,} observations...")
    print("-" * 70)

    train_losses = []
    val_losses = []
    val_rmses = []
    lrs = []

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, FT_MAX_EPOCHS + 1):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_mse = mse_criterion(val_pred, y_val).item()
            val_huber = criterion(val_pred, y_val).item()
            val_rmse = np.sqrt(val_mse)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_huber)
        new_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(avg_train_loss)
        val_losses.append(val_mse)
        val_rmses.append(val_rmse)
        lrs.append(current_lr)

        # Early stopping
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_mse,
                "val_rmse": val_rmse,
                "architecture": arch,
                "finetune_config": {
                    "base_model": "Gujarat MLP v3",
                    "lr": FT_LEARNING_RATE,
                    "train_obs": len(X_train),
                    "train_points": len(df_train["point_id"].unique())
                    if "point_id" in df_train.columns else "N/A"
                }
            }, FT_MODEL_PATH)
        else:
            patience_counter += 1

        lr_changed = new_lr != current_lr
        if epoch % 10 == 0 or epoch == 1 or lr_changed or \
           patience_counter >= FT_EARLY_STOP_PATIENCE:
            lr_note = f" <- LR reduced to {new_lr:.1e}" if lr_changed else ""
            best_note = " *" if epoch == best_epoch else ""
            print(f"  Epoch {epoch:>4d} | Train Huber: {avg_train_loss:.4f} | "
                  f"Val MSE: {val_mse:.4f} | Val RMSE: {val_rmse:.3f} m/s | "
                  f"LR: {current_lr:.1e}{lr_note}{best_note}")

        if patience_counter >= FT_EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}. "
                  f"Best epoch: {best_epoch} "
                  f"(Val RMSE: {np.sqrt(best_val_loss):.3f} m/s)")
            break

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Fine-tuning complete in {elapsed:.1f}s")
    print(f"Best model saved: {FT_MODEL_PATH}")
    print()

    # ── Load best model and evaluate on TEST set ──────────────────────
    print("=" * 60)
    print("STEP 5: EVALUATE ON HELD-OUT TEST SET")
    print("=" * 60)

    best_checkpoint = torch.load(FT_MODEL_PATH, map_location=device,
                                 weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        ft_test_pred = model(X_test.to(device)).cpu().numpy()

    y_test_np = y_test.numpy()

    # Metrics
    def compute_metrics(y_true, y_pred):
        residuals = y_pred - y_true
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        bias = np.mean(residuals)
        return {"RMSE": rmse, "MAE": mae, "R2": r2, "Bias": bias}

    ft_m = compute_metrics(y_test_np, ft_test_pred)
    zs_m = compute_metrics(y_test_np, zs_test_pred)

    # Save results
    results_path = os.path.join(OUTPUT_DIR, "finetune_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        comparison = f"""
{'=' * 65}
FINE-TUNING RESULTS: Gujarat Model -> Maharashtra
{'=' * 65}
Training:   {len(X_train):,} Maharashtra observations ({len(df_train_raw['point_id'].unique())} points)
Validation: {len(X_val):,} observations ({len(df_val_raw['point_id'].unique())} points)
Test:       {len(X_test):,} observations ({len(df_test_raw['point_id'].unique())} points)

Fine-tune config: LR={FT_LEARNING_RATE}, epochs={best_epoch}, HuberLoss(delta=2.0)

{'=' * 65}
                    Zero-Shot        Fine-Tuned       Improvement
{'=' * 65}
  RMSE (m/s):       {zs_m['RMSE']:>8.3f}          {ft_m['RMSE']:>8.3f}          {zs_m['RMSE'] - ft_m['RMSE']:>+8.3f}
  MAE (m/s):        {zs_m['MAE']:>8.3f}          {ft_m['MAE']:>8.3f}          {zs_m['MAE'] - ft_m['MAE']:>+8.3f}
  R2:               {zs_m['R2']:>8.4f}          {ft_m['R2']:>8.4f}          {ft_m['R2'] - zs_m['R2']:>+8.4f}
  Bias (m/s):       {zs_m['Bias']:>+8.3f}          {ft_m['Bias']:>+8.3f}
{'=' * 65}

Gujarat model's own val RMSE: {checkpoint['val_rmse']:.3f} m/s (home turf)
"""
        print(comparison)
        f.write(comparison)

        # Stratified by wind speed
        bin_indices = np.clip(
            np.digitize(y_test_np, WIND_SPEED_BINS) - 1,
            0, len(WIND_SPEED_LABELS) - 1
        )

        ws_header = "\nSTRATIFIED BY WIND SPEED BIN (Test Set)\n" + "-" * 65
        print(ws_header)
        f.write(ws_header + "\n")

        fmt = "{:<8} {:>5}  {:>10} {:>10} {:>10} {:>10}"
        hdr = fmt.format("Bin", "N", "ZS RMSE", "FT RMSE", "ZS Bias", "FT Bias")
        print(hdr)
        f.write(hdr + "\n")

        for i, label in enumerate(WIND_SPEED_LABELS):
            mask = bin_indices == i
            n = mask.sum()
            if n == 0:
                continue
            zs_bm = compute_metrics(y_test_np[mask], zs_test_pred[mask])
            ft_bm = compute_metrics(y_test_np[mask], ft_test_pred[mask])
            row = fmt.format(
                label, n,
                f"{zs_bm['RMSE']:.3f}", f"{ft_bm['RMSE']:.3f}",
                f"{zs_bm['Bias']:+.3f}", f"{ft_bm['Bias']:+.3f}"
            )
            print(row)
            f.write(row + "\n")

        # Stratified by season
        seasons = df_test_raw["month"].map(SEASON_MAP).values

        s_header = "\nSTRATIFIED BY SEASON (Test Set)\n" + "-" * 65
        print(s_header)
        f.write(s_header + "\n")

        for season in ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]:
            mask = seasons == season
            n = mask.sum()
            if n == 0:
                continue
            zs_sm = compute_metrics(y_test_np[mask], zs_test_pred[mask])
            ft_sm = compute_metrics(y_test_np[mask], ft_test_pred[mask])
            row = fmt.format(
                season[:8], n,
                f"{zs_sm['RMSE']:.3f}", f"{ft_sm['RMSE']:.3f}",
                f"{zs_sm['Bias']:+.3f}", f"{ft_sm['Bias']:+.3f}"
            )
            print(row)
            f.write(row + "\n")

    print(f"\nResults saved: {results_path}")

    # ── Plots ─────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    # 1. Training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(train_losses, label="Train Huber", alpha=0.8)
    axes[0].plot(val_losses, label="Val MSE", alpha=0.8)
    axes[0].axvline(best_epoch - 1, color="red", linestyle="--",
                    alpha=0.5, label=f"Best epoch ({best_epoch})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Fine-Tuning Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_rmses, color="green", alpha=0.8)
    axes[1].axhline(np.sqrt(best_val_loss), color="red", linestyle="--",
                    alpha=0.5, label=f"Best: {np.sqrt(best_val_loss):.3f} m/s")
    axes[1].axhline(zs_val_rmse, color="gray", linestyle=":",
                    alpha=0.5, label=f"Zero-shot: {zs_val_rmse:.3f} m/s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE (m/s)")
    axes[1].set_title("Validation RMSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(lrs, color="orange", alpha=0.8)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "finetune_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # 2. Scatter comparison: zero-shot vs fine-tuned (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, pred, name, color, metrics in [
        (axes[0], zs_test_pred, "Zero-Shot Transfer", "gray", zs_m),
        (axes[1], ft_test_pred, "Fine-Tuned", "steelblue", ft_m),
    ]:
        ax.scatter(y_test_np, pred, alpha=0.15, s=5, color=color,
                   rasterized=True)
        lims = [0, max(y_test_np.max(), pred.max()) + 1]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("ERA5 100m Wind Speed (m/s)")
        ax.set_ylabel("Predicted Wind Speed (m/s)")
        ax.set_title(f"{name}\nRMSE={metrics['RMSE']:.3f} m/s | "
                     f"R2={metrics['R2']:.4f}")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Maharashtra Test Set: Zero-Shot vs Fine-Tuned",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "finetune_scatter_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # 3. Spatial pattern comparison (3 panels: ERA5, zero-shot, fine-tuned)
    df_test_plot = df_test_raw.copy()
    df_test_plot["zs_pred"] = zs_test_pred
    df_test_plot["ft_pred"] = ft_test_pred

    per_point = df_test_plot.groupby(["point_id", "latitude", "longitude"]).agg(
        mean_era5=(TARGET, "mean"),
        mean_zs=("zs_pred", "mean"),
        mean_ft=("ft_pred", "mean"),
    ).reset_index()

    if len(per_point) >= 4:  # Need enough points for interpolation
        grid_lon = np.linspace(per_point.longitude.min(),
                               per_point.longitude.max(), 150)
        grid_lat = np.linspace(per_point.latitude.min(),
                               per_point.latitude.max(), 150)
        grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)
        points = per_point[["longitude", "latitude"]].values

        vmin = min(per_point["mean_era5"].min(),
                   per_point["mean_zs"].min(),
                   per_point["mean_ft"].min())
        vmax = max(per_point["mean_era5"].max(),
                   per_point["mean_zs"].max(),
                   per_point["mean_ft"].max())

        fig, axes = plt.subplots(1, 3, figsize=(20, 7))

        for ax, col, title in [
            (axes[0], "mean_era5", "ERA5 (Truth)"),
            (axes[1], "mean_zs", "Zero-Shot"),
            (axes[2], "mean_ft", "Fine-Tuned"),
        ]:
            grid_z = griddata(points, per_point[col].values,
                              (grid_lon2d, grid_lat2d), method="cubic")
            im = ax.pcolormesh(grid_lon, grid_lat, grid_z, cmap="jet",
                               shading="auto", vmin=vmin, vmax=vmax)
            ax.set_title(f"{title} (m/s)", fontweight="bold")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect("equal")
            plt.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle("Maharashtra Test Points: Spatial Pattern Comparison",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "finetune_spatial_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    # Save predictions
    df_test_raw["zero_shot_pred"] = zs_test_pred
    df_test_raw["finetuned_pred"] = ft_test_pred
    df_test_raw.to_csv(FT_PREDICTIONS_PATH, index=False)
    print(f"Predictions saved: {FT_PREDICTIONS_PATH}")

    print()
    print("=" * 65)
    print("FINE-TUNING COMPLETE")
    print("=" * 65)
    print(f"  Zero-shot RMSE:  {zs_m['RMSE']:.3f} m/s  (R2={zs_m['R2']:.4f})")
    print(f"  Fine-tuned RMSE: {ft_m['RMSE']:.3f} m/s  (R2={ft_m['R2']:.4f})")
    print(f"  Improvement:     {zs_m['RMSE'] - ft_m['RMSE']:.3f} m/s "
          f"({(1 - ft_m['RMSE']/zs_m['RMSE'])*100:.1f}% reduction)")
    print(f"\nAll outputs in: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    finetune()