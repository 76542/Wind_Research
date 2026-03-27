"""
Step 5c: Train Improved MLP (v3)
=================================
Keeps what worked from v2 (BatchNorm, wider layers, LeakyReLU).
Fixes what didn't: removes aggressive sample weighting, uses
Huber loss instead (robust to outliers without distorting the
training distribution).

Changes from v2:
  - Regular shuffled DataLoader (no weighted sampler)
  - HuberLoss(delta=2.0) instead of MSELoss — less sensitive to
    large errors from extreme winds, so the model doesn't collapse
    to predicting the mean to avoid big penalties
  - Slightly higher dropout (0.2) since we're back to normal sampling

Usage:
  cd /path/to/wind-research
  python scripts/ml/train_mlp_v3.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.ml.config_ml import (
    TRAIN_DATA_PATH, VAL_DATA_PATH, MODELS_DIR, OUTPUTS_DIR,
    ALL_FEATURES, TARGET, ensure_dirs
)

# ---------------------------------------------------------------------------
# V3 hyperparameters
# ---------------------------------------------------------------------------
V3_HIDDEN_LAYERS = [256, 128, 64, 32]
V3_DROPOUT = 0.2
V3_LEARNING_RATE = 1e-3
V3_BATCH_SIZE = 256
V3_MAX_EPOCHS = 500
V3_EARLY_STOP_PATIENCE = 25
V3_LR_REDUCE_PATIENCE = 10
V3_LR_REDUCE_FACTOR = 0.5
V3_MIN_LR = 1e-6
V3_HUBER_DELTA = 2.0  # Transition point between L1 and L2 loss

V3_MODEL_PATH = os.path.join(MODELS_DIR, "mlp_v3_wind_model.pth")


# =========================================================================
# MODEL (same architecture as v2)
# =========================================================================
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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x).squeeze(-1)


# =========================================================================
# TRAINING
# =========================================================================
def train_model():
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    # Load data
    print("Loading data...")
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_val = pd.read_csv(VAL_DATA_PATH)

    X_train = torch.FloatTensor(df_train[ALL_FEATURES].values)
    y_train = torch.FloatTensor(df_train[TARGET].values)
    X_val = torch.FloatTensor(df_val[ALL_FEATURES].values).to(device)
    y_val = torch.FloatTensor(df_val[TARGET].values).to(device)

    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"  Val:   {X_val.shape[0]:,} samples")
    print()

    # Normal DataLoader (no weighted sampling)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=V3_BATCH_SIZE,
        shuffle=True
    )

    # Model
    model = WindSpeedMLPv3(
        input_dim=len(ALL_FEATURES),
        hidden_layers=V3_HIDDEN_LAYERS,
        dropout_rate=V3_DROPOUT
    ).to(device)

    # Huber loss — quadratic for small errors, linear for large errors
    # This means a 10 m/s error doesn't dominate 100x more than a 1 m/s error
    criterion = nn.HuberLoss(delta=V3_HUBER_DELTA)
    # We still evaluate with MSE for fair comparison
    mse_criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=V3_LEARNING_RATE,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=V3_LR_REDUCE_PATIENCE,
        factor=V3_LR_REDUCE_FACTOR,
        min_lr=V3_MIN_LR
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model v3 architecture:")
    print(f"  Hidden:     {V3_HIDDEN_LAYERS}")
    print(f"  BatchNorm:  Yes")
    print(f"  Activation: LeakyReLU(0.01)")
    print(f"  Dropout:    {V3_DROPOUT}")
    print(f"  Loss:       HuberLoss(delta={V3_HUBER_DELTA})")
    print(f"  Parameters: {n_params:,}")
    print()

    # Training loop
    print(f"Training for up to {V3_MAX_EPOCHS} epochs "
          f"(early stop patience: {V3_EARLY_STOP_PATIENCE})...")
    print("-" * 70)

    train_losses = []
    val_losses = []
    val_rmses = []
    lrs = []

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, V3_MAX_EPOCHS + 1):
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

        # Validate (using MSE for fair RMSE comparison)
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_mse = mse_criterion(val_pred, y_val).item()
            val_huber = criterion(val_pred, y_val).item()
            val_rmse = np.sqrt(val_mse)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_huber)  # Schedule on Huber (training loss)
        new_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(avg_train_loss)
        val_losses.append(val_mse)  # Store MSE for plotting
        val_rmses.append(val_rmse)
        lrs.append(current_lr)

        # Early stopping on MSE (what we actually care about)
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_mse,
                "val_rmse": val_rmse,
                "architecture": {
                    "input_dim": len(ALL_FEATURES),
                    "hidden_layers": V3_HIDDEN_LAYERS,
                    "dropout": V3_DROPOUT,
                    "features": ALL_FEATURES,
                    "version": "v3"
                }
            }, V3_MODEL_PATH)
        else:
            patience_counter += 1

        lr_changed = new_lr != current_lr
        if epoch % 10 == 0 or epoch == 1 or lr_changed or patience_counter >= V3_EARLY_STOP_PATIENCE:
            lr_note = f" <- LR reduced to {new_lr:.1e}" if lr_changed else ""
            best_note = " *" if epoch == best_epoch else ""
            print(f"  Epoch {epoch:>4d} | Train Huber: {avg_train_loss:.4f} | "
                  f"Val MSE: {val_mse:.4f} | Val RMSE: {val_rmse:.3f} m/s | "
                  f"LR: {current_lr:.1e}{lr_note}{best_note}")

        if patience_counter >= V3_EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}. "
                  f"Best epoch: {best_epoch} (Val RMSE: {np.sqrt(best_val_loss):.3f} m/s)")
            break

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Best model saved: {V3_MODEL_PATH}")
    print(f"  Best val RMSE: {np.sqrt(best_val_loss):.3f} m/s")
    print()

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(train_losses, label="Train Huber", alpha=0.8)
    axes[0].plot(val_losses, label="Val MSE", alpha=0.8)
    axes[0].axvline(best_epoch - 1, color="red", linestyle="--",
                    alpha=0.5, label=f"Best epoch ({best_epoch})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss (v3)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_rmses, color="green", alpha=0.8)
    axes[1].axhline(np.sqrt(best_val_loss), color="red", linestyle="--",
                    alpha=0.5, label=f"Best: {np.sqrt(best_val_loss):.3f} m/s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE (m/s)")
    axes[1].set_title("Validation RMSE (v3)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(lrs, color="orange", alpha=0.8)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule (v3)")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "mlp_v3_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved: {path}")

    print()
    print("=" * 50)
    print("COMPARISON")
    print("=" * 50)
    print(f"  MLP v1 Val RMSE:  1.525 m/s")
    print(f"  MLP v2 Val RMSE:  1.848 m/s")
    print(f"  MLP v3 Val RMSE:  {np.sqrt(best_val_loss):.3f} m/s")
    print(f"  RF Val RMSE:      0.990 m/s")


if __name__ == "__main__":
    train_model()