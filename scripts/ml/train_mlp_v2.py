"""
Step 5b: Train Improved MLP (v2)
=================================
Improvements over v1:
  1. Wider architecture: 256 -> 128 -> 64 -> 32 (vs 128 -> 64 -> 32)
  2. Batch Normalization after each layer (stabilizes training, often
     the single biggest improvement for MLPs on tabular data)
  3. LeakyReLU instead of ReLU (prevents dead neurons)
  4. Sample weighting — upweights high-wind observations so the model
     doesn't just predict towards the mean
  5. Weight initialization (Kaiming)

Usage:
  cd /path/to/wind-research
  python scripts/ml/train_mlp_v2.py

Inputs:
  data/processed/ml_train.csv
  data/processed/ml_val.csv

Outputs:
  models/mlp_v2_wind_model.pth
  outputs/ml/mlp_v2_training_curves.png
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.ml.config_ml import (
    TRAIN_DATA_PATH, VAL_DATA_PATH, MODELS_DIR, OUTPUTS_DIR,
    ALL_FEATURES, TARGET, ensure_dirs
)

# ---------------------------------------------------------------------------
# V2-specific hyperparameters (overrides config_ml defaults)
# ---------------------------------------------------------------------------
V2_HIDDEN_LAYERS = [256, 128, 64, 32]
V2_DROPOUT = 0.15
V2_LEARNING_RATE = 5e-4
V2_BATCH_SIZE = 512
V2_MAX_EPOCHS = 600
V2_EARLY_STOP_PATIENCE = 30
V2_LR_REDUCE_PATIENCE = 12
V2_LR_REDUCE_FACTOR = 0.5
V2_MIN_LR = 1e-6

V2_MODEL_PATH = os.path.join(MODELS_DIR, "mlp_v2_wind_model.pth")


# =========================================================================
# IMPROVED MODEL DEFINITION
# =========================================================================
class WindSpeedMLPv2(nn.Module):
    """
    Improved MLP with Batch Normalization and LeakyReLU.

    Key differences from v1:
    - BatchNorm before activation — normalizes layer inputs, reduces
      internal covariate shift, allows higher learning rates.
    - LeakyReLU(0.01) — prevents dead neurons that ReLU can create.
    - Kaiming initialization — proper weight init for ReLU-family
      activations.
    - Wider first layer (256) to capture more feature interactions.
    """

    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super(WindSpeedMLPv2, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Kaiming initialization
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
# SAMPLE WEIGHTS — upweight high winds
# =========================================================================
def compute_sample_weights(y_train):
    """
    Create sample weights that give more importance to high-wind observations.

    Problem: 65% of data is below 6 m/s. The model learns to predict
    ~5 m/s for everything because that minimizes average loss.

    Solution: Weight each sample inversely proportional to how common
    its wind speed bin is. High winds (rare) get higher weight.
    """
    bins = [0, 3, 6, 9, 12, 15, 40]
    bin_indices = np.digitize(y_train, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

    # Count samples per bin
    bin_counts = np.bincount(bin_indices, minlength=len(bins) - 1)
    # Weight = total_samples / (n_bins * bin_count)
    bin_weights = len(y_train) / (len(bins) * bin_counts.astype(float) + 1)

    sample_weights = bin_weights[bin_indices]

    print("  Sample weight multipliers by wind speed bin:")
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if bin_counts[i] > 0:
            print(f"    {lo:>2}-{hi:<2} m/s: {bin_counts[i]:>5} samples, "
                  f"weight = {bin_weights[i]:.2f}x")

    return torch.DoubleTensor(sample_weights)


# =========================================================================
# TRAINING LOOP
# =========================================================================
def train_model():
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_val = pd.read_csv(VAL_DATA_PATH)

    X_train = torch.FloatTensor(df_train[ALL_FEATURES].values)
    y_train = torch.FloatTensor(df_train[TARGET].values)
    X_val = torch.FloatTensor(df_val[ALL_FEATURES].values).to(device)
    y_val = torch.FloatTensor(df_val[TARGET].values).to(device)

    print(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"  Val:   {X_val.shape[0]:,} samples")
    print()

    # ------------------------------------------------------------------
    # Weighted sampler — high winds sampled more often
    # ------------------------------------------------------------------
    print("Computing sample weights...")
    sample_weights = compute_sample_weights(df_train[TARGET].values)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=V2_BATCH_SIZE,
        sampler=sampler  # Weighted sampling instead of shuffle
    )
    print()

    # ------------------------------------------------------------------
    # Initialize model
    # ------------------------------------------------------------------
    model = WindSpeedMLPv2(
        input_dim=len(ALL_FEATURES),
        hidden_layers=V2_HIDDEN_LAYERS,
        dropout_rate=V2_DROPOUT
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=V2_LEARNING_RATE,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=V2_LR_REDUCE_PATIENCE,
        factor=V2_LR_REDUCE_FACTOR,
        min_lr=V2_MIN_LR
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model v2 architecture:")
    print(f"  Input:      {len(ALL_FEATURES)} features")
    print(f"  Hidden:     {V2_HIDDEN_LAYERS}")
    print(f"  BatchNorm:  Yes")
    print(f"  Activation: LeakyReLU(0.01)")
    print(f"  Dropout:    {V2_DROPOUT}")
    print(f"  Output:     1 (wind speed, m/s)")
    print(f"  Parameters: {n_params:,}")
    print()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"Training for up to {V2_MAX_EPOCHS} epochs "
          f"(early stop patience: {V2_EARLY_STOP_PATIENCE})...")
    print("-" * 70)

    train_losses = []
    val_losses = []
    val_rmses = []
    lrs = []

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, V2_MAX_EPOCHS + 1):
        # --- Train ---
        model.train()
        epoch_train_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_train_loss / n_batches

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            val_loss = criterion(val_predictions, y_val).item()
            val_rmse = np.sqrt(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_rmses.append(val_rmse)
        lrs.append(current_lr)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "architecture": {
                    "input_dim": len(ALL_FEATURES),
                    "hidden_layers": V2_HIDDEN_LAYERS,
                    "dropout": V2_DROPOUT,
                    "features": ALL_FEATURES,
                    "version": "v2"
                }
            }, V2_MODEL_PATH)
        else:
            patience_counter += 1

        lr_changed = new_lr != current_lr
        if epoch % 10 == 0 or epoch == 1 or lr_changed or patience_counter >= V2_EARLY_STOP_PATIENCE:
            lr_note = f" <- LR reduced to {new_lr:.1e}" if lr_changed else ""
            best_note = " *" if epoch == best_epoch else ""
            print(f"  Epoch {epoch:>4d} | Train MSE: {avg_train_loss:.4f} | "
                  f"Val MSE: {val_loss:.4f} | Val RMSE: {val_rmse:.3f} m/s | "
                  f"LR: {current_lr:.1e}{lr_note}{best_note}")

        if patience_counter >= V2_EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}. "
                  f"Best epoch: {best_epoch} (Val RMSE: {np.sqrt(best_val_loss):.3f} m/s)")
            break

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Best model saved: {V2_MODEL_PATH}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val RMSE: {np.sqrt(best_val_loss):.3f} m/s")
    print()

    # ------------------------------------------------------------------
    # Plot training curves
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(train_losses, label="Train MSE", alpha=0.8)
    axes[0].plot(val_losses, label="Val MSE", alpha=0.8)
    axes[0].axvline(best_epoch - 1, color="red", linestyle="--",
                    alpha=0.5, label=f"Best epoch ({best_epoch})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training & Validation Loss (v2)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_rmses, color="green", alpha=0.8)
    axes[1].axhline(np.sqrt(best_val_loss), color="red", linestyle="--",
                    alpha=0.5, label=f"Best: {np.sqrt(best_val_loss):.3f} m/s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE (m/s)")
    axes[1].set_title("Validation RMSE (v2)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(lrs, color="orange", alpha=0.8)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule (v2)")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = os.path.join(OUTPUTS_DIR, "mlp_v2_training_curves.png")
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved: {curves_path}")

    # ------------------------------------------------------------------
    # Compare with v1
    # ------------------------------------------------------------------
    print()
    print("=" * 50)
    print("COMPARISON")
    print("=" * 50)
    print(f"  MLP v1 Val RMSE:  1.525 m/s")
    print(f"  MLP v2 Val RMSE:  {np.sqrt(best_val_loss):.3f} m/s")
    print(f"  RF Val RMSE:      0.990 m/s")


if __name__ == "__main__":
    train_model()