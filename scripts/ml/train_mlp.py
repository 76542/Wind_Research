"""
Step 5: Train MLP Model
========================
Trains a Multi-Layer Perceptron (feedforward neural network) to predict
100m hub-height wind speed from SAR backscatter features.

Architecture:
  Input(11) -> Dense(128, ReLU) -> Dropout -> Dense(64, ReLU) -> Dropout
  -> Dense(32, ReLU) -> Dense(1, linear)

Training strategy:
  - Adam optimizer with learning rate scheduling (reduce on plateau)
  - Early stopping on validation loss (patience = 20 epochs)
  - Batch training with DataLoader

Usage:
  cd /path/to/wind-research
  python scripts/ml/train_mlp.py

Inputs:
  data/processed/ml_train.csv
  data/processed/ml_val.csv

Outputs:
  models/mlp_wind_model.pth
  outputs/ml/mlp_training_curves.png
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
    TRAIN_DATA_PATH, VAL_DATA_PATH, MLP_MODEL_PATH, OUTPUTS_DIR,
    ALL_FEATURES, TARGET,
    MLP_HIDDEN_LAYERS, MLP_DROPOUT, MLP_LEARNING_RATE, MLP_BATCH_SIZE,
    MLP_MAX_EPOCHS, MLP_EARLY_STOP_PATIENCE, MLP_LR_REDUCE_PATIENCE,
    MLP_LR_REDUCE_FACTOR, MLP_MIN_LR, ensure_dirs
)


# =========================================================================
# MODEL DEFINITION
# =========================================================================
class WindSpeedMLP(nn.Module):
    """
    Multi-Layer Perceptron for wind speed regression.

    Architecture is configurable via config_ml.MLP_HIDDEN_LAYERS.
    Default: [128, 64, 32] -> three hidden layers with decreasing width.

    Why this architecture:
    - Decreasing width acts as a feature funnel — forces the network
      to learn increasingly abstract representations.
    - Dropout after each hidden layer prevents overfitting on 26K
      training samples.
    - Linear output (no activation) because wind speed is continuous
      and can take any positive value.
    - ReLU activation — standard for regression, no vanishing gradient
      issues, fast to compute.
    """

    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super(WindSpeedMLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer — single neuron, no activation (regression)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


# =========================================================================
# DATA LOADING
# =========================================================================
def load_split(csv_path):
    """Load a preprocessed CSV and return feature tensor + target tensor."""
    df = pd.read_csv(csv_path)
    X = torch.FloatTensor(df[ALL_FEATURES].values)
    y = torch.FloatTensor(df[TARGET].values)
    return X, y


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
    X_train, y_train = load_split(TRAIN_DATA_PATH)
    X_val, y_val = load_split(VAL_DATA_PATH)
    print(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"  Val:   {X_val.shape[0]:,} samples")
    print()

    # DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=MLP_BATCH_SIZE,
                              shuffle=True)

    # Val as single batch (small enough)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    # ------------------------------------------------------------------
    # Initialize model, loss, optimizer, scheduler
    # ------------------------------------------------------------------
    model = WindSpeedMLP(
        input_dim=len(ALL_FEATURES),
        hidden_layers=MLP_HIDDEN_LAYERS,
        dropout_rate=MLP_DROPOUT
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=MLP_LR_REDUCE_PATIENCE,
        factor=MLP_LR_REDUCE_FACTOR,
        min_lr=MLP_MIN_LR
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model architecture:")
    print(f"  Input:   {len(ALL_FEATURES)} features")
    print(f"  Hidden:  {MLP_HIDDEN_LAYERS}")
    print(f"  Dropout: {MLP_DROPOUT}")
    print(f"  Output:  1 (wind speed, m/s)")
    print(f"  Total trainable parameters: {n_params:,}")
    print()

    # ------------------------------------------------------------------
    # Training loop with early stopping
    # ------------------------------------------------------------------
    print(f"Training for up to {MLP_MAX_EPOCHS} epochs "
          f"(early stop patience: {MLP_EARLY_STOP_PATIENCE})...")
    print("-" * 70)

    train_losses = []
    val_losses = []
    val_rmses = []
    lrs = []

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, MLP_MAX_EPOCHS + 1):
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

        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        # Record history
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_rmses.append(val_rmse)
        lrs.append(current_lr)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "architecture": {
                    "input_dim": len(ALL_FEATURES),
                    "hidden_layers": MLP_HIDDEN_LAYERS,
                    "dropout": MLP_DROPOUT,
                    "features": ALL_FEATURES
                }
            }, MLP_MODEL_PATH)
        else:
            patience_counter += 1

        # Print progress every 10 epochs or on LR change or at end
        lr_changed = new_lr != current_lr
        if epoch % 10 == 0 or epoch == 1 or lr_changed or patience_counter >= MLP_EARLY_STOP_PATIENCE:
            lr_note = f" <- LR reduced to {new_lr:.1e}" if lr_changed else ""
            best_note = " *" if epoch == best_epoch else ""
            print(f"  Epoch {epoch:>4d} | Train MSE: {avg_train_loss:.4f} | "
                  f"Val MSE: {val_loss:.4f} | Val RMSE: {val_rmse:.3f} m/s | "
                  f"LR: {current_lr:.1e}{lr_note}{best_note}")

        # Stop if patience exhausted
        if patience_counter >= MLP_EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}. "
                  f"Best epoch: {best_epoch} (Val RMSE: {np.sqrt(best_val_loss):.3f} m/s)")
            break

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Best model saved: {MLP_MODEL_PATH}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val RMSE: {np.sqrt(best_val_loss):.3f} m/s")
    print()

    # ------------------------------------------------------------------
    # Plot training curves
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curves
    axes[0].plot(train_losses, label="Train MSE", alpha=0.8)
    axes[0].plot(val_losses, label="Val MSE", alpha=0.8)
    axes[0].axvline(best_epoch - 1, color="red", linestyle="--",
                    alpha=0.5, label=f"Best epoch ({best_epoch})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Val RMSE
    axes[1].plot(val_rmses, color="green", alpha=0.8)
    axes[1].axhline(np.sqrt(best_val_loss), color="red", linestyle="--",
                    alpha=0.5, label=f"Best: {np.sqrt(best_val_loss):.3f} m/s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE (m/s)")
    axes[1].set_title("Validation RMSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning rate
    axes[2].plot(lrs, color="orange", alpha=0.8)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = os.path.join(OUTPUTS_DIR, "mlp_training_curves.png")
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved: {curves_path}")


if __name__ == "__main__":
    train_model()