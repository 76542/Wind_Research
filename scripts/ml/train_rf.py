"""
Step 6: Train Random Forest Baseline
======================================
Trains a Random Forest regressor as a comparison baseline for the MLP.

Why Random Forest as baseline:
  - Strong ML baseline for tabular data
  - Provides feature importance rankings (useful for thesis)
  - If MLP can't beat RF, the neural network isn't adding value

Usage:
  cd /path/to/wind-research
  python scripts/ml/train_rf.py

Inputs:
  data/processed/ml_train.csv
  data/processed/ml_val.csv

Outputs:
  models/rf_wind_model.pkl
  outputs/ml/rf_feature_importance.png
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.ml.config_ml import (
    TRAIN_DATA_PATH, VAL_DATA_PATH, RF_MODEL_PATH, OUTPUTS_DIR,
    ALL_FEATURES, TARGET,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_SPLIT,
    RF_MIN_SAMPLES_LEAF, RF_N_JOBS, RF_RANDOM_STATE, ensure_dirs
)


def train_rf():
    ensure_dirs()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_val = pd.read_csv(VAL_DATA_PATH)

    X_train = df_train[ALL_FEATURES].values
    y_train = df_train[TARGET].values
    X_val = df_val[ALL_FEATURES].values
    y_val = df_val[TARGET].values

    print(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"  Val:   {X_val.shape[0]:,} samples")
    print()

    # ------------------------------------------------------------------
    # Train Random Forest
    # ------------------------------------------------------------------
    print(f"Training Random Forest ({RF_N_ESTIMATORS} trees, "
          f"max_depth={RF_MAX_DEPTH})...")

    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=RF_N_JOBS,
        random_state=RF_RANDOM_STATE,
        verbose=0
    )

    start_time = time.time()
    rf.fit(X_train, y_train)
    elapsed = time.time() - start_time

    print(f"  Training complete in {elapsed:.1f}s")
    print()

    # ------------------------------------------------------------------
    # Evaluate on train and validation set
    # ------------------------------------------------------------------
    train_pred = rf.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

    val_pred = rf.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    val_bias = np.mean(val_pred - y_val)

    print("Results:")
    print(f"  Train RMSE: {train_rmse:.3f} m/s")
    print(f"  Val RMSE:   {val_rmse:.3f} m/s")
    print(f"  Val MAE:    {val_mae:.3f} m/s")
    print(f"  Val R2:     {val_r2:.4f}")
    print(f"  Val Bias:   {val_bias:+.3f} m/s")
    print()

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    with open(RF_MODEL_PATH, "wb") as f:
        pickle.dump(rf, f)
    print(f"Model saved: {RF_MODEL_PATH}")

    # ------------------------------------------------------------------
    # Feature importance plot
    # ------------------------------------------------------------------
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nFeature Importance Ranking:")
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {ALL_FEATURES[idx]:<25} {importances[idx]:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_idx = np.argsort(importances)
    ax.barh(range(len(ALL_FEATURES)),
            importances[sorted_idx],
            color="steelblue", edgecolor="white")
    ax.set_yticks(range(len(ALL_FEATURES)))
    ax.set_yticklabels([ALL_FEATURES[i] for i in sorted_idx])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest — Feature Importance for Wind Speed Prediction")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUTS_DIR, "rf_feature_importance.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFeature importance plot saved: {plot_path}")

    # ------------------------------------------------------------------
    # Quick comparison note
    # ------------------------------------------------------------------
    print()
    print("=" * 50)
    print("QUICK COMPARISON (Validation Set)")
    print("=" * 50)
    print(f"  Random Forest Val RMSE: {val_rmse:.3f} m/s")
    print(f"  (Compare with MLP Val RMSE from train_mlp.py)")
    print()
    print("Next step: python scripts/ml/evaluate.py")


if __name__ == "__main__":
    train_rf()