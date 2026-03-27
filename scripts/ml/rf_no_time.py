"""
RF without time features — ablation experiment
================================================
Trains RF using ONLY SAR + spatial features (no sin/cos month/doy).
This tests whether RF is actually learning SAR-wind physics
or just memorizing "it's July = high wind."

Usage:
  python scripts/ml/train_rf_no_time.py
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.ml.config_ml import (
    TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, MODELS_DIR,
    SAR_FEATURES, SPATIAL_FEATURES, TARGET,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_SPLIT,
    RF_MIN_SAMPLES_LEAF, RF_N_JOBS, RF_RANDOM_STATE, ensure_dirs
)

# Only SAR + spatial — NO time features
FEATURES_NO_TIME = SAR_FEATURES + SPATIAL_FEATURES  # 7 features
RF_NO_TIME_PATH = os.path.join(MODELS_DIR, "rf_no_time_model.pkl")


def main():
    ensure_dirs()

    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_val = pd.read_csv(VAL_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    print(f"Features: {FEATURES_NO_TIME}")
    print(f"({len(FEATURES_NO_TIME)} features — no time/seasonal info)\n")

    X_train = df_train[FEATURES_NO_TIME].values
    y_train = df_train[TARGET].values
    X_val = df_val[FEATURES_NO_TIME].values
    y_val = df_val[TARGET].values
    X_test = df_test[FEATURES_NO_TIME].values
    y_test = df_test[TARGET].values

    print(f"Training RF ({RF_N_ESTIMATORS} trees)...")
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=RF_N_JOBS,
        random_state=RF_RANDOM_STATE
    )

    start = time.time()
    rf.fit(X_train, y_train)
    print(f"  Done in {time.time() - start:.1f}s\n")

    # Save
    with open(RF_NO_TIME_PATH, "wb") as f:
        pickle.dump(rf, f)

    # Evaluate on val AND test
    for name, X, y in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
        pred = rf.predict(X)
        rmse = np.sqrt(mean_squared_error(y, pred))
        mae = mean_absolute_error(y, pred)
        r2 = r2_score(y, pred)
        bias = np.mean(pred - y)
        print(f"{name} Results:")
        print(f"  RMSE:  {rmse:.3f} m/s")
        print(f"  MAE:   {mae:.3f} m/s")
        print(f"  R2:    {r2:.4f}")
        print(f"  Bias:  {bias:+.3f} m/s")
        print()

    # Feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("Feature Importance (no time features):")
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {FEATURES_NO_TIME[idx]:<25} {importances[idx]:.4f}")

    # Comparison
    print()
    print("=" * 55)
    print("COMPARISON (Val RMSE)")
    print("=" * 55)
    print(f"  RF with time features:     0.990 m/s")
    val_pred = rf.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"  RF WITHOUT time features:  {val_rmse:.3f} m/s")
    print(f"  MLP v3:                    1.285 m/s")


if __name__ == "__main__":
    main()