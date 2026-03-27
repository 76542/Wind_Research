"""
Steps 1-4: Preprocessing Pipeline
===================================
Takes the raw era5_collocated.csv and produces train/val/test splits
ready for model training.

Operations:
  1. Data cleaning — remove physically impossible / noise-floor observations
  2. Feature engineering — add cyclical time encodings
  3. Spatial train/val/test split — hold out entire point_ids
  4. Feature scaling — StandardScaler fit on train only

Usage:
  cd /path/to/wind-research
  python scripts/ml/preprocess.py

Inputs:
  data/processed/era5_collocated.csv

Outputs:
  data/processed/ml_clean.csv
  data/processed/ml_train.csv
  data/processed/ml_val.csv
  data/processed/ml_test.csv
  models/feature_scaler.pkl
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Path setup — ensure imports work regardless of where the script is run from
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.ml.config_ml import (
    RAW_DATA_PATH, CLEAN_DATA_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH,
    TEST_DATA_PATH, SCALER_PATH,
    VV_MAX, VV_MIN, VH_VV_RATIO_MIN, ERA5_MIN, ERA5_MAX,
    SAR_FEATURES, SPATIAL_FEATURES, TIME_FEATURES, ALL_FEATURES, TARGET,
    TRAIN_FRAC, VAL_FRAC, TEST_FRAC, SPLIT_SEED,
    SEASON_MAP, ensure_dirs
)


# =========================================================================
# STEP 1: DATA CLEANING
# =========================================================================
def clean_data(df):
    """
    Remove physically impossible or unreliable observations.

    Criteria (with justification):
    - VV > 0 dB: Non-physical over open ocean. Likely land contamination
      or processing artifact. (2 rows)
    - VV < -40 dB: Below Sentinel-1 IW mode noise floor. Backscatter
      values here are dominated by instrument noise, not ocean signal. (70 rows)
    - VH/VV ratio < 0: Mathematically impossible for power ratios when
      both channels are in dB. Indicates corrupted data. (2 rows, same as VV > 0)
    - ERA5 < 0.3 m/s: Near-calm conditions where SAR backscatter has
      almost zero sensitivity to wind speed changes. Including these
      would add noise without signal. (222 rows)
    - ERA5 > 35 m/s: Safety cap. Our cyclone data maxes at ~28 m/s
      which we KEEP — real Biparjoy observations are valuable. This
      threshold only catches potential ERA5 processing errors.
    """
    n_before = len(df)

    # Track removals by reason (for the thesis)
    reasons = {}

    mask_vv_high = df["VV"] > VV_MAX
    reasons["VV > 0 dB (non-physical)"] = mask_vv_high.sum()

    mask_vv_low = df["VV"] < VV_MIN
    reasons["VV < -40 dB (noise floor)"] = mask_vv_low.sum()

    mask_ratio = df["VH_VV_ratio"] < VH_VV_RATIO_MIN
    reasons["VH/VV ratio < 0 (corrupted)"] = mask_ratio.sum()

    mask_calm = df[TARGET] < ERA5_MIN
    reasons["ERA5 < 0.3 m/s (near-calm)"] = mask_calm.sum()

    mask_extreme = df[TARGET] > ERA5_MAX
    reasons["ERA5 > 35 m/s (safety cap)"] = mask_extreme.sum()

    # Combined mask — remove if ANY condition is true
    remove_mask = mask_vv_high | mask_vv_low | mask_ratio | mask_calm | mask_extreme
    df_clean = df[~remove_mask].copy()

    n_after = len(df_clean)

    # Report
    print("=" * 60)
    print("STEP 1: DATA CLEANING")
    print("=" * 60)
    print(f"Input rows:   {n_before:,}")
    for reason, count in reasons.items():
        if count > 0:
            print(f"  Removed {count:>4} — {reason}")
    print(f"Output rows:  {n_after:,} ({n_before - n_after} removed, "
          f"{(n_before - n_after) / n_before * 100:.2f}%)")
    print()

    return df_clean


# =========================================================================
# STEP 2: FEATURE ENGINEERING
# =========================================================================
def engineer_features(df):
    """
    Add cyclical time encodings so the MLP can learn seasonal patterns.

    Why cyclical encoding?
    - Month 12 (December) and Month 1 (January) are adjacent seasons,
      but as raw integers they appear maximally distant (12 vs 1).
    - sin/cos encoding places them correctly on a circle.
    - Same logic for day_of_year.

    We do NOT one-hot encode months — that would add 12 sparse features
    and lose the continuity information.
    """
    print("=" * 60)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 60)

    # Cyclical month encoding
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    # Cyclical day-of-year encoding
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Add season column (for stratified evaluation later, not a model feature)
    df["season"] = df["month"].map(SEASON_MAP)

    print(f"Added features: {TIME_FEATURES}")
    print(f"Added column:   season (for evaluation only)")
    print(f"Total model input features: {len(ALL_FEATURES)}")
    print(f"  SAR:     {SAR_FEATURES}")
    print(f"  Spatial: {SPATIAL_FEATURES}")
    print(f"  Time:    {TIME_FEATURES}")
    print()

    return df


# =========================================================================
# STEP 3: SPATIAL TRAIN/VAL/TEST SPLIT
# =========================================================================
def spatial_split(df):
    """
    Split data by holding out entire point_ids.

    Why spatial split instead of random split?
    - Observations from the same point on nearby dates have correlated
      wind speeds (weather persistence) and identical SAR geometry
      (same orbit, same incidence angle).
    - A random split would leak this correlation into the validation
      and test sets, giving an inflated performance estimate.
    - By holding out entire points, we test whether the model
      generalizes to UNSEEN LOCATIONS — which is the real use case
      (predicting wind at new offshore sites).

    Stratification: We stratify by offshore_distance_km to ensure each
    split has a representative mix of 20/40/60/80 km points.
    """
    print("=" * 60)
    print("STEP 3: SPATIAL TRAIN/VAL/TEST SPLIT")
    print("=" * 60)

    # Get unique points with their distance band
    points = df.groupby("point_id").agg(
        offshore_distance_km=("offshore_distance_km", "first"),
        n_obs=("VV", "count")
    ).reset_index()

    rng = np.random.RandomState(SPLIT_SEED)

    train_points = []
    val_points = []
    test_points = []

    # Stratify by distance band
    for dist, group in points.groupby("offshore_distance_km"):
        point_ids = group["point_id"].tolist()
        rng.shuffle(point_ids)

        n = len(point_ids)
        n_train = max(1, int(n * TRAIN_FRAC))
        n_val = max(1, int(n * VAL_FRAC))
        # Test gets the remainder

        train_points.extend(point_ids[:n_train])
        val_points.extend(point_ids[n_train:n_train + n_val])
        test_points.extend(point_ids[n_train + n_val:])

    # Create splits
    df_train = df[df["point_id"].isin(train_points)].copy()
    df_val = df[df["point_id"].isin(val_points)].copy()
    df_test = df[df["point_id"].isin(test_points)].copy()

    # Report
    print(f"Total points: {len(points)}")
    print(f"  Train: {len(train_points):>4} points -> {len(df_train):>6,} observations")
    print(f"  Val:   {len(val_points):>4} points -> {len(df_val):>6,} observations")
    print(f"  Test:  {len(test_points):>4} points -> {len(df_test):>6,} observations")
    print()

    # Verify no overlap
    assert len(set(train_points) & set(val_points)) == 0, "Train/Val overlap!"
    assert len(set(train_points) & set(test_points)) == 0, "Train/Test overlap!"
    assert len(set(val_points) & set(test_points)) == 0, "Val/Test overlap!"

    # Show distance distribution per split
    print("Distance band distribution (% of points in each split):")
    for split_name, split_pts in [("Train", train_points),
                                   ("Val", val_points),
                                   ("Test", test_points)]:
        split_df = points[points["point_id"].isin(split_pts)]
        dist_counts = split_df["offshore_distance_km"].value_counts().sort_index()
        dist_pcts = (dist_counts / len(split_df) * 100).round(1)
        print(f"  {split_name}: { {int(k): f'{v}%' for k, v in dist_pcts.items()} }")
    print()

    # Show ERA5 distribution per split (sanity check)
    print("ERA5 wind speed stats per split:")
    for name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        era5 = split_df[TARGET]
        print(f"  {name}: mean={era5.mean():.2f}, std={era5.std():.2f}, "
              f"median={era5.median():.2f}")
    print()

    return df_train, df_val, df_test


# =========================================================================
# STEP 4: FEATURE SCALING
# =========================================================================
def scale_features(df_train, df_val, df_test):
    """
    Apply StandardScaler (zero mean, unit variance) to all input features.

    Critical rules:
    - Fit the scaler on TRAINING data only.
    - Transform val and test using the SAME fitted scaler.
    - This prevents data leakage from val/test statistics into training.
    - The scaler is saved to disk so it can be reused at inference time.

    Why StandardScaler over MinMaxScaler?
    - MLP with ReLU activation benefits from zero-centered inputs.
    - StandardScaler is more robust to outliers than MinMax.
    - The SAR features (VV, VH) already have a roughly normal distribution.
    """
    print("=" * 60)
    print("STEP 4: FEATURE SCALING")
    print("=" * 60)

    scaler = StandardScaler()

    # Fit on train only
    df_train[ALL_FEATURES] = scaler.fit_transform(df_train[ALL_FEATURES])

    # Transform val and test with the same scaler
    df_val[ALL_FEATURES] = scaler.transform(df_val[ALL_FEATURES])
    df_test[ALL_FEATURES] = scaler.transform(df_test[ALL_FEATURES])

    # Save scaler
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Scaler fit on {len(ALL_FEATURES)} features from training set")
    print(f"Scaler saved to: {SCALER_PATH}")
    print()

    # Show scaled ranges (sanity check)
    print("Scaled feature ranges (train set):")
    for feat in ALL_FEATURES:
        vals = df_train[feat]
        print(f"  {feat:<25} mean={vals.mean():>7.3f}  std={vals.std():>6.3f}  "
              f"range=[{vals.min():>7.2f}, {vals.max():>6.2f}]")
    print()

    return df_train, df_val, df_test


# =========================================================================
# MAIN PIPELINE
# =========================================================================
def main():
    ensure_dirs()

    # Load raw data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns from:")
    print(f"  {RAW_DATA_PATH}")
    print()

    # Step 1: Clean
    df = clean_data(df)

    # Step 2: Feature engineering
    df = engineer_features(df)

    # Save the full clean dataset (before splitting, useful for EDA)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Clean dataset saved: {CLEAN_DATA_PATH}")
    print()

    # Step 3: Spatial split
    df_train, df_val, df_test = spatial_split(df)

    # Step 4: Scale features
    df_train, df_val, df_test = scale_features(df_train, df_val, df_test)

    # Save splits
    df_train.to_csv(TRAIN_DATA_PATH, index=False)
    df_val.to_csv(VAL_DATA_PATH, index=False)
    df_test.to_csv(TEST_DATA_PATH, index=False)

    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Clean data:  {CLEAN_DATA_PATH}")
    print(f"  Train split: {TRAIN_DATA_PATH} ({len(df_train):,} rows)")
    print(f"  Val split:   {VAL_DATA_PATH} ({len(df_val):,} rows)")
    print(f"  Test split:  {TEST_DATA_PATH} ({len(df_test):,} rows)")
    print(f"  Scaler:      {SCALER_PATH}")
    print()
    print("Next step: python scripts/ml/train_mlp.py")


if __name__ == "__main__":
    main()