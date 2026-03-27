"""
ML Pipeline Configuration
=========================
Central configuration for the wind speed prediction ML pipeline.
All hyperparameters, paths, feature definitions, and cleaning thresholds
are defined here so nothing is hardcoded in individual scripts.
"""

import os

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "ml")

# Input file (the collocated SAR + ERA5 dataset)
RAW_DATA_PATH = os.path.join(DATA_DIR, "processed", "era5_collocated.csv")

# Output files from preprocessing
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "processed", "ml_clean.csv")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "processed", "ml_train.csv")
VAL_DATA_PATH = os.path.join(DATA_DIR, "processed", "ml_val.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "processed", "ml_test.csv")
SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.pkl")

# Model save paths
MLP_MODEL_PATH = os.path.join(MODELS_DIR, "mlp_wind_model.pth")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_wind_model.pkl")

# =============================================================================
# DATA CLEANING THRESHOLDS
# =============================================================================
# Based on physical analysis of the dataset:
# - VV > 0 dB over ocean is non-physical (2 rows)
# - VV < -40 dB is below noise floor for Sentinel-1 IW mode (70 rows)
# - Negative VH/VV ratio is non-physical (same 2 rows as VV > 0)
# - ERA5 > 25 m/s kept — these are real cyclone events (Biparjoy, June 2023)
# - ERA5 < 0.3 m/s removed — SAR is unreliable at near-calm conditions
VV_MAX = 0.0          # Remove VV > 0 dB
VV_MIN = -40.0        # Remove VV < -40 dB (noise floor)
VH_VV_RATIO_MIN = 0.0 # Remove negative ratios
ERA5_MIN = 0.3        # Remove near-calm (SAR unreliable)
ERA5_MAX = 35.0       # Safety cap (keep cyclone data up to 35 m/s)

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================
# Raw SAR features from the dataset
SAR_FEATURES = ["VV", "VH", "VH_VV_ratio", "incidence_angle"]

# Spatial/contextual features
SPATIAL_FEATURES = ["offshore_distance_km", "latitude", "longitude"]

# Engineered cyclical time features (added during preprocessing)
TIME_FEATURES = ["sin_month", "cos_month", "sin_doy", "cos_doy"]

# All input features for the model (11 total)
ALL_FEATURES = SAR_FEATURES + SPATIAL_FEATURES + TIME_FEATURES

# Target variable
TARGET = "ERA5_WindSpeed_100m_ms"

# =============================================================================
# TRAIN/VAL/TEST SPLIT
# =============================================================================
# Spatial split — hold out entire point_ids to prevent data leakage
# from spatial autocorrelation
TRAIN_FRAC = 0.70   # 70% of points for training
VAL_FRAC = 0.15     # 15% of points for validation
TEST_FRAC = 0.15    # 15% of points for testing
SPLIT_SEED = 42     # Reproducibility

# =============================================================================
# MLP HYPERPARAMETERS
# =============================================================================
MLP_HIDDEN_LAYERS = [128, 64, 32]  # Neurons per hidden layer
MLP_DROPOUT = 0.2
MLP_LEARNING_RATE = 1e-3
MLP_BATCH_SIZE = 256
MLP_MAX_EPOCHS = 500
MLP_EARLY_STOP_PATIENCE = 20      # Stop if val loss doesn't improve for N epochs
MLP_LR_REDUCE_PATIENCE = 10       # Reduce LR if val loss plateaus for N epochs
MLP_LR_REDUCE_FACTOR = 0.5        # Multiply LR by this when reducing
MLP_MIN_LR = 1e-6

# =============================================================================
# RANDOM FOREST HYPERPARAMETERS
# =============================================================================
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_SPLIT = 10
RF_MIN_SAMPLES_LEAF = 5
RF_N_JOBS = -1  # Use all CPU cores
RF_RANDOM_STATE = 42

# =============================================================================
# EVALUATION
# =============================================================================
# Wind speed bins for stratified evaluation (m/s)
WIND_SPEED_BINS = [0, 3, 6, 9, 12, 15, 35]
WIND_SPEED_LABELS = ["0-3", "3-6", "6-9", "9-12", "12-15", "15+"]

# Seasons for seasonal evaluation
SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Pre-Monsoon", 4: "Pre-Monsoon", 5: "Pre-Monsoon",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon"
}


def ensure_dirs():
    """Create all necessary directories if they don't exist."""
    for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR,
              os.path.join(DATA_DIR, "processed")]:
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input features ({len(ALL_FEATURES)}): {ALL_FEATURES}")
    print(f"Target: {TARGET}")
    print(f"Split: {TRAIN_FRAC}/{VAL_FRAC}/{TEST_FRAC}")