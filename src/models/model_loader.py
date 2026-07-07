import os
import joblib
from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger("models_loader")

def load_rf_artifacts():
    """Loads and returns Random Forest artifacts: model, scaler, encoder, features."""
    filepath = os.path.join(MODELS_DIR, 'random_forest.pkl')
    if not os.path.exists(filepath):
        logger.error(f"Random Forest pkl not found at {filepath}")
        raise FileNotFoundError(f"RF artifacts missing: run train_rf.py first.")
    try:
        artifacts = joblib.load(filepath)
        logger.info("Loaded Random Forest artifacts successfully.")
        return artifacts
    except Exception as e:
        logger.error(f"Error loading Random Forest artifacts from {filepath}: {e}")
        raise

def load_xgb_artifacts():
    """Loads and returns XGBoost artifacts: model, scaler, encoder, features."""
    filepath = os.path.join(MODELS_DIR, 'xgboost.pkl')
    if not os.path.exists(filepath):
        logger.error(f"XGBoost pkl not found at {filepath}")
        raise FileNotFoundError(f"XGBoost artifacts missing: run train_xgboost.py first.")
    try:
        artifacts = joblib.load(filepath)
        logger.info("Loaded XGBoost artifacts successfully.")
        return artifacts
    except Exception as e:
        logger.error(f"Error loading XGBoost artifacts from {filepath}: {e}")
        raise
