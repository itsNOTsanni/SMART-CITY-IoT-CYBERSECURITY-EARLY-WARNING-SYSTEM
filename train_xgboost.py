import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.clean_data import get_cleaned_ciciot2023
from src.preprocessing.feature_engineering import engineer_ciciot2023_features
from src.preprocessing.data_split import split_dataset
from src.models.xgboost_model import train_xgb
from src.utils.logger import get_logger

logger = get_logger("train_xgboost_cli")

def main():
    logger.info("Starting XGBoost Model Training Pipeline...")
    
    # 1. Load and clean data
    df = get_cleaned_ciciot2023()
    
    # 2. Extract features and scale
    X, y, scaler, label_encoder = engineer_ciciot2023_features(df)
    
    # 3. Train-test split
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    # 4. Train XGB model
    train_xgb(X_train, y_train, X_test, y_test, scaler, label_encoder)
    
    logger.info("XGBoost Training Pipeline execution completed successfully.")

if __name__ == '__main__':
    main()
