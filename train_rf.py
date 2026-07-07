import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.clean_data import get_cleaned_iot23
from src.preprocessing.feature_engineering import engineer_iot23_features
from src.preprocessing.data_split import split_dataset
from src.models.random_forest import train_rf
from src.utils.logger import get_logger

logger = get_logger("train_rf_cli")

def main():
    logger.info("Starting Random Forest Model Training Pipeline...")
    
    # 1. Load and clean data
    df = get_cleaned_iot23()
    
    # 2. Extract features and scale
    X, y, scaler, label_encoder = engineer_iot23_features(df)
    
    # 3. Train-test split
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    # 4. Train RF model
    train_rf(X_train, y_train, X_test, y_test, scaler, label_encoder)
    
    logger.info("Random Forest Training Pipeline execution completed successfully.")

if __name__ == '__main__':
    main()
