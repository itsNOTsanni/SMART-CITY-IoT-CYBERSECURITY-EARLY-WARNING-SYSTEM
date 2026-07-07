from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger

logger = get_logger("preprocessing_split")

def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Splits the features and target arrays into training and testing splits.
    Stratified split is performed if targets are discrete and balanced.
    """
    logger.info(f"Splitting dataset: size={X.shape[0]} rows, test_ratio={test_size}.")
    
    try:
        # Perform stratified split if y is not None
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state, 
                stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, 
                test_size=test_size, 
                random_state=random_state
            )
            
        logger.info(f"Split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error during train-test split: {e}")
        # Fallback to non-stratified split if stratification fails
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
        logger.warning("Performed non-stratified split due to error.")
        return X_train, X_test, y_train, y_test
