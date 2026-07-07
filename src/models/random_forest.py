import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.utils.config import RF_PARAMS, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger("models_rf")

class TemperatureScaler:
    """Confidence Calibration using Temperature Scaling for Multi-class predictions."""
    def __init__(self):
        self.T = 1.0
        
    def fit(self, probs, y_true):
        from scipy.optimize import minimize
        from sklearn.metrics import log_loss
        
        # Clamp probabilities to avoid log(0) or log(1) instability
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        logits = np.log(probs)
        
        def loss_fn(T):
            scaled_logits = logits / T[0]
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            scaled_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            return log_loss(y_true, scaled_probs)
            
        res = minimize(loss_fn, x0=[1.0], bounds=[(1e-3, 10.0)], method='L-BFGS-B')
        self.T = float(res.x[0])
        logger.info(f"Optimal calibration temperature T = {self.T:.4f}")
        return self.T
        
    def predict(self, probs):
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        logits = np.log(probs)
        scaled_logits = logits / self.T
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def train_rf(X_train, y_train, X_test, y_test, scaler, label_encoder):
    """
    Trains the Random Forest model and saves it along with the scaler, label encoder, and calibrated temperature scaler.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold, train_test_split
    
    # 1. Temperature Calibration on Validation Set
    logger.info("Splitting training data for calibration validation set...")
    X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    logger.info("Fitting calibration base Random Forest model...")
    cal_model = RandomForestClassifier(**RF_PARAMS)
    cal_model.fit(X_tr_sub, y_tr_sub)
    
    val_probs = cal_model.predict_proba(X_val)
    temp_scaler = TemperatureScaler()
    temp_scaler.fit(val_probs, y_val)
    
    # 2. Combine train and test to get the full dataset for CV
    if isinstance(X_train, pd.DataFrame):
        X_full = pd.concat([X_train, X_test]).reset_index(drop=True)
    else:
        X_full = np.concatenate([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])
    
    # Perform 5-fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracies = []
    
    logger.info("Executing 5-Fold Stratified Cross-Validation for Random Forest...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        if isinstance(X_full, pd.DataFrame):
            X_tr, X_va = X_full.iloc[train_idx], X_full.iloc[val_idx]
        else:
            X_tr, X_va = X_full[train_idx], X_full[val_idx]
        y_tr, y_va = y_full[train_idx], y_full[val_idx]
        
        fold_model = RandomForestClassifier(**RF_PARAMS)
        fold_model.fit(X_tr, y_tr)
        
        # Cross-validation reports calibrated probabilities
        raw_probs = fold_model.predict_proba(X_va)
        cal_probs = temp_scaler.predict(raw_probs)
        preds = np.argmax(cal_probs, axis=1)
        
        acc = accuracy_score(y_va, preds)
        cv_accuracies.append(acc)
        logger.info(f"Random Forest Fold {fold+1}/5 Calibrated Accuracy: {acc * 100:.2f}%")
        
    mean_acc = np.mean(cv_accuracies)
    std_acc = np.std(cv_accuracies)
    logger.info(f"Random Forest 5-Fold CV Calibrated Accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    
    # Fit final model on the training split
    logger.info("Initializing final Random Forest Classifier...")
    model = RandomForestClassifier(**RF_PARAMS)
    
    logger.info("Fitting final Random Forest Model...")
    model.fit(X_train, y_train)
    
    # Evaluate final calibrated model on test split
    raw_test_probs = model.predict_proba(X_test)
    cal_test_probs = temp_scaler.predict(raw_test_probs)
    y_pred = np.argmax(cal_test_probs, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Random Forest Final Calibrated Test Accuracy: {accuracy * 100:.2f}%")
    
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)
    
    # Save the cross-validation statistics in results/
    cv_summary_path = os.path.join(os.path.dirname(MODELS_DIR), 'results', 'rf_cv_results.txt')
    os.makedirs(os.path.dirname(cv_summary_path), exist_ok=True)
    with open(cv_summary_path, 'w') as f:
        f.write(f"RF CV Accuracies: {cv_accuracies}\n")
        f.write(f"RF Mean CV Accuracy: {mean_acc:.6f}\n")
        f.write(f"RF CV Accuracy Std: {std_acc:.6f}\n")
    
    # Package model artifacts
    save_path = os.path.join(MODELS_DIR, 'random_forest.pkl')
    artifacts = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'temperature_scaler': temp_scaler,
        'features': list(X_train.columns)
    }
    
    logger.info(f"Saving Random Forest artifacts to {save_path}...")
    joblib.dump(artifacts, save_path)
    
    return model, accuracy, report, matrix


