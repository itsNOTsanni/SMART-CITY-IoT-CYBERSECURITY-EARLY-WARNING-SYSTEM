import os
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.utils.config import XGB_PARAMS, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger("models_xgb")

class TemperatureScaler:
    """Confidence Calibration using Temperature Scaling for Multi-class predictions."""
    def __init__(self):
        self.T = 1.0
        
    def fit(self, probs, y_true):
        from scipy.optimize import minimize
        from sklearn.metrics import log_loss
        import numpy as np
        
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
        import numpy as np
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        logits = np.log(probs)
        scaled_logits = logits / self.T
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def train_xgb(X_train, y_train, X_test, y_test, scaler, label_encoder):
    """
    Trains the XGBoost model and saves it along with the scaler, label encoder, and calibrated temperature scaler.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold, train_test_split
    
    # 1. Temperature Calibration on Validation Set
    logger.info("Splitting training data for calibration validation set...")
    X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    logger.info("Fitting calibration base XGBoost model...")
    cal_model = XGBClassifier(**X_xgb_params_adapt(X_tr_sub, y_tr_sub))
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
    
    logger.info("Executing 5-Fold Stratified Cross-Validation for XGBoost...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        if isinstance(X_full, pd.DataFrame):
            X_tr, X_va = X_full.iloc[train_idx], X_full.iloc[val_idx]
        else:
            X_tr, X_va = X_full[train_idx], X_full[val_idx]
        y_tr, y_va = y_full[train_idx], y_full[val_idx]
        
        fold_model = XGBClassifier(**X_xgb_params_adapt(X_tr, y_tr))
        fold_model.fit(X_tr, y_tr)
        
        # Cross-validation reports calibrated probabilities
        raw_probs = fold_model.predict_proba(X_va)
        cal_probs = temp_scaler.predict(raw_probs)
        preds = np.argmax(cal_probs, axis=1)
        
        acc = accuracy_score(y_va, preds)
        cv_accuracies.append(acc)
        logger.info(f"XGBoost Fold {fold+1}/5 Calibrated Accuracy: {acc * 100:.2f}%")
        
    mean_acc = np.mean(cv_accuracies)
    std_acc = np.std(cv_accuracies)
    logger.info(f"XGBoost 5-Fold CV Calibrated Accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    
    # Fit final model on the training split
    logger.info("Initializing final XGBoost Classifier...")
    model = XGBClassifier(**X_xgb_params_adapt(X_train, y_train))
    
    logger.info("Fitting final XGBoost Model...")
    model.fit(X_train, y_train)
    
    # Evaluate final calibrated model on test split
    raw_test_probs = model.predict_proba(X_test)
    cal_test_probs = temp_scaler.predict(raw_test_probs)
    y_pred = np.argmax(cal_test_probs, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"XGBoost Final Calibrated Test Accuracy: {accuracy * 100:.2f}%")
    
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)
    
    # Save the cross-validation statistics in results/
    cv_summary_path = os.path.join(os.path.dirname(MODELS_DIR), 'results', 'xgb_cv_results.txt')
    os.makedirs(os.path.dirname(cv_summary_path), exist_ok=True)
    with open(cv_summary_path, 'w') as f:
        f.write(f"XGB CV Accuracies: {cv_accuracies}\n")
        f.write(f"XGB Mean CV Accuracy: {mean_acc:.6f}\n")
        f.write(f"XGB CV Accuracy Std: {std_acc:.6f}\n")
    
    # Package model artifacts
    save_path = os.path.join(MODELS_DIR, 'xgboost.pkl')
    artifacts = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'temperature_scaler': temp_scaler,
        'features': list(X_train.columns)
    }
    
    logger.info(f"Saving XGBoost artifacts to {save_path}...")
    joblib.dump(artifacts, save_path)
    
    return model, accuracy, report, matrix



def X_xgb_params_adapt(X_train, y_train):
    """Adapts config XGB params to training characteristics."""
    params = XGB_PARAMS.copy()
    # Check if multiclass or binary
    num_classes = len(np.unique(y_train))
    if num_classes > 2:
        params['objective'] = 'multi:softprob'
        params['num_class'] = num_classes
    else:
        params['objective'] = 'binary:logistic'
    return params

import numpy as np # import needed for np.unique
