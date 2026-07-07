import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from src.utils.config import MODELS_DIR, CICIOT2023_FILE
from src.utils.logger import get_logger

logger = get_logger("explainability_shap")

class ShapExplainer:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.explainer = None
        self.load_model_artifacts()
        
    def load_model_artifacts(self):
        if self.model_type == 'xgboost':
            filepath = os.path.join(MODELS_DIR, 'xgboost.pkl')
        else:
            filepath = os.path.join(MODELS_DIR, 'random_forest.pkl')
            
        if not os.path.exists(filepath):
            logger.error(f"Model artifacts not found at {filepath}")
            raise FileNotFoundError(f"Missing artifacts: train models first.")
            
        try:
            artifacts = joblib.load(filepath)
            self.model = artifacts['model']
            self.scaler = artifacts['scaler']
            self.label_encoder = artifacts['label_encoder']
            self.features = artifacts['features']
            
            # Create a small background dataset for TreeExplainer
            if os.path.exists(CICIOT2023_FILE):
                df = pd.read_csv(CICIOT2023_FILE)
                # Keep numerical features
                num_cols = ['flow_duration', 'packet_count', 'flow_bytes', 'packet_rate']
                # If we are explaining XGBoost, we scale numericals
                bg_df = df.copy()
                bg_df[num_cols] = self.scaler.transform(df[num_cols])
                
                # Check features alignment
                cat_cols = ['protocol', 'src_port', 'dst_port']
                available_cols = num_cols + cat_cols
                # Match features columns list exactly
                self.background_X = bg_df[self.features].head(50)
            else:
                self.background_X = pd.DataFrame(
                    np.random.normal(0, 1, size=(50, len(self.features))), 
                    columns=self.features
                )
                
            # Initialize TreeExplainer
            self.explainer = shap.TreeExplainer(self.model, data=self.background_X)
            logger.info(f"SHAP TreeExplainer initialized for {self.model_type}.")
        except Exception as e:
            logger.error(f"Error loading SHAP artifacts: {e}")
            raise
            
    def explain_instance(self, feature_dict, predicted_class_idx):
        """
        Calculates SHAP values for a single feature dictionary and 
        returns the feature contributions for the predicted class.
        """
        df = pd.DataFrame([feature_dict])
        
        # Scale numerical features
        num_cols = ['flow_duration', 'packet_count', 'flow_bytes', 'packet_rate']
        # If we have those columns, scale them
        for col in num_cols:
            if col in df.columns:
                df[col] = self.scaler.transform(df[num_cols])[:, num_cols.index(col)]
                
        # Reorder columns to match feature list
        X_instance = df[self.features]
        
        # Compute SHAP values
        shap_values = self.explainer(X_instance)
        
        # Multiclass models will output shape: (samples, features, classes)
        if len(shap_values.shape) == 3:
            class_shap_values = shap_values[0, :, predicted_class_idx]
        else:
            # Binary or single output class
            class_shap_values = shap_values[0]
            
        feats = X_instance.columns.tolist()
        vals = class_shap_values.values
        base_value = float(class_shap_values.base_values) if hasattr(class_shap_values, 'base_values') else 0.0
        
        contributions = []
        for f, v in zip(feats, vals):
            contributions.append({
                'feature': f,
                'value': float(feature_dict[f]) if f in feature_dict else 0.0,
                'shap_value': float(v)
            })
            
        # Sort by absolute SHAP values descending
        contributions = sorted(contributions, key=lambda x: abs(x['shap_value']), reverse=True)
        
        # Generate plot
        fig, ax = plt.subplots(figsize=(6, 4))
        y_pos = np.arange(len(feats))
        
        plot_contributions = sorted(contributions, key=lambda x: x['shap_value'])
        plot_feats = [x['feature'] for x in plot_contributions]
        plot_vals = [x['shap_value'] for x in plot_contributions]
        
        colors = ['#EF5350' if v > 0 else '#42A5F5' for v in plot_vals]
        
        ax.barh(y_pos, plot_vals, color=colors, align='center', height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_feats)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel('SHAP Value (Prediction Impact)')
        ax.set_title(f"SHAP local features explanation for {self.label_encoder.classes_[predicted_class_idx]}")
        
        plt.tight_layout()
        
        return contributions, fig, base_value
