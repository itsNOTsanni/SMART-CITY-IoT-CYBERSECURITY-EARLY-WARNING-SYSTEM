import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils.logger import get_logger

logger = get_logger("explainability_lime")

class LimeExplainer:
    def __init__(self, model_wrapper):
        """
        Custom local surrogate explainer (LIME concept) that operates
        by perturbing numerical input features around the instance.
        """
        self.model_wrapper = model_wrapper
        logger.info("Custom LIME Explainer initialized.")

    def explain_instance(self, feature_dict, prediction_fn, num_perturbations=500):
        """
        Generates local surrogate explanations by perturbing numerical features,
        fitting a weighted ridge regression model locally.
        """
        features = list(feature_dict.keys())
        # We perturb numerical values slightly
        instance_vec = np.array([float(feature_dict[f]) for f in features])
        
        # Standard deviation for perturbations (10% of value or baseline 1.0)
        stds = np.array([max(0.1, abs(x) * 0.1) for x in instance_vec])
        
        perturbations = []
        for _ in range(num_perturbations):
            noise = np.random.normal(0, stds)
            perturbed_vec = instance_vec + noise
            perturbations.append(perturbed_vec)
            
        perturbations = np.array(perturbations)
        
        # Predict class probabilities for perturbed samples
        probs = []
        for p in perturbations:
            sample_dict = dict(zip(features, p))
            # Format according to model input requirements
            prob = prediction_fn(sample_dict)
            probs.append(prob)
            
        probs = np.array(probs) # [num_perturbations, num_classes]
        
        # Compute distances and weights (exponential kernel)
        distances = np.linalg.norm(perturbations - instance_vec, axis=1)
        kernel_width = np.sqrt(len(features)) * 0.75
        weights = np.exp(- (distances ** 2) / (kernel_width ** 2))
        
        # Fit a simple weighted linear model for each class
        # coefficients will represent feature importance weights
        num_classes = probs.shape[1] if len(probs.shape) > 1 else 2
        
        coefs = []
        # Center perturbations
        X_centered = perturbations - instance_vec
        
        # Weighted Least Squares: Beta = (X^T * W * X)^-1 * X^T * W * y
        W = np.diag(weights)
        XT_W = X_centered.T.dot(W)
        XT_W_X = XT_W.dot(X_centered)
        # Add small regularization term to ensure invertibility
        XT_W_X += np.eye(len(features)) * 1e-4
        
        if len(probs.shape) > 1:
            for c in range(num_classes):
                y = probs[:, c]
                beta = np.linalg.solve(XT_W_X, XT_W.dot(y))
                coefs.append(beta)
        else:
            beta = np.linalg.solve(XT_W_X, XT_W.dot(probs))
            coefs.append(beta)
            
        coefs = np.array(coefs)
        
        # Create a local report
        results = {}
        for c in range(len(coefs)):
            results[c] = sorted(
                [{'feature': f, 'weight': float(coefs[c][i])} for i, f in enumerate(features)],
                key=lambda x: abs(x['weight']),
                reverse=True
            )
            
        # Draw explanation plot
        fig, ax = plt.subplots(figsize=(6, 4))
        y_pos = np.arange(len(features))
        
        # We plot the explanation weights for class index 0 or predicted class
        plot_coefs = coefs[0]
        sorted_indices = np.argsort(plot_coefs)
        
        ax.barh(y_pos, plot_coefs[sorted_indices], color='#8884d8', align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([features[i] for i in sorted_indices])
        ax.set_xlabel('Local Surrogate Coefficient (Impact)')
        ax.set_title('LIME Feature Perturbation Importance')
        plt.tight_layout()
        
        return results, fig
