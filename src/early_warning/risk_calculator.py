from src.utils.logger import get_logger
from src.impact_assessment.critical_assets import AssetCriticalityManager

logger = get_logger("early_warning_risk")

class RiskCalculator:
    def __init__(self):
        self.asset_manager = AssetCriticalityManager()
        
    def calculate_risk(self, node_id, attack_probability, centrality=0.1, cvss_base=8.5):
        """
        Calculates dynamic risk based on the proposed multi-factor equation:
        Risk = (lambda_1 * P_attack + lambda_2 * (Criticality/10) + lambda_3 * Centrality + lambda_4 * (CVSS/10)) * 100
        Weights sum to 1.0: lambda_1=0.4, lambda_2=0.2, lambda_3=0.2, lambda_4=0.2
        Returns a score in range 0 - 100, and a risk level classification.
        """
        criticality = self.asset_manager.get_criticality(node_id)
        
        # Factor normalizations to [0, 1] range:
        p_factor = attack_probability
        c_factor = criticality / 10.0
        g_factor = min(max(centrality, 0.0), 1.0)
        v_factor = cvss_base / 10.0
        
        # Weighted risk calculation
        score = (0.4 * p_factor + 0.2 * c_factor + 0.2 * g_factor + 0.2 * v_factor) * 100.0
        score = min(100.0, max(0.0, score))
        
        if score < 30.0:
            level = 'LOW'
        elif score < 60.0:
            level = 'MEDIUM'
        elif score < 85.0:
            level = 'HIGH'
        else:
            level = 'CRITICAL'
            
        logger.info(f"Risk calculated for {node_id}: score={score:.2f}, level={level}")
        
        return {
            'node_id': node_id,
            'criticality': criticality,
            'probability': attack_probability,
            'risk_score': round(score, 2),
            'risk_level': level
        }

