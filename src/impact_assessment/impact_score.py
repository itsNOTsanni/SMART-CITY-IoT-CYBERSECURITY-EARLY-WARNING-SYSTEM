import os
import pandas as pd
from src.utils.config import IMPACT_FILE
from src.utils.logger import get_logger
from src.impact_assessment.critical_assets import AssetCriticalityManager

logger = get_logger("impact_score")

class ImpactAssessor:
    def __init__(self, impact_path=IMPACT_FILE):
        self.impact_path = impact_path
        self.asset_manager = AssetCriticalityManager()
        self.impact_matrix = {}
        self.load_impact_matrix()
        
    def load_impact_matrix(self):
        """Loads baseline impact weights for attack-asset combinations."""
        if os.path.exists(self.impact_path):
            try:
                df = pd.read_csv(self.impact_path)
                for _, row in df.iterrows():
                    self.impact_matrix[(row['attack_type'], row['asset_type'])] = float(row['impact'])
                logger.info(f"Loaded {len(self.impact_matrix)} impact weight mappings.")
            except Exception as e:
                logger.error(f"Error loading impact matrix: {e}")
        else:
            logger.warning(f"Impact matrix CSV not found at {self.impact_path}. Using standard baseline values.")
            
    def get_base_impact(self, attack_type, asset_type):
        """Returns the base impact weight (0 to 1) for the attack on the asset type."""
        # Clean prefix if it has digits or underscores
        asset_type_clean = ''.join([i for i in asset_type.split('_')[0] if not i.isdigit()])
        
        # Exact match
        if (attack_type, asset_type_clean) in self.impact_matrix:
            return self.impact_matrix[(attack_type, asset_type_clean)]
            
        # Fallbacks depending on attack severity
        if attack_type in ['DDoS', 'DoS']:
            return 0.8 # heavy disruption
        elif attack_type in ['Botnet']:
            return 0.7 # control and data leak
        elif attack_type in ['Brute Force', 'Spoofing']:
            return 0.5 # unauthorized access
        return 0.3 # light threat (Reconnaissance)

    def assess_impacts(self, compromised_node, propagation_preds, attack_type, confidence):
        """
        Assesses node-level and sector-level impacts.
        Formula: Node Impact = Base Impact * Propagation Probability * (Criticality / 10.0) * Confidence
        """
        sector_mappings = {
            'TrafficLight': 'Traffic Management',
            'CCTV': 'Surveillance System',
            'SmartParking': 'Parking System',
            'SmartMeter': 'Energy System',
            'EnvironmentalSensor': 'Environmental Monitor',
            'EdgeGateway': 'Network Infrastructure',
            'ControlServer': 'Core Datacenter',
            'EmergencySystem': 'Emergency Response'
        }
        
        node_impacts = []
        
        # 1. Assess compromised seed node (prob = 1.0)
        comp_prefix = compromised_node.split('_')[0]
        comp_type = ''.join([i for i in comp_prefix if not i.isdigit()])
        comp_crit = self.asset_manager.get_criticality(compromised_node)
        comp_base_imp = self.get_base_impact(attack_type, comp_type)
        
        comp_impact = comp_base_imp * 1.0 * (comp_crit / 10.0) * confidence
        node_impacts.append({
            'node': compromised_node,
            'sector': sector_mappings.get(comp_type, 'General Infrastructure'),
            'probability': 100.0,
            'impact_score': round(comp_impact * 100, 2)
        })
        
        # 2. Assess propagation targets
        for pred in propagation_preds:
            node_id = pred['node']
            prob = pred['probability'] / 100.0
            
            node_prefix = node_id.split('_')[0]
            node_type = ''.join([i for i in node_prefix if not i.isdigit()])
            crit = self.asset_manager.get_criticality(node_id)
            base_imp = self.get_base_impact(attack_type, node_type)
            
            # Impact score calculation
            impact_score = base_imp * prob * (crit / 10.0) * confidence
            
            node_impacts.append({
                'node': node_id,
                'sector': sector_mappings.get(node_type, 'General Infrastructure'),
                'probability': pred['probability'],
                'impact_score': round(impact_score * 100, 2)
            })
            
        # 3. Aggregate sector disruption levels (using maximum node impact in each sector)
        sector_impacts = {}
        for item in node_impacts:
            sec = item['sector']
            score = item['impact_score']
            sector_impacts[sec] = max(sector_impacts.get(sec, 0), score)
            
        formatted_sectors = [
            {'sector': k, 'impact': round(v, 2)}
            for k, v in sector_impacts.items()
        ]
        formatted_sectors = sorted(formatted_sectors, key=lambda x: x['impact'], reverse=True)
        
        return {
            'node_impacts': node_impacts,
            'sector_impacts': formatted_sectors
        }
