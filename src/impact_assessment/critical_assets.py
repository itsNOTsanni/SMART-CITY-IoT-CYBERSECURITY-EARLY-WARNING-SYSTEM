import os
import pandas as pd
from src.utils.config import CRITICALITY_FILE, DEVICE_TYPES
from src.utils.logger import get_logger

logger = get_logger("impact_assets")

def load_criticality_map():
    """Loads a mapping of node IDs to criticality scores (1-10 scale)."""
    criticality_map = {}
    if os.path.exists(CRITICALITY_FILE):
        try:
            df = pd.read_csv(CRITICALITY_FILE)
            criticality_map = dict(zip(df['id'], df['criticality']))
            logger.info(f"Loaded {len(criticality_map)} assets from criticality file.")
        except Exception as e:
            logger.error(f"Error reading asset criticality file: {e}")
    else:
        logger.warning(f"Criticality file not found at {CRITICALITY_FILE}. Using fallback rules.")
        
    return criticality_map

class AssetCriticalityManager:
    def __init__(self):
        self.criticality_map = load_criticality_map()
        
    def get_criticality(self, node_id):
        """Returns the criticality score (1 to 10) for a given node ID."""
        if node_id in self.criticality_map:
            return float(self.criticality_map[node_id])
            
        # Fallback to prefix matching
        prefix = node_id.split('_')[0] if '_' in node_id else node_id
        # Remove numbers from prefix
        prefix_clean = ''.join([i for i in prefix if not i.isdigit()])
        
        # Check standard types and map to 1-10
        # Map device type weight (0-1) to 1-10 scale
        if prefix_clean in DEVICE_TYPES:
            return float(DEVICE_TYPES[prefix_clean] * 10.0)
            
        return 5.0 # baseline moderate criticality
