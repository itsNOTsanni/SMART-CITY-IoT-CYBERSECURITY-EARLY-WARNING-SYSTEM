import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
MODELS_DIR = os.path.join(BASE_DIR, 'trained_models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Ensure directories exist
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Datasets
CICIOT2023_FILE = os.path.join(DATASETS_DIR, 'CICIoT2023', 'CICIoT2023_subset.csv')
IOT23_FILE = os.path.join(DATASETS_DIR, 'IoT23', 'IoT23_subset.csv')
TOPOLOGY_FILE = os.path.join(DATASETS_DIR, 'smart_city_topology.json')
CRITICALITY_FILE = os.path.join(DATASETS_DIR, 'asset_criticality.csv')
PROPAGATION_FILE = os.path.join(DATASETS_DIR, 'attack_propagation.csv')
IMPACT_FILE = os.path.join(DATASETS_DIR, 'impact_assessment.csv')

# Attack Classes
ATTACK_CLASSES = ['Normal', 'DDoS', 'DoS', 'Spoofing', 'Botnet', 'Brute Force', 'Reconnaissance']

# Smart City Device Categories & Criticality Weights (0.0 to 1.0)
DEVICE_TYPES = {
    'CCTV': 0.7,
    'TrafficLight': 0.8,
    'SmartParking': 0.4,
    'SmartMeter': 0.6,
    'EnvironmentalSensor': 0.3,
    'EdgeGateway': 0.9,
    'ControlServer': 1.0,
    'EmergencySystem': 0.95
}

# Machine Learning config
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 8,
    'random_state': 42,
    'n_jobs': -1
}

XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'random_state': 42,
    'eval_metric': 'mlogloss',
    'n_jobs': -1
}

# GNN model config
GNN_PARAMS = {
    'in_features': 6,  # CPU, Memory, Traffic Rate, Packet Count, Attack Prob, Risk Score
    'hidden_dim': 16,
    'embed_dim': 8,
    'epochs': 150,
    'lr': 0.01
}
