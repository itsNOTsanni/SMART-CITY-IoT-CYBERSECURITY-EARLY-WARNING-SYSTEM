import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.logger import get_logger
from src.preprocessing.clean_data import check_and_bootstrap_datasets
import train_rf
import train_xgboost
import train_gnn
import evaluate

logger = get_logger("main_pipeline")

def run_pipeline():
    logger.info("=================================================================")
    logger.info("  STARTING SMART CITY IoT SECURITY RESEARCH PIPELINE END-TO-END  ")
    logger.info("=================================================================")
    
    # Phase 1: Bootstrapping datasets
    logger.info("--- Phase 1: Checking and Bootstrapping Datasets ---")
    check_and_bootstrap_datasets()
    
    # Phase 2: Training Random Forest model
    logger.info("--- Phase 2: Training Random Forest Classifier (IoT-23) ---")
    train_rf.main()
    
    # Phase 3: Training XGBoost model
    logger.info("--- Phase 3: Training XGBoost Classifier (CICIoT2023) ---")
    train_xgboost.main()
    
    # Phase 4: Training GNN model
    logger.info("--- Phase 4: Training GNN propagation Predictor (Topology) ---")
    train_gnn.main()
    
    # Phase 5: Generating reports
    logger.info("--- Phase 5: Running Evaluations and Writing Reports ---")
    evaluate.main()
    
    logger.info("=================================================================")
    logger.info("  PIPELINE EXECUTED SUCCESSFULLY                                ")
    logger.info("  Trained Models Saved: trained_models/                         ")
    logger.info("  Evaluation Reports Saved: results/                            ")
    logger.info("  To launch the Dashboard:                                      ")
    logger.info("     streamlit run dashboard/app.py                             ")
    logger.info("=================================================================")

if __name__ == '__main__':
    run_pipeline()
