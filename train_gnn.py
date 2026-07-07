import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.gnn.train_gnn import train_gnn
from src.utils.logger import get_logger

logger = get_logger("train_gnn_cli")

def main():
    logger.info("Starting GNN Model Training Pipeline...")
    train_gnn()
    logger.info("GNN Training Pipeline execution completed successfully.")

if __name__ == '__main__':
    main()
