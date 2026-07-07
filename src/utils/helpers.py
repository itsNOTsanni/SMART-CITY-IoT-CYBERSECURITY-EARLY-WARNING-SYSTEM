import json
import os
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("helpers")

def load_json(filepath):
    """Loads a JSON file with error handling."""
    if not os.path.exists(filepath):
        logger.error(f"JSON file not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {filepath}: {e}")
        raise

def save_json(data, filepath):
    """Saves data to a JSON file with pretty printing."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Successfully saved JSON to {filepath}")
    except Exception as e:
        logger.error(f"Error saving JSON file {filepath}: {e}")
        raise

def load_csv(filepath):
    """Loads a CSV file into a DataFrame."""
    if not os.path.exists(filepath):
        logger.error(f"CSV file not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Error loading CSV file {filepath}: {e}")
        raise

def save_csv(df, filepath):
    """Saves a DataFrame to a CSV file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Successfully saved CSV to {filepath}")
    except Exception as e:
        logger.error(f"Error saving CSV file {filepath}: {e}")
        raise
