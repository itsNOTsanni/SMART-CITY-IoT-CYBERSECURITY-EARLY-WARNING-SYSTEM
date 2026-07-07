import os
import pandas as pd
from src.utils.config import CICIOT2023_FILE, IOT23_FILE, TOPOLOGY_FILE
from src.utils.logger import get_logger

logger = get_logger("preprocessing_clean")

def check_and_bootstrap_datasets():
    """Checks if dataset files exist. If not, generates synthetic datasets."""
    required_files = [CICIOT2023_FILE, IOT23_FILE, TOPOLOGY_FILE]
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        logger.info(f"Missing dataset files: {missing}. Bootstrapping synthetic datasets...")
        try:
            from datasets import generate_synthetic_data as gen
            logger.info("Generating CICIoT2023...")
            gen.generate_ciciot2023()
            logger.info("Generating TON_IoT...")
            gen.generate_ton_iot()
            logger.info("Generating IoT23...")
            gen.generate_iot23()
            logger.info("Generating Network Topology and Criticality Assets...")
            gen.generate_smart_city_topology_and_assets()
            logger.info("Bootstrapping complete.")
        except ImportError:
            logger.warning("datasets.generate_synthetic_data not found in path. Please run the generation script manually.")
        except Exception as e:
            logger.error(f"Failed to bootstrap datasets: {e}")
            raise

def clean_dataframe(df, dataset_name="Dataset"):
    """
    Cleans a dataframe by:
    1. Identifying and dropping duplicates.
    2. Filling missing values (median for numerical, mode for categorical).
    3. Applying Isolation Forest to detect and remove outliers in numerical features.
    """
    initial_shape = df.shape
    
    # Drop duplicates
    df = df.drop_duplicates()
    duplicate_count = initial_shape[0] - df.shape[0]
    if duplicate_count > 0:
        logger.info(f"[{dataset_name}] Removed {duplicate_count} duplicate rows.")
        
    # Check and handle missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.info(f"[{dataset_name}] Found {missing_count} missing values. Handling...")
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                else:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty() else "Unknown"
                    df[col] = df[col].fillna(mode_val)
        logger.info(f"[{dataset_name}] Missing values imputed successfully.")
        
    # Outlier detection and removal using Isolation Forest
    # Identify continuous numerical columns
    num_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and col not in ['src_port', 'dst_port', 'protocol', 'label']:
            num_cols.append(col)
            
    if num_cols and df.shape[0] > 100:
        from sklearn.ensemble import IsolationForest
        logger.info(f"[{dataset_name}] Fitting Isolation Forest for outlier detection on: {num_cols}...")
        try:
            iso = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
            preds = iso.fit_predict(df[num_cols].fillna(0.0))
            outliers_mask = (preds == -1)
            outliers_count = outliers_mask.sum()
            if outliers_count > 0:
                df = df[~outliers_mask].reset_index(drop=True)
                logger.info(f"[{dataset_name}] Removed {outliers_count} outliers using Isolation Forest.")
        except Exception as e:
            logger.warning(f"[{dataset_name}] Failed to run Isolation Forest: {e}")
        
    return df

def map_ciciot2023_columns(df):
    """Maps raw official UNB CICIoT2023 columns to standard project features."""
    rename_dict = {
        'Protocol': 'protocol',
        'Source_Port': 'src_port',
        'Destination_Port': 'dst_port',
        'Flow_Bytes_s': 'flow_bytes',
        'Flow_Packets_s': 'packet_rate',
        'Tot Fwd Pkts': 'packet_count',
        'Tot Pkts': 'packet_count',
        'flow_duration': 'flow_duration',
        'label': 'label'
    }
    # Auto-fallback case-insensitive check
    for col in df.columns:
        if col not in rename_dict and col.lower() in rename_dict:
            rename_dict[col] = rename_dict[col.lower()]
            
    df = df.rename(columns=rename_dict)
    return df

def map_iot23_columns(df):
    """Maps raw official IoT-23 columns to standard project features."""
    rename_dict = {
        'duration': 'duration',
        'orig_bytes': 'orig_bytes',
        'resp_bytes': 'resp_bytes',
        'conn_state': 'conn_state',
        'label': 'label'
    }
    # Auto-fallback case-insensitive check
    for col in df.columns:
        if col not in rename_dict and col.lower() in rename_dict:
            rename_dict[col] = rename_dict[col.lower()]
            
    df = df.rename(columns=rename_dict)
    return df

def get_cleaned_ciciot2023():
    """Loads, maps, and cleans the CICIoT2023 dataset."""
    check_and_bootstrap_datasets()
    df = pd.read_csv(CICIOT2023_FILE)
    df = map_ciciot2023_columns(df)
    return clean_dataframe(df, "CICIoT2023")

def get_cleaned_iot23():
    """Loads, maps, and cleans the IoT23 dataset."""
    check_and_bootstrap_datasets()
    df = pd.read_csv(IOT23_FILE)
    df = map_iot23_columns(df)
    return clean_dataframe(df, "IoT23")

if __name__ == '__main__':
    # Test cleaning routines
    check_and_bootstrap_datasets()
    ciciot_df = get_cleaned_ciciot2023()
    iot_df = get_cleaned_iot23()
    print(f"CICIoT2023 shape: {ciciot_df.shape}")
    print(f"IoT23 shape: {iot_df.shape}")
