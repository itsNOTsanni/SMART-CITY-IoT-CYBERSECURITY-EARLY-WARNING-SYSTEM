import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.utils.logger import get_logger

logger = get_logger("preprocessing_features")

def engineer_ciciot2023_features(df, scaler=None, label_encoder=None):
    """
    Standardizes and encodes features for the CICIoT2023 dataset.
    If scaler and label_encoder are provided, they are used in transform-only mode.
    """
    df_processed = df.copy()
    
    num_cols = ['flow_duration', 'packet_count', 'flow_bytes', 'packet_rate']
    cat_cols = ['protocol', 'src_port', 'dst_port']
    
    # Scale numerical columns
    if scaler is None:
        scaler = StandardScaler()
        df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])
        logger.info("Fitted new StandardScaler for CICIoT2023.")
    else:
        df_processed[num_cols] = scaler.transform(df_processed[num_cols])
        
    # Encode label column if present
    if 'label' in df_processed.columns:
        if label_encoder is None:
            label_encoder = LabelEncoder()
            df_processed['label_encoded'] = label_encoder.fit_transform(df_processed['label'])
            logger.info("Fitted new LabelEncoder for CICIoT2023.")
        else:
            df_processed['label_encoded'] = label_encoder.transform(df_processed['label'])
            
    # Keep numerical + categorical columns
    X = df_processed[num_cols + cat_cols]
    y = df_processed['label_encoded'] if 'label_encoded' in df_processed.columns else None
    
    return X, y, scaler, label_encoder

def engineer_iot23_features(df, scaler=None, label_encoder=None):
    """
    Standardizes and encodes features for the IoT-23 dataset.
    """
    df_processed = df.copy()
    
    num_cols = ['duration', 'orig_bytes', 'resp_bytes']
    
    # Check what columns are available
    if 'history' in df_processed.columns:
        df_processed = df_processed.drop(columns=['history'])
        
    # Categorical columns
    cat_cols = [c for c in df_processed.columns if c.startswith('conn_state_')]
    if not cat_cols and 'conn_state' in df_processed.columns:
        df_processed = pd.get_dummies(df_processed, columns=['conn_state'], drop_first=True)
        cat_cols = [c for c in df_processed.columns if c.startswith('conn_state_')]
        
    # Scale numerical columns
    if scaler is None:
        scaler = StandardScaler()
        df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])
        logger.info("Fitted new StandardScaler for IoT-23.")
    else:
        df_processed[num_cols] = scaler.transform(df_processed[num_cols])
        
    # Encode label column if present
    if 'label' in df_processed.columns:
        if label_encoder is None:
            label_encoder = LabelEncoder()
            df_processed['label_encoded'] = label_encoder.fit_transform(df_processed['label'])
            logger.info("Fitted new LabelEncoder for IoT-23.")
        else:
            df_processed['label_encoded'] = label_encoder.transform(df_processed['label'])
            
    feature_cols = num_cols + cat_cols
    X = df_processed[feature_cols]
    y = df_processed['label_encoded'] if 'label_encoded' in df_processed.columns else None
    
    return X, y, scaler, label_encoder
