"""Data loading utilities"""

import pandas as pd
from pathlib import Path


def load_raw_data(filepath):
    """
    Load raw data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    data = pd.read_csv(filepath)
    return data


def validate_data(df):
    """
    Validate the loaded data.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        bool: True if data is valid
    """
    required_columns = ['age', 'salary', 'experience']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Expected: {required_columns}")
    
    if df.isnull().any().any():
        raise ValueError("Data contains null values")
    
    return True
