import pandas as pd
import numpy as np
from typing import List, Tuple

def extract_spectral_bands(df: pd.DataFrame, band_start: int = 113, band_end: int = 212) -> pd.DataFrame:
    """
    Extract specific spectral bands from hyperspectral data
    
    Args:
        df: DataFrame containing hyperspectral data
        band_start: Starting band number (inclusive)
        band_end: Ending band number (inclusive)
    
    Returns:
        DataFrame with only the specified spectral bands
    """
    required_bands = [f"C{str(i).zfill(3)}" for i in range(band_start, band_end + 1)]
    
    # Check if all required bands exist
    missing_bands = [band for band in required_bands if band not in df.columns]
    if missing_bands:
        raise ValueError(f"Missing required bands: {missing_bands}")
    
    return df[required_bands].copy()

def validate_and_clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate and clean spectral data
    
    Args:
        df: DataFrame with spectral data
    
    Returns:
        Tuple of (cleaned DataFrame, list of warnings)
    """
    warnings = []
    
    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check for NaN values
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        warnings.append(f"Found {nan_count} non-numeric values, converted to NaN")
    
    # Check for infinite values
    inf_count = np.isinf(df.values).sum()
    if inf_count > 0:
        warnings.append(f"Found {inf_count} infinite values")
        df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with any NaN values
    original_rows = len(df)
    df = df.dropna()
    dropped_rows = original_rows - len(df)
    
    if dropped_rows > 0:
        warnings.append(f"Dropped {dropped_rows} rows with invalid data")
    
    return df, warnings

def preprocess_csv(csv_path: str, band_start: int = 113, band_end: int = 212) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for CSV files
    
    Args:
        csv_path: Path to CSV file
        band_start: Starting band number
        band_end: Ending band number
    
    Returns:
        Preprocessed DataFrame ready for prediction
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Extract required bands
    df_bands = extract_spectral_bands(df, band_start, band_end)
    
    # Validate and clean
    df_clean, warnings = validate_and_clean_data(df_bands)
    
    if warnings:
        print("Preprocessing warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    return df_clean
