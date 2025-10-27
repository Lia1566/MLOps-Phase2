"""
Data Loading Module
    
This module contains functions for loading datasets from various sources.
"""
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_raw_data(data_path: Path, filename: str = 'student_entry_performance_original.csv') -> pd.DataFrame:
    """
    Load raw dataset from CSV file.
    
    Parameters
    ----------
    data_path : Path
        Path to the data directory
    filename : str, optional
        Name of the CSV file to load
        
    Returns
    -------
    pd.DataFrame
        Loaded dataframe
        
    Raises
    ------
    FileNotFoundError
        If the specified file doesn't exist
    """
    file_path = data_path / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    return df


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load processed dataset from CSV file.
    
    Parameters
    ----------
    data_path : Path
        Full path to the CSV file
        
    Returns
    -------
    pd.DataFrame
        Loaded dataframe
        
    Raises
    -------
    FileNotFoundError
        If the specified file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {file_path}")
    
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    return df

def load_processed_data(data_path: Path, filename: str) -> pd.DataFrame:
    """
    Load processed dataset from CSV file.
    
    Parameters
    ----------
    data_path : Path
        Path to the processed data directory
    filename : str
        Name of the processed CSV file
        
    Returns
    -------
    pd.DataFrame
        Loaded dataframe
    """
    file_path = data_path / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {file_path}")
    
    logger.info(f"Loading processed data from {file_path}")
    df = pd.read_csv(file_path)
    
    return df


def load_train_test_data(data_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load pre-split train and test datasets.
    
    Parameters
    ----------
    data_path : Path
        Path to the processed data directory
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    logger.info("Loading train/test split data")
    
    # Load train data
    train_data = load_processed_data(data_path, 'student_performance_train.csv')
    X_train = train_data.drop('Performance_Binary', axis=1)
    y_train = train_data['Performance_Binary']
    
    # Load test data
    test_data = load_processed_data(data_path, 'student_performance_test.csv')
    X_test = test_data.drop('Performance_Binary', axis=1)
    y_test = test_data['Performance_Binary']
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def get_feature_names(data_path: Path) -> list:
    """
    Load feature names from text file.
    
    Parameters
    ----------
    data_path : Path
        Path to the processed data directory
        
    Returns
    -------
    list
        List of feature names
    """
    feature_file = data_path / 'feature_names.txt'
    
    if feature_file.exists():
        with open(feature_file, 'r') as f:
            features = [line.strip() for line in f.readlines()]
        logger.info(f"Loaded {len(features)} feature names")
        return features
    else:
        logger.warning("Feature names file not found")
        return []


def save_dataframe(df: pd.DataFrame, output_path: Path, filename: str) -> None:
    """
    Save dataframe to CSV file.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save
    output_path : Path
        Directory to save the file
    filename : str
        Name of the output file
    """
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / filename
    
    df.to_csv(file_path, index=False)
    logger.info(f"Saved dataframe to {file_path} ({df.shape})")


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get summary information about the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    dict
        Dictionary containing dataset information
    """
    info = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'column_names': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    
    return info