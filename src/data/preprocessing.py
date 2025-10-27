"""
Data Preprocessing Module

This module contains functions for cleaning and preprocessing data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names by removing spaces and extra characters.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    pd.DataFrame
        Dataframe with cleaned column names
    """
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip().str.replace(' ', '')
    logger.info("Column names cleaned")
    return df_clean


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    pd.DataFrame
        Dataframe with duplicates removed
    """
    n_duplicates = df.duplicated().sum()
    df_clean = df.drop_duplicates()
    
    logger.info(f"Removed {n_duplicates} duplicate rows ({n_duplicates/len(df)*100:.2f}%)")
    logger.info(f"Shape after deduplication: {df_clean.shape}")
    
    return df_clean


def create_binary_target(df: pd.DataFrame, 
                        target_col: str = 'Performance',
                        new_col: str = 'Performance_Binary',
                        high_classes: list = ['Excellent', 'Vg']) -> pd.DataFrame:
    """
    Create binary target variable from multi-class target.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the original target column
    new_col : str
        Name of the new binary target column
    high_classes : list
        List of classes to map to 1 (high performance)
        
    Returns
    -------
    pd.DataFrame
        Dataframe with binary target added
    """
    df_binary = df.copy()
    df_binary[new_col] = df_binary[target_col].apply(
        lambda x: 1 if x in high_classes else 0
    )
    
    class_dist = df_binary[new_col].value_counts()
    logger.info(f"Binary target created: {new_col}")
    logger.info(f"Class 0: {class_dist[0]} ({class_dist[0]/len(df_binary)*100:.1f}%)")
    logger.info(f"Class 1: {class_dist[1]} ({class_dist[1]/len(df_binary)*100:.1f}%)")
    
    return df_binary


def group_rare_categories(df: pd.DataFrame, 
                         column: str,
                         rare_values: list,
                         new_value: str) -> pd.DataFrame:
    """
    Group rare category values into a single category.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to process
    rare_values : list
        List of values to group together
    new_value : str
        New value for grouped categories
        
    Returns
    -------
    pd.DataFrame
        Dataframe with grouped categories
    """
    df_grouped = df.copy()
    df_grouped[column] = df_grouped[column].apply(
        lambda x: new_value if x in rare_values else x
    )
    
    logger.info(f"Grouped {len(rare_values)} rare categories in '{column}' into '{new_value}'")
    return df_grouped


def encode_ordinal_features(df: pd.DataFrame, 
                           mappings: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Encode ordinal features with specified mappings.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    mappings : dict
        Dictionary of {column_name: {value: encoded_value}}
        
    Returns
    -------
    Tuple[pd.DataFrame, dict]
        Encoded dataframe and applied mappings
    """
    df_encoded = df.copy()
    applied_mappings = {}
    
    for col, mapping in mappings.items():
        if col in df_encoded.columns:
            new_col = f"{col}_Encoded"
            df_encoded[new_col] = df_encoded[col].map(mapping)
            applied_mappings[new_col] = mapping
            logger.info(f"Ordinal encoding applied to '{col}' -> '{new_col}'")
    
    return df_encoded, applied_mappings


def encode_nominal_features(df: pd.DataFrame, 
                           columns: list,
                           drop_first: bool = True) -> pd.DataFrame:
    """
    One-hot encode nominal categorical features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        List of column names to encode
    drop_first : bool
        Whether to drop first category to avoid multicollinearity
        
    Returns
    -------
    pd.DataFrame
        Dataframe with one-hot encoded features
    """
    df_encoded = pd.get_dummies(df, columns=columns, drop_first=drop_first, dtype=int)
    
    n_new_cols = len(df_encoded.columns) - len(df.columns) + len(columns)
    logger.info(f"One-hot encoding: {len(columns)} columns -> {n_new_cols} binary features")
    
    return df_encoded


def scale_features(X: pd.DataFrame, 
                  feature_cols: list,
                  scaler: Optional[StandardScaler] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale specified features using StandardScaler.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix
    feature_cols : list
        List of columns to scale
    scaler : StandardScaler, optional
        Pre-fitted scaler (for test data)
        
    Returns
    -------
    Tuple[pd.DataFrame, StandardScaler]
        Scaled dataframe and fitted scaler
    """
    X_scaled = X.copy()
    
    if scaler is None:
        scaler = StandardScaler()
        X_scaled[feature_cols] = scaler.fit_transform(X[feature_cols])
        logger.info(f"Fitted and transformed {len(feature_cols)} features")
    else:
        X_scaled[feature_cols] = scaler.transform(X[feature_cols])
        logger.info(f"Transformed {len(feature_cols)} features using existing scaler")
    
    return X_scaled, scaler


def split_train_test(X: pd.DataFrame, 
                    y: pd.Series,
                    test_size: float = 0.2,
                    random_state: int = 42,
                    stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    test_size : float
        Proportion of data for test set
    random_state : int
        Random seed for reproducibility
    stratify : bool
        Whether to stratify split by target
        
    Returns
    -------
    Tuple
        X_train, X_test, y_train, y_test
    """
    stratify_var = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_var
    )
    
    logger.info(f"Train/test split: {len(X_train)}/{len(X_test)} ({(1-test_size)*100:.0f}%/{test_size*100:.0f}%)")
    logger.info(f"Train class distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Test class distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def get_feature_columns(df: pd.DataFrame, 
                       exclude_cols: list) -> list:
    """
    Get list of feature columns excluding specified columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    exclude_cols : list
        Columns to exclude
        
    Returns
    -------
    list
        List of feature column names
    """
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    logger.info(f"Selected {len(feature_cols)} feature columns")
    return feature_cols