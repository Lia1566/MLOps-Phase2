"""
Feature Engineering Module

This module contains functions for creating and transforming features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


# Default mappings for student performance dataset
DEFAULT_GRADE_MAPPING = {
    'Average': 1,
    'Good': 2,
    'Vg': 3,
    'Excellent': 4
}

DEFAULT_TIME_MAPPING = {
    'ONE': 1,
    'TWO': 2,
    'THREE': 3,
    'FOUR_PLUS': 4
}


def get_ordinal_mappings() -> Dict[str, Dict]:
    """
    Get default ordinal feature mappings for student dataset.
    
    Returns
    -------
    Dict[str, Dict]
        Dictionary of feature mappings
    """
    return {
        'Class_X_Percentage': DEFAULT_GRADE_MAPPING,
        'Class_XII_Percentage': DEFAULT_GRADE_MAPPING,
        'time': DEFAULT_TIME_MAPPING
    }


def get_nominal_columns() -> List[str]:
    """
    Get default nominal columns for one-hot encoding.
    
    Returns
    -------
    List[str]
        List of nominal column names
    """
    return [
        'Gender',
        'Caste',
        'coaching',
        'Class_ten_education',
        'twelve_education',
        'medium',
        'Father_occupation',
        'Mother_occupation'
    ]


def create_interaction_features(df: pd.DataFrame,
                                feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Create interaction features between feature pairs.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_pairs : List[Tuple[str, str]]
        List of feature pairs to interact
        
    Returns
    -------
    pd.DataFrame
        Dataframe with interaction features added
    """
    df_interactions = df.copy()
    
    for feat1, feat2 in feature_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            interaction_name = f"{feat1}_x_{feat2}"
            df_interactions[interaction_name] = df[feat1] * df[feat2]
            logger.info(f"Created interaction feature: {interaction_name}")
    
    return df_interactions


def create_polynomial_features(df: pd.DataFrame,
                               columns: List[str],
                               degree: int = 2) -> pd.DataFrame:
    """
    Create polynomial features for specified columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        Columns to create polynomial features for
    degree : int
        Polynomial degree
        
    Returns
    -------
    pd.DataFrame
        Dataframe with polynomial features added
    """
    df_poly = df.copy()
    
    for col in columns:
        if col in df.columns:
            for d in range(2, degree + 1):
                poly_name = f"{col}_pow{d}"
                df_poly[poly_name] = df[col] ** d
                logger.info(f"Created polynomial feature: {poly_name}")
    
    return df_poly


def bin_continuous_feature(df: pd.DataFrame,
                           column: str,
                           bins: int = 5,
                           labels: List[str] = None) -> pd.DataFrame:
    """
    Bin a continuous feature into categories.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column to bin
    bins : int
        Number of bins
    labels : List[str], optional
        Labels for bins
        
    Returns
    -------
    pd.DataFrame
        Dataframe with binned feature
    """
    df_binned = df.copy()
    bin_column = f"{column}_binned"
    
    df_binned[bin_column] = pd.cut(df[column], bins=bins, labels=labels)
    logger.info(f"Binned '{column}' into {bins} categories")
    
    return df_binned


def aggregate_features(df: pd.DataFrame,
                      group_col: str,
                      agg_cols: List[str],
                      agg_funcs: List[str] = ['mean', 'std']) -> pd.DataFrame:
    """
    Create aggregated features by grouping.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    group_col : str
        Column to group by
    agg_cols : List[str]
        Columns to aggregate
    agg_funcs : List[str]
        Aggregation functions
        
    Returns
    -------
    pd.DataFrame
        Dataframe with aggregated features
    """
    df_agg = df.copy()
    
    for col in agg_cols:
        for func in agg_funcs:
            agg_name = f"{col}_{func}_by_{group_col}"
            agg_values = df.groupby(group_col)[col].transform(func)
            df_agg[agg_name] = agg_values
            logger.info(f"Created aggregated feature: {agg_name}")
    
    return df_agg


def create_ratio_features(df: pd.DataFrame,
                          numerator_cols: List[str],
                          denominator_cols: List[str]) -> pd.DataFrame:
    """
    Create ratio features between columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    numerator_cols : List[str]
        Columns for numerator
    denominator_cols : List[str]
        Columns for denominator
        
    Returns
    -------
    pd.DataFrame
        Dataframe with ratio features
    """
    df_ratios = df.copy()
    
    for num_col in numerator_cols:
        for den_col in denominator_cols:
            if num_col in df.columns and den_col in df.columns:
                ratio_name = f"{num_col}_div_{den_col}"
                # Avoid division by zero
                df_ratios[ratio_name] = df[num_col] / (df[den_col] + 1e-10)
                logger.info(f"Created ratio feature: {ratio_name}")
    
    return df_ratios


def get_feature_statistics(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Get summary statistics for features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str], optional
        Specific columns to analyze (default: all numeric)
        
    Returns
    -------
    pd.DataFrame
        Feature statistics
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    stats = df[columns].describe().T
    stats['missing'] = df[columns].isnull().sum()
    stats['missing_pct'] = (df[columns].isnull().sum() / len(df) * 100).round(2)
    
    logger.info(f"Generated statistics for {len(columns)} features")
    
    return stats