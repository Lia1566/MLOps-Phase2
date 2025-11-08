"""
Unit Tests for Data Preprocessing Module
Tests for src/data/preprocessing.py
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from data.preprocessing import (
        remove_duplicates,
        create_binary_target,
        encode_ordinal_features,
        encode_categorical_features,
        split_features_target,
        split_train_test
    )
    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not PREPROCESSING_AVAILABLE, reason="Preprocessing module not available")
class TestDataPreprocessing:
    """Test suite for data preprocessing functions."""
    
    def test_remove_duplicates(self, sample_raw_data):
        """Test duplicate removal."""
        # Add duplicates
        df_with_duplicates = pd.concat([sample_raw_data, sample_raw_data.iloc[:5]], ignore_index=True)
        
        # Remove duplicates
        df_cleaned = remove_duplicates(df_with_duplicates)
        
        # Assertions
        assert len(df_cleaned) == len(sample_raw_data), "Should remove duplicates"
        assert df_cleaned.duplicated().sum() == 0, "Should have no duplicates"
    
    def test_create_binary_target(self, sample_raw_data):
        """Test binary target creation."""
        df = sample_raw_data.copy()
        
        # Create binary target
        df = create_binary_target(df, target_col='Performance')
        
        # Assertions
        assert 'Performance_Binary' in df.columns, "Should create binary target column"
        assert df['Performance_Binary'].isin([0, 1]).all(), "Should only contain 0 and 1"
        
        # Check mapping logic (Excellent/Very Good = 1, others = 0)
        high_perf = df[df['Performance'].isin(['Excellent', 'Very Good'])]
        if len(high_perf) > 0:
            assert (high_perf['Performance_Binary'] == 1).all(), "High performance should be 1"
    
    def test_encode_ordinal_features(self, sample_raw_data):
        """Test ordinal feature encoding."""
        df = sample_raw_data.copy()
        
        # Encode ordinal features
        df = encode_ordinal_features(df)
        
        # Check that ordinal columns are numeric
        ordinal_cols = ['Class_X_Percentage', 'Class_XII_Percentage', 'Study_Hours']
        for col in ordinal_cols:
            if col in df.columns:
                assert pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric"
    
    def test_encode_categorical_features(self, sample_raw_data):
        """Test categorical feature encoding (one-hot)."""
        df = sample_raw_data.copy()
        
        # Select categorical columns
        categorical_cols = ['Gender', 'Caste', 'Coaching', 'Medium']
        
        # Encode
        df_encoded = encode_categorical_features(df, categorical_cols)
        
        # Assertions
        assert len(df_encoded) == len(df), "Row count should not change"
        
        # Check that original categorical columns are replaced with binary columns
        for col in categorical_cols:
            assert col not in df_encoded.columns or df_encoded[col].dtype in ['int64', 'uint8'], \
                f"{col} should be encoded or removed"
    
    def test_split_features_target(self, sample_processed_data):
        """Test splitting features and target."""
        df = sample_processed_data.copy()
        
        X, y = split_features_target(df, target_col='Performance_Binary')
        
        # Assertions
        assert isinstance(X, pd.DataFrame), "X should be DataFrame"
        assert isinstance(y, (pd.Series, np.ndarray)), "y should be Series or array"
        assert 'Performance_Binary' not in X.columns, "Target should not be in features"
        assert len(X) == len(y), "X and y should have same length"
    
    def test_split_train_test(self, sample_processed_data):
        """Test train-test splitting."""
        df = sample_processed_data.copy()
        
        X, y = split_features_target(df, target_col='Performance_Binary')
        
        X_train, X_test, y_train, y_test = split_train_test(
            X, y, 
            test_size=0.2, 
            random_state=42,
            stratify=y
        )
        
        # Assertions
        assert len(X_train) + len(X_test) == len(X), "Should split all data"
        assert len(y_train) + len(y_test) == len(y), "Should split all labels"
        
        # Check approximate split ratio (80/20)
        train_ratio = len(X_train) / len(X)
        assert 0.75 <= train_ratio <= 0.85, "Should be approximately 80/20 split"
        
        # Check stratification (distribution should be similar)
        train_dist = y_train.value_counts(normalize=True).sort_index()
        test_dist = y_test.value_counts(normalize=True).sort_index()
        
        # Allow 10% difference in distributions
        for val in train_dist.index:
            if val in test_dist.index:
                diff = abs(train_dist[val] - test_dist[val])
                assert diff < 0.15, f"Stratification failed for value {val}"


@pytest.mark.unit
class TestPreprocessingEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, KeyError, IndexError)):
            remove_duplicates(df) if PREPROCESSING_AVAILABLE else None
    
    def test_missing_values(self, sample_raw_data):
        """Test handling of missing values."""
        df = sample_raw_data.copy()
        
        # Introduce missing values
        df.loc[0:5, 'Class_X_Percentage'] = np.nan
        
        # Preprocessing should handle this
        # (Either impute or drop - depends on implementation)
        assert df.isnull().sum().sum() > 0, "Should have missing values"
    
    def test_single_class_target(self):
        """Test error handling when target has only one class."""
        df = pd.DataFrame({
            'feature_1': [1, 2, 3, 4],
            'Performance_Binary': [1, 1, 1, 1]  # Only one class
        })
        
        X, y = df.drop('Performance_Binary', axis=1), df['Performance_Binary']
        
        # Should raise error or warning when stratifying with single class
        with pytest.raises((ValueError, Exception)):
            if PREPROCESSING_AVAILABLE:
                split_train_test(X, y, test_size=0.2, stratify=y)


@pytest.mark.unit
class TestPreprocessingDataTypes:
    """Test data type conversions and validations."""
    
    def test_numeric_conversion(self, sample_raw_data):
        """Test that percentage columns are numeric."""
        df = sample_raw_data.copy()
        
        assert pd.api.types.is_numeric_dtype(df['Class_X_Percentage'])
        assert pd.api.types.is_numeric_dtype(df['Class_XII_Percentage'])
    
    def test_categorical_encoding_dtype(self, sample_raw_data):
        """Test that encoded categorical features are numeric."""
        df = sample_raw_data.copy()
        
        if PREPROCESSING_AVAILABLE:
            df_encoded = encode_categorical_features(df, ['Gender', 'Coaching'])
            
            # All columns should be numeric after encoding
            for col in df_encoded.columns:
                assert pd.api.types.is_numeric_dtype(df_encoded[col]), \
                    f"{col} should be numeric after encoding"


# ================== PERFORMANCE TESTS ==================

@pytest.mark.unit
@pytest.mark.slow
class TestPreprocessingPerformance:
    """Test preprocessing performance."""
    
    def test_large_dataset_performance(self):
        """Test preprocessing on larger dataset."""
        # Create large dataset
        np.random.seed(42)
        large_df = pd.DataFrame({
            'Gender': np.random.choice(['Male', 'Female'], 10000),
            'Class_X_Percentage': np.random.uniform(50, 95, 10000),
            'Performance': np.random.choice(['Excellent', 'Good', 'Poor'], 10000)
        })
        
        import time
        start = time.time()
        
        if PREPROCESSING_AVAILABLE:
            df_clean = remove_duplicates(large_df)
            df_target = create_binary_target(df_clean, 'Performance')
        
        elapsed = time.time() - start
        
        # Should complete in reasonable time (<5 seconds)
        assert elapsed < 5.0, f"Preprocessing took too long: {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])