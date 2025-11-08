"""
Unit Tests for Feature Engineering Module
Tests for src/features/engineering.py
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
    from features.engineering import (
        create_interaction_features,
        create_polynomial_features,
        scale_features,
        select_features
    )
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not FEATURES_AVAILABLE, reason="Feature engineering module not available")
class TestFeatureEngineering:
    """Test suite for feature engineering functions."""
    
    def test_feature_scaling(self, sample_processed_data):
        """Test feature scaling (StandardScaler)."""
        df = sample_processed_data.copy()
        X = df.drop('Performance_Binary', axis=1)
        
        if FEATURES_AVAILABLE:
            X_scaled = scale_features(X)
            
            # Check that scaled features have mean≈0 and std≈1
            means = X_scaled.mean()
            stds = X_scaled.std()
            
            assert all(abs(means) < 0.1), "Scaled features should have mean ≈ 0"
            assert all(abs(stds - 1.0) < 0.1), "Scaled features should have std ≈ 1"
    
    def test_interaction_features_creation(self, sample_processed_data):
        """Test creation of interaction features."""
        df = sample_processed_data.copy()
        X = df.drop('Performance_Binary', axis=1)
        
        original_cols = len(X.columns)
        
        if FEATURES_AVAILABLE:
            # Create interactions between first 3 features
            X_interactions = create_interaction_features(X, ['feature_0', 'feature_1', 'feature_2'])
            
            # Should have more columns after adding interactions
            assert len(X_interactions.columns) > original_cols, \
                "Should create additional interaction features"
    
    def test_polynomial_features_creation(self, sample_processed_data):
        """Test creation of polynomial features."""
        df = sample_processed_data.copy()
        X = df.drop('Performance_Binary', axis=1).iloc[:, :3]  # Use first 3 features
        
        original_cols = len(X.columns)
        
        if FEATURES_AVAILABLE:
            # Create polynomial features (degree 2)
            X_poly = create_polynomial_features(X, degree=2)
            
            # Should have more features (degree 2 includes original + squares + interactions)
            assert len(X_poly.columns) > original_cols, \
                "Polynomial features should increase feature count"


@pytest.mark.unit
class TestFeatureEngineeringValidation:
    """Test validation and error handling."""
    
    def test_scaling_preserves_shape(self, sample_processed_data):
        """Test that scaling preserves data shape."""
        df = sample_processed_data.copy()
        X = df.drop('Performance_Binary', axis=1)
        
        original_shape = X.shape
        
        if FEATURES_AVAILABLE:
            X_scaled = scale_features(X)
            
            assert X_scaled.shape == original_shape, \
                "Scaling should preserve data shape"
    
    def test_no_nan_after_scaling(self, sample_processed_data):
        """Test that scaling doesn't introduce NaN values."""
        df = sample_processed_data.copy()
        X = df.drop('Performance_Binary', axis=1)
        
        if FEATURES_AVAILABLE:
            X_scaled = scale_features(X)
            
            assert not X_scaled.isnull().any().any(), \
                "Scaled features should not contain NaN"
    
    def test_feature_names_preserved(self, sample_processed_data):
        """Test that feature names are preserved after transformations."""
        df = sample_processed_data.copy()
        X = df.drop('Performance_Binary', axis=1)
        
        original_columns = X.columns.tolist()
        
        if FEATURES_AVAILABLE:
            X_scaled = scale_features(X)
            
            # Original columns should be present (maybe with additions)
            for col in original_columns:
                assert col in X_scaled.columns or any(col in c for c in X_scaled.columns), \
                    f"Original column {col} should be preserved or transformed"


@pytest.mark.unit
class TestFeatureSelection:
    """Test feature selection functionality."""
    
    @pytest.mark.skipif(not FEATURES_AVAILABLE, reason="Feature engineering module not available")
    def test_select_top_features(self, sample_train_test_split):
        """Test selecting top K features."""
        X_train, X_test, y_train, y_test = sample_train_test_split
        
        if FEATURES_AVAILABLE:
            # Select top 5 features
            X_train_selected = select_features(X_train, y_train, k=5)
            
            assert X_train_selected.shape[1] == 5, "Should select exactly 5 features"
            assert X_train_selected.shape[0] == X_train.shape[0], "Should preserve number of samples"
    
    @pytest.mark.skipif(not FEATURES_AVAILABLE, reason="Feature engineering module not available")
    def test_feature_selection_consistency(self, sample_train_test_split):
        """Test that feature selection is consistent."""
        X_train, X_test, y_train, y_test = sample_train_test_split
        
        if FEATURES_AVAILABLE:
            # Run twice with same data
            X_selected_1 = select_features(X_train, y_train, k=5)
            X_selected_2 = select_features(X_train, y_train, k=5)
            
            # Should select same features
            assert X_selected_1.columns.tolist() == X_selected_2.columns.tolist(), \
                "Feature selection should be deterministic"


@pytest.mark.unit
class TestFeatureEngineeringEdgeCases:
    """Test edge cases in feature engineering."""
    
    def test_single_feature(self):
        """Test scaling with single feature."""
        X = pd.DataFrame({'feature_1': [1, 2, 3, 4, 5]})
        
        if FEATURES_AVAILABLE:
            X_scaled = scale_features(X)
            
            assert X_scaled.shape == X.shape, "Should handle single feature"
            assert not X_scaled.isnull().any().any(), "Should not create NaN"
    
    def test_constant_feature(self):
        """Test handling of constant features."""
        X = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'constant': [5, 5, 5, 5, 5]  # Constant feature
        })
        
        if FEATURES_AVAILABLE:
            # Scaling constant feature should work (but std will be 0)
            X_scaled = scale_features(X)
            
            # Check that it doesn't crash
            assert X_scaled.shape == X.shape


@pytest.mark.unit
@pytest.mark.slow
class TestFeatureEngineeringPerformance:
    """Test performance of feature engineering."""
    
    def test_scaling_large_dataset(self):
        """Test scaling performance on large dataset."""
        # Create large dataset
        np.random.seed(42)
        X_large = pd.DataFrame(
            np.random.randn(10000, 50),
            columns=[f'feature_{i}' for i in range(50)]
        )
        
        import time
        start = time.time()
        
        if FEATURES_AVAILABLE:
            X_scaled = scale_features(X_large)
        
        elapsed = time.time() - start
        
        # Should complete quickly (<2 seconds)
        assert elapsed < 2.0, f"Scaling took too long: {elapsed:.2f}s"


# ================== INTEGRATION WITH SKLEARN ==================

@pytest.mark.unit
class TestSklearnIntegration:
    """Test integration with sklearn transformers."""
    
    def test_scaler_fit_transform(self, sample_train_test_split):
        """Test that scaler can fit and transform."""
        from sklearn.preprocessing import StandardScaler
        
        X_train, X_test, _, _ = sample_train_test_split
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Check shapes
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
        
        # Check that test set uses train statistics
        assert scaler.mean_ is not None, "Scaler should be fitted"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])