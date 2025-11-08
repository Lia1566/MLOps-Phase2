"""
Unit Tests for Model Inference
Tests for model loading and prediction functionality
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.mark.unit
class TestModelInference:
    """Test suite for model inference."""
    
    def test_model_predict(self, sample_model, sample_train_test_split):
        """Test that model can make predictions."""
        _, X_test, _, _ = sample_train_test_split
        
        predictions = sample_model.predict(X_test)
        
        # Assertions
        assert len(predictions) == len(X_test), "Should predict for all samples"
        assert all(p in [0, 1] for p in predictions), "Predictions should be binary"
    
    def test_model_predict_proba(self, sample_model, sample_train_test_split):
        """Test that model can predict probabilities."""
        _, X_test, _, _ = sample_train_test_split
        
        probabilities = sample_model.predict_proba(X_test)
        
        # Assertions
        assert probabilities.shape[0] == len(X_test), "Should predict for all samples"
        assert probabilities.shape[1] == 2, "Should have probabilities for 2 classes"
        assert all((0 <= p <= 1).all() for p in probabilities), "Probabilities should be [0,1]"
        assert all(abs(p.sum() - 1.0) < 0.01 for p in probabilities), "Probabilities should sum to 1"
    
    def test_pipeline_predict(self, sample_pipeline, sample_train_test_split):
        """Test that sklearn pipeline can make predictions."""
        _, X_test, _, _ = sample_train_test_split
        
        predictions = sample_pipeline.predict(X_test)
        
        # Assertions
        assert len(predictions) == len(X_test), "Should predict for all samples"
        assert all(p in [0, 1] for p in predictions), "Predictions should be binary"
    
    def test_model_save_load(self, sample_model, temp_model_path):
        """Test saving and loading model."""
        # Save model
        joblib.dump(sample_model, temp_model_path)
        
        # Check file exists
        assert temp_model_path.exists(), "Model file should exist"
        
        # Load model
        loaded_model = joblib.load(temp_model_path)
        
        # Test loaded model works
        X_test = np.random.randn(10, 15)
        predictions = loaded_model.predict(X_test)
        
        assert len(predictions) == 10, "Loaded model should make predictions"
    
    def test_model_reproducibility(self, sample_train_test_split):
        """Test that model predictions are reproducible."""
        from sklearn.linear_model import LogisticRegression
        
        X_train, X_test, y_train, _ = sample_train_test_split
        
        # Train two identical models
        model1 = LogisticRegression(random_state=42, max_iter=1000)
        model1.fit(X_train, y_train)
        
        model2 = LogisticRegression(random_state=42, max_iter=1000)
        model2.fit(X_train, y_train)
        
        # Predictions should be identical
        pred1 = model1.predict(X_test)
        pred2 = model2.predict(X_test)
        
        assert np.array_equal(pred1, pred2), "Predictions should be reproducible"


@pytest.mark.unit
class TestInferenceValidation:
    """Test input validation for inference."""
    
    def test_wrong_feature_count(self, sample_model):
        """Test error handling for wrong number of features."""
        # Model expects 15 features, give it 10
        X_wrong = np.random.randn(5, 10)
        
        with pytest.raises((ValueError, Exception)):
            sample_model.predict(X_wrong)
    
    def test_nan_in_features(self, sample_model):
        """Test handling of NaN values in features."""
        X_with_nan = np.random.randn(10, 15)
        X_with_nan[0, 0] = np.nan
        
        # Should either handle or raise error
        with pytest.raises((ValueError, Exception)):
            sample_model.predict(X_with_nan)
    
    def test_inf_in_features(self, sample_model):
        """Test handling of infinite values in features."""
        X_with_inf = np.random.randn(10, 15)
        X_with_inf[0, 0] = np.inf
        
        # Should either handle or raise error (sklearn usually raises)
        try:
            predictions = sample_model.predict(X_with_inf)
            # If it doesn't raise, check predictions are still valid
            assert not np.isnan(predictions).any(), "Predictions should not be NaN"
        except (ValueError, Exception):
            # It's okay if it raises an error
            pass
    
    def test_single_sample_prediction(self, sample_model):
        """Test prediction on single sample."""
        X_single = np.random.randn(1, 15)
        
        prediction = sample_model.predict(X_single)
        
        assert len(prediction) == 1, "Should predict for single sample"
        assert prediction[0] in [0, 1], "Prediction should be binary"


@pytest.mark.unit
class TestBatchInference:
    """Test batch prediction functionality."""
    
    def test_batch_prediction(self, sample_model):
        """Test prediction on batch of samples."""
        batch_sizes = [1, 10, 100, 1000]
        
        for batch_size in batch_sizes:
            X_batch = np.random.randn(batch_size, 15)
            predictions = sample_model.predict(X_batch)
            
            assert len(predictions) == batch_size, \
                f"Should predict for batch size {batch_size}"
    
    def test_batch_vs_single_consistency(self, sample_model):
        """Test that batch prediction equals individual predictions."""
        X_batch = np.random.randn(5, 15)
        
        # Batch prediction
        batch_predictions = sample_model.predict(X_batch)
        
        # Individual predictions
        individual_predictions = [
            sample_model.predict(X_batch[i:i+1])[0] 
            for i in range(len(X_batch))
        ]
        
        assert np.array_equal(batch_predictions, individual_predictions), \
            "Batch and individual predictions should match"


@pytest.mark.unit
class TestPipelineInference:
    """Test inference with sklearn pipelines."""
    
    def test_pipeline_preprocessing(self, sample_pipeline):
        """Test that pipeline applies preprocessing."""
        # Pipeline has StandardScaler
        X_raw = np.random.randn(10, 15) * 100 + 50  # Unscaled data
        
        # Pipeline should handle unscaled data
        predictions = sample_pipeline.predict(X_raw)
        
        assert len(predictions) == 10, "Pipeline should handle raw data"
        assert all(p in [0, 1] for p in predictions), "Predictions should be binary"
    
    def test_pipeline_transform(self, sample_pipeline):
        """Test that pipeline can transform data."""
        X_raw = np.random.randn(10, 15)
        
        # Get preprocessing steps only (exclude classifier)
        if hasattr(sample_pipeline, 'named_steps'):
            scaler = sample_pipeline.named_steps['scaler']
            X_transformed = scaler.transform(X_raw)
            
            # Check that transformation was applied
            assert X_transformed.shape == X_raw.shape
            assert not np.array_equal(X_transformed, X_raw), \
                "Transformed data should differ from raw data"


@pytest.mark.unit
@pytest.mark.slow
class TestInferencePerformance:
    """Test inference performance."""
    
    def test_prediction_speed(self, sample_model):
        """Test that predictions are fast enough."""
        import time
        
        X_large = np.random.randn(10000, 15)
        
        start = time.time()
        predictions = sample_model.predict(X_large)
        elapsed = time.time() - start
        
        # Should predict 10k samples in <1 second
        assert elapsed < 1.0, f"Prediction too slow: {elapsed:.2f}s"
        
        # Calculate predictions per second
        pred_per_sec = len(predictions) / elapsed
        assert pred_per_sec > 5000, f"Should predict >5000 samples/sec, got {pred_per_sec:.0f}"


@pytest.mark.unit
class TestModelMetadata:
    """Test model metadata and attributes."""
    
    def test_model_has_classes(self, sample_model):
        """Test that model has classes_ attribute."""
        assert hasattr(sample_model, 'classes_'), "Model should have classes_ attribute"
        assert len(sample_model.classes_) == 2, "Should have 2 classes"
        assert all(c in [0, 1] for c in sample_model.classes_), "Classes should be 0 and 1"
    
    def test_model_has_feature_count(self, sample_model):
        """Test that model knows feature count."""
        if hasattr(sample_model, 'n_features_in_'):
            assert sample_model.n_features_in_ == 15, "Model should expect 15 features"
    
    def test_pipeline_has_steps(self, sample_pipeline):
        """Test that pipeline has named steps."""
        assert hasattr(sample_pipeline, 'named_steps'), "Pipeline should have named_steps"
        assert 'scaler' in sample_pipeline.named_steps, "Pipeline should have scaler"
        assert 'classifier' in sample_pipeline.named_steps, "Pipeline should have classifier"


@pytest.mark.unit
class TestInferenceErrorHandling:
    """Test error handling in inference."""
    
    def test_empty_input(self, sample_model):
        """Test prediction with empty input."""
        X_empty = np.array([]).reshape(0, 15)
        
        # Should handle empty input gracefully
        predictions = sample_model.predict(X_empty)
        assert len(predictions) == 0, "Should return empty predictions for empty input"
    
    def test_invalid_dtype(self, sample_model):
        """Test prediction with invalid data types."""
        # Create array with strings (invalid)
        X_invalid = np.array([['a'] * 15])
        
        with pytest.raises((ValueError, TypeError, Exception)):
            sample_model.predict(X_invalid)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])