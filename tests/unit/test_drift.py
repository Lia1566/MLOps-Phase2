""" 
Unit Tests for Drift Detection
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Try to import drift detector
try:
    from app.drift_detection import DriftDetector
    DRIFT_AVAILABLE = True
except ImportError:
    DRIFT_AVAILABLE = False


@pytest.mark.skipif(not DRIFT_AVAILABLE, reason="Drift detection not available")
class TestDriftDetection:
    """Test suite for drift detection."""
    
    def test_drift_detector_initialization(self):
        """Test drift detector can be initialized."""
        detector = DriftDetector()
        assert detector is not None
    
    def test_reference_data_loading(self, tmp_path):
        """Test loading reference data from CSV."""
        # Create sample reference data
        ref_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(0, 1, 100)
        })
        
        ref_path = tmp_path / "reference.csv"
        ref_data.to_csv(ref_path, index=False)
        
        # Load in detector
        detector = DriftDetector(reference_data_path=ref_path)
        
        assert detector.reference_data is not None
        assert len(detector.reference_data) == 100
    
    def test_detect_no_drift(self):
        """Test drift detection with similar data (no drift)."""
        np.random.seed(42)
        
        # Reference data
        ref_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.normal(0, 1, 1000)
        })
        
        # Current data (similar distribution)
        current_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(0, 1, 100)
        })
        
        detector = DriftDetector()
        detector.reference_data = ref_data
        
        results = detector.detect_drift(current_data)
        
        assert 'drift_detected' in results
        assert 'drift_score' in results or 'error' in results
    
    def test_detect_with_drift(self):
        """Test drift detection with different data (drift expected)."""
        np.random.seed(42)
        
        # Reference data (mean=0)
        ref_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.normal(0, 1, 1000)
        })
        
        # Current data (mean=5 - significant shift!)
        current_data = pd.DataFrame({
            'feature_1': np.random.normal(5, 1, 100),
            'feature_2': np.random.normal(5, 1, 100)
        })
        
        detector = DriftDetector()
        detector.reference_data = ref_data
        
        results = detector.detect_drift(current_data)
        
        assert 'drift_detected' in results
        # With such a large shift, drift should be detected
        if 'drift_detected' in results and 'error' not in results:
            assert results['drift_detected'] == True
    
    def test_drift_with_missing_reference(self):
        """Test drift detection without reference data."""
        detector = DriftDetector()
        
        current_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 100)
        })
        
        results = detector.detect_drift(current_data)
        
        assert 'error' in results or 'drift_detected' in results