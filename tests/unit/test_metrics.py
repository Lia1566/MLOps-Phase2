"""
Unit Tests for Metrics Calculation
Tests for model evaluation metrics
"""

import pytest
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


@pytest.mark.unit
class TestMetricsCalculation:
    """Test suite for metrics calculation."""
    
    def test_accuracy_calculation(self, sample_predictions):
        """Test accuracy metric calculation."""
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']
        
        accuracy = accuracy_score(y_true, y_pred)
        
        # Assertions
        assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
        assert isinstance(accuracy, (float, np.floating)), "Accuracy should be float"
    
    def test_precision_calculation(self, sample_predictions):
        """Test precision metric calculation."""
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        
        # Assertions
        assert 0 <= precision <= 1, "Precision should be between 0 and 1"
        assert isinstance(precision, (float, np.floating)), "Precision should be float"
    
    def test_recall_calculation(self, sample_predictions):
        """Test recall metric calculation."""
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']
        
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # Assertions
        assert 0 <= recall <= 1, "Recall should be between 0 and 1"
        assert isinstance(recall, (float, np.floating)), "Recall should be float"
    
    def test_f1_calculation(self, sample_predictions):
        """Test F1-score calculation."""
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Assertions
        assert 0 <= f1 <= 1, "F1-score should be between 0 and 1"
        assert isinstance(f1, (float, np.floating)), "F1-score should be float"
        
        # F1 is harmonic mean of precision and recall
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        if precision + recall > 0:
            expected_f1 = 2 * (precision * recall) / (precision + recall)
            assert abs(f1 - expected_f1) < 0.01, "F1 should be harmonic mean of P and R"
    
    def test_roc_auc_calculation(self, sample_predictions):
        """Test ROC-AUC calculation."""
        y_true = sample_predictions['y_true']
        y_pred_proba = sample_predictions['y_pred_proba']
        
        # Need at least one sample of each class
        if len(np.unique(y_true)) == 2:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            
            # Assertions
            assert 0 <= roc_auc <= 1, "ROC-AUC should be between 0 and 1"
            assert isinstance(roc_auc, (float, np.floating)), "ROC-AUC should be float"
    
    def test_confusion_matrix_calculation(self, sample_predictions):
        """Test confusion matrix calculation."""
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Assertions
        assert cm.shape == (2, 2), "Confusion matrix should be 2x2 for binary classification"
        assert cm.sum() == len(y_true), "CM sum should equal number of samples"
        assert (cm >= 0).all(), "CM values should be non-negative"
        
        # Extract TN, FP, FN, TP
        tn, fp, fn, tp = cm.ravel()
        
        # Verify accuracy calculation from CM
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        expected_accuracy = accuracy_score(y_true, y_pred)
        
        assert abs(accuracy - expected_accuracy) < 0.01, \
            "Accuracy from CM should match sklearn accuracy"


@pytest.mark.unit
class TestMetricsEdgeCases:
    """Test edge cases in metrics calculation."""
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        
        assert accuracy_score(y_true, y_pred) == 1.0, "Perfect predictions should have accuracy 1.0"
        assert precision_score(y_true, y_pred) == 1.0, "Perfect predictions should have precision 1.0"
        assert recall_score(y_true, y_pred) == 1.0, "Perfect predictions should have recall 1.0"
        assert f1_score(y_true, y_pred) == 1.0, "Perfect predictions should have F1 1.0"
    
    def test_worst_predictions(self):
        """Test metrics with completely wrong predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        
        assert accuracy_score(y_true, y_pred) == 0.0, "Wrong predictions should have accuracy 0.0"
    
    def test_all_positive_predictions(self):
        """Test metrics when all predictions are positive."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 1])
        
        # Recall should be 1.0 (all positives captured)
        assert recall_score(y_true, y_pred) == 1.0
        
        # Precision should be 0.5 (half are false positives)
        assert precision_score(y_true, y_pred) == 0.5
    
    def test_all_negative_predictions(self):
        """Test metrics when all predictions are negative."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0])
        
        # Recall should be 0.0 (no positives captured)
        assert recall_score(y_true, y_pred, zero_division=0) == 0.0
        
        # Precision is undefined (0/0), with zero_division=0
        precision = precision_score(y_true, y_pred, zero_division=0)
        assert precision == 0.0
    
    def test_single_class_in_predictions(self):
        """Test metrics when predictions have only one class."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0])
        
        # Should handle gracefully with zero_division parameter
        f1 = f1_score(y_true, y_pred, zero_division=0)
        assert 0 <= f1 <= 1, "F1 should be valid"


@pytest.mark.unit
class TestMetricsValidation:
    """Test validation of metrics inputs."""
    
    def test_mismatched_lengths(self):
        """Test error handling for mismatched lengths."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1])  # Different length
        
        with pytest.raises(ValueError):
            accuracy_score(y_true, y_pred)
    
    def test_invalid_class_labels(self):
        """Test handling of invalid class labels."""
        y_true = np.array([0, 1, 2])  # 3 classes
        y_pred = np.array([0, 1, 1])
        
        # Should work for multi-class
        accuracy = accuracy_score(y_true, y_pred)
        assert 0 <= accuracy <= 1
    
    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])
        
        with pytest.raises((ValueError, ZeroDivisionError)):
            accuracy_score(y_true, y_pred)


@pytest.mark.unit
class TestClassificationReport:
    """Test classification report generation."""
    
    def test_classification_report_dict(self, sample_predictions):
        """Test classification report as dictionary."""
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']
        
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Assertions
        assert isinstance(report, dict), "Report should be dictionary"
        assert '0' in report, "Report should have class 0"
        assert '1' in report, "Report should have class 1"
        assert 'accuracy' in report, "Report should have accuracy"
        assert 'macro avg' in report, "Report should have macro avg"
        assert 'weighted avg' in report, "Report should have weighted avg"
        
        # Check metrics for each class
        for class_label in ['0', '1']:
            assert 'precision' in report[class_label]
            assert 'recall' in report[class_label]
            assert 'f1-score' in report[class_label]
            assert 'support' in report[class_label]
    
    def test_classification_report_string(self, sample_predictions):
        """Test classification report as string."""
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']
        
        report = classification_report(y_true, y_pred, zero_division=0)
        
        # Assertions
        assert isinstance(report, str), "Report should be string"
        assert 'precision' in report, "Report should contain precision"
        assert 'recall' in report, "Report should contain recall"
        assert 'f1-score' in report, "Report should contain f1-score"


@pytest.mark.unit
class TestMetricsConsistency:
    """Test consistency between different metric calculations."""
    
    def test_accuracy_from_confusion_matrix(self):
        """Test that accuracy from CM matches sklearn accuracy."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
        
        # Calculate accuracy directly
        accuracy_direct = accuracy_score(y_true, y_pred)
        
        # Calculate from confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        accuracy_from_cm = np.trace(cm) / np.sum(cm)
        
        assert abs(accuracy_direct - accuracy_from_cm) < 1e-10, \
            "Accuracy from CM should match direct calculation"
    
    def test_f1_from_precision_recall(self):
        """Test that F1 matches harmonic mean of precision and recall."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1_direct = f1_score(y_true, y_pred, zero_division=0)
        
        if precision + recall > 0:
            f1_calculated = 2 * (precision * recall) / (precision + recall)
            assert abs(f1_direct - f1_calculated) < 1e-10, \
                "F1 should match harmonic mean calculation"


@pytest.mark.unit
class TestCrossValidationMetrics:
    """Test metrics in cross-validation context."""
    
    def test_cv_score_aggregation(self):
        """Test aggregation of cross-validation scores."""
        # Simulate CV scores from 5 folds
        cv_scores = np.array([0.75, 0.80, 0.78, 0.82, 0.77])
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        assert 0 <= mean_score <= 1, "Mean CV score should be [0,1]"
        assert std_score >= 0, "Std should be non-negative"
        assert 0.75 <= mean_score <= 0.82, "Mean should be within range of scores"
    
    def test_cv_score_consistency(self):
        """Test that CV scores have reasonable variance."""
        # Good model should have low variance
        good_cv_scores = np.array([0.85, 0.86, 0.84, 0.85, 0.86])
        good_std = np.std(good_cv_scores)
        
        # Unstable model has high variance
        unstable_cv_scores = np.array([0.60, 0.85, 0.50, 0.90, 0.55])
        unstable_std = np.std(unstable_cv_scores)
        
        assert unstable_std > good_std, "Unstable model should have higher variance"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])