"""
Integration Tests for End-to-End ML Pipeline
Tests the complete flow: load → preprocess → train → predict → evaluate
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


@pytest.mark.integration
class TestEndToEndPipeline:
    """Test complete ML pipeline from data to predictions."""
    
    def test_complete_pipeline_flow(self, sample_raw_data, tmp_path):
        """Test complete pipeline: preprocess → train → predict → evaluate."""
        
        # Step 1: Preprocess data
        df = sample_raw_data.copy()
        
        # Create binary target
        df['Performance_Binary'] = df['Performance'].apply(
            lambda x: 1 if x in ['Excellent', 'Very Good'] else 0
        )
        
        # Simple preprocessing: drop non-numeric
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols]
        
        # Step 2: Split data
        from sklearn.model_selection import train_test_split
        
        X = df_numeric.drop('Performance_Binary', axis=1)
        y = df_numeric['Performance_Binary']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Step 3: Train model
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Step 4: Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)
        
        # Step 5: Evaluate
        from sklearn.metrics import accuracy_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Assertions
        assert 0 <= accuracy <= 1, "Accuracy should be [0,1]"
        assert 0 <= f1 <= 1, "F1 should be [0,1]"
        assert len(y_pred) == len(y_test), "Predictions should match test size"
        
        # Step 6: Save model
        model_path = tmp_path / "pipeline.pkl"
        joblib.dump(pipeline, model_path)
        
        # Step 7: Load and verify
        loaded_pipeline = joblib.load(model_path)
        y_pred_loaded = loaded_pipeline.predict(X_test)
        
        assert np.array_equal(y_pred, y_pred_loaded), \
            "Loaded model should produce same predictions"
    
    def test_pipeline_with_cross_validation(self, sample_processed_data):
        """Test pipeline with cross-validation."""
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        df = sample_processed_data.copy()
        X = df.drop('Performance_Binary', axis=1)
        y = df['Performance_Binary']
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        # Perform cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
        
        # Assertions
        assert len(cv_scores) == 5, "Should have 5 CV scores"
        assert all(0 <= score <= 1 for score in cv_scores), "Scores should be [0,1]"
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        assert 0 <= mean_score <= 1, "Mean score should be [0,1]"
        assert std_score >= 0, "Std should be non-negative"


@pytest.mark.integration
class TestDataPreprocessingIntegration:
    """Test integrated data preprocessing workflow."""
    
    def test_preprocessing_pipeline(self, sample_raw_data):
        """Test complete preprocessing pipeline."""
        df = sample_raw_data.copy()
        
        # Step 1: Remove duplicates
        df_clean = df.drop_duplicates()
        
        # Step 2: Create target
        df_clean['Performance_Binary'] = df_clean['Performance'].apply(
            lambda x: 1 if x in ['Excellent', 'Very Good'] else 0
        )
        
        # Step 3: Encode categorical
        df_encoded = pd.get_dummies(df_clean, columns=['Gender', 'Coaching'])
        
        # Step 4: Split features and target
        y = df_encoded['Performance_Binary']
        X = df_encoded.drop(['Performance', 'Performance_Binary'], axis=1, errors='ignore')
        
        # Assertions
        assert len(df_clean) <= len(df), "Cleaned data should have <= rows"
        assert 'Performance_Binary' in df_encoded.columns
        assert len(X.columns) > 0, "Should have features"
        assert len(y) == len(X), "Features and target should align"


@pytest.mark.integration
class TestModelTrainingIntegration:
    """Test integrated model training workflow."""
    
    def test_multiple_models_training(self, sample_train_test_split):
        """Test training multiple models and comparing."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
        
        X_train, X_test, y_train, y_test = sample_train_test_split
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=10),
            'SVM': SVC(random_state=42, kernel='linear')
        }
        
        results = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
        
        # Assertions
        assert len(results) == 3, "Should have results for 3 models"
        
        for name, accuracy in results.items():
            assert 0 <= accuracy <= 1, f"{name} accuracy should be [0,1]"
        
        # At least one model should have decent performance (>0.4)
        assert any(acc > 0.4 for acc in results.values()), \
            "At least one model should have accuracy >0.4"


@pytest.mark.integration
class TestPipelineConsistency:
    """Test consistency across pipeline runs."""
    
    def test_reproducibility_with_seed(self, sample_processed_data):
        """Test that setting seed produces reproducible results."""
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        
        df = sample_processed_data.copy()
        X = df.drop('Performance_Binary', axis=1)
        y = df['Performance_Binary']
        
        # Run 1
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model1 = LogisticRegression(random_state=42, max_iter=1000)
        model1.fit(X_train1, y_train1)
        pred1 = model1.predict(X_test1)
        
        # Run 2 with same seed
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model2 = LogisticRegression(random_state=42, max_iter=1000)
        model2.fit(X_train2, y_train2)
        pred2 = model2.predict(X_test2)
        
        # Assertions
        assert np.array_equal(X_train1, X_train2), "Splits should be identical"
        assert np.array_equal(pred1, pred2), "Predictions should be identical"


@pytest.mark.integration
@pytest.mark.slow
class TestLargeDatasetIntegration:
    """Test pipeline on larger datasets."""
    
    def test_pipeline_scales_with_data(self):
        """Test that pipeline works with varying data sizes."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split
        import time
        
        sizes = [100, 500, 1000]
        times = []
        
        for size in sizes:
            # Generate data
            np.random.seed(42)
            X = np.random.randn(size, 15)
            y = np.random.choice([0, 1], size)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Create and train pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ])
            
            start = time.time()
            pipeline.fit(X_train, y_train)
            pipeline.predict(X_test)
            elapsed = time.time() - start
            
            times.append(elapsed)
        
        # Training time should scale reasonably
        # Larger datasets should take longer, but not exponentially
        assert all(t < 2.0 for t in times), "All training should complete in <2s"


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Test error handling in integrated pipeline."""
    
    def test_pipeline_with_missing_values(self):
        """Test pipeline behavior with missing values."""
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split
        
        # Create data with missing values
        np.random.seed(42)
        X = np.random.randn(100, 10)
        X[np.random.rand(100, 10) < 0.1] = np.nan  # 10% missing
        y = np.random.choice([0, 1], 100)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Pipeline with imputation
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        # Should handle missing values
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        # Assertions
        assert len(predictions) == len(X_test), "Should predict for all samples"
        assert not np.isnan(predictions).any(), "Predictions should not be NaN"


@pytest.mark.integration
class TestPipelineExport:
    """Test exporting and importing pipeline."""
    
    def test_export_import_pipeline(self, sample_train_test_split, tmp_path):
        """Test that pipeline can be exported and imported."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        X_train, X_test, y_train, y_test = sample_train_test_split
        
        # Create and train pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        pipeline.fit(X_train, y_train)
        
        # Export
        export_path = tmp_path / "exported_pipeline.pkl"
        joblib.dump(pipeline, export_path)
        
        # Import
        imported_pipeline = joblib.load(export_path)
        
        # Verify predictions match
        orig_pred = pipeline.predict(X_test)
        imported_pred = imported_pipeline.predict(X_test)
        
        assert np.array_equal(orig_pred, imported_pred), \
            "Imported pipeline should produce same predictions"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])