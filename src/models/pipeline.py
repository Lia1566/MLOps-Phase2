"""
ML Pipeline Module

This module provides Scikit-Learn Pipeline Implementation that automates
the entire ML workflow: preprocessing → feature engineering → training → evaluation.

The pipelines ensure reproducibility, modularity, and best practices in ML development.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


class MLPipeline:
    """
    Machine Learning Pipeline wrapper for streamlined training and evaluation.
    
    This class encapsulates the entire ML workflow using Scikit-Learn pipelines,
    ensuring reproducibility and following best practices.
    """
    
    def __init__(
        self,
        model_name: str,
        model: Any,
        preprocessor: Optional[Any] = None,
        random_state: int = 42
    ):
        """
        Initialize the ML Pipeline.
        
        Args:
            model_name: Name of the model (e.g., 'Logistic Regression')
            model: Scikit-learn model instance
            preprocessor: Optional preprocessing steps (if None, uses StandardScaler)
            random_state: Random state for reproducibility
        """
        self.model_name = model_name
        self.random_state = random_state
        
        # Create preprocessor if not provided
        if preprocessor is None:
            preprocessor = StandardScaler()
        
        # Create the pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MLPipeline':
        """
        Fit the pipeline on training data.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            Self (fitted pipeline)
        """
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the fitted pipeline.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using the fitted pipeline.
        
        Args:
            X: Features to predict on
            
        Returns:
            Probability predictions array
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        
        if hasattr(self.pipeline.named_steps['model'], 'predict_proba'):
            return self.pipeline.predict_proba(X)
        else:
            raise AttributeError(f"{self.model_name} does not support probability predictions")
    
    def get_params(self) -> Dict[str, Any]:
        """Get pipeline parameters."""
        return self.pipeline.get_params()
    
    def set_params(self, **params) -> 'MLPipeline':
        """Set pipeline parameters."""
        self.pipeline.set_params(**params)
        return self


def create_preprocessing_pipeline(scaler_type: str = 'standard') -> Pipeline:
    """
    Create a preprocessing pipeline.
    
    Args:
        scaler_type: Type of scaler ('standard', 'robust', 'minmax')
        
    Returns:
        Preprocessing pipeline
    """
    scalers = {
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'minmax': MinMaxScaler()
    }
    
    if scaler_type not in scalers:
        raise ValueError(f"Unknown scaler type: {scaler_type}. Choose from {list(scalers.keys())}")
    
    preprocessor = Pipeline([
        ('scaler', scalers[scaler_type])
    ])
    
    return preprocessor


def create_model_pipeline(
    model_name: str,
    model: Any,
    scaler_type: str = 'standard'
) -> MLPipeline:
    """
    Create a complete ML pipeline with preprocessing and model.
    
    Args:
        model_name: Name of the model
        model: Scikit-learn model instance
        scaler_type: Type of scaler to use
        
    Returns:
        MLPipeline instance
    """
    preprocessor = create_preprocessing_pipeline(scaler_type)
    return MLPipeline(model_name, model, preprocessor)


def get_baseline_pipelines(random_state: int = 42) -> Dict[str, MLPipeline]:
    """
    Get dictionary of baseline ML pipelines with preprocessing.
    
    Each pipeline includes:
    1. Preprocessing (StandardScaler)
    2. Model training
    
    Args:
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with model names as keys and MLPipeline instances as values
    """
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=random_state,
            max_iter=1000
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            random_state=random_state
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            random_state=random_state
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=random_state
        )
    }
    
    # Create pipelines for each model
    pipelines = {}
    for model_name, model in models.items():
        pipelines[model_name] = create_model_pipeline(
            model_name,
            model,
            scaler_type='standard'
        )
    
    return pipelines


def train_pipeline_with_cv(
    pipeline: MLPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[MLPipeline, np.ndarray]:
    """
    Train a pipeline with cross-validation.
    
    Args:
        pipeline: MLPipeline instance
        X_train: Training features
        y_train: Training target
        cv_folds: Number of cross-validation folds
        random_state: Random state for CV splitting
        
    Returns:
        Tuple of (fitted pipeline, cross-validation scores)
    """
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Perform cross-validation
    cv_scores = cross_val_score(
        pipeline.pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    # Fit on full training set
    pipeline.fit(X_train, y_train)
    
    return pipeline, cv_scores


def evaluate_pipeline(
    pipeline: MLPipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate a trained pipeline on test data.
    
    Args:
        pipeline: Fitted MLPipeline instance
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary of evaluation metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score
    )
    
    # Generate predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    # Add ROC-AUC if model supports probability predictions
    try:
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    except:
        metrics['roc_auc'] = 0.0
    
    return metrics


def train_baseline_pipelines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    experiment_name: str = "baseline_pipeline_experiment",
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Dict[str, MLPipeline], pd.DataFrame]:
    """
    Train and evaluate multiple baseline pipelines with MLflow tracking.
    
    This function implements the complete pipeline workflow:
    1. Data preprocessing (scaling)
    2. Model training with cross-validation
    3. Model evaluation on test set
    4. MLflow experiment tracking
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        experiment_name: MLflow experiment name
        cv_folds: Number of cross-validation folds
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (trained pipelines dict, results dataframe)
    """
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Get baseline pipelines
    pipelines = get_baseline_pipelines(random_state)
    
    # Store results
    results = []
    trained_pipelines = {}
    
    print(f"\n{'='*80}")
    print(f"TRAINING {len(pipelines)} BASELINE PIPELINES")
    print(f"{'='*80}")
    print("\nPipeline Structure:")
    print(" 1. Preprocessing: StandardScaler")
    print(" 2. Model Training: Various algorithms")
    print(" 3. Evaluation: Cross-validation + Test set\n")
    
    for model_name, pipeline in pipelines.items():
        print(f"\n{'─'*80}")
        print(f"Training Pipeline: {model_name}")
        print(f"{'─'*80}")
        
        with mlflow.start_run(run_name=f"{model_name}_pipeline"):
            # Train with cross-validation
            trained_pipeline, cv_scores = train_pipeline_with_cv(
                pipeline, X_train, y_train, cv_folds, random_state
            )
            
            # Evaluate on training set
            train_metrics = evaluate_pipeline(trained_pipeline, X_train, y_train)
            
            # Evaluate on test set
            test_metrics = evaluate_pipeline(trained_pipeline, X_test, y_test)
            
            # Log pipeline parameters
            pipeline_params = {
                'preprocessing': 'StandardScaler',
                'model_type': model_name,
                **trained_pipeline.pipeline.named_steps['model'].get_params()
            }
            mlflow.log_params(pipeline_params)
            
            # Log metrics
            mlflow.log_metric("cv_mean", cv_scores.mean())
            mlflow.log_metric("cv_std", cv_scores.std())
            mlflow.log_metric("train_accuracy", train_metrics['accuracy'])
            mlflow.log_metric("test_accuracy", test_metrics['accuracy'])
            mlflow.log_metric("test_precision", test_metrics['precision'])
            mlflow.log_metric("test_recall", test_metrics['recall'])
            mlflow.log_metric("test_f1", test_metrics['f1'])
            mlflow.log_metric("test_roc_auc", test_metrics['roc_auc'])
            
            # Log the complete pipeline
            signature = infer_signature(X_train, trained_pipeline.predict(X_train))
            mlflow.sklearn.log_model(
                trained_pipeline.pipeline,
                artifact_path="pipeline",
                signature=signature
            )
            
            # Store results
            result = {
                'Model': model_name,
                'Pipeline': 'StandardScaler + Model',
                'CV Mean': cv_scores.mean(),
                'CV Std': cv_scores.std(),
                'Train Accuracy': train_metrics['accuracy'],
                'Test Accuracy': test_metrics['accuracy'],
                'Test Precision': test_metrics['precision'],
                'Test Recall': test_metrics['recall'],
                'Test F1': test_metrics['f1'],
                'Test ROC-AUC': test_metrics['roc_auc']
            }
            results.append(result)
            trained_pipelines[model_name] = trained_pipeline
            
            # Print progress
            print(f"  ✓ Preprocessing: StandardScaler applied")
            print(f"  ✓ Cross-Validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print(f"  ✓ Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  ✓ Test F1-Score: {test_metrics['f1']:.4f}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test Accuracy', ascending=False).reset_index(drop=True)
    
    print(f"\n{'='*80}")
    print("PIPELINE TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"\nBest Pipeline: {results_df.iloc[0]['Model']}")
    print(f" Test Accuracy: {results_df.iloc[0]['Test Accuracy']:.4f}")
    print(f" Test F1-Score: {results_df.iloc[0]['Test F1']:.4f}")
    print(f"\nResults Summary:")
    print(results_df[['Model', 'Test Accuracy', 'Test F1', 'CV Mean']].to_string(index=False))
    print(f"\n{'='*80}\n")
    
    return trained_pipelines, results_df


def save_pipeline(
    pipeline: MLPipeline,
    output_path: Path,
    pipeline_name: Optional[str] = None
) -> None:
    """
    Save a trained pipeline to disk.
    
    Args:
        pipeline: Fitted MLPipeline instance
        output_path: Path to save the pipeline
        pipeline_name: Optional name for logging
    """
    import joblib
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline.pipeline, output_path)
    
    if pipeline_name:
        print(f"✓ {pipeline_name} pipeline saved to: {output_path}")
    else:
        print(f"✓ Pipeline saved to: {output_path}")


def load_pipeline(pipeline_path: Path) -> Pipeline:
    """
    Load a saved pipeline from disk.
    
    Args:
        pipeline_path: Path to the saved pipeline
        
    Returns:
        Loaded pipeline
    """
    import joblib
    
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline not found at: {pipeline_path}")
    
    pipeline = joblib.load(pipeline_path)
    print(f"✓ Pipeline loaded from: {pipeline_path}")
    return pipeline


def get_pipeline_info(pipeline: MLPipeline) -> Dict[str, Any]:
    """
    Get detailed information about a pipeline.
    
    Args:
        pipeline: MLPipeline instance
        
    Returns:
        Dictionary with pipeline information
    """
    info = {
        'model_name': pipeline.model_name,
        'is_fitted': pipeline.is_fitted,
        'steps': list(pipeline.pipeline.named_steps.keys()),
        'preprocessing': str(pipeline.pipeline.named_steps.get('preprocessor', 'None')),
        'model': str(pipeline.pipeline.named_steps.get('model', 'None')),
        'parameters': pipeline.get_params()
    }
    
    return info


def print_pipeline_structure(pipeline: MLPipeline) -> None:
    """
    Print the structure of a pipeline in a readable format.
    
    Args:
        pipeline: MLPipeline instance
    """
    print(f"\n{'='*80}")
    print(f"PIPELINE STRUCTURE: {pipeline.model_name}")
    print(f"{'='*80}\n")
    
    for i, (name, step) in enumerate(pipeline.pipeline.named_steps.items(), 1):
        print(f"Step {i}: {name}")
        print(f"  Type: {type(step).__name__}")
        if hasattr(step, 'get_params'):
            params = step.get_params()
            if params:
                print(f"  Parameters:")
                for key, value in list(params.items())[:5]:  # Show first 5 params
                    print(f"    - {key}: {value}")
                if len(params) > 5:
                    print(f"    ... ({len(params) - 5} more parameters)")
        print()
    
    print(f"{'='*80}\n")