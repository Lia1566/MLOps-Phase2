"""
Model Training Module

This module provides functions for training machine learning models
with MLflow experiment tracking and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, make_scorer
)


def get_baseline_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Get a dictionary of baseline models to train.
    
    Args:
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with model names as keys and model instances as values
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
    return models


def train_model_with_cv(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Any, np.ndarray]:
    """
    Train a model with cross-validation.
    
    Args:
        model: Sklearn model instance
        X_train: Training features
        y_train: Training target
        cv_folds: Number of cross-validation folds
        random_state: Random state for CV splitting
        
    Returns:
        Tuple of (trained model, cross-validation scores)
    """
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Perform cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring='accuracy', n_jobs=-1
    )
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    return model, cv_scores


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    # Add ROC-AUC if model supports probability predictions
    if hasattr(model, 'predict_proba'):
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        except:
            metrics['roc_auc'] = 0.0
    else:
        metrics['roc_auc'] = 0.0
    
    return metrics


def train_baseline_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    experiment_name: str = "baseline_experiment",
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Train and evaluate multiple baseline models with MLflow tracking.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        experiment_name: MLflow experiment name
        cv_folds: Number of cross-validation folds
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (trained models dict, results dataframe)
    """
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Get baseline models
    models = get_baseline_models(random_state)
    
    # Store results
    results = []
    trained_models = {}
    
    print(f"\nTraining {len(models)} baseline models...")
    print("=" * 80)
    
    for model_name, model in models.items():
        print(f"\nTraining: {model_name}")
        
        with mlflow.start_run(run_name=model_name):
            # Train with cross-validation
            trained_model, cv_scores = train_model_with_cv(
                model, X_train, y_train, cv_folds, random_state
            )
            
            # Evaluate on training set
            train_metrics = evaluate_model(trained_model, X_train, y_train)
            
            # Evaluate on test set
            test_metrics = evaluate_model(trained_model, X_test, y_test)
            
            # Log parameters
            if hasattr(trained_model, 'get_params'):
                mlflow.log_params(trained_model.get_params())
            
            # Log metrics
            mlflow.log_metric("cv_mean", cv_scores.mean())
            mlflow.log_metric("cv_std", cv_scores.std())
            mlflow.log_metric("train_accuracy", train_metrics['accuracy'])
            mlflow.log_metric("test_accuracy", test_metrics['accuracy'])
            mlflow.log_metric("test_precision", test_metrics['precision'])
            mlflow.log_metric("test_recall", test_metrics['recall'])
            mlflow.log_metric("test_f1", test_metrics['f1'])
            mlflow.log_metric("test_roc_auc", test_metrics['roc_auc'])
            
            # Log model
            signature = infer_signature(X_train, trained_model.predict(X_train))
            mlflow.sklearn.log_model(
                trained_model,
                artifact_path="model",
                signature=signature
            )
            
            # Store results
            result = {
                'Model': model_name,
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
            trained_models[model_name] = trained_model
            
            print(f"  CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test Accuracy', ascending=False).reset_index(drop=True)
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best model: {results_df.iloc[0]['Model']} "
          f"(Test Accuracy: {results_df.iloc[0]['Test Accuracy']:.4f})")
    
    return trained_models, results_df


def get_hyperparameter_grids() -> Dict[str, Dict[str, List]]:
    """
    Get hyperparameter grids for model tuning.
    
    Returns:
        Dictionary with model names as keys and parameter grids as values
    """
    param_grids = {
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear', 'saga']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5]
        }
    }
    return param_grids


def tune_model(
    model: Any,
    model_name: str,
    param_grid: Dict[str, List],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
    n_jobs: int = -1
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Tune a model using GridSearchCV.
    
    Args:
        model: Base model instance
        model_name: Name of the model
        param_grid: Hyperparameter grid
        X_train: Training features
        y_train: Training target
        cv_folds: Number of cross-validation folds
        random_state: Random state for CV
        n_jobs: Number of parallel jobs
        
    Returns:
        Tuple of (best model, best parameters, best CV score)
    """
    print(f"\nTuning {model_name}...")
    print(f"Parameter combinations: {np.prod([len(v) for v in param_grid.values()])}")
    
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True
    )
    
    # Fit
    grid_search.fit(X_train, y_train)
    
    print(f"Best CV Score: {grid_search.best_score_:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def tune_top_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    top_models: List[str],
    experiment_name: str = "tuning_experiment",
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Tune top performing models with GridSearchCV and MLflow tracking.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        top_models: List of model names to tune
        experiment_name: MLflow experiment name
        cv_folds: Number of cross-validation folds
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (tuned models dict, results dataframe)
    """
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Get base models and parameter grids
    base_models = get_baseline_models(random_state)
    param_grids = get_hyperparameter_grids()
    
    # Filter to top models
    models_to_tune = {
        name: base_models[name]
        for name in top_models
        if name in base_models and name in param_grids
    }
    
    # Store results
    results = []
    tuned_models = {}
    
    print(f"\nTuning {len(models_to_tune)} models...")
    print("=" * 80)
    
    for model_name, base_model in models_to_tune.items():
        with mlflow.start_run(run_name=f"{model_name}_tuned"):
            # Tune model
            best_model, best_params, best_cv_score = tune_model(
                base_model,
                model_name,
                param_grids[model_name],
                X_train,
                y_train,
                cv_folds,
                random_state
            )
            
            # Evaluate on test set
            test_metrics = evaluate_model(best_model, X_test, y_test)
            
            # Log parameters
            mlflow.log_params(best_params)
            
            # Log metrics
            mlflow.log_metric("best_cv_score", best_cv_score)
            mlflow.log_metric("test_accuracy", test_metrics['accuracy'])
            mlflow.log_metric("test_precision", test_metrics['precision'])
            mlflow.log_metric("test_recall", test_metrics['recall'])
            mlflow.log_metric("test_f1", test_metrics['f1'])
            mlflow.log_metric("test_roc_auc", test_metrics['roc_auc'])
            
            # Log model
            signature = infer_signature(X_train, best_model.predict(X_train))
            mlflow.sklearn.log_model(
                best_model,
                artifact_path="model",
                signature=signature
            )
            
            # Store results
            result = {
                'Model': model_name,
                'Best CV Score': best_cv_score,
                'Test Accuracy': test_metrics['accuracy'],
                'Test Precision': test_metrics['precision'],
                'Test Recall': test_metrics['recall'],
                'Test F1': test_metrics['f1'],
                'Test ROC-AUC': test_metrics['roc_auc'],
                'Best Params': str(best_params)
            }
            results.append(result)
            tuned_models[model_name] = best_model
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test Accuracy', ascending=False).reset_index(drop=True)
    
    print("\n" + "=" * 80)
    print("Tuning complete!")
    print(f"Best tuned model: {results_df.iloc[0]['Model']} "
          f"(Test Accuracy: {results_df.iloc[0]['Test Accuracy']:.4f})")
    
    return tuned_models, results_df


def save_model(
    model: Any,
    model_path: Path,
    model_name: Optional[str] = None
) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model
        model_path: Path to save the model
        model_name: Optional name for logging
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    
    if model_name:
        print(f"{model_name} saved to: {model_path}")
    else:
        print(f"Model saved to: {model_path}")


def load_model(model_path: Path) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    return model
