"""
Model Evaluation Module

This module provides functions for comprehensive model evaluation
including metrics calculation, confusion matrix, ROC curves, and error analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
import json
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # Add probability-based metrics if available
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
        except:
            metrics['roc_auc'] = None
            metrics['avg_precision'] = None
    else:
        metrics['roc_auc'] = None
        metrics['avg_precision'] = None
    
    return metrics


def calculate_confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate confusion matrix and related metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with confusion matrix values and derived metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate rates
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    metrics = {
        'confusion_matrix': cm,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'specificity': specificity,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        'total_errors': int(fp + fn),
        'error_rate': (fp + fn) / len(y_true)
    }
    
    return metrics


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list] = None
) -> str:
    """
    Generate a classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional names for target classes
        
    Returns:
        Classification report as string
    """
    if target_names is None:
        target_names = ['Class 0', 'Class 1']
    
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=4
    )
    
    return report


def analyze_errors(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Analyze misclassified samples.
    
    Args:
        X: Feature matrix
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        DataFrame with misclassified samples
    """
    # Find misclassified samples
    misclassified_mask = y_true != y_pred
    
    # Create error analysis dataframe
    error_df = X[misclassified_mask].copy()
    error_df['true_label'] = y_true[misclassified_mask]
    error_df['predicted_label'] = y_pred[misclassified_mask]
    
    if y_pred_proba is not None:
        error_df['prediction_confidence'] = y_pred_proba[misclassified_mask]
    
    # Add error type
    error_df['error_type'] = error_df.apply(
        lambda row: 'False Positive' if row['predicted_label'] == 1 else 'False Negative',
        axis=1
    )
    
    return error_df


def get_roc_curve_data(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Calculate ROC curve data.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Tuple of (fpr, tpr, thresholds, auc_score)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    return fpr, tpr, thresholds, auc_score


def get_precision_recall_curve_data(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Calculate precision-recall curve data.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Tuple of (precision, recall, thresholds, avg_precision)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    return precision, recall, thresholds, avg_precision


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    target_names: Optional[list] = None
) -> Dict[str, Any]:
    """
    Perform comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        target_names: Optional names for target classes
        
    Returns:
        Dictionary with all evaluation results
    """
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Confusion matrix metrics
    cm_metrics = calculate_confusion_matrix_metrics(y_test, y_pred)
    
    # Classification report
    classification_rep = get_classification_report(y_test, y_pred, target_names)
    
    # Error analysis
    error_analysis = analyze_errors(X_test, y_test, y_pred, y_pred_proba)
    
    # ROC and PR curves (if probabilities available)
    roc_data = None
    pr_data = None
    if y_pred_proba is not None:
        try:
            fpr, tpr, roc_thresholds, auc = get_roc_curve_data(y_test, y_pred_proba)
            roc_data = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': roc_thresholds,
                'auc': auc
            }
            
            precision, recall, pr_thresholds, avg_prec = get_precision_recall_curve_data(
                y_test, y_pred_proba
            )
            pr_data = {
                'precision': precision,
                'recall': recall,
                'thresholds': pr_thresholds,
                'avg_precision': avg_prec
            }
        except:
            pass
    
    # Compile results
    results = {
        'metrics': metrics,
        'confusion_matrix_metrics': cm_metrics,
        'classification_report': classification_rep,
        'error_analysis': error_analysis,
        'roc_data': roc_data,
        'pr_data': pr_data,
        'predictions': {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    }
    
    return results


def get_feature_importance(
    model: Any,
    feature_names: list,
    top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Extract feature importance from model.
    
    Args:
        model: Trained model with feature_importances_ or coef_ attribute
        feature_names: List of feature names
        top_n: Number of top features to return (None for all)
        
    Returns:
        DataFrame with feature importance sorted by importance
    """
    # Try to get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return None
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Add rank
    importance_df['rank'] = range(1, len(importance_df) + 1)
    
    # Filter to top_n if specified
    if top_n is not None:
        importance_df = importance_df.head(top_n)
    
    importance_df = importance_df.reset_index(drop=True)
    
    return importance_df


def assess_model_performance(
    metrics: Dict[str, float],
    threshold_accuracy: float = 0.7,
    threshold_f1: float = 0.7
) -> Tuple[str, bool]:
    """
    Assess model performance and provide recommendation.
    
    Args:
        metrics: Dictionary of evaluation metrics
        threshold_accuracy: Minimum acceptable accuracy
        threshold_f1: Minimum acceptable F1 score
        
    Returns:
        Tuple of (assessment message, deployment_ready flag)
    """
    accuracy = metrics.get('accuracy', 0)
    f1 = metrics.get('f1_score', 0)
    
    if accuracy >= 0.85 and f1 >= 0.85:
        assessment = "Excellent performance! Model is ready for deployment."
        deployment_ready = True
    elif accuracy >= threshold_accuracy and f1 >= threshold_f1:
        assessment = "Good performance. Model meets minimum requirements for deployment."
        deployment_ready = True
    elif accuracy >= 0.6 or f1 >= 0.6:
        assessment = "Moderate performance. Consider additional tuning or feature engineering."
        deployment_ready = False
    else:
        assessment = "Poor performance. Significant improvements needed before deployment."
        deployment_ready = False
    
    return assessment, deployment_ready


def create_model_card(
    model: Any,
    model_name: str,
    evaluation_results: Dict[str, Any],
    model_path: Path,
    output_path: Path
) -> None:
    """
    Create a model card with comprehensive documentation.
    
    Args:
        model: Trained model
        model_name: Name of the model
        evaluation_results: Results from evaluate_model
        model_path: Path where model is saved
        output_path: Path to save the model card
    """
    metrics = evaluation_results['metrics']
    cm_metrics = evaluation_results['confusion_matrix_metrics']
    
    # Assess performance
    assessment, deployment_ready = assess_model_performance(metrics)
    
    # Create model card content
    card_content = f"""
{'='*80}
MODEL CARD
{'='*80}

Model Information:
-----------------
Model Name: {model_name}
Model Type: {model.__class__.__name__}
Creation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Path: {model_path}

Model Parameters:
----------------
{model.get_params() if hasattr(model, 'get_params') else 'N/A'}

Performance Metrics:
-------------------
Accuracy:       {metrics['accuracy']:.4f}
Precision:      {metrics['precision']:.4f}
Recall:         {metrics['recall']:.4f}
F1-Score:       {metrics['f1_score']:.4f}
Specificity:    {cm_metrics['specificity']:.4f}
ROC-AUC:        {metrics['roc_auc'] if metrics['roc_auc'] is None else f"{metrics['roc_auc']:.4f}"}
Avg Precision:  {metrics['avg_precision'] if metrics['avg_precision'] is None else f"{metrics['avg_precision']:.4f}"}

Confusion Matrix:
----------------
True Negatives:  {cm_metrics['true_negatives']}
False Positives: {cm_metrics['false_positives']}
False Negatives: {cm_metrics['false_negatives']}
True Positives:  {cm_metrics['true_positives']}

Error Analysis:
--------------
Total Errors:    {cm_metrics['total_errors']}
Error Rate:      {cm_metrics['error_rate']:.4f}
FP Rate:         {cm_metrics['false_positive_rate']:.4f}
FN Rate:         {cm_metrics['false_negative_rate']:.4f}

Classification Report:
---------------------
{evaluation_results['classification_report']}

Assessment:
----------
{assessment}
Deployment Ready: {'Yes' if deployment_ready else 'No'}

{'='*80}
"""
    
    # Save model card
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(card_content)
    
    print(f"- Model card saved to: {output_path}")


def save_evaluation_results(
    evaluation_results: Dict[str, Any],
    model_name: str,
    output_dir: Path
) -> Dict[str, Path]:
    """
    Save evaluation results to files.
    
    Args:
        evaluation_results: Results from evaluate_model
        model_name: Name of the model
        output_dir: Directory to save results
        
    Returns:
        Dictionary of saved file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = {}
    
    # Save metrics as JSON
    metrics_combined = {
        **evaluation_results['metrics'],
        **{k: v for k, v in evaluation_results['confusion_matrix_metrics'].items() 
           if k != 'confusion_matrix'}
    }
    
    json_path = output_dir / f"{model_name.lower().replace(' ', '_')}_metrics.json"
    with open(json_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in metrics_combined.items() if v is not None}
        json.dump(json_metrics, f, indent=4)
    saved_files['metrics_json'] = json_path
    
    # Save metrics as CSV
    csv_path = output_dir / f"{model_name.lower().replace(' ', '_')}_metrics.csv"
    pd.DataFrame([metrics_combined]).to_csv(csv_path, index=False)
    saved_files['metrics_csv'] = csv_path
    
    # Save error analysis
    if len(evaluation_results['error_analysis']) > 0:
        error_path = output_dir / f"{model_name.lower().replace(' ', '_')}_error_analysis.csv"
        evaluation_results['error_analysis'].to_csv(error_path, index=False)
        saved_files['error_analysis'] = error_path
    
    # Save classification report
    report_path = output_dir / f"{model_name.lower().replace(' ', '_')}_classification_report.txt"
    with open(report_path, 'w') as f:
        f.write(evaluation_results['classification_report'])
    saved_files['classification_report'] = report_path
    
    print(f"- Evaluation results saved to: {output_dir}")
    
    return saved_files






