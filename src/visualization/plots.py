"""
Visualization Module

This module provides functions for creating visualizations
for model training, evaluation, and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime


def set_plot_style():
    """Set default plotting style."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix array
        class_names: Names for classes
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = ['Class 0', 'Class 1']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        ax=ax,
        linewidths=1,
        linecolor='gray'
    )
    
    # Labels and title
    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
    ax.set_ylabel('True Label', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14, pad=20)
    
    # Add percentages as text
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                   ha='center', va='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"- Confusion matrix saved to: {save_path}")
    
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: AUC score
        model_name: Name of the model
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='#2ecc71', lw=2, 
            label=f'{model_name} (AUC = {auc_score:.4f})')
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', 
            label='Random Classifier (AUC = 0.5)')
    
    # Labels and title
    ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    ax.set_title(f'ROC Curve - {model_name}', fontweight='bold', fontsize=14)
    
    # Legend and grid
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    # Set limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"- ROC curve saved to: {save_path}")
    
    return fig


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    avg_precision: float,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Args:
        precision: Precision values
        recall: Recall values
        avg_precision: Average precision score
        model_name: Name of the model
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot PR curve
    ax.plot(recall, precision, color='#3498db', lw=2,
            label=f'{model_name} (AP = {avg_precision:.4f})')
    
    # Labels and title
    ax.set_xlabel('Recall', fontweight='bold', fontsize=12)
    ax.set_ylabel('Precision', fontweight='bold', fontsize=12)
    ax.set_title(f'Precision-Recall Curve - {model_name}', 
                fontweight='bold', fontsize=14)
    
    # Legend and grid
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(alpha=0.3)
    
    # Set limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"- PR curve saved to: {save_path}")
    
    return fig


def plot_roc_and_pr_curves(
    roc_data: Dict[str, np.ndarray],
    pr_data: Dict[str, np.ndarray],
    model_name: str = "Model",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot ROC and PR curves side by side.
    
    Args:
        roc_data: Dictionary with ROC curve data
        pr_data: Dictionary with PR curve data
        model_name: Name of the model
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ROC Curve
    ax1 = axes[0]
    ax1.plot(roc_data['fpr'], roc_data['tpr'], color='#2ecc71', lw=2,
            label=f"AUC = {roc_data['auc']:.4f}")
    ax1.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--',
            label='Random (AUC = 0.5)')
    ax1.set_xlabel('False Positive Rate', fontweight='bold', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontweight='bold', fontsize=11)
    ax1.set_title('ROC Curve', fontweight='bold', fontsize=13)
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    
    # PR Curve
    ax2 = axes[1]
    ax2.plot(pr_data['recall'], pr_data['precision'], color='#3498db', lw=2,
            label=f"AP = {pr_data['avg_precision']:.4f}")
    ax2.set_xlabel('Recall', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Precision', fontweight='bold', fontsize=11)
    ax2.set_title('Precision-Recall Curve', fontweight='bold', fontsize=13)
    ax2.legend(loc='lower left')
    ax2.grid(alpha=0.3)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    
    fig.suptitle(f'{model_name} - ROC & PR Curves', 
                fontweight='bold', fontsize=15, y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"- ROC and PR curves saved to: {save_path}")
    
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with feature importance
        top_n: Number of top features to show
        model_name: Name of the model
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    # Get top N features
    plot_df = importance_df.head(top_n).copy()
    plot_df = plot_df.sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar chart
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(plot_df)))
    bars = ax.barh(plot_df['feature'], plot_df['importance'], 
                   color=colors, edgecolor='black', linewidth=0.5)
    
    # Labels and title
    ax.set_xlabel('Importance', fontweight='bold', fontsize=12)
    ax.set_ylabel('Features', fontweight='bold', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance - {model_name}',
                fontweight='bold', fontsize=14, pad=20)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f'{width:.4f}',
               ha='left', va='center', fontsize=9, 
               fontweight='bold', color='black')
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"- Feature importance plot saved to: {save_path}")
    
    return fig


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = 'Test Accuracy',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot comparison of multiple models.
    
    Args:
        results_df: DataFrame with model comparison results
        metric: Metric to compare
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by metric
    plot_df = results_df.sort_values(metric, ascending=False)
    
    # Create bar chart
    colors = ['#2ecc71' if i == 0 else '#3498db' 
              for i in range(len(plot_df))]
    bars = ax.bar(plot_df['Model'], plot_df[metric], 
                  color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
               f'{height:.4f}',
               ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Labels and title
    ax.set_xlabel('Model', fontweight='bold', fontsize=12)
    ax.set_ylabel(metric, fontweight='bold', fontsize=12)
    ax.set_title(f'Model Comparison - {metric}', 
                fontweight='bold', fontsize=14, pad=20)
    
    # Rotate x-labels
    plt.xticks(rotation=45, ha='right')
    
    # Add threshold line if accuracy/f1
    if 'accuracy' in metric.lower() or 'f1' in metric.lower():
        ax.axhline(y=0.7, color='red', linestyle='--', 
                  linewidth=2, label='Target (70%)')
        ax.legend()
    
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Model comparison plot saved to: {save_path}")
    
    return fig


def plot_training_comparison(
    results_df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create comprehensive training comparison visualization.
    
    Args:
        results_df: DataFrame with model training results
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Model Training Comparison', fontweight='bold', fontsize=16)
    
    # Sort by test accuracy
    plot_df = results_df.sort_values('Test Accuracy', ascending=False)
    
    # 1. Test Accuracy comparison
    ax1 = axes[0, 0]
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(plot_df))]
    bars1 = ax1.bar(plot_df['Model'], plot_df['Test Accuracy'],
                    color=colors, edgecolor='black')
    ax1.set_ylabel('Test Accuracy', fontweight='bold')
    ax1.set_title('Test Accuracy Comparison', fontweight='bold')
    ax1.set_xticklabels(plot_df['Model'], rotation=45, ha='right')
    ax1.axhline(y=0.7, color='red', linestyle='--', linewidth=2)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.0])
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    # 2. CV Mean comparison
    ax2 = axes[0, 1]
    ax2.bar(plot_df['Model'], plot_df['CV Mean'], 
            color='#f39c12', edgecolor='black')
    ax2.errorbar(range(len(plot_df)), plot_df['CV Mean'], 
                yerr=plot_df['CV Std'], fmt='none', 
                ecolor='black', capsize=5, linewidth=2)
    ax2.set_ylabel('CV Mean Accuracy', fontweight='bold')
    ax2.set_title('Cross-Validation Performance', fontweight='bold')
    ax2.set_xticklabels(plot_df['Model'], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1.0])
    
    # 3. Precision, Recall, F1
    ax3 = axes[1, 0]
    x = np.arange(len(plot_df))
    width = 0.25
    ax3.bar(x - width, plot_df['Test Precision'], width, 
            label='Precision', color='#e74c3c', edgecolor='black')
    ax3.bar(x, plot_df['Test Recall'], width, 
            label='Recall', color='#f39c12', edgecolor='black')
    ax3.bar(x + width, plot_df['Test F1'], width, 
            label='F1-Score', color='#9b59b6', edgecolor='black')
    ax3.set_ylabel('Score', fontweight='bold')
    ax3.set_title('Precision, Recall, F1-Score', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(plot_df['Model'], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, 1.0])
    
    # 4. Train vs Test (overfitting analysis)
    ax4 = axes[1, 1]
    x = np.arange(len(plot_df))
    width = 0.35
    ax4.bar(x - width/2, plot_df['Train Accuracy'], width,
            label='Train', color='#3498db', edgecolor='black', alpha=0.7)
    ax4.bar(x + width/2, plot_df['Test Accuracy'], width,
            label='Test', color='#2ecc71', edgecolor='black', alpha=0.7)
    ax4.set_ylabel('Accuracy', fontweight='bold')
    ax4.set_title('Train vs Test Accuracy (Overfitting Check)', 
                 fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(plot_df['Model'], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"- Training comparison plot saved to: {save_path}")
    
    return fig


def plot_prediction_confidence(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot prediction confidence distribution.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Decision threshold
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Split by true label
    class_0_proba = y_pred_proba[y_true == 0]
    class_1_proba = y_pred_proba[y_true == 1]
    
    # 1. Histogram of prediction probabilities
    ax1 = axes[0]
    ax1.hist(class_0_proba, bins=30, alpha=0.7, color='#3498db', 
            label='True Class 0', edgecolor='black')
    ax1.hist(class_1_proba, bins=30, alpha=0.7, color='#2ecc71', 
            label='True Class 1', edgecolor='black')
    ax1.axvline(x=threshold, color='red', linestyle='--', 
               linewidth=2, label=f'Threshold ({threshold})')
    ax1.set_xlabel('Predicted Probability', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('Prediction Probability Distribution', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Confidence by correctness
    ax2 = axes[1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    correct_mask = y_true == y_pred
    
    correct_proba = y_pred_proba[correct_mask]
    incorrect_proba = y_pred_proba[~correct_mask]
    
    ax2.hist(correct_proba, bins=30, alpha=0.7, color='#2ecc71',
            label=f'Correct ({len(correct_proba)})', edgecolor='black')
    ax2.hist(incorrect_proba, bins=30, alpha=0.7, color='#e74c3c',
            label=f'Incorrect ({len(incorrect_proba)})', edgecolor='black')
    ax2.set_xlabel('Predicted Probability', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Confidence by Prediction Correctness', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"- Prediction confidence plot saved to: {save_path}")
    
    return fig


# Initialize plotting style when module is imported
set_plot_style()
