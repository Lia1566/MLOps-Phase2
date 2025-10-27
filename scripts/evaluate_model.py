#!/usr/bin/env python3
"""
Evaluation Script

This script evaluates trained machine learning models using the refactored codebase.
It generates comprehensive evaluation reports, visualizations, and model cards.

Usage:
    python scripts/evaluate_model.py --model models/best_model_tuned.pkl
    python scripts/evaluate_model.py --model models/best_model_baseline.pkl --output-dir reports/evaluation
    python scripts/evaluate_model.py --config config/config.yaml --model models/best_model_tuned.pkl
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.data.loader import load_data
from src.models.train import load_model
from src.models.evaluate import (
    evaluate_model,
    get_feature_importance,
    create_model_card,
    save_evaluation_results
)
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_roc_and_pr_curves,
    plot_feature_importance,
    plot_prediction_confidence
)


def evaluate(config, args):
    """Evaluate a trained model."""
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n✗ Error: Model not found at {model_path}")
        print("  Please train a model first or specify correct path")
        sys.exit(1)
    
    print(f"\nLoading model from: {model_path}")
    model = load_model(model_path)
    model_name = model.__class__.__name__
    
    # Load test data
    print("\nLoading test data...")
    test_data = load_data(config.data_processed / config.get('data.test_file'))
    
    target_col = config.get('data.target_column')
    X_test = test_data.drop(target_col, axis=1)
    y_test = test_data[target_col].values
    
    print(f"✓ Test data loaded")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X_test.shape[1]}")
    print(f"  Class distribution: {pd.Series(y_test).value_counts().to_dict()}")
    
    # Get class names
    eval_config = config.get('evaluation', {})
    class_names = eval_config.get('class_names', ['Class 0', 'Class 1'])
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluation_results = evaluate_model(
        model, X_test, y_test,
        target_names=class_names
    )
    
    # Print metrics
    metrics = evaluation_results['metrics']
    cm_metrics = evaluation_results['confusion_matrix_metrics']
    
    print("\n" + "-"*80)
    print("PERFORMANCE METRICS")
    print("-"*80)
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"F1-Score:       {metrics['f1_score']:.4f}")
    print(f"Specificity:    {cm_metrics['specificity']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"ROC-AUC:        {metrics['roc_auc']:.4f}")
    if metrics['avg_precision'] is not None:
        print(f"Avg Precision:  {metrics['avg_precision']:.4f}")
    
    print("\n" + "-"*80)
    print("CONFUSION MATRIX")
    print("-"*80)
    print(f"True Negatives:  {cm_metrics['true_negatives']}")
    print(f"False Positives: {cm_metrics['false_positives']}")
    print(f"False Negatives: {cm_metrics['false_negatives']}")
    print(f"True Positives:  {cm_metrics['true_positives']}")
    print(f"Total Errors:    {cm_metrics['total_errors']}")
    print(f"Error Rate:      {cm_metrics['error_rate']:.4f}")
    
    print("\n" + "-"*80)
    print("CLASSIFICATION REPORT")
    print("-"*80)
    print(evaluation_results['classification_report'])
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config.reports_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figures_dir = output_dir / 'figures' if args.output_dir else config.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Save evaluation results
    print("\nSaving evaluation results...")
    saved_files = save_evaluation_results(
        evaluation_results,
        model_name,
        output_dir
    )
    
    # Create model card
    model_card_path = output_dir / 'model_card.txt'
    create_model_card(
        model, model_name,
        evaluation_results,
        model_path,
        model_card_path
    )
    
    # Create visualizations
    print("\nCreating visualizations...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Confusion Matrix
    cm = cm_metrics['confusion_matrix']
    cm_path = figures_dir / f'confusion_matrix_{timestamp}.png'
    plot_confusion_matrix(
        cm,
        class_names=class_names,
        title=f'Confusion Matrix - {model_name}',
        save_path=cm_path
    )
    
    # 2. ROC and PR Curves (if available)
    if evaluation_results['roc_data'] is not None:
        curves_path = figures_dir / f'roc_pr_curves_{timestamp}.png'
        plot_roc_and_pr_curves(
            evaluation_results['roc_data'],
            evaluation_results['pr_data'],
            model_name=model_name,
            save_path=curves_path
        )
    
    # 3. Prediction Confidence
    y_pred_proba = evaluation_results['predictions']['y_pred_proba']
    if y_pred_proba is not None:
        confidence_path = figures_dir / f'prediction_confidence_{timestamp}.png'
        plot_prediction_confidence(
            y_test,
            y_pred_proba,
            save_path=confidence_path
        )
    
    # 4. Feature Importance (if available)
    feature_importance = get_feature_importance(
        model,
        X_test.columns.tolist(),
        top_n=eval_config.get('top_n_features', 20)
    )
    
    if feature_importance is not None:
        # Save to CSV
        importance_path = output_dir / 'feature_importance.csv'
        feature_importance.to_csv(importance_path, index=False)
        print(f"✓ Feature importance saved to: {importance_path}")
        
        # Create plot
        importance_plot_path = figures_dir / f'feature_importance_{timestamp}.png'
        plot_feature_importance(
            feature_importance,
            top_n=eval_config.get('top_n_features', 20),
            model_name=model_name,
            save_path=importance_plot_path
        )
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10)[['feature', 'importance']].to_string(index=False))
    
    # Error analysis
    error_analysis = evaluation_results['error_analysis']
    if len(error_analysis) > 0:
        print(f"\n✓ Found {len(error_analysis)} misclassified samples")
        print(f" False Positives: {(error_analysis['error_type'] == 'False Positive').sum()}")
        print(f" False Negatives: {(error_analysis['error_type'] == 'False Negative').sum()}")
        
        # Show sample of errors
        if len(error_analysis) > 0:
            print("\nSample of Misclassified Cases:")
            sample_size = min(5, len(error_analysis))
            print(error_analysis[['true_label', 'predicted_label', 'error_type']].head(sample_size).to_string(index=False))
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nModel: {model_name}")
    print(f"Model Path: {model_path}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1-Score: {metrics['f1_score']:.4f}")
    
    # Assessment
    threshold_acc = eval_config.get('threshold_accuracy', 0.7)
    threshold_f1 = eval_config.get('threshold_f1', 0.7)
    
    if metrics['accuracy'] >= 0.85 and metrics['f1_score'] >= 0.85:
        print("\n✓ Assessment: EXCELLENT - Model is ready for deployment")
    elif metrics['accuracy'] >= threshold_acc and metrics['f1_score'] >= threshold_f1:
        print("\n✓ Assessment: GOOD - Model meets minimum requirements")
    elif metrics['accuracy'] >= 0.6 or metrics['f1_score'] >= 0.6:
        print("\n⚠ Assessment: MODERATE - Consider additional tuning")
    else:
        print("\n✗ Assessment: POOR - Significant improvements needed")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Visualizations saved to: {figures_dir}")
    print(f"Model card: {model_card_path}")
    print("="*80 + "\n")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained machine learning models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate best tuned model
  python scripts/evaluate_model.py --model models/best_model_tuned.pkl
  
  # Evaluate with custom output directory
  python scripts/evaluate_model.py --model models/best_model_baseline.pkl --output-dir results
  
  # Use custom config
  python scripts/evaluate_model.py --config my_config.yaml --model models/best_model_tuned.pkl
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file (.pkl)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: from config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(Path(args.config))
        print("\n✓ Configuration loaded successfully")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("Please create a configuration file or specify --config")
        sys.exit(1)
    
    # Create necessary directories
    config.create_directories()
    
    # Run evaluation
    evaluate(config, args)


if __name__ == '__main__':
    main()


