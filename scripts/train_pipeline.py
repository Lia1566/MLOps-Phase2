#!/usr/bin/env python3
"""
Training Script with Scikit-Learn Pipelines

This script trains machine learning models using Scikit-Learn Pipelines
for automated preprocessing, training, and evaluation.

The pipeline ensures reproducibility and follows ML best practices:
1. Preprocessing (StandardScaler)
2. Model Training
3. Cross-Validation
4. MLflow Tracking

Usage:
    python scripts/train_pipeline.py --mode baseline
    python scripts/train_pipeline.py --config config/config.yaml
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.data.loader import load_data
from src.models.pipeline import (
    train_baseline_pipelines,
    save_pipeline,
    get_pipeline_info,
    print_pipeline_structure
)
from src.visualization.plots import (
    plot_model_comparison,
    plot_training_comparison
)


def setup_mlflow(config):
    """Setup MLflow tracking."""
    mlflow_config = config.get_mlflow_config()
    
    # Set tracking URI
    tracking_uri = mlflow_config.get('tracking_uri')
    if tracking_uri is None:
        tracking_uri = f"file://{config.mlflow_dir}"
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"✓ MLflow tracking URI: {mlflow.get_tracking_uri()}")


def train_pipelines(config, args):
    """Train models using Scikit-Learn pipelines."""
    print("\n" + "="*80)
    print("SCIKIT-LEARN PIPELINE TRAINING")
    print("="*80)
    print("\nPipeline Components:")
    print("  1. Preprocessing: StandardScaler (normalize features)")
    print("  2. Model Training: Multiple algorithms")
    print("  3. Cross-Validation: 5-fold stratified")
    print("  4. Evaluation: Test set performance")
    print("  5. MLflow Tracking: All experiments logged")
    
    # Setup MLflow
    setup_mlflow(config)
    experiment_name = config.get_mlflow_config().get(
        'baseline_experiment',
        'baseline_experiment'
    ) + '_pipeline'
    
    # Load data
    print("\n" + "-"*80)
    print("LOADING DATA")
    print("-"*80)
    train_data = load_data(config.data_processed / config.get('data.train_file'))
    test_data = load_data(config.data_processed / config.get('data.test_file'))
    
    target_col = config.get('data.target_column')
    X_train = train_data.drop(target_col, axis=1)
    y_train = train_data[target_col]
    X_test = test_data.drop(target_col, axis=1)
    y_test = test_data[target_col]
    
    print(f"✓ Data loaded successfully")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Target: {target_col}")
    
    # Get training parameters
    training_config = config.get_training_config()
    cv_folds = training_config.get('cv_folds', 5)
    random_state = training_config.get('random_state', 42)
    
    # Train pipelines
    trained_pipelines, results_df = train_baseline_pipelines(
        X_train, y_train, X_test, y_test,
        experiment_name=experiment_name,
        cv_folds=cv_folds,
        random_state=random_state
    )
    
    # Save results
    print("\n" + "-"*80)
    print("SAVING RESULTS")
    print("-"*80)
    
    results_path = config.reports_dir / 'pipeline_baseline_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"✓ Results saved to: {results_path}")
    
    # Save best pipeline
    best_model_name = results_df.iloc[0]['Model']
    best_pipeline = trained_pipelines[best_model_name]
    best_pipeline_path = config.models_dir / 'best_pipeline_baseline.pkl'
    save_pipeline(best_pipeline, best_pipeline_path, best_model_name)
    
    # Print best pipeline structure
    print_pipeline_structure(best_pipeline)
    
    # Save all pipelines
    print("Saving all pipelines...")
    for model_name, pipeline in trained_pipelines.items():
        pipeline_filename = model_name.lower().replace(' ', '_') + '_pipeline.pkl'
        pipeline_path = config.models_dir / pipeline_filename
        save_pipeline(pipeline, pipeline_path)
    print(f"✓ All {len(trained_pipelines)} pipelines saved to: {config.models_dir}")
    
    # Create visualizations
    print("\n" + "-"*80)
    print("CREATING VISUALIZATIONS")
    print("-"*80)
    
    # Model comparison plot
    comparison_path = config.figures_dir / 'pipeline_baseline_comparison.png'
    plot_model_comparison(
        results_df,
        metric='Test Accuracy',
        save_path=comparison_path
    )
    print(f"✓ Model comparison saved")
    
    # Training comparison plot
    training_comp_path = config.figures_dir / 'pipeline_training_comparison.png'
    plot_training_comparison(
        results_df,
        save_path=training_comp_path
    )
    print(f"✓ Training comparison saved")
    
    # Create pipeline documentation
    doc_path = config.reports_dir / 'pipeline_documentation.txt'
    with open(doc_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SCIKIT-LEARN PIPELINE DOCUMENTATION\n")
        f.write("="*80 + "\n\n")
        
        f.write("Pipeline Structure:\n")
        f.write("-"*80 + "\n")
        f.write("1. Preprocessing Step:\n")
        f.write("   - StandardScaler: Standardizes features by removing mean and scaling to unit variance\n")
        f.write("   - Formula: z = (x - mean) / std\n")
        f.write("   - Benefits: Ensures all features are on the same scale\n\n")
        
        f.write("2. Model Training Step:\n")
        f.write("   - Various algorithms: Logistic Regression, Random Forest, SVM, etc.\n")
        f.write("   - Each model trained on preprocessed data\n")
        f.write("   - Hyperparameters logged to MLflow\n\n")
        
        f.write("3. Cross-Validation:\n")
        f.write(f"   - {cv_folds}-fold stratified cross-validation\n")
        f.write("   - Ensures robust performance estimates\n")
        f.write("   - Maintains class distribution in each fold\n\n")
        
        f.write("="*80 + "\n")
        f.write("TRAINED PIPELINES\n")
        f.write("="*80 + "\n\n")
        
        for i, row in results_df.iterrows():
            f.write(f"{i+1}. {row['Model']}\n")
            f.write(f"Pipeline: {row['Pipeline']}\n")
            f.write(f"Test Accuracy: {row['Test Accuracy']:.4f}\n")
            f.write(f"Test F1-Score: {row['Test F1']:.4f}\n")
            f.write(f"CV Score: {row['CV Mean']:.4f} (+/- {row['CV Std']:.4f})\n\n")
        
        f.write("="*80 + "\n")
        f.write("BEST PIPELINE\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {best_model_name}\n")
        f.write(f"Test Accuracy: {results_df.iloc[0]['Test Accuracy']:.4f}\n")
        f.write(f"Test F1-Score: {results_df.iloc[0]['Test F1']:.4f}\n")
        f.write(f"Saved to: {best_pipeline_path}\n\n")
        
        f.write("="*80 + "\n")
        f.write("REPRODUCIBILITY\n")
        f.write("="*80 + "\n\n")
        f.write("To reproduce these results:\n\n")
        f.write("1. Load the pipeline:\n")
        f.write("   from src.models.pipeline import load_pipeline\n")
        f.write(f"   pipeline = load_pipeline('{best_pipeline_path}')\n\n")
        f.write("2. Make predictions:\n")
        f.write("   predictions = pipeline.predict(X_new)\n")
        f.write("   probabilities = pipeline.predict_proba(X_new)\n\n")
        f.write("The pipeline automatically handles preprocessing!\n\n")
    
    print(f"✓ Pipeline documentation saved to: {doc_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("PIPELINE TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest Pipeline: {best_model_name}")
    print(f" Structure: StandardScaler → {best_model_name}")
    print(f" Test Accuracy: {results_df.iloc[0]['Test Accuracy']:.4f}")
    print(f" Test F1-Score: {results_df.iloc[0]['Test F1']:.4f}")
    print(f" Cross-Val Score: {results_df.iloc[0]['CV Mean']:.4f} (+/- {results_df.iloc[0]['CV Std']:.4f})")
    
    print(f"\nOutputs:")
    print(f"  Pipelines: {config.models_dir}")
    print(f"  Results: {results_path}")
    print(f"  Visualizations: {config.figures_dir}")
    print(f"  Documentation: {doc_path}")
    
    print(f"\nMLflow:")
    print(f"  Experiment: {experiment_name}")
    print(f"  View UI: mlflow ui")
    print(f"  Navigate to: http://localhost:5000")
    
    print("\n" + "="*80)
    print("✓ All pipelines trained and saved successfully!")
    print("✓ Each pipeline includes preprocessing + model in a single object")
    print("✓ Pipelines are reproducible and ready for deployment")
    print("="*80 + "\n")


def main():
    """Main pipeline training script."""
    parser = argparse.ArgumentParser(
        description='Train ML models using Scikit-Learn Pipelines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train baseline pipelines
  python scripts/train_pipeline.py
  
  # Use custom config
  python scripts/train_pipeline.py --config my_config.yaml
  
What are Pipelines?
  Pipelines chain preprocessing and modeling steps into a single object:
  - Ensures preprocessing is applied consistently
  - Prevents data leakage
  - Makes deployment easier
  - Improves reproducibility
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(Path(args.config))
        print("\n" + "="*80)
        print("CONFIGURATION LOADED")
        print("="*80)
        print(f"Config file: {args.config}")
        print(f"Project root: {config.project_root}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please create a configuration file or specify --config")
        sys.exit(1)
    
    # Create necessary directories
    config.create_directories()
    
    # Run pipeline training
    train_pipelines(config, args)


if __name__ == '__main__':
    main()