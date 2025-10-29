#!/usr/bin/env python3
"""
Training Script

This script trains machine learning models using the refactored codebase.
It can train baseline models or perform hyperparameter tuning.

Usage:
    python scripts/train_model.py --mode baseline
    python scripts/train_model.py --mode tuning --top-n 3
    python scripts/train_model.py --config config/config.yaml --mode baseline
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
from src.models.train import (train_baseline_models, tune_top_models, save_model)
from src.visualization.plots import (
    plot_model_comparison,
    plot_training_comparison
)

# DVC Integration
try:
    from src.utils.dvc_manager import DVCManager
    DVC_AVAILABLE = True
except ImportError:
    print("Warning: DVC manager is not available")
    DVC_AVAILABLE = False
    
def track_models_with_dvc(config, trained_models, results_df, mode='baseline', use_dvc=True):
    """
    Track trained models with DVC.
    
    Args:
        config: Configuration object
        trained_models: Dictionary of trained models
        results_df: DataFrame with results
        mode: Training mode ('baseline' or 'tuning')
        use_dvc: Whether to use DVC tracking
    """
    if not use_dvc or not DVC_AVAILABLE:
        return
    
    print("\n" + "="*80)
    print("DVC MODEL TRACKING")
    print("="*80)
    
    try:
        dvc = DVCManager(project_root=config.project_root)
        
        if not dvc.is_initialized():
            print("Initializing DVC...")
            dvc.initialize()
        
        # Track all model files
        print("\nTracking model files with DVC...")
        models_dir = config.models_dir
        
        tracked_count = 0
        for model_file in models_dir.glob('*.pkl'):
            print(f"Tracking: {model_file.name}")
            success = dvc.track_model(model_file)
            if success:
                tracked_count += 1
                print(f" {model_file.name} tracked")
        
        print(f"\nTracked {tracked_count} model file(s)")
        
        # Track results files
        print("\nTracking results files...")
        results_file = config.reports_dir / f'{mode}_results.csv'
        if results_file.exists():
            dvc.track_file(results_file)
            print(f"    {results_file.name} tracked")
        
        # Update params.yaml with model parameters
        print("\nUpdating parameters...")
        best_model_name = results_df.iloc[0]['Model']
        model_params = {
            'model': {
                'algorithm': best_model_name,
                'mode': mode,
                'random_state': config.get('training.random_state', 42),
                'best_accuracy': float(results_df.iloc[0]['Test Accuracy']),
                'best_f1': float(results_df.iloc[0]['Test F1']),
            }
        }
        dvc.update_params(model_params)
        print("Parameters updated")
        
        # Create a git tag for this training run
        try:
            import subprocess
            from datetime import datetime
            tag_name = f"{mode}-{best_model_name.lower().replace(' ', '-')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            subprocess.run(
                ['git', 'tag', '-a', tag_name, '-m', f'{mode} training: {best_model_name}'],
                cwd=config.project_root,
                capture_output=True,
                check=False
            )
            print(f"\nGit tag created: {tag_name}")
        except Exception as e:
            print(f"\nCould not create git tag: {e}")
        
        # Show status
        print("\nDVC Status:")
        dvc.status()
        
        # Check for remotes
        remotes = dvc.list_remotes()
        if remotes:
            print(f"\nDVC remote configured")
            print("Push models with: dvc push")
        else:
            print("\nNo DVC remote configured")
            print("Models are tracked locally only")
        
        print("\n" + "="*80)
        print("✓ DVC MODEL TRACKING COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"\nDVC tracking failed: {e}")
        print("Continuing without DVC tracking...")


def setup_mlflow(config):
    """Setup MLflow tracking."""
    mlflow_config = config.get_mlflow_config()
    
    # Set tracking URI
    tracking_uri = mlflow_config.get('tracking_uri')
    if tracking_uri is None:
        tracking_uri = f"file://{config.mlflow_dir}"
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"- MLflow tracking URI: {mlflow.get_tracking_uri()}")


def train_baseline(config, args):
    """Train baseline models."""
    print("\n" + "="*80)
    print("BASELINE MODEL TRAINING")
    print("="*80)
    
    # Setup MLflow
    setup_mlflow(config)
    experiment_name = config.get_mlflow_config().get(
        'baseline_experiment', 
        'baseline_experiment'
    )
    
    # Load data
    print("\nLoading data...")
    train_data = load_data(config.data_processed / config.get('data.train_file'))
    test_data = load_data(config.data_processed / config.get('data.test_file'))
    
    target_col = config.get('data.target_column')
    X_train = train_data.drop(target_col, axis=1)
    y_train = train_data[target_col]
    X_test = test_data.drop(target_col, axis=1)
    y_test = test_data[target_col]
    
    print(f"- Data loaded")
    print(f" Training samples: {len(X_train)}")
    print(f" Test samples: {len(X_test)}")
    print(f" Features: {X_train.shape[1]}")
    
    # Get training parameters
    training_config = config.get_training_config()
    cv_folds = training_config.get('cv_folds', 5)
    random_state = training_config.get('random_state', 42)
    
    # Train models
    trained_models, results_df = train_baseline_models(
        X_train, y_train, X_test, y_test,
        experiment_name=experiment_name,
        cv_folds=cv_folds,
        random_state=random_state
    )
    
    # Save results
    results_path = config.reports_dir / 'baseline_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Save best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    best_model_path = config.models_dir / 'best_model_baseline.pkl'
    save_model(best_model, best_model_path, best_model_name)
    
    # Save all models
    for model_name, model in trained_models.items():
        model_filename = model_name.lower().replace(' ', '_') + '_baseline.pkl'
        model_path = config.models_dir / model_filename
        save_model(model, model_path)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Model comparison plot
    comparison_path = config.figures_dir / 'baseline_model_comparison.png'
    plot_model_comparison(
        results_df,
        metric='Test Accuracy',
        save_path=comparison_path
    )
    
    # Training comparison plot
    training_comp_path = config.figures_dir / 'baseline_training_comparison.png'
    plot_training_comparison(
        results_df,
        save_path=training_comp_path
    )
    
    # DVC tracking (if enabled)
    if args.track_with_dvc:
        track_models_with_dvc(config, trained_models, results_df, mode='baseline', use_dvc=True)
    
    # Print summary
    print("\n" + "="*80)
    print("BASELINE TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest Model: {best_model_name}")
    print(f"Test Accuracy: {results_df.iloc[0]['Test Accuracy']:.4f}")
    print(f"Test F1-Score: {results_df.iloc[0]['Test F1']:.4f}")
    print(f"\nModels saved to: {config.models_dir}")
    print(f"Results saved to: {results_path}")
    print(f"Visualizations saved to: {config.figures_dir}")
    
    if args.track_with_dvc and DVC_AVAILABLE:
        print(f"\nModels tracked with DVC")
    
    print(f"\nView MLflow UI: mlflow ui")
    print("="*80 + "\n")


def train_tuned(config, args):
    """Train models with hyperparameter tuning."""
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING")
    print("="*80)
    
    # Setup MLflow
    setup_mlflow(config)
    experiment_name = config.get_mlflow_config().get(
        'tuning_experiment',
        'tuning_experiment'
    )
    
    # Load data
    print("\nLoading data...")
    train_data = load_data(config.data_processed / config.get('data.train_file'))
    test_data = load_data(config.data_processed / config.get('data.test_file'))
    
    target_col = config.get('data.target_column')
    X_train = train_data.drop(target_col, axis=1)
    y_train = train_data[target_col]
    X_test = test_data.drop(target_col, axis=1)
    y_test = test_data[target_col]
    
    print(f"Data loaded")
    
    # Load baseline results to determine top models
    baseline_results_path = config.reports_dir / 'baseline_results.csv'
    if not baseline_results_path.exists():
        print("\nWarning: Baseline results not found!")
        print("Please run baseline training first: python scripts/train_model.py --mode baseline")
        return
    
    baseline_results = pd.read_csv(baseline_results_path)
    top_n = args.top_n if args.top_n else config.get('training.top_n_models', 3)
    top_models = baseline_results.head(top_n)['Model'].tolist()
    
    print(f"\nTop {top_n} models selected for tuning:")
    for i, model_name in enumerate(top_models, 1):
        acc = baseline_results[baseline_results['Model'] == model_name]['Test Accuracy'].values[0]
        print(f"  {i}. {model_name} (Baseline Accuracy: {acc:.4f})")
    
    # Get training parameters
    training_config = config.get_training_config()
    cv_folds = training_config.get('cv_folds', 5)
    random_state = training_config.get('random_state', 42)
    
    # Tune models
    tuned_models, results_df = tune_top_models(
        X_train, y_train, X_test, y_test,
        top_models=top_models,
        experiment_name=experiment_name,
        cv_folds=cv_folds,
        random_state=random_state
    )
    
    # Save results
    results_path = config.reports_dir / 'tuning_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Tuning results saved to: {results_path}")
    
    # Save best tuned model
    best_model_name = results_df.iloc[0]['Model']
    best_model = tuned_models[best_model_name]
    best_model_path = config.models_dir / 'best_model_tuned.pkl'
    save_model(best_model, best_model_path, best_model_name)
    
    # Save all tuned models
    for model_name, model in tuned_models.items():
        model_filename = model_name.lower().replace(' ', '_') + '_tuned.pkl'
        model_path = config.models_dir / model_filename
        save_model(model, model_path)
    
    # Save hyperparameters
    params_path = config.reports_dir / 'best_hyperparameters.txt'
    with open(params_path, 'w') as f:
        f.write("BEST HYPERPARAMETERS\n")
        f.write("="*80 + "\n\n")
        for _, row in results_df.iterrows():
            f.write(f"{row['Model']}:\n")
            f.write(f"  Test Accuracy: {row['Test Accuracy']:.4f}\n")
            f.write(f"  Parameters: {row['Best Params']}\n\n")
    print(f"Best parameters saved to: {params_path}")
    
    # Create comparison with baseline
    if baseline_results_path.exists():
        comparison_data = []
        for _, row in results_df.iterrows():
            model_name = row['Model']
            baseline_acc = baseline_results[
                baseline_results['Model'] == model_name
            ]['Test Accuracy'].values[0]
            
            comparison_data.append({
                'Model': model_name,
                'Baseline Accuracy': baseline_acc,
                'Tuned Accuracy': row['Test Accuracy'],
                'Improvement': row['Test Accuracy'] - baseline_acc
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = config.reports_dir / 'baseline_vs_tuned_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        print(f"✓ Comparison saved to: {comparison_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    comparison_path = config.figures_dir / 'tuned_model_comparison.png'
    plot_model_comparison(
        results_df,
        metric='Test Accuracy',
        save_path=comparison_path
    )
    
    # DVC tracking (if enabled)
    if args.track_with_dvc:
        track_models_with_dvc(config, tuned_models, results_df, mode='tuning', use_dvc=True)
    
    # Print summary
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("="*80)
    print(f"\nBest Tuned Model: {best_model_name}")
    print(f"Test Accuracy: {results_df.iloc[0]['Test Accuracy']:.4f}")
    print(f"Test F1-Score: {results_df.iloc[0]['Test F1']:.4f}")
    
    if baseline_results_path.exists():
        baseline_acc = baseline_results[
            baseline_results['Model'] == best_model_name
        ]['Test Accuracy'].values[0]
        improvement = results_df.iloc[0]['Test Accuracy'] - baseline_acc
        print(f"Improvement over baseline: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    print(f"\nModels saved to: {config.models_dir}")
    print(f"Results saved to: {results_path}")
    print(f"Visualizations saved to: {config.figures_dir}")
    if args.track_with_dvc and DVC_AVAILABLE:
        print(f"\nModels tracked with DVC")
    print(f"\nView MLflow UI: mlflow ui")
    print("="*80 + "\n")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description='Train machine learning models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train baseline models
  python scripts/train_model.py --mode baseline
  
  # Tune top 3 models
  python scripts/train_model.py --mode tuning --top-n 3
  
  # Use custom config
  python scripts/train_model.py --config my_config.yaml --mode baseline
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['baseline', 'tuning'],
        required=True,
        help='Training mode: baseline or tuning'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=None,
        help='Number of top models to tune (default: from config)'
    )
    
    parser.add_argument(
        '--track-with-dvc',
        action='store_true',
        help='Enable DVC tracking for models and results'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(Path(args.config))
        print("\n✓ Configuration loaded successfully")
        config.print_config()
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("Please create a configuration file or specify --config")
        sys.exit(1)
    
    # Create necessary directories
    config.create_directories()
    
    # Run training based on mode
    if args.mode == 'baseline':
        train_baseline(config, args)
    elif args.mode == 'tuning':
        train_tuned(config, args)


if __name__ == '__main__':
    main()
