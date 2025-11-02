"""
Feature Importance Analysis using Permutation Importance
Works for any model type (SVM, Neural Networks, etc.)
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from pathlib import Path
import yaml
import mlflow

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
# Set MLflow expeirment
mlflow.set_experiment('feature_importance_analysis')

# Load test data
print("Loading test data...")
test_data_path = 'data/processed/student_performance_test.csv'
test_df = pd.read_csv(test_data_path)

# Separate features and target
X_test = test_df.drop(columns=['Performance_Binary'])
y_test = test_df['Performance_Binary']

print(f"Features: {X_test.shape[1]}")
print(f"Samples: {X_test.shape[0]}")

# Load model
model_path = 'models/best_model_baseline.pkl'
print(f"\nLoading model from: {model_path}")
model = joblib.load(model_path)

# Start MLflow run 
with mlflow.start_run(run_name='feature_importance'):
    
    # Log parameters
    mlflow.log_param('model_path', model_path)
    mlflow.log_param('n_features', X_test.shape[1])
    mlflow.log_param('n_samples', X_test.shape[0])
    mlflow.log_param('n_repeats', 10)   

    # Calculate permutation importance
    print("\nCalculating permutation importance...")
    print("This may take a minute...")

    result = permutation_importance(
        model, 
        X_test, 
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    # Create feature importance DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance': result.importances_mean,
        'std': result.importances_std
    }).sort_values('importance', ascending=False)
    
    # Log top features as metrics
    for i, row in feature_importance_df.head(10).iterrows():
        mlflow.log_metric(f"importance_{row['feature']}", row['importance'])

    # Save to CSV
    output_path = 'reports/feature_importance.csv'
    feature_importance_df.to_csv(output_path, index=False)
    print(f"\n✓ Feature importance saved to: {output_path}")
    
    # Log CSV as artifact
    mlflow.log_artifact(output_path)

    # Display top 10
    print("\nTop 10 Most Important Features:")
    print(feature_importance_df.head(10)[['feature', 'importance']].to_string(index=False))

    # Create visualization
    plt.figure(figsize=(12, 8))
    top_n = min(15, len(feature_importance_df))

    # Plot with error bars
    plt.barh(
        range(top_n),
        feature_importance_df['importance'].head(top_n),
        xerr=feature_importance_df['std'].head(top_n),
        color='steelblue',
        alpha=0.8
    )
    plt.yticks(range(top_n), feature_importance_df['feature'].head(top_n))
    plt.xlabel('Permutation Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Most Important Features (SVM Model)\nPermutation Importance Method')
    plt.tight_layout()

    # Save plot
    plot_path = 'reports/figures/feature_importance.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    # log plot to MLflow
    mlflow.log_figure(plt.gcf(), "feature_importance.png")
    
    plt.close()

print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("="*80)
print("Results logged to MLflow - view with mlflow ui")
