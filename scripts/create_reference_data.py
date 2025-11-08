""" 
Create Reference Data for Drift Detection
Generates a baseline dataset for monitoring
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def create_reference_data(n_samples=1000):
    """
    Create reference data that represents "normal" data distribution.
    
    Args:
        n_samples: Number of samples to generate
    """
    print(f"Creating reference data with {n_samples} samples...")
    
    # Generate realistic student data
    data = {
        'Class_X_Percentage': np.random.normal(75, 10, n_samples).clip(50, 100),
        'Class_XII_Percentage': np.random.normal(75, 10, n_samples).clip(50, 100),
        'Study_Hours': np.random.gamma(3, 1.5, n_samples).clip(0, 12),
        'Gender_Male': np.random.choice([0, 1], n_samples, p=[0.48, 0.52]),
        'Caste_General': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'Caste_OBC': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'Caste_SC': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'Coaching_Yes': np.random.choice([0, 1], n_samples, p=[0.35, 0.65]),
        'Medium_English': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    }
    
    df = pd.DataFrame(data)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'data' / 'reference'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save reference data
    output_path = output_dir / 'reference_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Reference data saved to: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"\nSample statistics:")
    print(df.describe())
    
    return df


if __name__ == "__main__":
    create_reference_data(n_samples=1000)
