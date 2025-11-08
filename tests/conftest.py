"""
Pytest Configuration and Fixtures
Shared fixtures for all tests in the project
"""

import os
import sys
from pathlib import Path 
import pytest
import pandas as pd
import numpy as np 
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Add project roo to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# PATH FIXTURES

@pytest.fixture(scope='session')
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT

@pytest.fixture(scope='session')
def data_dir(project_root):
    """Resturn the data directory path."""
    return project_root / 'data'

@pytest.fixture(scope='session')
def models_dir(project_root):
    """Return the models directory path."""
    return project_root / 'models'

@pytest.fixture(scope='session')
def config_dir(project_root):
    """Return the config directory path."""
    return project_root / 'config'


# DATA FIXTURES

@pytest.fixture
def sample_raw_data():
    """Create sample raw student performance data."""
    np.random.seed(42)
    data = {
        'Gender': np.random.choice(['Male', 'Female'], 100),
        'Caste': np.random.choice(['General', 'OBC', 'SC', 'ST'], 100),
        'Class_X_Percentage': np.random.uniform(50, 95, 100),
        'Class_XII_Percentage': np.random.uniform(50, 95, 100),
        'Coaching': np.random.choice(['Yes', 'No'], 100),
        'Study_Hours': np.random.choice(['0-2', '2-4', '4-6', '6+'], 100),
        'Medium': np.random.choice(['English', 'Hindi'], 100),
        'Father_Education': np.random.choice(['Primary', 'Secondary', 'Graduate', 'Post-Graduate'], 100),
        'Mother_Education': np.random.choice(['Primary', 'Secondary', 'Graduate', 'Post-Graduate'], 100),
        'Father_Occupation': np.random.choice(['Business', 'Service', 'Labor', 'Other'], 100),
        'Mother_Occupation': np.random.choice(['Business', 'Service', 'Housewife', 'Other'], 100),
        'Performance': np.random.choice(['Excellent', 'Very Good', 'Good', 'Average', 'Poor'], 100)
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_processed_data():
    """Create sample processed data."""
    np.random.seed(42)
    
    # Create features
    n_samples = 100
    n_features = 15
    X = np.random.randn(n_samples, n_features)
    
    # Create binary target (0 or 1)
    y = np.random.choice([0,1], n_samples)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['Performance_Binary'] = y
    
    return df

@pytest.fixture
def sample_train_test_split(sample_processed_data):
    """Create sample train/test split"""
    df = sample_processed_data
    
    # 80/20 split
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Separate features and target
    X_train = train_df.drop('Performance_Binary', axis=1)
    y_train = train_df['Performance_Binary']
    X_test = test_df.drop('Performance_Binary', axis=1)
    y_test = test_df['Performance_Binary']
    
    return X_train, X_test, y_train, y_test

@pytest.fixture
def sample_inference_data():
    """Create sample data for model inference."""
    np.random.seed(42)
    
    data = {
        'Gender': 'Male',
        'Caste': 'General',
        'Class_X_Percentage': 85.5,
        'Class_XII_Percentage': 88.2,
        'Coaching': 'Yes',
        'Study_Hours': '4-6',
        'Medium': 'English',
        'Father_Education': 'Graduate',
        'Mother_Education': 'Graduate',
        'Father_Occupation': 'Service',
        'Mother_Occupation': 'Service'
    }
    
    return data

# MODEL FIXTURES

@pytest.fixture
def sample_model():
    """Create a sample trained model."""
    np.random.seed(42)
    
    # Create simple training data
    X_train = np.random.randn(100, 15)
    y_train = np.random.choice([0,1], 100)
    
    # Train a simple logistic regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    return model

@pytest.fixture
def sample_pipeline():
    """Create a sample sklearn pipeline."""
    np.random.seed(42)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Train pipeline
    X_train = np.random.randn(100, 15)
    y_train = np.random.choice([0,1], 100)
    pipeline.fit(X_train, y_train)
    
    return pipeline

@pytest.fixture
def temp_model_path(tmp_path):
    """Create a temporary path for saving models."""
    model_file = tmp_path / "test_model.pkl"
    return model_file

# CONFIGURATION FIXTURES

@pytest.fixture
def sample_config():
    """Create a sample configuration dictionary."""
    return {
        'project': {
            'name': 'student_performance_prediction',
            'version': '1.0.0'
        },
        'data': {
            'target_column': 'Performance_Binary',
            'test_size': 0.2,
            'random_state': 42,
            'stratify': True
        },
        'training': {
            'cv_folds': 5,
            'random_state': 42,
            'n_jobs': -1
        },
        'mlflow': {
            'baseline_experiment': 'student_performance_baseline',
            'tracking_uri': None
        }
    }
    
# METRICS FIXTURES

@pytest.fixture
def sample_predictions():
    """Create sample predictions and true labels."""
    np.random.seed(42)
    
    y_true = np.random.choice([0,1], 100)
    y_pred = np.random.choice([0,1], 100)
    y_pred_proba = np.random.rand(100)
    
    return {
        'y_true': y_true, 
        'y_pred': y_pred, 
        'y_pred_proba': y_pred_proba
    }
    
# API FIXTURES

@pytest.fixture
def sample_api_request():
    """Create a sample API request payload."""
    return {
        "Class_X_Percentage": 85.5,
        "Class_XII_Percentage": 78.0,
        "Study_Hours": 5.0,  # Float, not string
        "Gender": "Male",
        "Caste": "General",
        "Coaching": "Yes",
        "Medium": "English"
    }
    
# DRIFT DETECTION FIXTURES

@pytest.fixture
def sample_reference_data():
    """Create reference data for drift detection."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.normal(0, 1, 1000),
        'feature_3': np.random.uniform(0, 100, 1000),
    })


@pytest.fixture
def sample_drifted_data():
    """Create drifted data for drift detection testing."""
    np.random.seed(42)
    # Mean shifted by 2 standard deviations
    return pd.DataFrame({
        'feature_1': np.random.normal(2, 1, 1000),  # Drift!
        'feature_2': np.random.normal(0, 1, 1000),  # No drift
        'feature_3': np.random.uniform(20, 120, 1000),  # Drift!
    })

# CLEANUP FIXTURES

@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    import random
    random.seed(42)
    
# SESSION FIXTURES

@pytest.fixture(scope="session", autouse=True)
def test_environment():
    """Set up test environment variables."""
    os.environ['TESTING'] = 'true'
    os.environ['MLFLOW_TRACKING_URI'] = ''  # Disable MLflow during tests
    
    yield
    
    # Cleanup
    if 'TESTING' in os.environ:
        del os.environ['TESTING']
        

@pytest.fixture(scope='session', autouse=True)
def ensure_test_model(project_root):
    """Ensure test model exists before running API tests."""
    model_path = project_root / 'models' / 'pipeline_baseline.pkl'
    
    if not model_path.exists():
        # Create a simple test model
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        import joblib
        
        np.random.seed(42)
        X = np.random.randn(100, 9)
        y = np.random.choice([0, 1], 100)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        pipeline.fit(X, y)
        
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(pipeline, model_path)