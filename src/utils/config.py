"""
Configuration Management Module

This module provides functions for loading and managing
project configuration from YAML files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os


class Config:
    """Configuration class for managing project settings."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        if config_path is None:
            # Try to find config file in standard locations
            config_path = self._find_config_file()
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_paths()
    
    def _find_config_file(self) -> Path:
        """
        Find configuration file in standard locations.
        
        Returns:
            Path to configuration file
        """
        # Check common locations
        possible_paths = [
            Path('config/config.yaml'),
            Path('../config/config.yaml'),
            Path('../../config/config.yaml'),
            Path.cwd() / 'config' / 'config.yaml',
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        raise FileNotFoundError(
            "Configuration file not found. Please specify config_path or "
            "place config.yaml in config/ directory."
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _setup_paths(self) -> None:
        """Setup and validate project paths."""
        paths_config = self.config.get('paths', {})
        
        # Get project root
        self.project_root = Path(paths_config.get('project_root', '.')).resolve()
        
        # Setup data paths
        data_dir = self.project_root / paths_config.get('data', 'data')
        self.data_raw = data_dir / 'raw'
        self.data_processed = data_dir / 'processed'
        self.data_external = data_dir / 'external'
        
        # Setup model paths
        self.models_dir = self.project_root / paths_config.get('models', 'models')
        
        # Setup output paths
        reports_dir = self.project_root / paths_config.get('reports', 'reports')
        self.reports_dir = reports_dir
        self.figures_dir = reports_dir / 'figures'
        
        # Setup MLflow path
        self.mlflow_dir = self.project_root / paths_config.get('mlflow', 'mlruns')
        
        # Setup logs path
        self.logs_dir = self.project_root / paths_config.get('logs', 'logs')
    
    def create_directories(self) -> None:
        """Create all necessary project directories."""
        directories = [
            self.data_raw,
            self.data_processed,
            self.data_external,
            self.models_dir,
            self.reports_dir,
            self.figures_dir,
            self.mlflow_dir,
            self.logs_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("All project directories created")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'model.random_state')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get hyperparameters for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of hyperparameters
        """
        models_config = self.config.get('models', {})
        return models_config.get(model_name, {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        Get training configuration.
        
        Returns:
            Training configuration dictionary
        """
        return self.config.get('training', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Get data configuration.
        
        Returns:
            Data configuration dictionary
        """
        return self.config.get('data', {})
    
    def get_mlflow_config(self) -> Dict[str, Any]:
        """
        Get MLflow configuration.
        
        Returns:
            MLflow configuration dictionary
        """
        return self.config.get('mlflow', {})
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(config_path={self.config_path})"
    
    def print_config(self) -> None:
        """Print current configuration."""
        print("\n" + "="*80)
        print("PROJECT CONFIGURATION")
        print("="*80)
        print(f"\nConfig File: {self.config_path}")
        print(f"Project Root: {self.project_root}")
        print("\nPaths:")
        print(f"  Data Raw: {self.data_raw}")
        print(f"  Data Processed: {self.data_processed}")
        print(f"  Models: {self.models_dir}")
        print(f"  Reports: {self.reports_dir}")
        print(f"  Figures: {self.figures_dir}")
        print(f"  MLflow: {self.mlflow_dir}")
        print(f"  Logs: {self.logs_dir}")
        
        print("\nTraining Config:")
        training = self.get_training_config()
        for key, value in training.items():
            print(f"  {key}: {value}")
        
        print("\nMLflow Config:")
        mlflow_cfg = self.get_mlflow_config()
        for key, value in mlflow_cfg.items():
            print(f"  {key}: {value}")
        
        print("\n" + "="*80 + "\n")


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        Config object
    """
    return Config(config_path)


def save_config(config: Dict[str, Any], output_path: Path) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"- Configuration saved to: {output_path}")


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration template.
    
    Returns:
        Default configuration dictionary
    """
    config = {
        'project': {
            'name': 'ml_project',
            'version': '1.0.0',
            'description': 'Machine Learning Project'
        },
        
        'paths': {
            'project_root': '.',
            'data': 'data',
            'models': 'models',
            'reports': 'reports',
            'mlflow': 'mlruns',
            'logs': 'logs'
        },
        
        'data': {
            'target_column': 'target',
            'test_size': 0.2,
            'random_state': 42,
            'stratify': True
        },
        
        'training': {
            'cv_folds': 5,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 1
        },
        
        'models': {
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear', 'saga'],
                'max_iter': 1000
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5]
            }
        },
        
        'mlflow': {
            'experiment_name': 'default_experiment',
            'tracking_uri': None,  # Will use mlruns directory
            'log_models': True,
            'log_metrics': True,
            'log_params': True
        },
        
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            'threshold_accuracy': 0.7,
            'threshold_f1': 0.7
        }
    }
    
    return config


def create_default_config_file(output_path: Path) -> None:
    """
    Create a default configuration file.
    
    Args:
        output_path: Path to save configuration file
    """
    config = get_default_config()
    save_config(config, output_path)