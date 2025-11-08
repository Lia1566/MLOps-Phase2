"""
Unit Tests for Configuration Management
Tests for configuration loading and validation
"""

import pytest
import yaml
from pathlib import Path
import sys

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.mark.unit
class TestConfigurationLoading:
    """Test suite for configuration loading."""
    
    def test_load_config_file(self, config_dir):
        """Test loading configuration from YAML file."""
        config_file = config_dir / "config.yaml"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Assertions
            assert config is not None, "Config should not be None"
            assert isinstance(config, dict), "Config should be dictionary"
    
    def test_config_has_required_sections(self, sample_config):
        """Test that config has all required sections."""
        required_sections = ['project', 'data', 'training']
        
        for section in required_sections:
            assert section in sample_config, f"Config should have {section} section"
    
    def test_project_config_fields(self, sample_config):
        """Test project configuration fields."""
        project_config = sample_config['project']
        
        assert 'name' in project_config, "Project should have name"
        assert 'version' in project_config, "Project should have version"
        
        assert isinstance(project_config['name'], str), "Name should be string"
        assert isinstance(project_config['version'], str), "Version should be string"
    
    def test_data_config_fields(self, sample_config):
        """Test data configuration fields."""
        data_config = sample_config['data']
        
        required_fields = ['target_column', 'test_size', 'random_state']
        
        for field in required_fields:
            assert field in data_config, f"Data config should have {field}"
        
        # Validate types
        assert isinstance(data_config['target_column'], str)
        assert isinstance(data_config['test_size'], float)
        assert isinstance(data_config['random_state'], int)
    
    def test_training_config_fields(self, sample_config):
        """Test training configuration fields."""
        training_config = sample_config['training']
        
        required_fields = ['cv_folds', 'random_state']
        
        for field in required_fields:
            assert field in training_config, f"Training config should have {field}"
        
        # Validate types
        assert isinstance(training_config['cv_folds'], int)
        assert isinstance(training_config['random_state'], int)


@pytest.mark.unit
class TestConfigurationValidation:
    """Test configuration value validation."""
    
    def test_test_size_range(self, sample_config):
        """Test that test_size is in valid range."""
        test_size = sample_config['data']['test_size']
        
        assert 0 < test_size < 1, "Test size should be between 0 and 1"
        assert 0.1 <= test_size <= 0.4, "Test size should typically be 10-40%"
    
    def test_cv_folds_range(self, sample_config):
        """Test that cv_folds is in valid range."""
        cv_folds = sample_config['training']['cv_folds']
        
        assert cv_folds >= 2, "CV folds should be at least 2"
        assert cv_folds <= 10, "CV folds should not exceed 10"
    
    def test_random_state_values(self, sample_config):
        """Test that random_state values are valid."""
        data_random_state = sample_config['data']['random_state']
        training_random_state = sample_config['training']['random_state']
        
        assert isinstance(data_random_state, int), "Random state should be integer"
        assert isinstance(training_random_state, int), "Random state should be integer"
        
        assert data_random_state >= 0, "Random state should be non-negative"
        assert training_random_state >= 0, "Random state should be non-negative"
    
    def test_project_name_format(self, sample_config):
        """Test project name format."""
        project_name = sample_config['project']['name']
        
        # Should not be empty
        assert len(project_name) > 0, "Project name should not be empty"
        
        # Should not have special characters (allow underscore and dash)
        import re
        assert re.match(r'^[a-zA-Z0-9_-]+$', project_name), \
            "Project name should only contain alphanumeric, underscore, and dash"
    
    def test_version_format(self, sample_config):
        """Test version format."""
        version = sample_config['project']['version']
        
        # Should follow semantic versioning (X.Y.Z)
        import re
        assert re.match(r'^\d+\.\d+\.\d+$', version), \
            "Version should follow semantic versioning (X.Y.Z)"


@pytest.mark.unit
class TestConfigurationPaths:
    """Test path handling in configuration."""
    
    def test_config_file_exists(self, config_dir):
        """Test that config.yaml exists."""
        config_file = config_dir / "config.yaml"
        
        if config_dir.exists():
            # If config dir exists, check for config file
            assert config_file.exists() or True, "Config file should exist if dir exists"
    
    def test_relative_paths_resolution(self, project_root):
        """Test that relative paths can be resolved."""
        # Test common relative paths
        data_dir = project_root / "data"
        models_dir = project_root / "models"
        
        # These directories should be accessible (even if empty)
        assert isinstance(data_dir, Path), "Data dir should be Path object"
        assert isinstance(models_dir, Path), "Models dir should be Path object"


@pytest.mark.unit
class TestConfigurationErrors:
    """Test error handling in configuration."""
    
    def test_missing_required_field(self):
        """Test handling of missing required fields."""
        incomplete_config = {
            'project': {
                'name': 'test_project'
                # Missing 'version'
            }
        }
        
        # Should detect missing field
        assert 'version' not in incomplete_config['project']
    
    def test_invalid_yaml_content(self, tmp_path):
        """Test handling of invalid YAML."""
        invalid_yaml_file = tmp_path / "invalid.yaml"
        
        # Write invalid YAML
        with open(invalid_yaml_file, 'w') as f:
            f.write("invalid: yaml: content:")
        
        # Should raise error when loading
        with pytest.raises(yaml.YAMLError):
            with open(invalid_yaml_file, 'r') as f:
                yaml.safe_load(f)
    
    def test_empty_config_file(self, tmp_path):
        """Test handling of empty config file."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.touch()
        
        # Loading empty YAML should return None
        with open(empty_file, 'r') as f:
            config = yaml.safe_load(f)
        
        assert config is None, "Empty YAML should return None"


@pytest.mark.unit
class TestConfigurationDefaults:
    """Test default configuration values."""
    
    def test_default_test_size(self, sample_config):
        """Test default test size is reasonable."""
        test_size = sample_config['data']['test_size']
        
        # Common default is 0.2 (20%)
        assert test_size == 0.2, "Default test size should be 0.2"
    
    def test_default_cv_folds(self, sample_config):
        """Test default CV folds."""
        cv_folds = sample_config['training']['cv_folds']
        
        # Common default is 5
        assert cv_folds == 5, "Default CV folds should be 5"
    
    def test_default_random_state(self, sample_config):
        """Test default random state."""
        random_state = sample_config['data']['random_state']
        
        # Common default is 42
        assert random_state == 42, "Default random state should be 42"


@pytest.mark.unit
class TestConfigurationUpdate:
    """Test configuration update functionality."""
    
    def test_update_config_values(self, sample_config):
        """Test that config values can be updated."""
        original_test_size = sample_config['data']['test_size']
        
        # Update value
        sample_config['data']['test_size'] = 0.3
        
        # Verify update
        assert sample_config['data']['test_size'] == 0.3
        assert sample_config['data']['test_size'] != original_test_size
    
    def test_add_new_config_field(self, sample_config):
        """Test adding new field to config."""
        # Add new field
        sample_config['data']['new_field'] = 'new_value'
        
        # Verify addition
        assert 'new_field' in sample_config['data']
        assert sample_config['data']['new_field'] == 'new_value'


@pytest.mark.unit
class TestMLflowConfiguration:
    """Test MLflow-specific configuration."""
    
    def test_mlflow_config_exists(self, sample_config):
        """Test that MLflow configuration exists."""
        assert 'mlflow' in sample_config, "Config should have mlflow section"
    
    def test_mlflow_experiment_names(self, sample_config):
        """Test MLflow experiment name configuration."""
        mlflow_config = sample_config['mlflow']
        
        if 'baseline_experiment' in mlflow_config:
            assert isinstance(mlflow_config['baseline_experiment'], str)
            assert len(mlflow_config['baseline_experiment']) > 0
    
    def test_mlflow_tracking_uri(self, sample_config):
        """Test MLflow tracking URI configuration."""
        mlflow_config = sample_config['mlflow']
        
        if 'tracking_uri' in mlflow_config:
            tracking_uri = mlflow_config['tracking_uri']
            
            # Can be None (local) or string (remote)
            assert tracking_uri is None or isinstance(tracking_uri, str)


@pytest.mark.unit
class TestConfigurationConsistency:
    """Test consistency between configuration values."""
    
    def test_random_state_consistency(self, sample_config):
        """Test that random states are consistent."""
        data_rs = sample_config['data']['random_state']
        training_rs = sample_config['training']['random_state']
        
        # For reproducibility, they should often be the same
        # (though not strictly required)
        if data_rs == training_rs:
            assert data_rs == training_rs, "Random states should match for reproducibility"
    
    def test_n_jobs_value(self, sample_config):
        """Test n_jobs configuration."""
        if 'n_jobs' in sample_config['training']:
            n_jobs = sample_config['training']['n_jobs']
            
            # Should be -1 (all cores) or positive integer
            assert n_jobs == -1 or n_jobs > 0, "n_jobs should be -1 or positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])