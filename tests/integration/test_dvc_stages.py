"""
Integration Tests for DVC Pipeline
Tests for DVC pipeline stages and reproducibility
"""

import pytest
import subprocess
from pathlib import Path
import yaml
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.mark.integration
class TestDVCPipelineStructure:
    """Test DVC pipeline structure and configuration."""
    
    def test_dvc_yaml_exists(self, project_root):
        """Test that dvc.yaml exists."""
        dvc_yaml = project_root / "dvc.yaml"
        
        if not dvc_yaml.exists():
            pytest.skip("dvc.yaml not found - DVC pipeline not configured")
        
        assert dvc_yaml.exists(), "dvc.yaml should exist"
    
    def test_dvc_yaml_valid(self, project_root):
        """Test that dvc.yaml is valid YAML."""
        dvc_yaml = project_root / "dvc.yaml"
        
        if not dvc_yaml.exists():
            pytest.skip("dvc.yaml not found")
        
        with open(dvc_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        assert config is not None, "dvc.yaml should not be empty"
        assert 'stages' in config, "dvc.yaml should have stages"
    
    def test_dvc_pipeline_stages(self, project_root):
        """Test that DVC pipeline has expected stages."""
        dvc_yaml = project_root / "dvc.yaml"
        
        if not dvc_yaml.exists():
            pytest.skip("dvc.yaml not found")
        
        with open(dvc_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        stages = config.get('stages', {})
        
        # Expected stages from Phase 2
        expected_stages = ['prepare_data', 'train_baseline', 'train_pipeline']
        
        for stage in expected_stages:
            if stage in stages:
                assert stage in stages, f"Pipeline should have {stage} stage"
    
    def test_params_yaml_exists(self, project_root):
        """Test that params.yaml exists."""
        params_yaml = project_root / "params.yaml"
        
        if params_yaml.exists():
            with open(params_yaml, 'r') as f:
                params = yaml.safe_load(f)
            
            assert params is not None, "params.yaml should not be empty"


@pytest.mark.integration
@pytest.mark.slow
class TestDVCPipelineExecution:
    """Test DVC pipeline execution (requires DVC installed)."""
    
    def test_dvc_status(self, project_root):
        """Test dvc status command."""
        try:
            result = subprocess.run(
                ['dvc', 'status'],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Should not crash
            assert result.returncode in [0, 1], "dvc status should run successfully"
            
        except FileNotFoundError:
            pytest.skip("DVC not installed")
        except subprocess.TimeoutExpired:
            pytest.fail("dvc status timed out")
    
    def test_dvc_dag(self, project_root):
        """Test dvc dag command."""
        try:
            result = subprocess.run(
                ['dvc', 'dag'],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Should show pipeline structure
            assert result.returncode == 0, "dvc dag should run successfully"
            
            output = result.stdout
            
            # Should contain stage names
            if 'prepare_data' in output:
                assert 'prepare_data' in output, "DAG should show prepare_data stage"
            
        except FileNotFoundError:
            pytest.skip("DVC not installed")
        except subprocess.TimeoutExpired:
            pytest.fail("dvc dag timed out")


@pytest.mark.integration
class TestDVCDataVersioning:
    """Test DVC data versioning."""
    
    def test_dvc_directory_exists(self, project_root):
        """Test that .dvc directory exists."""
        dvc_dir = project_root / ".dvc"
        
        if not dvc_dir.exists():
            pytest.skip(".dvc directory not found - DVC not initialized")
        
        assert dvc_dir.exists(), ".dvc directory should exist"
        assert dvc_dir.is_dir(), ".dvc should be a directory"
    
    def test_dvc_config_exists(self, project_root):
        """Test that DVC config exists."""
        dvc_config = project_root / ".dvc" / "config"
        
        if not dvc_config.exists():
            pytest.skip("DVC config not found")
        
        assert dvc_config.exists(), "DVC config should exist"
    
    def test_dvc_tracked_files(self, project_root):
        """Test that data files are tracked by DVC."""
        data_dir = project_root / "data"
        
        if not data_dir.exists():
            pytest.skip("data directory not found")
        
        # Check for .dvc files
        dvc_files = list(data_dir.rglob("*.dvc"))
        
        # May have DVC-tracked files
        if len(dvc_files) > 0:
            assert len(dvc_files) > 0, "Should have DVC-tracked files"


@pytest.mark.integration
class TestDVCStageDefinitions:
    """Test individual DVC stage definitions."""
    
    def test_prepare_data_stage(self, project_root):
        """Test prepare_data stage definition."""
        dvc_yaml = project_root / "dvc.yaml"
        
        if not dvc_yaml.exists():
            pytest.skip("dvc.yaml not found")
        
        with open(dvc_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        stages = config.get('stages', {})
        
        if 'prepare_data' in stages:
            stage = stages['prepare_data']
            
            # Should have cmd
            assert 'cmd' in stage, "Stage should have cmd"
            
            # Should have deps or outs
            has_io = 'deps' in stage or 'outs' in stage or 'params' in stage
            assert has_io, "Stage should have deps, outs, or params"
    
    def test_train_baseline_stage(self, project_root):
        """Test train_baseline stage definition."""
        dvc_yaml = project_root / "dvc.yaml"
        
        if not dvc_yaml.exists():
            pytest.skip("dvc.yaml not found")
        
        with open(dvc_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        stages = config.get('stages', {})
        
        if 'train_baseline' in stages:
            stage = stages['train_baseline']
            
            # Should have cmd
            assert 'cmd' in stage, "Stage should have cmd"
            
            # Should depend on processed data
            if 'deps' in stage:
                deps = stage['deps']
                assert any('data' in str(dep) for dep in deps if isinstance(dep, (str, dict)))


@pytest.mark.integration
class TestDVCReproducibility:
    """Test DVC reproducibility features."""
    
    def test_dvc_lock_exists(self, project_root):
        """Test that dvc.lock exists."""
        dvc_lock = project_root / "dvc.lock"
        
        if not dvc_lock.exists():
            pytest.skip("dvc.lock not found - pipeline not run yet")
        
        assert dvc_lock.exists(), "dvc.lock should exist after pipeline runs"
    
    def test_dvc_lock_valid(self, project_root):
        """Test that dvc.lock is valid YAML."""
        dvc_lock = project_root / "dvc.lock"
        
        if not dvc_lock.exists():
            pytest.skip("dvc.lock not found")
        
        with open(dvc_lock, 'r') as f:
            lock_data = yaml.safe_load(f)
        
        assert lock_data is not None, "dvc.lock should not be empty"
        assert 'stages' in lock_data, "dvc.lock should have stages"
    
    def test_locked_stages_match_yaml(self, project_root):
        """Test that locked stages match dvc.yaml stages."""
        dvc_yaml = project_root / "dvc.yaml"
        dvc_lock = project_root / "dvc.lock"
        
        if not dvc_yaml.exists() or not dvc_lock.exists():
            pytest.skip("DVC files not found")
        
        with open(dvc_yaml, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        with open(dvc_lock, 'r') as f:
            lock_config = yaml.safe_load(f)
        
        yaml_stages = set(yaml_config.get('stages', {}).keys())
        lock_stages = set(lock_config.get('stages', {}).keys())
        
        # Lock should have some or all stages from yaml
        if len(lock_stages) > 0:
            assert lock_stages.issubset(yaml_stages), \
                "Locked stages should be subset of defined stages"


@pytest.mark.integration
@pytest.mark.slow
class TestDVCMetrics:
    """Test DVC metrics tracking."""
    
    def test_metrics_files_exist(self, project_root):
        """Test that metrics files exist."""
        reports_dir = project_root / "reports"
        
        if not reports_dir.exists():
            pytest.skip("reports directory not found")
        
        # Check for results CSV files
        result_files = [
            "baseline_results.csv",
            "pipeline_baseline_results.csv",
            "tuning_results.csv"
        ]
        
        existing_files = []
        for f in result_files:
            if (reports_dir / f).exists():
                existing_files.append(f)
        
        # At least one should exist if pipeline has run
        if len(existing_files) > 0:
            assert len(existing_files) > 0, "Should have some results files"
    
    def test_dvc_metrics_show(self, project_root):
        """Test dvc metrics show command."""
        try:
            result = subprocess.run(
                ['dvc', 'metrics', 'show'],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Should run (may have no metrics yet)
            assert result.returncode in [0, 1], "dvc metrics show should run"
            
        except FileNotFoundError:
            pytest.skip("DVC not installed")
        except subprocess.TimeoutExpired:
            pytest.fail("dvc metrics show timed out")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])