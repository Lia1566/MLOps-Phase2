"""
DVC Management Module

This module provides functions for versioning data, models, and managing
DVC pipelines for reproducibility in this ML project.

It handles:
- Data versioning (dvc add, dvc push, dvc pull)
- Model versioning
- Pipeline stage management
- Remote storage configuration
- Metrics tracking
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import yaml

logger = logging.getLogger(__name__)


class DVCManager:
    """Manager class for DVC operations."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize DVC Manager.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.dvc_dir = self.project_root / '.dvc'
        self.dvc_yaml = self.project_root / 'dvc.yaml'
        self.params_yaml = self.project_root / 'params.yaml'
        
    def is_initialized(self) -> bool:
        """
        Check if DVC is initialized in the project.
        
        Returns:
            True if DVC is initialized, False otherwise
        """
        return self.dvc_dir.exists()
    
    def initialize(self) -> bool:
        """
        Initialize DVC in the project.
        
        Returns:
            True if successful, False otherwise
        """
        if self.is_initialized():
            logger.info("DVC is already initialized")
            return True
        
        try:
            result = subprocess.run(
                ['dvc', 'init'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("DVC initialized successfully")
            logger.debug(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to initialize DVC: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("DVC is not installed. Please install with: pip install dvc")
            return False
    
    def add_remote(self, remote_name: str, remote_url: str) -> bool:
        """
        Add a DVC remote storage.
        
        Args:
            remote_name: Name for the remote (e.g., 'myremote', 'storage')
            remote_url: URL of remote storage (e.g., '/path/to/storage', 's3://bucket/path')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add remote
            subprocess.run(
                ['dvc', 'remote', 'add', '-d', remote_name, remote_url],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Remote '{remote_name}' added successfully: {remote_url}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to add remote: {e.stderr}")
            return False
    
    def list_remotes(self) -> List[str]:
        """
        List all configured DVC remotes.
        
        Returns:
            List of remote names
        """
        try:
            result = subprocess.run(
                ['dvc', 'remote', 'list'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            remotes = [line.split()[0] for line in result.stdout.strip().split('\n') if line]
            return remotes
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to list remotes: {e.stderr}")
            return []
    
    def track_file(self, file_path: Path, commit: bool = True) -> bool:
        """
        Track a file with DVC.
        
        Args:
            file_path: Path to file to track
            commit: Whether to git commit the .dvc file
            
        Returns:
            True if successful, False otherwise
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        try:
            # Add file to DVC
            result = subprocess.run(
                ['dvc', 'add', str(file_path)],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"File tracked with DVC: {file_path}")
            logger.debug(result.stdout)
            
            # Git add the .dvc file
            dvc_file = file_path.with_suffix(file_path.suffix + '.dvc')
            if commit and dvc_file.exists():
                subprocess.run(
                    ['git', 'add', str(dvc_file), '.gitignore'],
                    cwd=self.project_root,
                    capture_output=True,
                    check=False  # Don't fail if git is not initialized
                )
                logger.info(f"DVC file added to git: {dvc_file}")
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to track file: {e.stderr}")
            return False
    
    def track_directory(self, dir_path: Path, commit: bool = True) -> bool:
        """
        Track a directory with DVC.
        
        Args:
            dir_path: Path to directory to track
            commit: Whether to git commit the .dvc file
            
        Returns:
            True if successful, False otherwise
        """
        return self.track_file(dir_path, commit)
    
    def push(self, remote: Optional[str] = None) -> bool:
        """
        Push tracked files to DVC remote.
        
        Args:
            remote: Remote name (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = ['dvc', 'push']
            if remote:
                cmd.extend(['-r', remote])
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Files pushed to DVC remote successfully")
            logger.debug(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to push to remote: {e.stderr}")
            return False
    
    def pull(self, remote: Optional[str] = None) -> bool:
        """
        Pull tracked files from DVC remote.
        
        Args:
            remote: Remote name (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = ['dvc', 'pull']
            if remote:
                cmd.extend(['-r', remote])
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Files pulled from DVC remote successfully")
            logger.debug(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull from remote: {e.stderr}")
            return False
    
    def track_model(self, model_path: Path, version: Optional[str] = None) -> bool:
        """
        Track a trained model with DVC.
        
        Args:
            model_path: Path to model file
            version: Optional version identifier
            
        Returns:
            True if successful, False otherwise
        """
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        
        success = self.track_file(model_path, commit=True)
        
        if success and version:
            # Create a git tag for the model version
            try:
                subprocess.run(
                    ['git', 'tag', f'model-{version}'],
                    cwd=self.project_root,
                    capture_output=True,
                    check=False
                )
                logger.info(f"Model tagged with version: {version}")
            except Exception as e:
                logger.warning(f"Could not create git tag: {e}")
        
        return success
    
    def track_metrics(self, metrics_file: Path) -> bool:
        """
        Track a metrics file with DVC (without caching).
        
        Args:
            metrics_file: Path to metrics file (CSV, JSON, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        if not metrics_file.exists():
            logger.error(f"Metrics file not found: {metrics_file}")
            return False
        
        try:
            # Add to DVC pipeline as metrics (no cache)
            result = subprocess.run(
                ['dvc', 'metrics', 'add', str(metrics_file)],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Metrics tracked: {metrics_file}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to track metrics: {e.stderr}")
            return False
    
    def run_pipeline(self, pipeline_name: Optional[str] = None) -> bool:
        """
        Run DVC pipeline.
        
        Args:
            pipeline_name: Optional specific pipeline/stage to run
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = ['dvc', 'repro']
            if pipeline_name:
                cmd.append(pipeline_name)
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("DVC pipeline executed successfully")
            logger.info(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Pipeline execution failed: {e.stderr}")
            return False
    
    def show_pipeline_dag(self) -> bool:
        """
        Display the pipeline DAG.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            result = subprocess.run(
                ['dvc', 'dag'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            print("\nDVC Pipeline DAG:")
            print("=" * 80)
            print(result.stdout)
            print("=" * 80)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to show DAG: {e.stderr}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all tracked metrics.
        
        Returns:
            Dictionary of metrics
        """
        try:
            result = subprocess.run(
                ['dvc', 'metrics', 'show'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            # Parse the output (format depends on DVC version)
            metrics = {}
            for line in result.stdout.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    metrics[key.strip()] = value.strip()
            return metrics
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get metrics: {e.stderr}")
            return {}
    
    def compare_experiments(self, experiments: Optional[List[str]] = None) -> bool:
        """
        Compare experiments/runs.
        
        Args:
            experiments: List of experiment names/commits to compare
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = ['dvc', 'metrics', 'diff']
            if experiments:
                cmd.extend(experiments)
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            print("\nExperiment Comparison:")
            print("=" * 80)
            print(result.stdout)
            print("=" * 80)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to compare experiments: {e.stderr}")
            return False
    
    def update_params(self, params: Dict[str, Any]) -> bool:
        """
        Update parameters in params.yaml.
        
        Args:
            params: Dictionary of parameters to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load existing params if file exists
            existing_params = {}
            if self.params_yaml.exists():
                with open(self.params_yaml, 'r') as f:
                    existing_params = yaml.safe_load(f) or {}
            
            # Update with new params
            existing_params.update(params)
            
            # Save updated params
            with open(self.params_yaml, 'w') as f:
                yaml.dump(existing_params, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Parameters updated in {self.params_yaml}")
            return True
        except Exception as e:
            logger.error(f"Failed to update params: {e}")
            return False
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get current parameters from params.yaml.
        
        Returns:
            Dictionary of parameters
        """
        try:
            if not self.params_yaml.exists():
                logger.warning(f"Params file not found: {self.params_yaml}")
                return {}
            
            with open(self.params_yaml, 'r') as f:
                params = yaml.safe_load(f) or {}
            
            return params
        except Exception as e:
            logger.error(f"Failed to load params: {e}")
            return {}
    
    def status(self) -> bool:
        """
        Show DVC status (tracked files, changes, etc.).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            result = subprocess.run(
                ['dvc', 'status'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            print("\nDVC Status:")
            print("=" * 80)
            print(result.stdout if result.stdout else "Everything is up to date")
            print("=" * 80)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get status: {e.stderr}")
            return False
    
    def checkout(self, target: Optional[str] = None) -> bool:
        """
        Checkout DVC tracked files.
        
        Args:
            target: Optional specific file or directory to checkout
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = ['dvc', 'checkout']
            if target:
                cmd.append(str(target))
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("DVC checkout completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Checkout failed: {e.stderr}")
            return False


def initialize_dvc_project(project_root: Path, remote_url: Optional[str] = None) -> DVCManager:
    """
    Initialize DVC in a project with optional remote.
    
    Args:
        project_root: Path to project root
        remote_url: Optional URL for remote storage
        
    Returns:
        DVCManager instance
    """
    dvc = DVCManager(project_root)
    
    if not dvc.is_initialized():
        print("Initializing DVC...")
        dvc.initialize()
    else:
        print("DVC already initialized")
    
    if remote_url:
        print(f"Adding remote storage: {remote_url}")
        dvc.add_remote('storage', remote_url)
    
    return dvc


def track_data_and_models(
    dvc: DVCManager,
    data_files: Optional[List[Path]] = None,
    model_files: Optional[List[Path]] = None
) -> None:
    """
    Track data files and models with DVC.
    
    Args:
        dvc: DVCManager instance
        data_files: List of data files to track
        model_files: List of model files to track
    """
    if data_files:
        print("\nTracking data files...")
        for data_file in data_files:
            if data_file.exists():
                dvc.track_file(data_file)
                print(f"  ✓ {data_file}")
    
    if model_files:
        print("\nTracking model files...")
        for model_file in model_files:
            if model_file.exists():
                dvc.track_model(model_file)
                print(f"  ✓ {model_file}")


def create_dvcignore(project_root: Path) -> None:
    """
    Create a .dvcignore file with common patterns.
    
    Args:
        project_root: Path to project root
    """
    dvcignore_path = project_root / '.dvcignore'
    
    dvcignore_content = """# Add patterns of files dvc should ignore, which could improve
# the performance. Learn more at
# https://dvc.org/doc/user-guide/dvcignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Logs
*.log
logs/

# MLflow
mlruns/
mlartifacts/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Temporary files
*.tmp
*.temp
.cache/
"""
    
    with open(dvcignore_path, 'w') as f:
        f.write(dvcignore_content)
    
    print(f"✓ Created .dvcignore at {dvcignore_path}")


# Convenience functions for common operations
def quick_track_and_push(
    project_root: Path,
    file_path: Path,
    message: str = "Track file with DVC"
) -> bool:
    """
    Quick function to track a file and push to remote.
    
    Args:
        project_root: Project root path
        file_path: File to track
        message: Git commit message
        
    Returns:
        True if successful, False otherwise
    """
    dvc = DVCManager(project_root)
    
    # Track file
    if not dvc.track_file(file_path):
        return False
    
    # Push to remote
    if not dvc.push():
        logger.warning("Could not push to remote (may not be configured)")
    
    return True


if __name__ == '__main__':
    # Example usage
    print("DVC Manager Module")
    print("=" * 80)
    print("\nThis module provides comprehensive DVC management functionality.")
    print("\nExample usage:")
    print("""
from src.utils.dvc_manager import DVCManager, initialize_dvc_project

# Initialize DVC in your project
dvc = initialize_dvc_project(
    project_root=Path('.'),
    remote_url='/path/to/remote/storage'
)

# Track data files
dvc.track_file(Path('data/processed/train.csv'))
dvc.track_directory(Path('data/processed'))

# Track models
dvc.track_model(Path('models/best_model.pkl'), version='1.0')

# Push to remote
dvc.push()

# Run pipeline
dvc.run_pipeline()

# Show metrics
metrics = dvc.get_metrics()
print(metrics)
    """)


























