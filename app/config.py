"""
Configuration for FastAPI Application
"""

from pathlib import Path
from typing import Optional
import yaml


class APIConfig:
    """Configuration class for FastAPI application."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    MODEL_DIR = PROJECT_ROOT / "models"
    CONFIG_DIR = PROJECT_ROOT / "config"
    
    # API metadata
    API_TITLE = "Student Performance Prediction API"
    API_DESCRIPTION = """
    ## 🎓 Student Performance Prediction API
    
    This API predicts student academic performance based on various features.
    
    ### Features:
    * **Predict**: Get performance predictions for students
    * **Health Check**: Verify API is running
    * **Model Info**: Get information about the loaded model
    
    ### Model:
    - Trained on student performance dataset
    - Binary classification (High/Low performance)
    - Returns prediction and probability
    """
    API_VERSION = "1.0.0"
    
    # Model configuration
    MODEL_FILENAME = "pipeline_baseline.pkl"  # Default model
    MODEL_PATH: Optional[Path] = None
    
    # Features expected by the model
    EXPECTED_FEATURES = [
        "Class_X_Percentage",
        "Class_XII_Percentage", 
        "Study_Hours",
        "Gender_Male",
        "Caste_General",
        "Caste_OBC",
        "Caste_SC",
        "Coaching_Yes",
        "Medium_English"
    ]
    
    # Prediction labels
    LABELS = {
        0: "Low Performance",
        1: "High Performance"
    }
    
    @classmethod
    def load_config(cls) -> dict:
        """Load configuration from params.yaml if it exists."""
        config_file = cls.CONFIG_DIR / "config.yaml"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    @classmethod
    def get_model_path(cls, model_name: Optional[str] = None) -> Path:
        """Get the path to the model file."""
        if model_name is None:
            model_name = cls.MODEL_FILENAME
        
        model_path = cls.MODEL_DIR / model_name
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                f"Please train a model first or specify correct model path."
            )
        
        return model_path


# Create a global config instance
config = APIConfig()