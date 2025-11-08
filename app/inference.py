"""
Model Inference Logic
Handles model loading and predictions
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

from app.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInference:
    """Class to handle model loading and inference."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the inference class.
        
        Args:
            model_path: Path to the model file. If None, uses default from config.
        """
        self.model_path = model_path or config.get_model_path()
        self.model = None
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load the trained model from disk."""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            self.model = joblib.load(self.model_path)
            self.model_loaded = True
            logger.info("Model loaded successfully!")
            
            # Log model info
            model_type = type(self.model).__name__
            logger.info(f"Model type: {model_type}")
            
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            self.model_loaded = False
            raise
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model_loaded = False
            raise
    
    def preprocess_features(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess input features for prediction.
        
        Args:
            features: Dictionary of input features
            
        Returns:
            DataFrame with preprocessed features
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Encode categorical features (one-hot encoding)
        # Gender
        df['Gender_Male'] = (df['Gender'] == 'Male').astype(int)
        
        # Caste (one-hot encode)
        df['Caste_General'] = (df['Caste'] == 'General').astype(int)
        df['Caste_OBC'] = (df['Caste'] == 'OBC').astype(int)
        df['Caste_SC'] = (df['Caste'] == 'SC').astype(int)
        # ST is reference category (all zeros)
        
        # Coaching
        df['Coaching_Yes'] = (df['Coaching'] == 'Yes').astype(int)
        
        # Medium
        df['Medium_English'] = (df['Medium'] == 'English').astype(int)
        
        # Drop original categorical columns
        df = df.drop(['Gender', 'Caste', 'Coaching', 'Medium'], axis=1)
        
        # Ensure correct column order (match training data)
        expected_cols = [
            'Class_X_Percentage',
            'Class_XII_Percentage',
            'Study_Hours',
            'Gender_Male',
            'Caste_General',
            'Caste_OBC',
            'Caste_SC',
            'Coaching_Yes',
            'Medium_English'
        ]
        
        # Reorder columns
        df = df[expected_cols]
        
        logger.info(f"Preprocessed features shape: {df.shape}")
        logger.info(f"Features: {df.columns.tolist()}")
        
        return df
    
    def predict(self, features: Dict[str, Any]) -> Tuple[int, np.ndarray]:
        """
        Make a prediction.
        
        Args:
            features: Dictionary of input features
            
        Returns:
            Tuple of (prediction, probabilities)
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Cannot make predictions.")
        
        try:
            # Preprocess features
            X = self.preprocess_features(features)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            logger.info(f"Prediction: {prediction}, Probabilities: {probabilities}")
            
            return int(prediction), probabilities
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.model_loaded:
            return {
                "model_loaded": False,
                "error": "Model not loaded"
            }
        
        model_type = type(self.model).__name__
        
        # Try to get feature names (if pipeline)
        features = config.EXPECTED_FEATURES
        if hasattr(self.model, 'feature_names_in_'):
            features = self.model.feature_names_in_.tolist()
        
        # Try to get classes
        classes = config.LABELS
        if hasattr(self.model, 'classes_'):
            classes = {int(i): config.LABELS.get(i, f"Class {i}") 
                      for i in self.model.classes_}
        
        return {
            "model_loaded": True,
            "model_type": model_type,
            "model_path": str(self.model_path),
            "features": features,
            "target_classes": classes,
            "n_features": len(features)
        }


# Global model instance (singleton pattern)
_model_instance: Optional[ModelInference] = None


def get_model() -> ModelInference:
    """
    Get or create the global model instance.
    
    Returns:
        ModelInference instance
    """
    global _model_instance
    
    if _model_instance is None:
        logger.info("Creating new model instance...")
        _model_instance = ModelInference()
    
    return _model_instance


def reload_model(model_path: Optional[Path] = None):
    """
    Reload the model (useful for model updates).
    
    Args:
        model_path: Path to new model file
    """
    global _model_instance
    
    logger.info("Reloading model...")
    _model_instance = ModelInference(model_path)
    logger.info("Model reloaded successfully!")