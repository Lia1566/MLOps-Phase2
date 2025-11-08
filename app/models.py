"""
Pydantic Models for Request/Response Validation
Pydantic v2 Compatible
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime


class StudentFeatures(BaseModel):
    """Input features for student performance prediction."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Class_X_Percentage": 85.5,
                "Class_XII_Percentage": 78.0,
                "Study_Hours": 5.0,
                "Gender": "Male",
                "Caste": "General",
                "Coaching": "Yes",
                "Medium": "English"
            }
        }
    )
    
    Class_X_Percentage: float = Field(
        ..., 
        ge=0, 
        le=100,
        description="Percentage in Class X (10th grade)",
        examples=[85.5]
    )
    Class_XII_Percentage: float = Field(
        ..., 
        ge=0, 
        le=100,
        description="Percentage in Class XII (12th grade)",
        examples=[78.0]
    )
    Study_Hours: float = Field(
        ..., 
        ge=0, 
        le=24,
        description="Daily study hours",
        examples=[5.0]
    )
    Gender: str = Field(
        ...,
        description="Student gender (Male/Female)",
        examples=["Male"]
    )
    Caste: str = Field(
        ...,
        description="Caste category (General/OBC/SC/ST)",
        examples=["General"]
    )
    Coaching: str = Field(
        ...,
        description="Coaching attendance (Yes/No)",
        examples=["Yes"]
    )
    Medium: str = Field(
        ...,
        description="Medium of instruction (English/Hindi)",
        examples=["English"]
    )
    
    @field_validator('Gender')
    @classmethod
    def validate_gender(cls, v):
        """Validate gender field."""
        allowed = ['Male', 'Female']
        if v not in allowed:
            raise ValueError(f"Gender must be one of {allowed}")
        return v
    
    @field_validator('Caste')
    @classmethod
    def validate_caste(cls, v):
        """Validate caste field."""
        allowed = ['General', 'OBC', 'SC', 'ST']
        if v not in allowed:
            raise ValueError(f"Caste must be one of {allowed}")
        return v
    
    @field_validator('Coaching')
    @classmethod
    def validate_coaching(cls, v):
        """Validate coaching field."""
        allowed = ['Yes', 'No']
        if v not in allowed:
            raise ValueError(f"Coaching must be one of {allowed}")
        return v
    
    @field_validator('Medium')
    @classmethod
    def validate_medium(cls, v):
        """Validate medium field."""
        allowed = ['English', 'Hindi']
        if v not in allowed:
            raise ValueError(f"Medium must be one of {allowed}")
        return v


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": {
                    "Class_X_Percentage": 85.5,
                    "Class_XII_Percentage": 78.0,
                    "Study_Hours": 5.0,
                    "Gender": "Male",
                    "Caste": "General",
                    "Coaching": "Yes",
                    "Medium": "English"
                }
            }
        }
    )
    
    features: StudentFeatures = Field(
        ...,
        description="Student features for prediction"
    )


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 1,
                "prediction_label": "High Performance",
                "probability": 0.85,
                "probabilities": {
                    "Low Performance": 0.15,
                    "High Performance": 0.85
                },
                "timestamp": "2025-01-07T16:30:00"
            }
        }
    )
    
    prediction: int = Field(
        ...,
        description="Predicted class (0: Low Performance, 1: High Performance)"
    )
    prediction_label: str = Field(
        ...,
        description="Human-readable prediction label"
    )
    probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of positive class (High Performance)"
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probabilities for all classes"
    )
    timestamp: str = Field(
        ...,
        description="Prediction timestamp"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "timestamp": "2025-01-07T16:30:00",
                "model_loaded": True,
                "version": "1.0.0"
            }
        }
    )
    
    status: str = Field(
        ...,
        description="API health status"
    )
    timestamp: str = Field(
        ...,
        description="Health check timestamp"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether model is loaded successfully"
    )
    version: str = Field(
        ...,
        description="API version"
    )


class ModelInfoResponse(BaseModel):
    """Response model for model info endpoint."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_name": "pipeline_baseline.pkl",
                "model_type": "Pipeline",
                "version": "1.0.0",
                "features": [
                    "Class_X_Percentage",
                    "Class_XII_Percentage",
                    "Study_Hours",
                    "Gender_Male",
                    "Caste_General"
                ],
                "target_classes": {
                    0: "Low Performance",
                    1: "High Performance"
                },
                "model_path": "/path/to/models/pipeline_baseline.pkl"
            }
        }
    )
    
    model_name: str = Field(
        ...,
        description="Name of the loaded model"
    )
    model_type: str = Field(
        ...,
        description="Type of model (e.g., LogisticRegression, Pipeline)"
    )
    version: str = Field(
        ...,
        description="Model version"
    )
    features: List[str] = Field(
        ...,
        description="List of features expected by the model"
    )
    target_classes: Dict[int, str] = Field(
        ...,
        description="Mapping of class indices to labels"
    )
    model_path: str = Field(
        ...,
        description="Path to the model file"
    )


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Prediction failed",
                "detail": "Invalid input format",
                "timestamp": "2025-01-07T16:30:00"
            }
        }
    )
    
    error: str = Field(
        ...,
        description="Error message"
    )
    detail: Optional[str] = Field(
        None,
        description="Detailed error information"
    )
    timestamp: str = Field(
        ...,
        description="Error timestamp"
    )
    
# DRIFT DETECTION MODELS

class DriftDetectionRequest(BaseModel):
    """Request model for drift detection."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "current_data": [
                    {
                        "Class_X_Percentage": 85.5,
                        "Class_XII_Percentage": 78.0,
                        "Study_Hours": 5.0,
                        "Gender_Male": 1,
                        "Caste_General": 1,
                        "Caste_OBC": 0,
                        "Caste_SC": 0,
                        "Coaching_Yes": 1,
                        "Medium_English": 1
                    }
                ]
            }
        }
    )
    
    current_data: List[Dict[str, Any]] = Field(
        ...,
        description="Current data samples to check for drift"
    )


class DriftDetectionResponse(BaseModel):
    """Response model for drift detection."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "drift_detected": True,
                "drift_share": 0.35,
                "drifted_columns_count": 3,
                "drift_score": 0.35,
                "threshold": 0.1,
                "timestamp": "2025-01-07T16:30:00",
                "reference_size": 1000,
                "message": "Drift detected in 3 columns"
            }
        }
    )
    
    drift_detected: bool = Field(
        ...,
        description="Whether drift was detected"
    )
    drift_share: Optional[float] = Field(
        None,
        description="Share of drifted features (0-1)"
    )
    drifted_columns_count: Optional[int] = Field(
        None,
        description="Number of columns with drift"
    )
    drift_score: Optional[float] = Field(
        None,
        description="Overall drift score"
    )
    threshold: float = Field(
        ...,
        description="Drift detection threshold"
    )
    timestamp: str = Field(
        ...,
        description="Detection timestamp"
    )
    reference_size: int = Field(
        ...,
        description="Size of reference dataset"
    )
    message: Optional[str] = Field(
        None,
        description="Additional information"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if detection failed"
    )


class MonitoringStatsResponse(BaseModel):
    """Response model for monitoring statistics."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "uptime_seconds": 3600,
                "total_predictions": 150,
                "model_loaded": True,
                "drift_monitoring_enabled": True,
                "last_drift_check": "2025-01-07T16:30:00",
                "drift_status": "no_drift",
                "timestamp": "2025-01-07T16:30:00"
            }
        }
    )
    
    uptime_seconds: float = Field(
        ...,
        description="API uptime in seconds"
    )
    total_predictions: int = Field(
        ...,
        description="Total number of predictions made"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether model is loaded"
    )
    drift_monitoring_enabled: bool = Field(
        ...,
        description="Whether drift monitoring is enabled"
    )
    last_drift_check: Optional[str] = Field(
        None,
        description="Timestamp of last drift check"
    )
    drift_status: str = Field(
        ...,
        description="Current drift status (no_drift/drift_detected/unknown)"
    )
    timestamp: str = Field(
        ...,
        description="Current timestamp"
    )