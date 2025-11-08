""" 
FastAPI Main Application
Student Performance Prediction API
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import logging
from typing import Dict, Any
import time

from app.config import config
from app.models import (
    StudentFeatures, PredictionResponse, HealthResponse, ModelInfoResponse, ErrorResponse, 
    DriftDetectionRequest, DriftDetectionResponse, MonitoringStatsResponse
)
from app.inference import get_model, ModelInference
from app.drift_detection import get_drift_detector

# Monitoring variables
startup_time = None
prediction_count = 0
last_drift_check = None
current_drift_status = "unknown"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model: ModelInference = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model, startup_time
    try:
        logger.info("Starting up API...")
        model = get_model()
        logger.info("Model loaded successfully on startup!")
        
        # Initialize drift detector
        try:
            drift_detector = get_drift_detector()
            logger.info("Drift detector initialized")
        except Exception as e:
            logger.warning(f"Drift detector initialization failed: {e}")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")
        logger.warning("API will start but predictions will fail until model is loaded")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API...")


@app.get("/", tags=["General"])
async def root():
    """
    Welcome endpoint.
    
    Returns basic information about the API.
    """
    return {
        "message": "Welcome to Student Performance Prediction API",
        "version": config.API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "/predict",
            "model_info": "/model-info",
            "health": "/health"
        }
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["General"],
    summary="Health Check",
    description="Check if the API is running and model is loaded"
)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with API status and model loading status
    """
    try:
        model_loaded = model is not None and model.model_loaded
        
        return HealthResponse(
            status="healthy" if model_loaded else "degraded",
            timestamp=datetime.now().isoformat(),
            model_loaded=model_loaded,
            version=config.API_VERSION
        )
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            model_loaded=False,
            version=config.API_VERSION
        )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Make Prediction",
    description="Predict student performance based on input features",
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid input"},
        500: {"description": "Prediction error"}
    }
)
async def predict(features: StudentFeatures):
    """
    Make a prediction for student performance.
    
    Args:
        features: StudentFeatures with all required input features
        
    Returns:
        PredictionResponse with prediction and probabilities
        
    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    try:
        # Check if model is loaded
        if model is None or not model.model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please try again later."
            )
        
        # Convert features to dict
        features_dict = features.dict()
        
        logger.info(f"Received prediction request: {features_dict}")
        
        # Make prediction
        prediction, probabilities = model.predict(features_dict)
        
        # Get prediction label
        prediction_label = config.LABELS.get(prediction, f"Class {prediction}")
        
        # Format probabilities
        prob_dict = {
            config.LABELS.get(0, "Class 0"): float(probabilities[0]),
            config.LABELS.get(1, "Class 1"): float(probabilities[1])
        }
        
        # Get probability for positive class (High Performance)
        probability = float(probabilities[1])
        
        global prediction_count
        prediction_count += 1
        
        logger.info(f"Prediction: {prediction} ({prediction_label}), Probability: {probability:.4f}")
        
        return PredictionResponse(
            prediction=prediction,
            prediction_label=prediction_label,
            probability=probability,
            probabilities=prob_dict,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Model Information",
    description="Get information about the loaded model"
)
async def model_info():
    """
    Get information about the loaded model.
    
    Returns:
        ModelInfoResponse with model metadata
        
    Raises:
        HTTPException: If model not loaded
    """
    try:
        if model is None or not model.model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        info = model.get_model_info()
        
        return ModelInfoResponse(
            model_name=config.MODEL_FILENAME,
            model_type=info.get("model_type", "Unknown"),
            version=config.API_VERSION,
            features=info.get("features", []),
            target_classes=info.get("target_classes", {}),
            model_path=info.get("model_path", "")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# DRIFT DETECTION ENDPOINT
@app.post(
    "/detect-drift",
    response_model=DriftDetectionResponse,
    tags=["Monitoring"],
    summary="Detect Data Drift",
    description="Detect drift between current data and reference data"
)
async def detect_drift(request: DriftDetectionRequest):
    """
    Detect data drift in current data compared to reference data.
    
    Args:
        request: DriftDetectionRequest with current data samples
        
    Returns:
        DriftDetectionResponse with drift metrics
    """
    global last_drift_check, current_drift_status
    
    try:
        # Get drift detector
        drift_detector = get_drift_detector()
        
        # Convert request data to DataFrame
        import pandas as pd
        current_df = pd.DataFrame(request.current_data)
        
        logger.info(f"Detecting drift for {len(current_df)} samples")
        
        # Detect drift
        drift_results = drift_detector.detect_drift(current_df)
        
        # Update monitoring variables
        last_drift_check = datetime.now().isoformat()
        current_drift_status = "drift_detected" if drift_results.get("drift_detected") else "no_drift"
        
        # Add message
        if drift_results.get("drift_detected"):
            drift_results["message"] = f"Drift detected in {drift_results.get('drifted_columns_count', 0)} columns"
        else:
            drift_results["message"] = "No significant drift detected"
        
        return DriftDetectionResponse(**drift_results)
        
    except Exception as e:
        logger.error(f"Drift detection error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drift detection failed: {str(e)}"
        )


@app.get(
    "/monitoring/stats",
    response_model=MonitoringStatsResponse,
    tags=["Monitoring"],
    summary="Get Monitoring Statistics",
    description="Get API monitoring statistics including drift status"
)
async def get_monitoring_stats():
    """
    Get monitoring statistics.
    
    Returns:
        MonitoringStatsResponse with monitoring metrics
    """
    global startup_time, prediction_count, last_drift_check, current_drift_status
    
    uptime = time.time() - startup_time if startup_time else 0
    
    drift_detector = get_drift_detector()
    drift_enabled = drift_detector.reference_data is not None
    
    return MonitoringStatsResponse(
        uptime_seconds=uptime,
        total_predictions=prediction_count,
        model_loaded=model is not None and model.model_loaded,
        drift_monitoring_enabled=drift_enabled,
        last_drift_check=last_drift_check,
        drift_status=current_drift_status,
        timestamp=datetime.now().isoformat()
    )


# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )