"""
Integration Tests for FastAPI Application
Tests for API endpoints (will be implemented in Phase 3B)
"""

import pytest
from pathlib import Path
import sys

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import FastAPI dependencies
try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Try to import app 
try:
    from app.main import app
    APP_AVAILABLE = True
except ImportError:
    APP_AVAILABLE = False


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
@pytest.mark.skipif(not APP_AVAILABLE, reason="FastAPI app not created yet")
class TestFastAPIEndpoints:
    """Test suite for FastAPI endpoints."""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create test client."""
        if APP_AVAILABLE:
            return TestClient(app)
        return None
    
    def test_health_endpoint(self, client):
        """Test /health endpoint."""
        if client is None:
            pytest.skip("FastAPI app not available")
        
        response = client.get("/health")
        
        # Assertions
        assert response.status_code == 200, "Health check should return 200"
        assert response.json()["status"] == "healthy"
    
    def test_predict_endpoint(self, client, sample_api_request):
        """Test POST /predict endpoint."""
        if client is None:
            pytest.skip("FastAPI app not available")
        
        response = client.post("/predict", json=sample_api_request)
        
        # Assertions
        assert response.status_code == 200, "Predict should return 200"
        
        data = response.json()
        assert "prediction" in data, "Response should have prediction"
        assert "probability" in data, "Response should have probability"
        
        # Validate prediction
        assert data["prediction"] in [0, 1], "Prediction should be binary"
        assert 0 <= data["probability"] <= 1, "Probability should be [0,1]"
    
    def test_predict_validation_error(self, client):
        """Test prediction with invalid input."""
        if client is None:
            pytest.skip("FastAPI app not available")
        
        # Invalid request (missing fields)
        invalid_request = {"Gender": "Male"}
        
        response = client.post("/predict", json=invalid_request)
        
        # Should return 422 (Validation Error)
        assert response.status_code == 422, "Should return validation error"
    
    def test_model_info_endpoint(self, client):
        """Test /model-info endpoint."""
        if client is None:
            pytest.skip("FastAPI app not available")
        
        response = client.get("/model-info")
        
        # Assertions
        assert response.status_code == 200
        
        data = response.json()
        assert "model_name" in data
        assert "version" in data
        assert "features" in data


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
@pytest.mark.skipif(not APP_AVAILABLE, reason="FastAPI app not created yet")
class TestAPIPerformance:
    """Test API performance."""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create test client."""
        if APP_AVAILABLE:
            return TestClient(app)
        return None
    
    def test_prediction_latency(self, client, sample_api_request):
        """Test that predictions are fast enough."""
        if client is None:
            pytest.skip("FastAPI app not available")
        
        import time
        
        # Warm up
        client.post("/predict", json=sample_api_request)
        
        # Measure latency
        start = time.time()
        response = client.post("/predict", json=sample_api_request)
        latency = time.time() - start
        
        # Assertions
        assert response.status_code == 200
        assert latency < 0.5, f"Prediction should be <500ms, got {latency*1000:.0f}ms"
    
    def test_batch_predictions(self, client, sample_api_request):
        """Test multiple predictions in sequence."""
        if client is None:
            pytest.skip("FastAPI app not available")
        
        num_requests = 10
        successes = 0
        
        for _ in range(num_requests):
            response = client.post("/predict", json=sample_api_request)
            if response.status_code == 200:
                successes += 1
        
        # All requests should succeed
        assert successes == num_requests, "All batch predictions should succeed"


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
@pytest.mark.skipif(not APP_AVAILABLE, reason="FastAPI app not created yet")
class TestAPIErrorHandling:
    """Test API error handling."""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create test client."""
        if APP_AVAILABLE:
            return TestClient(app)
        return None
    
    def test_invalid_method(self, client):
        """Test using wrong HTTP method."""
        if client is None:
            pytest.skip("FastAPI app not available")
        
        # Try GET on /predict (should be POST)
        response = client.get("/predict")
        
        # Should return 405 (Method Not Allowed)
        assert response.status_code == 405
    
    def test_invalid_endpoint(self, client):
        """Test accessing non-existent endpoint."""
        if client is None:
            pytest.skip("FastAPI app not available")
        
        response = client.get("/nonexistent")
        
        # Should return 404 (Not Found)
        assert response.status_code == 404
    
    def test_malformed_json(self, client):
        """Test sending malformed JSON."""
        if client is None:
            pytest.skip("FastAPI app not available")
        
        response = client.post(
            "/predict",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )
        
        # Should return 422 (Validation Error)
        assert response.status_code == 422


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create test client."""
        if APP_AVAILABLE:
            return TestClient(app)
        return None
    
    def test_openapi_schema(self, client):
        """Test that OpenAPI schema is available."""
        if client is None:
            pytest.skip("FastAPI app not available")
        
        response = client.get("/openapi.json")
        
        # Assertions
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
    
    def test_docs_endpoint(self, client):
        """Test that /docs endpoint is available."""
        if client is None:
            pytest.skip("FastAPI app not available")
        
        response = client.get("/docs")
        
        # Should return HTML documentation
        assert response.status_code == 200
        assert "swagger" in response.text.lower() or "openapi" in response.text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "api"])