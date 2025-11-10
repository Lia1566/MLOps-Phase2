"""Debug model loading directly"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 60)
print("Testing Model Loading")
print("=" * 60)

# Test 1: Direct model loading
print("\n1. Testing direct model load...")
try:
    from app.inference import ModelInference
    
    model_path = Path("models/pipeline_baseline.pkl")
    print(f"   Model path: {model_path}")
    print(f"   Exists: {model_path.exists()}")
    
    print("   Creating ModelInference...")
    inference = ModelInference(model_path)
    print(f"   ✓ Model loaded: {inference.model_loaded}")
    print(f"   ✓ Model type: {type(inference.model).__name__}")
    
    # Try a prediction
    print("\n2. Testing prediction...")
    test_features = {
        "Class_X_Percentage": 85.5,
        "Class_XII_Percentage": 78.0,
        "Study_Hours": 5.0,
        "Gender": "Male",
        "Caste": "General",
        "Coaching": "Yes",
        "Medium": "English"
    }
    
    pred, probs = inference.predict(test_features)
    print(f"   ✓ Prediction: {pred}")
    print(f"   ✓ Probabilities: {probs}")
    
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 2: App startup
print("\n3. Testing FastAPI app startup...")
try:
    from app.main import app
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    response = client.get("/health")
    print(f"   Health: {response.json()}")
    
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
