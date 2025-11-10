"""Debug test to check model loading"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print(f"Current directory: {Path.cwd()}")
print(f"Project root: {PROJECT_ROOT}")

# Check if model exists
model_path = Path("models/pipeline_baseline.pkl")
print(f"Model path: {model_path}")
print(f"Model exists: {model_path.exists()}")
print(f"Model absolute path: {model_path.absolute()}")

# Try to import and create app
try:
    from app.main import app
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    response = client.get("/health")
    print(f"Health response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
