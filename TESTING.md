# Testing Documentation

## Overview

This document describes the testing infrastructure for the MLOps Phase 3 project. We implement comprehensive unit and integration tests to ensure code quality, reliability, and reproducibility.

## Testing Strategy

### **Test Coverage Goals**
- **Unit Tests**: >80% code coverage
- **Integration Tests**: All critical workflows
- **API Tests**: All endpoints
- **Performance Tests**: Key operations

### **Testing Pyramid**
```
         /\
        /  \  Integration Tests (20%)
       /____\
      /      \
     /        \ Unit Tests (70%)
    /__________\
   /            \
  /              \ Manual Tests (10%)
 /________________\
```

---

## Test Structure
```
tests/
├── conftest.py                    # Shared fixtures
├── pytest.ini                     # Pytest configuration
│
├── unit/                          # Unit Tests (70% of tests)
│   ├── test_preprocessing.py      # Data preprocessing
│   ├── test_feature_engineering.py # Feature engineering
│   ├── test_model_inference.py    # Model predictions
│   ├── test_metrics.py            # Metrics calculation
│   └── test_config.py             # Configuration
│
└── integration/                   # Integration Tests (30% of tests)
    ├── test_pipeline_e2e.py       # End-to-end pipeline
    ├── test_api.py                # FastAPI endpoints
    └── test_dvc_stages.py         # DVC pipeline
```

---

## Running Tests

### **Install Testing Dependencies**
```bash
# Install all testing dependencies
pip install -r requirements-dev.txt
```

### **Run All Tests**
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run quietly (less output)
pytest -q
```

### **Run Specific Test Categories**
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only API tests
pytest -m api

# Run only slow tests
pytest -m slow

# Run fast tests only (exclude slow)
pytest -m "not slow"
```

### **Run Specific Test Files**
```bash
# Run specific test file
pytest tests/unit/test_preprocessing.py

# Run specific test class
pytest tests/unit/test_preprocessing.py::TestDataPreprocessing

# Run specific test function
pytest tests/unit/test_preprocessing.py::TestDataPreprocessing::test_remove_duplicates
```

### **Run Tests with Coverage**
```bash
# Run tests with coverage report
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### **Run Tests in Parallel**
```bash
# Run tests in parallel (faster)
pytest -n auto

# Run with 4 workers
pytest -n 4
```

---

## Test Markers

We use pytest markers to organize tests:

| Marker | Description | Example |
|--------|-------------|---------|
| `@pytest.mark.unit` | Unit tests | `@pytest.mark.unit` |
| `@pytest.mark.integration` | Integration tests | `@pytest.mark.integration` |
| `@pytest.mark.api` | API endpoint tests | `@pytest.mark.api` |
| `@pytest.mark.slow` | Slow tests (>1 second) | `@pytest.mark.slow` |
| `@pytest.mark.skipif` | Conditional skip | `@pytest.mark.skipif(...)` |

### **Using Markers**
```python
@pytest.mark.unit
def test_example():
    """This is a unit test."""
    assert True

@pytest.mark.integration
@pytest.mark.slow
def test_pipeline():
    """This is a slow integration test."""
    # Long-running test
    pass
```

---

## Test Fixtures

### **Available Fixtures**

Located in `tests/conftest.py`:

#### **Path Fixtures**
- `project_root`: Project root directory
- `data_dir`: Data directory path
- `models_dir`: Models directory path
- `config_dir`: Config directory path

#### **Data Fixtures**
- `sample_raw_data`: Raw student performance data (100 samples)
- `sample_processed_data`: Preprocessed data with features
- `sample_train_test_split`: Train/test split (80/20)
- `sample_inference_data`: Single sample for inference

#### **Model Fixtures**
- `sample_model`: Trained LogisticRegression model
- `sample_pipeline`: Trained sklearn Pipeline
- `temp_model_path`: Temporary path for saving models

#### **Configuration Fixtures**
- `sample_config`: Sample configuration dictionary

#### **Metrics Fixtures**
- `sample_predictions`: Predictions and true labels
- `sample_reference_data`: Reference data for drift detection
- `sample_drifted_data`: Drifted data for drift detection

#### **API Fixtures**
- `sample_api_request`: API request payload

### **Using Fixtures**
```python
def test_model_prediction(sample_model, sample_train_test_split):
    """Test using fixtures."""
    _, X_test, _, y_test = sample_train_test_split
    predictions = sample_model.predict(X_test)
    assert len(predictions) == len(y_test)
```

---

## Coverage Reports

### **Generate Coverage Report**
```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html --cov-report=term

# Coverage reports are saved to htmlcov/
```

### **Coverage Requirements**

| Component | Target Coverage | Current |
|-----------|----------------|---------|
| Data preprocessing | >85% | TBD |
| Feature engineering | >80% | TBD |
| Model inference | >90% | TBD |
| Metrics calculation | >95% | TBD |
| Configuration | >75% | TBD |
| **Overall** | **>80%** | **TBD** |

---

## Debugging Tests

### **Run Tests with pdb**
```bash
# Drop into debugger on failure
pytest --pdb

# Drop into debugger on first failure
pytest -x --pdb
```

### **Print Output**
```bash
# Show print statements
pytest -s

# Show print statements with verbose
pytest -sv
```

### **Show Locals on Failure**
```bash
# Show local variables on failure
pytest -l

# Show full trace
pytest --tb=long
```

---

## Performance Testing

### **Test Execution Time**
```bash
# Show slowest 10 tests
pytest --durations=10

# Show all test durations
pytest --durations=0
```

### **Timeout Tests**
```bash
# Set timeout for all tests (5 seconds)
pytest --timeout=5
```

---

## Continuous Integration

### **CI/CD Integration**

Our tests are designed to run in CI/CD pipelines:
```yaml
# Example GitHub Actions workflow
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run tests
        run: pytest --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

---

## Writing New Tests

### **Test Naming Conventions**
```python
# Good test names
def test_preprocessing_removes_duplicates():
    pass

def test_model_prediction_on_valid_input():
    pass

def test_api_returns_200_on_health_check():
    pass

# Bad test names
def test_1():
    pass

def test_stuff():
    pass
```

### **Test Structure**

Follow the **Arrange-Act-Assert** pattern:
```python
def test_example():
    # Arrange: Set up test data
    data = [1, 2, 3, 4, 5]
    
    # Act: Perform the operation
    result = sum(data)
    
    # Assert: Verify the result
    assert result == 15
```

### **Test Organization**
```python
@pytest.mark.unit
class TestFeature:
    """Test suite for a feature."""
    
    def test_normal_case(self):
        """Test normal operation."""
        pass
    
    def test_edge_case(self):
        """Test edge case."""
        pass
    
    def test_error_handling(self):
        """Test error handling."""
        pass
```

---

## Best Practices

### **1. Test Independence**
- Each test should be independent
- Tests should not rely on execution order
- Clean up after tests (use fixtures)

### **2. Use Fixtures**
- Reuse common setup code
- Keep tests DRY (Don't Repeat Yourself)

### **3. Test One Thing**
- Each test should test one specific behavior
- Use multiple small tests instead of one large test

### **4. Descriptive Names**
- Test names should describe what they test
- Use `test_<function>_<scenario>_<expected>` format

### **5. Fast Tests**
- Keep tests fast (<1 second if possible)
- Mark slow tests with `@pytest.mark.slow`

### **6. Clear Assertions**
- Use clear assertion messages
- Prefer `assert x == y, "Expected x to equal y"`

### **7. Test Edge Cases**
- Empty inputs
- Missing values
- Invalid inputs
- Boundary conditions

---

## Common Issues

### **Issue: "Module not found"**
```bash
# Solution: Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest
```

### **Issue: "Fixture not found"**
```bash
# Solution: Check conftest.py is in the right location
# Fixtures should be in tests/conftest.py
```

### **Issue: "Tests taking too long"**
```bash
# Solution 1: Run in parallel
pytest -n auto

# Solution 2: Skip slow tests
pytest -m "not slow"
```

### **Issue: "Coverage not accurate"**
```bash
# Solution: Make sure to include all source files
pytest --cov=src --cov-report=term-missing
```

---

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

---
## Test Checklist

Before committing code, ensure:

- All tests pass: `pytest`
- Coverage is >80%: `pytest --cov=src`
- No linting errors: `flake8 src/`
- Code is formatted: `black src/`
- New tests added for new features
- Tests are documented