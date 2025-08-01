# Test Fixtures

This directory contains test data and fixtures used by the test suite.

## Directory Structure

```
fixtures/
├── models/           # Sample model files for testing
│   ├── simple.onnx   # Minimal ONNX model
│   ├── resnet50.onnx # Computer vision model
│   └── bert.onnx     # NLP model
├── configs/          # Sample configuration files
│   ├── basic.yaml    # Basic service configuration
│   ├── gpu.yaml      # GPU-optimized configuration
│   └── ensemble.yaml # Multi-model configuration
├── data/             # Test input/output data
│   ├── images/       # Sample images for CV models
│   ├── text/         # Sample text for NLP models
│   └── expected/     # Expected outputs for validation
└── k8s/              # Kubernetes manifests for testing
    ├── namespace.yaml
    ├── service.yaml
    └── deployment.yaml
```

## Usage

These fixtures are automatically discovered by pytest and used in tests:

```python
def test_model_loading(fixture_model_path):
    # Test uses fixtures/models/simple.onnx
    pass

def test_configuration(fixture_config):
    # Test uses fixtures/configs/basic.yaml
    pass
```

## Adding New Fixtures

1. Add files to appropriate subdirectory
2. Update fixture functions in `conftest.py` if needed
3. Document the fixture purpose and usage
4. Keep fixtures minimal and focused on specific test scenarios

## Model Fixtures

All model fixtures are created programmatically to avoid large binary files in the repository. See `tests/conftest.py` for model generation code.

## Security

- No real API keys or credentials in fixtures
- All model files are synthetic/minimal
- Test data should not contain sensitive information