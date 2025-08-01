# Testing Guide

This document describes the testing strategy and practices for Nimify Anything.

## Testing Philosophy

We follow a comprehensive testing approach with multiple levels:

1. **Unit Tests**: Fast, isolated tests for individual components
2. **Integration Tests**: Test interactions between components
3. **End-to-End Tests**: Full workflow testing from CLI to deployment
4. **Performance Tests**: Load testing and benchmarking
5. **Contract Tests**: API contract validation

## Test Structure

```
tests/
├── unit/                 # Unit tests (fast, isolated)
├── integration/          # Integration tests (require services)
├── e2e/                  # End-to-end tests (full workflows)
├── performance/          # Performance and load tests
├── fixtures/             # Test data and utilities
└── conftest.py          # Shared fixtures and configuration
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
make test

# Run fast tests only (exclude slow/integration)
make test-fast

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m "not slow"        # Skip slow tests
```

### Advanced Test Options

```bash
# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_cli.py

# Run tests matching pattern
pytest -k "test_model_creation"

# Run tests with verbose output
pytest -v

# Run tests in parallel
pytest -n auto

# Run integration tests (requires setup)
pytest --run-integration --k8s-context=kind-test
```

## Test Categories and Markers

### Markers

- `@pytest.mark.unit`: Fast unit tests
- `@pytest.mark.integration`: Integration tests requiring services
- `@pytest.mark.slow`: Tests that take >1 second
- `@pytest.mark.gpu`: Tests requiring GPU hardware
- `@pytest.mark.docker`: Tests requiring Docker
- `@pytest.mark.performance`: Performance and load tests

### Example Usage

```python
import pytest
from nimify.core import Nimifier

@pytest.mark.unit
def test_model_config_creation():
    """Fast unit test for model configuration."""
    config = ModelConfig(name="test")
    assert config.name == "test"

@pytest.mark.integration
@pytest.mark.docker
def test_container_build(docker_client):
    """Integration test requiring Docker."""
    # Test actual container building
    pass

@pytest.mark.slow
@pytest.mark.performance
def test_high_throughput_inference():
    """Performance test for high load scenarios."""
    # Run load test scenarios
    pass
```

## Fixtures and Test Data

### Core Fixtures

Our test suite provides comprehensive fixtures:

```python
def test_with_fixtures(
    temp_dir,           # Temporary directory
    sample_model_config, # Basic model configuration
    nimifier,           # Nimifier instance
    mock_model_file,    # Fake ONNX model file
    mock_triton_client  # Mocked Triton client
):
    # Test implementation
    pass
```

### Integration Test Fixtures

For integration tests requiring real services:

```python
@pytest.mark.integration
def test_kubernetes_deployment(
    k8s_client,         # Real Kubernetes client
    test_namespace,     # Isolated test namespace
    sample_onnx_model   # Minimal real ONNX model
):
    # Test actual Kubernetes deployment
    pass
```

## Writing Good Tests

### Unit Test Guidelines

1. **Fast**: Each test should run in <100ms
2. **Isolated**: No external dependencies
3. **Deterministic**: Same input = same output
4. **Focused**: Test one behavior per test

```python
@pytest.mark.unit
def test_model_config_validation():
    """Test model configuration validation."""
    # Test valid configuration
    config = ModelConfig(name="valid-name")
    assert config.name == "valid-name"
    
    # Test invalid configuration
    with pytest.raises(ValueError):
        ModelConfig(name="")  # Empty name should fail
```

### Integration Test Guidelines

1. **Real services**: Use actual Docker/Kubernetes
2. **Isolated**: Clean up after each test
3. **Realistic**: Test real-world scenarios
4. **Conditional**: Skip if services unavailable

```python
@pytest.mark.integration
def test_model_deployment_workflow(
    docker_client, 
    k8s_client, 
    test_namespace
):
    """Test complete model deployment workflow."""
    # 1. Build container
    # 2. Deploy to Kubernetes
    # 3. Verify service is running
    # 4. Test inference endpoint
    # 5. Clean up resources
```

### Performance Test Guidelines

1. **Baseline**: Establish performance baselines
2. **Regression**: Detect performance regressions
3. **Load**: Test under realistic load
4. **Metrics**: Collect detailed metrics

```python
@pytest.mark.performance
def test_inference_latency(benchmark):
    """Benchmark inference latency."""
    
    def inference_call():
        # Perform inference
        return model.predict(input_data)
    
    result = benchmark(inference_call)
    
    # Assert performance requirements
    assert result.stats['mean'] < 0.100  # <100ms mean latency
```

## Mock Strategies

### External Service Mocking

```python
from unittest.mock import patch, Mock

@patch('nimify.core.tritonclient')
def test_model_serving(mock_triton):
    """Test model serving with mocked Triton."""
    mock_triton.InferenceServerClient.return_value.is_server_ready.return_value = True
    
    # Test logic with mocked Triton
    service = NIMService(config)
    assert service.is_ready()
```

### Environment Mocking

```python
@patch.dict(os.environ, {
    'NIMIFY_ENV': 'test',
    'GPU_AVAILABLE': 'false'
})
def test_cpu_only_mode():
    """Test CPU-only mode configuration."""
    # Test behavior with mocked environment
```

## Continuous Integration

### GitHub Actions Integration

Our CI pipeline runs comprehensive tests:

```yaml
# .github/workflows/test.yml
- name: Run Unit Tests
  run: pytest tests/ -m "unit" --cov=src

- name: Run Integration Tests
  run: pytest tests/ -m "integration" --run-integration
  if: github.event_name == 'push'

- name: Run Performance Tests
  run: pytest tests/ -m "performance"
  if: github.ref == 'refs/heads/main'
```

### Test Coverage Requirements

- **Minimum coverage**: 80%
- **New code coverage**: 90%
- **Critical paths**: 100%

## Debugging Tests

### Common Debugging Techniques

1. **Verbose output**: `pytest -v -s`
2. **PDB debugging**: `pytest --pdb`
3. **Log output**: `pytest --log-cli-level=DEBUG`
4. **Single test**: `pytest tests/test_file.py::test_function -s`

### Test Data Inspection

```python
def test_with_debug_output(caplog, tmp_path):
    """Example of debugging test data."""
    with caplog.at_level(logging.DEBUG):
        # Test code here
        pass
    
    # Inspect logs
    print(caplog.records)
    
    # Inspect temporary files
    for file in tmp_path.iterdir():
        print(f"Created: {file}")
```

## Performance Testing

### Load Testing Setup

```bash
# Install load testing tools
pip install locust pytest-benchmark

# Run performance tests
pytest tests/performance/ --benchmark-only

# Generate performance report
pytest tests/performance/ --benchmark-json=results.json
```

### Benchmark Examples

```python
@pytest.mark.performance
def test_model_loading_performance(benchmark):
    """Benchmark model loading time."""
    
    def load_model():
        return Nimifier.load_model("test_model.onnx")
    
    result = benchmark(load_model)
    
    # Performance assertions
    assert result.stats['mean'] < 2.0  # <2s mean loading time
    assert result.stats['stddev'] < 0.5  # Low variance
```

## Best Practices

### Test Organization

1. **File naming**: `test_*.py` or `*_test.py`
2. **Class naming**: `TestClassName`
3. **Method naming**: `test_descriptive_name`
4. **Descriptive docstrings**: Explain what is tested

### Test Data Management

1. **Fixtures**: Use pytest fixtures for reusable test data
2. **Factories**: Use factory functions for complex objects
3. **Parametrization**: Test multiple scenarios with `@pytest.mark.parametrize`
4. **Temporary data**: Always use temporary directories/files

### Error Testing

```python
def test_error_conditions():
    """Test error handling and edge cases."""
    
    # Test expected exceptions
    with pytest.raises(ValueError, match="Invalid model format"):
        Nimifier.load_model("invalid.txt")
    
    # Test error messages
    try:
        invalid_operation()
    except CustomError as e:
        assert "specific error message" in str(e)
```

## Troubleshooting

### Common Issues

1. **Slow tests**: Use `pytest-profiling` to identify bottlenecks
2. **Flaky tests**: Add retries or better synchronization
3. **Environment issues**: Use container-based testing
4. **Resource leaks**: Ensure proper cleanup in fixtures

### Getting Help

- Check test logs: `pytest --log-cli-level=DEBUG`
- Run single test: `pytest tests/specific_test.py::test_name -v -s`
- Use debugger: `pytest --pdb`
- Check CI logs for environment-specific issues