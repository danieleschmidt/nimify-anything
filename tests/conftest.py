"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
from unittest.mock import Mock, patch

from nimify.core import ModelConfig, Nimifier


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_model_config() -> ModelConfig:
    """Create a sample ModelConfig for testing."""
    return ModelConfig(
        name="test-model",
        max_batch_size=16,
        dynamic_batching=True,
        preferred_batch_sizes=[1, 4, 8, 16]
    )


@pytest.fixture
def nimifier(sample_model_config: ModelConfig) -> Nimifier:
    """Create a Nimifier instance for testing."""
    return Nimifier(sample_model_config)


@pytest.fixture
def mock_model_file(temp_dir: Path) -> Path:
    """Create a mock model file for testing."""
    model_file = temp_dir / "test_model.onnx"
    model_file.write_bytes(b"fake_model_data")
    return model_file


@pytest.fixture
def sample_schemas() -> Dict[str, Dict[str, str]]:
    """Sample input and output schemas for testing."""
    return {
        "input_schema": {"input": "float32[?,3,224,224]"},
        "output_schema": {"predictions": "float32[?,1000]"}
    }


@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        "NIMIFY_MODEL_CACHE": "/tmp/test_cache",
        "NIMIFY_LOG_LEVEL": "DEBUG",
        "CUDA_VISIBLE_DEVICES": "0"
    }):
        yield


@pytest.fixture
def mock_triton_client():
    """Mock Triton client for testing."""
    mock_client = Mock()
    mock_client.is_server_ready.return_value = True
    mock_client.get_model_metadata.return_value = {
        "name": "test-model",
        "versions": ["1"],
        "platform": "onnxruntime_onnx"
    }
    return mock_client


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean up environment after each test."""
    yield
    # Clean up any environment variables set during tests
    test_env_vars = [
        "NIMIFY_MODEL_CACHE",
        "NIMIFY_LOG_LEVEL", 
        "CUDA_VISIBLE_DEVICES"
    ]
    for var in test_env_vars:
        os.environ.pop(var, None)


# Pytest markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "docker: mark test as requiring Docker"
    )