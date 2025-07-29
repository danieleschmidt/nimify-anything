"""Tests for core functionality."""

import pytest
from nimify.core import ModelConfig, Nimifier


def test_model_config_defaults():
    """Test ModelConfig with default values."""
    config = ModelConfig(name="test-model")
    assert config.name == "test-model"
    assert config.max_batch_size == 32
    assert config.dynamic_batching is True
    assert config.preferred_batch_sizes == [1, 4, 8, 16, 32]


def test_model_config_custom():
    """Test ModelConfig with custom values."""
    config = ModelConfig(
        name="custom-model",
        max_batch_size=64,
        dynamic_batching=False,
        preferred_batch_sizes=[8, 16]
    )
    assert config.name == "custom-model"
    assert config.max_batch_size == 64
    assert config.dynamic_batching is False
    assert config.preferred_batch_sizes == [8, 16]


def test_nimifier_creation():
    """Test Nimifier instance creation."""
    config = ModelConfig(name="test-model")
    nimifier = Nimifier(config)
    assert nimifier.config == config


def test_wrap_model():
    """Test model wrapping functionality."""
    config = ModelConfig(name="test-model")
    nimifier = Nimifier(config)
    
    service = nimifier.wrap_model(
        model_path="test.onnx",
        input_schema={"input": "float32[?,3,224,224]"},
        output_schema={"predictions": "float32[?,1000]"}
    )
    
    assert service.config == config
    assert service.model_path == "test.onnx"
    assert service.input_schema == {"input": "float32[?,3,224,224]"}
    assert service.output_schema == {"predictions": "float32[?,1000]"}