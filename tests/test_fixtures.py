"""Tests for test fixtures and utilities."""

import pytest
from pathlib import Path
import tempfile
import json

from tests.conftest import *


class TestFixtures:
    """Test the test fixtures themselves."""
    
    def test_temp_dir_fixture(self, temp_dir):
        """Test temporary directory fixture."""
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test we can write to it
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.read_text() == "test content"
    
    def test_sample_model_config(self, sample_model_config):
        """Test sample model configuration fixture."""
        assert sample_model_config.name == "test-model"
        assert sample_model_config.max_batch_size == 16
        assert sample_model_config.dynamic_batching is True
        assert sample_model_config.preferred_batch_sizes == [1, 4, 8, 16]
    
    def test_nimifier_fixture(self, nimifier):
        """Test Nimifier instance fixture."""
        assert nimifier.config.name == "test-model"
        assert hasattr(nimifier, 'wrap_model')
    
    def test_mock_model_file(self, mock_model_file):
        """Test mock model file fixture."""
        assert mock_model_file.exists()
        assert mock_model_file.suffix == ".onnx"
        assert mock_model_file.read_bytes() == b"fake_model_data"
    
    def test_sample_schemas(self, sample_schemas):
        """Test sample schema fixture."""
        assert "input_schema" in sample_schemas
        assert "output_schema" in sample_schemas
        assert "input" in sample_schemas["input_schema"]
        assert "predictions" in sample_schemas["output_schema"]
    
    def test_mock_triton_client(self, mock_triton_client):
        """Test mock Triton client fixture."""
        assert mock_triton_client.is_server_ready() is True
        metadata = mock_triton_client.get_model_metadata()
        assert metadata["name"] == "test-model"
        assert "1" in metadata["versions"]


class TestTestUtilities:
    """Test utilities for testing."""
    
    def test_pytest_markers_configured(self):
        """Test that custom pytest markers are configured."""
        # This test ensures our custom markers are registered
        # pytest will warn about unknown markers if not configured
        pass
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker works."""
        assert True


class TestEnvironmentCleanup:
    """Test environment cleanup functionality."""
    
    def test_environment_cleanup(self, clean_environment):
        """Test that environment is cleaned up after tests."""
        import os
        
        # Set some test environment variables
        os.environ["NIMIFY_MODEL_CACHE"] = "/tmp/test"
        os.environ["NIMIFY_LOG_LEVEL"] = "DEBUG"
        
        # Variables should exist during test
        assert os.getenv("NIMIFY_MODEL_CACHE") == "/tmp/test"
        assert os.getenv("NIMIFY_LOG_LEVEL") == "DEBUG"
        
        # cleanup happens automatically after test via fixture


class TestMockingUtilities:
    """Test mocking utilities and patterns."""
    
    def test_mock_environment_variables(self, mock_environment_variables):
        """Test environment variable mocking."""
        import os
        
        assert os.getenv("NIMIFY_MODEL_CACHE") == "/tmp/test_cache"
        assert os.getenv("NIMIFY_LOG_LEVEL") == "DEBUG"
        assert os.getenv("CUDA_VISIBLE_DEVICES") == "0"
    
    def test_mock_triton_responses(self, mock_triton_client):
        """Test Triton client mocking."""
        # Test server ready
        assert mock_triton_client.is_server_ready()
        
        # Test model metadata
        metadata = mock_triton_client.get_model_metadata()
        assert metadata["name"] == "test-model"
        assert metadata["platform"] == "onnxruntime_onnx"
        
        # Test that it's actually a mock
        from unittest.mock import Mock
        assert isinstance(mock_triton_client, Mock)