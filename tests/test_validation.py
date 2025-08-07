"""Tests for validation and security utilities."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from nimify.validation import (
    ModelFileValidator, ServiceNameValidator, ConfigurationValidator,
    SecurityValidator, RequestValidator, ValidationError,
    sanitize_filename, validate_docker_image_name
)


class TestModelFileValidator:
    """Test model file validation."""
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        with pytest.raises(ValidationError, match="Model file not found"):
            ModelFileValidator.validate_file_path("nonexistent.onnx")
    
    def test_validate_directory(self, tmp_path):
        """Test validation of directory instead of file."""
        test_dir = tmp_path / "testdir"
        test_dir.mkdir()
        
        with pytest.raises(ValidationError, match="Path is not a file"):
            ModelFileValidator.validate_file_path(str(test_dir))
    
    def test_validate_invalid_extension(self, tmp_path):
        """Test validation of file with invalid extension."""
        test_file = tmp_path / "model.txt"
        test_file.write_text("fake model")
        
        with pytest.raises(ValidationError, match="Unsupported file extension"):
            ModelFileValidator.validate_file_path(str(test_file))
    
    def test_validate_too_large_file(self, tmp_path):
        """Test validation of file that's too large."""
        test_file = tmp_path / "model.onnx"
        
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value.st_size = ModelFileValidator.MAX_FILE_SIZE + 1
            test_file.touch()  # Create the file
            
            with pytest.raises(ValidationError, match="File too large"):
                ModelFileValidator.validate_file_path(str(test_file))
    
    def test_validate_too_small_file(self, tmp_path):
        """Test validation of file that's too small."""
        test_file = tmp_path / "model.onnx"
        test_file.write_text("x")  # Very small file
        
        with pytest.raises(ValidationError, match="File too small"):
            ModelFileValidator.validate_file_path(str(test_file))
    
    def test_validate_valid_file(self, tmp_path):
        """Test validation of valid file."""
        test_file = tmp_path / "model.onnx"
        test_file.write_text("x" * 2000)  # Valid size
        
        result = ModelFileValidator.validate_file_path(str(test_file))
        assert isinstance(result, Path)
        assert result.name == "model.onnx"
    
    def test_compute_file_hash(self, tmp_path):
        """Test file hash computation."""
        test_file = tmp_path / "model.onnx"
        test_file.write_text("test content")
        
        hash1 = ModelFileValidator.compute_file_hash(test_file)
        hash2 = ModelFileValidator.compute_file_hash(test_file)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length


class TestServiceNameValidator:
    """Test service name validation."""
    
    def test_validate_valid_names(self):
        """Test validation of valid service names."""
        valid_names = ["my-service", "test123", "a-b-c", "service-v2"]
        
        for name in valid_names:
            result = ServiceNameValidator.validate_service_name(name)
            assert result == name
    
    def test_validate_invalid_names(self):
        """Test validation of invalid service names."""
        invalid_names = [
            "A",  # Too short
            "a",  # Too short
            "-invalid",  # Starts with hyphen
            "invalid-",  # Ends with hyphen
            "Invalid",  # Uppercase
            "in valid",  # Space
            "in_valid_name_that_is_way_too_long_for_kubernetes_service_names_and_exceeds_the_limit"  # Too long
        ]
        
        for name in invalid_names:
            with pytest.raises(ValidationError):
                ServiceNameValidator.validate_service_name(name)
    
    def test_validate_reserved_names(self):
        """Test validation of reserved names."""
        reserved_names = ["nimify", "api", "admin", "root"]
        
        for name in reserved_names:
            with pytest.raises(ValidationError, match="reserved name"):
                ServiceNameValidator.validate_service_name(name)


class TestConfigurationValidator:
    """Test configuration validation."""
    
    def test_validate_valid_port(self):
        """Test validation of valid port numbers."""
        valid_ports = [1024, 8080, 9090, 65535]
        
        for port in valid_ports:
            result = ConfigurationValidator.validate_port(port)
            assert result == port
    
    def test_validate_invalid_ports(self):
        """Test validation of invalid port numbers."""
        invalid_ports = [80, 1023, 65536, 100000, "8080"]
        
        for port in invalid_ports:
            with pytest.raises(ValidationError):
                ConfigurationValidator.validate_port(port)
    
    def test_validate_batch_size(self):
        """Test batch size validation."""
        valid_sizes = [1, 32, 64, 1024]
        
        for size in valid_sizes:
            result = ConfigurationValidator.validate_batch_size(size)
            assert result == size
        
        invalid_sizes = [0, -1, 1025, "32"]
        
        for size in invalid_sizes:
            with pytest.raises(ValidationError):
                ConfigurationValidator.validate_batch_size(size)
    
    def test_validate_memory_limit(self):
        """Test memory limit validation."""
        valid_limits = ["512Mi", "1Gi", "2.5Gi", "1024Mi"]
        
        for limit in valid_limits:
            result = ConfigurationValidator.validate_memory_limit(limit)
            assert result == limit
        
        invalid_limits = ["512M", "1GB", "invalid", "512"]
        
        for limit in invalid_limits:
            with pytest.raises(ValidationError):
                ConfigurationValidator.validate_memory_limit(limit)


class TestSecurityValidator:
    """Test security validation."""
    
    def test_scan_dangerous_content(self):
        """Test scanning for dangerous content patterns."""
        dangerous_contents = [
            "SELECT * FROM users",
            "<script>alert('xss')</script>",
            "$(rm -rf /)",
            "../../../etc/passwd",
            "javascript:alert(1)"
        ]
        
        for content in dangerous_contents:
            matches = SecurityValidator.scan_for_dangerous_content(content)
            assert len(matches) > 0
    
    def test_scan_safe_content(self):
        """Test scanning safe content."""
        safe_contents = [
            "This is a normal string",
            "model prediction: [0.1, 0.9]",
            "HTTP status: 200 OK"
        ]
        
        for content in safe_contents:
            matches = SecurityValidator.scan_for_dangerous_content(content)
            assert len(matches) == 0
    
    def test_validate_environment_variables(self):
        """Test environment variable validation."""
        # Safe environment variables
        safe_env = {
            "MODEL_NAME": "my-model",
            "LOG_LEVEL": "INFO",
            "PORT": "8080"
        }
        
        result = SecurityValidator.validate_environment_variables(safe_env)
        assert result == safe_env
        
        # Dangerous environment variables
        dangerous_env = {
            "COMMAND": "rm -rf /"
        }
        
        with pytest.raises(ValidationError):
            SecurityValidator.validate_environment_variables(dangerous_env)


class TestRequestValidator:
    """Test request validation using Pydantic."""
    
    def test_valid_request(self):
        """Test valid request validation."""
        valid_data = {
            "input": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        }
        
        request = RequestValidator(**valid_data)
        assert request.input == valid_data["input"]
    
    def test_invalid_input_dimensions(self):
        """Test request with inconsistent input dimensions."""
        invalid_data = {
            "input": [[1.0, 2.0, 3.0], [4.0, 5.0]]  # Different lengths
        }
        
        with pytest.raises(ValueError, match="same length"):
            RequestValidator(**invalid_data)
    
    def test_nan_values(self):
        """Test request with NaN values."""
        invalid_data = {
            "input": [[1.0, float('nan'), 3.0]]
        }
        
        with pytest.raises(ValueError, match="Invalid value"):
            RequestValidator(**invalid_data)
    
    def test_infinite_values(self):
        """Test request with infinite values."""
        invalid_data = {
            "input": [[1.0, float('inf'), 3.0]]
        }
        
        with pytest.raises(ValueError, match="Invalid value"):
            RequestValidator(**invalid_data)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        test_cases = [
            ("normal.txt", "normal.txt"),
            ("file<>name.txt", "file__name.txt"),
            ("../../../etc/passwd", ".._.._.._.._etc_passwd"),
            ("", "unnamed"),
            ("." * 300, "." * 255),  # Length limit
        ]
        
        for input_name, expected in test_cases:
            result = sanitize_filename(input_name)
            assert result == expected
    
    def test_validate_docker_image_name(self):
        """Test Docker image name validation."""
        valid_names = [
            "nginx:latest",
            "registry.io/namespace/image:v1.0",
            "localhost:5000/my-app:dev"
        ]
        
        for name in valid_names:
            result = validate_docker_image_name(name)
            assert result == name
        
        invalid_names = [
            "Invalid:Name",  # Uppercase
            "name::",  # Invalid tag
            "",  # Empty
        ]
        
        for name in invalid_names:
            with pytest.raises(ValidationError):
                validate_docker_image_name(name)