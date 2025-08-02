"""Test utilities and helper functions."""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from unittest.mock import Mock, MagicMock
import tempfile
import subprocess
import pytest


class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def create_sample_onnx_metadata() -> Dict[str, Any]:
        """Create sample ONNX model metadata."""
        return {
            "ir_version": 8,
            "producer_name": "pytorch",
            "producer_version": "1.13.0",
            "domain": "",
            "model_version": 1,
            "doc_string": "Sample model for testing",
            "graph": {
                "name": "test_model",
                "input": [{
                    "name": "input",
                    "type": {
                        "tensor_type": {
                            "elem_type": 1,  # FLOAT
                            "shape": {
                                "dim": [
                                    {"dim_value": -1},  # batch_size
                                    {"dim_value": 3},   # channels
                                    {"dim_value": 224}, # height
                                    {"dim_value": 224}  # width
                                ]
                            }
                        }
                    }
                }],
                "output": [{
                    "name": "output",
                    "type": {
                        "tensor_type": {
                            "elem_type": 1,  # FLOAT
                            "shape": {
                                "dim": [
                                    {"dim_value": -1},    # batch_size
                                    {"dim_value": 1000}   # num_classes
                                ]
                            }
                        }
                    }
                }]
            }
        }
    
    @staticmethod
    def create_sample_tensorrt_metadata() -> Dict[str, Any]:
        """Create sample TensorRT engine metadata."""
        return {
            "name": "test_model_trt",
            "platform": "tensorrt_plan",
            "backend": "tensorrt",
            "version_policy": {"latest": {"num_versions": 1}},
            "input": [{
                "name": "input",
                "data_type": "TYPE_FP32",
                "dims": [-1, 3, 224, 224]
            }],
            "output": [{
                "name": "output", 
                "data_type": "TYPE_FP32",
                "dims": [-1, 1000]
            }],
            "dynamic_batching": {"preferred_batch_size": [1, 4, 8, 16]},
            "optimization": {"cuda": {"graphs": True}}
        }
    
    @staticmethod
    def create_test_model_file(
        file_path: Path, 
        model_type: str = "onnx", 
        size_bytes: int = 1024
    ) -> Path:
        """Create a test model file with specified properties."""
        if model_type == "onnx":
            # Create minimal ONNX file structure
            content = b"\\x08\\x01\\x12\\x0c\\x74\\x65\\x73\\x74\\x5f\\x6d\\x6f\\x64\\x65\\x6c"
        elif model_type == "tensorrt":
            # Create minimal TensorRT engine structure
            content = b"\\x50\\x54\\x58\\x20\\x01\\x00\\x00\\x00"
        else:
            # Generic binary content
            content = b"\\x00" * min(size_bytes, 64)
        
        # Pad to requested size
        if len(content) < size_bytes:
            content += b"\\x00" * (size_bytes - len(content))
        
        file_path.write_bytes(content)
        return file_path


class MockTritonServer:
    """Mock Triton server for testing."""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.models: Dict[str, Dict] = {}
        self.is_ready = False
    
    def start(self) -> None:
        """Mock server start."""
        self.is_ready = True
    
    def stop(self) -> None:
        """Mock server stop."""
        self.is_ready = False
        self.models.clear()
    
    def load_model(self, model_name: str, model_config: Dict[str, Any]) -> None:
        """Mock model loading."""
        self.models[model_name] = {
            "config": model_config,
            "status": "READY",
            "version": "1"
        }
    
    def unload_model(self, model_name: str) -> None:
        """Mock model unloading."""
        if model_name in self.models:
            del self.models[model_name]
    
    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """Mock getting model metadata."""
        if model_name not in self.models:
            raise Exception(f"Model {model_name} not found")
        return self.models[model_name]
    
    def infer(self, model_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Mock inference."""
        if model_name not in self.models:
            raise Exception(f"Model {model_name} not found")
        
        # Return mock prediction
        return {
            "outputs": {
                "predictions": [[0.1, 0.2, 0.7] for _ in range(len(inputs.get("input", [])))]
            }
        }


class PerformanceTimer:
    """Utility for measuring test performance."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
    
    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.start_time is None or self.end_time is None:
            raise ValueError("Timer not properly used with context manager")
        return self.end_time - self.start_time
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        return self.duration * 1000


class AsyncTestHelper:
    """Helper for testing async operations."""
    
    @staticmethod
    def run_async_test(coro):
        """Run an async test function."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    @staticmethod
    async def wait_for_condition(
        condition_func, 
        timeout: float = 5.0, 
        check_interval: float = 0.1
    ) -> bool:
        """Wait for a condition to become true."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            await asyncio.sleep(check_interval)
        return False


class ContainerTestHelper:
    """Helper for container-related tests."""
    
    @staticmethod
    def is_docker_available() -> bool:
        """Check if Docker is available."""
        try:
            subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    @staticmethod
    def is_kubernetes_available() -> bool:
        """Check if Kubernetes is available."""
        try:
            subprocess.run(
                ["kubectl", "version", "--client"], 
                capture_output=True, 
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    @staticmethod
    def cleanup_test_containers(name_pattern: str) -> None:
        """Clean up test containers matching pattern."""
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={name_pattern}", "--format", "{{.ID}}"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                container_ids = result.stdout.strip().split("\\n")
                subprocess.run(["docker", "rm", "-f"] + container_ids, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass  # Docker not available or no containers to clean


class ValidationHelper:
    """Helper for validation and assertion utilities."""
    
    @staticmethod
    def assert_file_exists(file_path: Union[str, Path]) -> None:
        """Assert that a file exists."""
        path = Path(file_path)
        assert path.exists(), f"File does not exist: {path}"
    
    @staticmethod
    def assert_directory_structure(base_path: Path, expected_structure: List[str]) -> None:
        """Assert that a directory has the expected structure."""
        for expected_path in expected_structure:
            full_path = base_path / expected_path
            assert full_path.exists(), f"Expected path does not exist: {full_path}"
    
    @staticmethod
    def assert_json_schema(data: Dict[str, Any], required_keys: List[str]) -> None:
        """Assert that JSON data has required keys."""
        for key in required_keys:
            assert key in data, f"Required key missing: {key}"
    
    @staticmethod
    def assert_performance_threshold(duration: float, max_duration: float) -> None:
        """Assert that operation completed within time threshold."""
        assert duration <= max_duration, f"Operation took {duration:.3f}s, expected <= {max_duration:.3f}s"


# Pytest fixtures using the utilities
@pytest.fixture
def test_data_generator():
    """Test data generator fixture."""
    return TestDataGenerator()


@pytest.fixture
def mock_triton_server():
    """Mock Triton server fixture."""
    server = MockTritonServer()
    yield server
    server.stop()


@pytest.fixture
def performance_timer():
    """Performance timer fixture."""
    return PerformanceTimer()


@pytest.fixture
def async_test_helper():
    """Async test helper fixture."""
    return AsyncTestHelper()


@pytest.fixture
def container_test_helper():
    """Container test helper fixture."""
    return ContainerTestHelper()


@pytest.fixture
def validation_helper():
    """Validation helper fixture."""
    return ValidationHelper()


# Skip decorators for conditional tests
skip_if_no_docker = pytest.mark.skipif(
    not ContainerTestHelper.is_docker_available(),
    reason="Docker not available"
)

skip_if_no_kubernetes = pytest.mark.skipif(
    not ContainerTestHelper.is_kubernetes_available(),
    reason="Kubernetes not available"
)

skip_if_no_gpu = pytest.mark.skipif(
    not Path("/proc/driver/nvidia/version").exists(),
    reason="NVIDIA GPU not available"
)