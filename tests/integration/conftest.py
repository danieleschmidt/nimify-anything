"""Integration test configuration and fixtures."""

import os
import pytest
import asyncio
import docker
import kubernetes
from pathlib import Path
from typing import Optional
from unittest.mock import Mock

# Skip integration tests if not explicitly enabled
def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless explicitly enabled."""
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


def pytest_addoption(parser):
    """Add command-line options for integration tests."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests"
    )
    parser.addoption(
        "--k8s-context",
        action="store",
        default=None,
        help="kubernetes context for integration tests"
    )


@pytest.fixture(scope="session")
def docker_client():
    """Docker client for integration tests."""
    try:
        client = docker.from_env()
        # Test connection
        client.ping()
        return client
    except Exception:
        pytest.skip("Docker not available")


@pytest.fixture(scope="session")
def k8s_client(request):
    """Kubernetes client for integration tests."""
    try:
        # Load kubeconfig
        kubernetes.config.load_kube_config(
            context=request.config.getoption("--k8s-context")
        )
        
        # Test connection
        v1 = kubernetes.client.CoreV1Api()
        v1.list_namespace()
        
        return kubernetes.client
    except Exception:
        pytest.skip("Kubernetes not available")


@pytest.fixture
def test_namespace(k8s_client):
    """Create isolated test namespace."""
    namespace_name = f"nimify-test-{os.getpid()}"
    
    v1 = k8s_client.CoreV1Api()
    
    # Create namespace
    namespace = kubernetes.client.V1Namespace(
        metadata=kubernetes.client.V1ObjectMeta(name=namespace_name)
    )
    v1.create_namespace(namespace)
    
    yield namespace_name
    
    # Cleanup namespace
    try:
        v1.delete_namespace(namespace_name)
    except:
        pass  # Best effort cleanup


@pytest.fixture
def sample_onnx_model():
    """Create a minimal ONNX model for integration testing."""
    try:
        import onnx
        import numpy as np
        from onnx import helper, TensorProto
        
        # Create a simple linear model: y = x * 2 + 1
        input_tensor = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, [1, 3]
        )
        output_tensor = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, [1, 3]
        )
        
        # Weight and bias tensors
        weight = helper.make_tensor(
            'weight', TensorProto.FLOAT, [3, 3], 
            np.eye(3).flatten() * 2
        )
        bias = helper.make_tensor(
            'bias', TensorProto.FLOAT, [3], [1, 1, 1]
        )
        
        # MatMul and Add nodes
        matmul_node = helper.make_node(
            'MatMul', ['input', 'weight'], ['matmul_output']
        )
        add_node = helper.make_node(
            'Add', ['matmul_output', 'bias'], ['output']
        )
        
        # Create graph
        graph = helper.make_graph(
            [matmul_node, add_node],
            'test_model',
            [input_tensor],
            [output_tensor],
            [weight, bias]
        )
        
        # Create model
        model = helper.make_model(graph)
        onnx.checker.check_model(model)
        
        return model
        
    except ImportError:
        pytest.skip("ONNX not available for integration tests")


@pytest.fixture
def integration_config():
    """Configuration for integration tests."""
    return {
        'registry': os.getenv('TEST_REGISTRY', 'localhost:5000'),
        'timeout': int(os.getenv('TEST_TIMEOUT', '300')),
        'cleanup': os.getenv('TEST_CLEANUP', 'true').lower() == 'true',
        'gpu_available': os.getenv('GPU_AVAILABLE', 'false').lower() == 'true'
    }


@pytest.fixture
async def async_client():
    """Async HTTP client for API testing."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            yield client
    except ImportError:
        pytest.skip("httpx not available for async tests")


@pytest.fixture
def mock_nim_registry():
    """Mock NIM registry for testing without external dependencies."""
    mock_registry = Mock()
    mock_registry.pull_image.return_value = True
    mock_registry.list_models.return_value = [
        {"name": "llama2-7b", "version": "1.0"},
        {"name": "mistral-7b", "version": "1.0"}
    ]
    return mock_registry


@pytest.fixture(scope="session", autouse=True)
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()