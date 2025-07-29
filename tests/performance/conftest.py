"""
Pytest configuration for performance tests.
"""

import os
import pytest
from typing import Generator


def pytest_configure(config):
    """Configure pytest for performance testing."""
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests (deselect with '-m \"not performance\"')"
    )
    config.addinivalue_line(
        "markers", "load: marks tests as load tests"
    )
    config.addinivalue_line(
        "markers", "memory: marks tests as memory tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmark tests"
    )


@pytest.fixture(scope="session")
def performance_config():
    """Performance test configuration from environment variables."""
    return {
        "concurrent_users": int(os.getenv("LOAD_TEST_USERS", "10")),
        "requests_per_user": int(os.getenv("LOAD_TEST_REQUESTS", "100")),
        "test_duration": int(os.getenv("LOAD_TEST_DURATION", "60")),
        "base_url": os.getenv("SERVICE_BASE_URL", "http://localhost:8080"),
        "timeout": int(os.getenv("REQUEST_TIMEOUT", "30")),
    }


@pytest.fixture(autouse=True)
def performance_test_environment():
    """Ensure performance test environment is properly configured."""
    # Check if service is running
    import httpx
    import time
    
    base_url = os.getenv("SERVICE_BASE_URL", "http://localhost:8080")
    
    # Wait for service to be ready
    for attempt in range(30):  # 30 second timeout
        try:
            with httpx.Client() as client:
                response = client.get(f"{base_url}/health", timeout=5.0)
                if response.status_code == 200:
                    break
        except Exception:
            pass
        time.sleep(1)
    else:
        pytest.skip(f"Service at {base_url} is not available for performance testing")


@pytest.fixture
def large_test_payload():
    """Generate large test payload for performance testing."""
    return {
        "input": [[float(i % 100) for i in range(224 * 224)] for _ in range(8)]
    }


@pytest.fixture
def small_test_payload():
    """Generate small test payload for performance testing."""
    return {
        "input": [[1.0, 2.0, 3.0, 4.0]]
    }