"""
Performance testing package for Nimify services.

This package provides comprehensive performance and load testing capabilities
for NVIDIA NIM microservices, including:

- Concurrent load testing
- Batch size optimization
- Memory usage monitoring
- Latency and throughput analysis
- Prometheus metrics validation

Usage:
    pytest tests/performance/ -m performance --verbose

Configuration:
    Set environment variables to customize test parameters:
    - LOAD_TEST_USERS: Number of concurrent users (default: 10)
    - LOAD_TEST_REQUESTS: Requests per user (default: 100)
    - LOAD_TEST_DURATION: Test duration in seconds (default: 60)
    - SERVICE_BASE_URL: Base URL for testing (default: http://localhost:8080)
"""

__version__ = "1.0.0"
__author__ = "Nimify Performance Team"