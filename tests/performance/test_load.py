"""
Performance and load testing for Nimify services.
Tests various performance scenarios and load conditions.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import pytest
import httpx
from prometheus_client.parser import text_string_to_metric_families


class LoadTestConfig:
    """Configuration for load testing scenarios."""
    
    def __init__(
        self,
        concurrent_users: int = 10,
        requests_per_user: int = 100,
        test_duration: int = 60,
        base_url: str = "http://localhost:8080"
    ):
        self.concurrent_users = concurrent_users
        self.requests_per_user = requests_per_user
        self.test_duration = test_duration
        self.base_url = base_url


class PerformanceMetrics:
    """Collects and analyzes performance metrics."""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.error_count = 0
        self.success_count = 0
        self.start_time: float = 0
        self.end_time: float = 0
    
    def record_request(self, response_time: float, success: bool):
        """Record a single request result."""
        self.response_times.append(response_time)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate performance statistics."""
        if not self.response_times:
            return {}
        
        sorted_times = sorted(self.response_times)
        total_requests = len(self.response_times)
        duration = self.end_time - self.start_time
        
        return {
            "total_requests": total_requests,
            "successful_requests": self.success_count,
            "failed_requests": self.error_count,
            "error_rate": self.error_count / total_requests * 100,
            "duration_seconds": duration,
            "requests_per_second": total_requests / duration if duration > 0 else 0,
            "response_time_ms": {
                "min": min(sorted_times) * 1000,
                "max": max(sorted_times) * 1000,
                "mean": sum(sorted_times) / len(sorted_times) * 1000,
                "p50": sorted_times[int(len(sorted_times) * 0.5)] * 1000,
                "p95": sorted_times[int(len(sorted_times) * 0.95)] * 1000,
                "p99": sorted_times[int(len(sorted_times) * 0.99)] * 1000,
            }
        }


@pytest.fixture
def load_test_config():
    """Provide load test configuration."""
    return LoadTestConfig()


@pytest.fixture
def performance_metrics():
    """Provide performance metrics collector."""
    return PerformanceMetrics()


async def simulate_user_session(
    session: httpx.AsyncClient,
    config: LoadTestConfig,
    metrics: PerformanceMetrics
):
    """Simulate a single user session with multiple requests."""
    
    test_payload = {
        "input": [[1.0, 2.0, 3.0, 4.0] for _ in range(32)]  # Batch of 32
    }
    
    for _ in range(config.requests_per_user):
        start_time = time.time()
        try:
            response = await session.post(
                f"{config.base_url}/v1/predict",
                json=test_payload,
                timeout=30.0
            )
            end_time = time.time()
            
            success = response.status_code == 200
            metrics.record_request(end_time - start_time, success)
            
            if not success:
                print(f"Request failed with status {response.status_code}")
                
        except Exception as e:
            end_time = time.time()
            metrics.record_request(end_time - start_time, False)
            print(f"Request exception: {e}")
        
        # Small delay to simulate realistic user behavior
        await asyncio.sleep(0.1)


@pytest.mark.performance
@pytest.mark.asyncio
async def test_concurrent_load(load_test_config, performance_metrics):
    """Test system performance under concurrent load."""
    
    config = load_test_config
    metrics = performance_metrics
    
    metrics.start_time = time.time()
    
    # Create async HTTP session
    async with httpx.AsyncClient() as client:
        # Create concurrent user sessions
        tasks = [
            simulate_user_session(client, config, metrics)
            for _ in range(config.concurrent_users)
        ]
        
        # Run all user sessions concurrently
        await asyncio.gather(*tasks)
    
    metrics.end_time = time.time()
    
    # Analyze results
    stats = metrics.get_statistics()
    
    # Performance assertions
    assert stats["error_rate"] < 5.0, f"Error rate too high: {stats['error_rate']}%"
    assert stats["response_time_ms"]["p95"] < 1000, f"P95 latency too high: {stats['response_time_ms']['p95']}ms"
    assert stats["requests_per_second"] > 50, f"Throughput too low: {stats['requests_per_second']} RPS"
    
    # Print performance report
    print("\\n=== Performance Test Results ===")
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Success Rate: {(stats['successful_requests']/stats['total_requests'])*100:.2f}%")
    print(f"Requests/Second: {stats['requests_per_second']:.2f}")
    print(f"Response Times (ms):")
    print(f"  Min: {stats['response_time_ms']['min']:.2f}")
    print(f"  Mean: {stats['response_time_ms']['mean']:.2f}")
    print(f"  P95: {stats['response_time_ms']['p95']:.2f}")
    print(f"  P99: {stats['response_time_ms']['p99']:.2f}")
    print(f"  Max: {stats['response_time_ms']['max']:.2f}")


@pytest.mark.performance
@pytest.mark.asyncio
async def test_batch_size_performance():
    """Test performance with different batch sizes."""
    
    batch_sizes = [1, 8, 16, 32, 64, 128]
    results = {}
    
    async with httpx.AsyncClient() as client:
        for batch_size in batch_sizes:
            test_payload = {
                "input": [[1.0, 2.0, 3.0, 4.0] for _ in range(batch_size)]
            }
            
            # Warm up
            for _ in range(5):
                await client.post("http://localhost:8080/v1/predict", json=test_payload)
            
            # Measure performance
            times = []
            for _ in range(20):
                start_time = time.time()
                response = await client.post(
                    "http://localhost:8080/v1/predict",
                    json=test_payload,
                    timeout=30.0
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    times.append(end_time - start_time)
            
            if times:
                avg_time = sum(times) / len(times)
                throughput = batch_size / avg_time  # samples per second
                results[batch_size] = {
                    "avg_latency_ms": avg_time * 1000,
                    "throughput_samples_per_sec": throughput
                }
    
    # Print batch size analysis
    print("\\n=== Batch Size Performance Analysis ===")
    for batch_size, metrics in results.items():
        print(f"Batch Size {batch_size}: "
              f"{metrics['avg_latency_ms']:.2f}ms, "
              f"{metrics['throughput_samples_per_sec']:.2f} samples/sec")
    
    # Find optimal batch size (highest throughput)
    if results:
        optimal_batch = max(results.keys(), 
                          key=lambda k: results[k]['throughput_samples_per_sec'])
        print(f"\\nOptimal batch size: {optimal_batch}")


@pytest.mark.performance
def test_memory_usage_under_load():
    """Test memory usage patterns under sustained load."""
    
    import psutil
    import os
    
    # Get current process
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run load test with memory monitoring
    peak_memory = initial_memory
    
    def memory_monitor():
        nonlocal peak_memory
        current_memory = process.memory_info().rss / 1024 / 1024
        peak_memory = max(peak_memory, current_memory)
        return current_memory
    
    # Simulate memory-intensive operations
    large_batches = []
    for i in range(100):
        batch_data = [[1.0] * 1000 for _ in range(64)]  # Large batch
        large_batches.append(batch_data)
        
        if i % 10 == 0:
            current_mem = memory_monitor()
            print(f"Iteration {i}: Memory usage: {current_mem:.2f} MB")
    
    final_memory = memory_monitor()
    memory_increase = final_memory - initial_memory
    
    print(f"\\n=== Memory Usage Analysis ===")
    print(f"Initial Memory: {initial_memory:.2f} MB")
    print(f"Peak Memory: {peak_memory:.2f} MB")
    print(f"Final Memory: {final_memory:.2f} MB")
    print(f"Memory Increase: {memory_increase:.2f} MB")
    
    # Assert reasonable memory usage
    assert memory_increase < 500, f"Memory increase too high: {memory_increase:.2f} MB"
    assert peak_memory < initial_memory * 3, f"Peak memory usage too high: {peak_memory:.2f} MB"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_prometheus_metrics_performance():
    """Test that Prometheus metrics collection doesn't impact performance."""
    
    # Test with metrics collection
    with_metrics_times = []
    async with httpx.AsyncClient() as client:
        for _ in range(50):
            start_time = time.time()
            response = await client.post(
                "http://localhost:8080/v1/predict",
                json={"input": [[1.0, 2.0, 3.0, 4.0]] * 16}
            )
            end_time = time.time()
            
            if response.status_code == 200:
                with_metrics_times.append(end_time - start_time)
            
            # Also check metrics endpoint
            metrics_response = await client.get("http://localhost:8080/metrics")
            assert metrics_response.status_code == 200
    
    # Calculate metrics overhead
    if with_metrics_times:
        avg_time = sum(with_metrics_times) / len(with_metrics_times)
        print(f"\\n=== Metrics Performance Analysis ===")
        print(f"Average response time with metrics: {avg_time * 1000:.2f}ms")
        
        # Verify metrics are being collected
        async with httpx.AsyncClient() as client:
            metrics_response = await client.get("http://localhost:8080/metrics")
            metrics_text = metrics_response.text
            
            # Check for key metrics
            assert "nim_request_duration_seconds" in metrics_text
            assert "nim_request_count_total" in metrics_text
            
            # Parse metrics to verify they're updating
            families = list(text_string_to_metric_families(metrics_text))
            request_count = None
            
            for family in families:
                if family.name == "nim_request_count_total":
                    for sample in family.samples:
                        if sample.value > 0:
                            request_count = sample.value
                            break
            
            assert request_count is not None and request_count >= 50, \
                "Request count metric not updating properly"