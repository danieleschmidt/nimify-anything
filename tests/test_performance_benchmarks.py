"""Performance benchmarks and load testing for the Nimify system."""

import asyncio
import time
import statistics
import concurrent.futures
from typing import List, Dict, Any
import pytest
import threading
from dataclasses import dataclass
from unittest.mock import Mock, patch

from src.nimify.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from src.nimify.rate_limiter import TokenBucketRateLimiter, RateLimitConfig
from src.nimify.caching_system import InMemoryCache, CacheConfig, CacheStrategy
from src.nimify.auto_scaler import IntelligentAutoScaler, ScalingConfig


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    operation: str
    total_operations: int
    duration_seconds: float
    operations_per_second: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    success_rate: float
    errors: int


class PerformanceBenchmark:
    """Performance benchmarking utility."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    async def benchmark_async_operation(
        self,
        operation_name: str,
        async_func,
        num_operations: int = 1000,
        concurrency: int = 10
    ) -> BenchmarkResult:
        """Benchmark an async operation with specified concurrency."""
        
        semaphore = asyncio.Semaphore(concurrency)
        latencies = []
        errors = 0
        
        async def timed_operation():
            nonlocal errors
            async with semaphore:
                start_time = time.time()
                try:
                    await async_func()
                    latency = (time.time() - start_time) * 1000  # Convert to ms
                    latencies.append(latency)
                except Exception:
                    errors += 1
        
        # Run benchmark
        start_time = time.time()
        tasks = [timed_operation() for _ in range(num_operations)]
        await asyncio.gather(*tasks, return_exceptions=True)
        total_duration = time.time() - start_time
        
        # Calculate statistics
        successful_ops = len(latencies)
        ops_per_second = successful_ops / total_duration if total_duration > 0 else 0
        avg_latency = statistics.mean(latencies) if latencies else 0
        p95_latency = statistics.quantiles(latencies, n=20)[18] if latencies else 0  # 95th percentile
        p99_latency = statistics.quantiles(latencies, n=100)[98] if latencies else 0  # 99th percentile
        success_rate = successful_ops / num_operations
        
        result = BenchmarkResult(
            operation=operation_name,
            total_operations=num_operations,
            duration_seconds=total_duration,
            operations_per_second=ops_per_second,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            success_rate=success_rate,
            errors=errors
        )
        
        self.results.append(result)
        return result
    
    def benchmark_sync_operation(
        self,
        operation_name: str,
        sync_func,
        num_operations: int = 1000,
        num_threads: int = 10
    ) -> BenchmarkResult:
        """Benchmark a synchronous operation with threading."""
        
        latencies = []
        errors = 0
        lock = threading.Lock()
        
        def timed_operation():
            nonlocal errors
            start_time = time.time()
            try:
                sync_func()
                latency = (time.time() - start_time) * 1000
                with lock:
                    latencies.append(latency)
            except Exception:
                with lock:
                    errors += 1
        
        # Run benchmark
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(timed_operation) for _ in range(num_operations)]
            concurrent.futures.wait(futures)
        
        total_duration = time.time() - start_time
        
        # Calculate statistics
        successful_ops = len(latencies)
        ops_per_second = successful_ops / total_duration if total_duration > 0 else 0
        avg_latency = statistics.mean(latencies) if latencies else 0
        p95_latency = statistics.quantiles(latencies, n=20)[18] if latencies else 0
        p99_latency = statistics.quantiles(latencies, n=100)[98] if latencies else 0
        success_rate = successful_ops / num_operations
        
        result = BenchmarkResult(
            operation=operation_name,
            total_operations=num_operations,
            duration_seconds=total_duration,
            operations_per_second=ops_per_second,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            success_rate=success_rate,
            errors=errors
        )
        
        self.results.append(result)
        return result
    
    def print_results(self):
        """Print benchmark results in a formatted table."""
        print("\n" + "="*120)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*120)
        print(f"{'Operation':<30} {'Ops':<8} {'Duration':<10} {'Ops/sec':<10} {'Avg(ms)':<10} {'P95(ms)':<10} {'P99(ms)':<10} {'Success%':<10}")
        print("-"*120)
        
        for result in self.results:
            print(f"{result.operation:<30} "
                  f"{result.total_operations:<8} "
                  f"{result.duration_seconds:<10.2f} "
                  f"{result.operations_per_second:<10.1f} "
                  f"{result.avg_latency_ms:<10.2f} "
                  f"{result.p95_latency_ms:<10.2f} "
                  f"{result.p99_latency_ms:<10.2f} "
                  f"{result.success_rate*100:<10.1f}")
        
        print("="*120)


class TestCircuitBreakerPerformance:
    """Test circuit breaker performance under load."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_circuit_breaker_throughput(self):
        """Test circuit breaker throughput with successful calls."""
        config = CircuitBreakerConfig(failure_threshold=10)
        cb = CircuitBreaker(config, "perf_test")
        
        benchmark = PerformanceBenchmark()
        
        async def successful_operation():
            await cb.acall(lambda: "success")
        
        result = await benchmark.benchmark_async_operation(
            "CircuitBreaker Success",
            successful_operation,
            num_operations=10000,
            concurrency=50
        )
        
        # Assertions for performance expectations
        assert result.success_rate > 0.99  # 99% success rate
        assert result.operations_per_second > 1000  # At least 1000 ops/sec
        assert result.avg_latency_ms < 10  # Average latency under 10ms
        
        benchmark.print_results()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_circuit_breaker_with_failures(self):
        """Test circuit breaker performance with mixed success/failure."""
        config = CircuitBreakerConfig(failure_threshold=5, timeout=1.0)
        cb = CircuitBreaker(config, "perf_fail_test")
        
        benchmark = PerformanceBenchmark()
        call_counter = 0
        
        async def mixed_operation():
            nonlocal call_counter
            call_counter += 1
            # 20% failure rate
            if call_counter % 5 == 0:
                await cb.acall(lambda: exec('raise Exception("test failure")'))
            else:
                await cb.acall(lambda: "success")
        
        result = await benchmark.benchmark_async_operation(
            "CircuitBreaker Mixed",
            mixed_operation,
            num_operations=5000,
            concurrency=25
        )
        
        # Should handle mixed load reasonably well
        assert result.operations_per_second > 500
        
        benchmark.print_results()
    
    @pytest.mark.performance
    def test_circuit_breaker_memory_usage(self):
        """Test circuit breaker memory usage with many instances."""
        import sys
        
        initial_objects = len([obj for obj in globals().values() if hasattr(obj, '__dict__')])
        
        # Create many circuit breakers
        breakers = []
        for i in range(1000):
            config = CircuitBreakerConfig(failure_threshold=3)
            cb = CircuitBreaker(config, f"test_{i}")
            breakers.append(cb)
            
            # Generate some activity
            try:
                cb.call(lambda: "success")
                if i % 10 == 0:  # Occasional failure
                    cb.call(lambda: exec('raise Exception("test")'))
            except:
                pass
        
        final_objects = len([obj for obj in globals().values() if hasattr(obj, '__dict__')])
        
        # Memory usage should be reasonable
        objects_created = final_objects - initial_objects
        assert objects_created < 2000  # Should not create excessive objects
        
        # Test metrics collection doesn't cause memory leaks
        for cb in breakers[:100]:  # Test subset
            metrics = cb.get_metrics()
            assert len(metrics) > 0


class TestRateLimiterPerformance:
    """Test rate limiter performance under load."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_token_bucket_throughput(self):
        """Test token bucket rate limiter throughput."""
        config = RateLimitConfig(
            max_requests=10000,
            burst_size=1000,
            refill_rate=1000  # High refill rate for testing
        )
        limiter = TokenBucketRateLimiter(config)
        
        benchmark = PerformanceBenchmark()
        
        async def rate_limit_check():
            allowed, _ = await limiter.is_allowed("perf_client")
            return allowed
        
        result = await benchmark.benchmark_async_operation(
            "RateLimiter Check",
            rate_limit_check,
            num_operations=50000,
            concurrency=100
        )
        
        # Should handle high throughput efficiently
        assert result.operations_per_second > 5000
        assert result.avg_latency_ms < 5
        
        benchmark.print_results()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_rate_limiter_multiple_clients(self):
        """Test rate limiter with multiple clients."""
        config = RateLimitConfig(max_requests=100, burst_size=50)
        limiter = TokenBucketRateLimiter(config)
        
        benchmark = PerformanceBenchmark()
        client_counter = 0
        
        async def multi_client_check():
            nonlocal client_counter
            client_counter += 1
            client_id = f"client_{client_counter % 100}"  # 100 different clients
            allowed, _ = await limiter.is_allowed(client_id)
            return allowed
        
        result = await benchmark.benchmark_async_operation(
            "RateLimiter MultiClient",
            multi_client_check,
            num_operations=20000,
            concurrency=50
        )
        
        # Should scale well with multiple clients
        assert result.operations_per_second > 2000
        
        benchmark.print_results()


class TestCachePerformance:
    """Test cache performance under various scenarios."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cache_read_performance(self):
        """Test cache read performance with high hit rate."""
        config = CacheConfig(max_size=10000, strategy=CacheStrategy.LRU)
        cache = InMemoryCache(config)
        
        # Pre-populate cache
        for i in range(1000):
            await cache.set(f"key_{i}", f"value_{i}")
        
        benchmark = PerformanceBenchmark()
        key_counter = 0
        
        async def cache_read():
            nonlocal key_counter
            key_counter += 1
            key = f"key_{key_counter % 1000}"  # Ensure high hit rate
            return await cache.get(key)
        
        result = await benchmark.benchmark_async_operation(
            "Cache Read (High Hit)",
            cache_read,
            num_operations=100000,
            concurrency=100
        )
        
        # Cache reads should be very fast
        assert result.operations_per_second > 10000
        assert result.avg_latency_ms < 1
        
        benchmark.print_results()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cache_write_performance(self):
        """Test cache write performance."""
        config = CacheConfig(max_size=50000, strategy=CacheStrategy.LRU)
        cache = InMemoryCache(config)
        
        benchmark = PerformanceBenchmark()
        write_counter = 0
        
        async def cache_write():
            nonlocal write_counter
            write_counter += 1
            key = f"write_key_{write_counter}"
            value = f"write_value_{write_counter}"
            return await cache.set(key, value)
        
        result = await benchmark.benchmark_async_operation(
            "Cache Write",
            cache_write,
            num_operations=50000,
            concurrency=50
        )
        
        # Cache writes should be reasonably fast
        assert result.operations_per_second > 5000
        assert result.avg_latency_ms < 5
        
        benchmark.print_results()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cache_eviction_performance(self):
        """Test cache performance under eviction pressure."""
        config = CacheConfig(max_size=1000, strategy=CacheStrategy.LRU)
        cache = InMemoryCache(config)
        
        benchmark = PerformanceBenchmark()
        eviction_counter = 0
        
        async def cache_with_eviction():
            nonlocal eviction_counter
            eviction_counter += 1
            # Write more items than cache can hold
            key = f"evict_key_{eviction_counter}"
            value = f"evict_value_{eviction_counter}"
            return await cache.set(key, value)
        
        result = await benchmark.benchmark_async_operation(
            "Cache Eviction",
            cache_with_eviction,
            num_operations=10000,  # 10x cache size
            concurrency=20
        )
        
        # Should handle eviction reasonably well
        assert result.operations_per_second > 1000
        assert result.success_rate > 0.95
        
        benchmark.print_results()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cache_compression_performance(self):
        """Test cache performance with compression enabled."""
        config = CacheConfig(
            max_size=1000,
            compression_enabled=True,
            compression_threshold=100  # Low threshold to test compression
        )
        cache = InMemoryCache(config)
        
        benchmark = PerformanceBenchmark()
        
        # Large values to trigger compression
        large_value = "x" * 1000
        
        async def cache_large_value():
            key = f"large_key_{time.time()}"
            await cache.set(key, large_value)
            return await cache.get(key)
        
        result = await benchmark.benchmark_async_operation(
            "Cache Compression",
            cache_large_value,
            num_operations=1000,
            concurrency=10
        )
        
        # Compression adds overhead but should still be reasonable
        assert result.operations_per_second > 100
        assert result.success_rate > 0.95
        
        benchmark.print_results()


class TestAutoScalerPerformance:
    """Test auto-scaler performance under load."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_metrics_collection_performance(self):
        """Test metrics collection performance."""
        config = ScalingConfig()
        scaler = IntelligentAutoScaler(config)
        
        benchmark = PerformanceBenchmark()
        
        async def collect_metrics():
            return await scaler.metrics_collector.collect_metrics()
        
        result = await benchmark.benchmark_async_operation(
            "Metrics Collection",
            collect_metrics,
            num_operations=1000,
            concurrency=10
        )
        
        # Metrics collection should be efficient
        assert result.operations_per_second > 100
        assert result.avg_latency_ms < 100
        
        benchmark.print_results()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_scaling_decision_performance(self):
        """Test scaling decision logic performance."""
        config = ScalingConfig()
        scaler = IntelligentAutoScaler(config)
        
        # Mock metrics
        current_metrics = {
            "cpu_utilization": 75.0,
            "memory_utilization": 60.0,
            "timestamp": time.time()
        }
        
        metrics_summary = {
            "cpu_utilization_avg": 70.0,
            "memory_utilization_avg": 65.0
        }
        
        predictions = {}
        
        benchmark = PerformanceBenchmark()
        
        async def make_decision():
            return await scaler._make_scaling_decision(
                current_metrics, metrics_summary, predictions
            )
        
        result = await benchmark.benchmark_async_operation(
            "Scaling Decision",
            make_decision,
            num_operations=10000,
            concurrency=20
        )
        
        # Decision making should be very fast
        assert result.operations_per_second > 1000
        assert result.avg_latency_ms < 10
        
        benchmark.print_results()


class TestIntegratedPerformance:
    """Test integrated system performance."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_full_stack_performance(self):
        """Test performance of all components working together."""
        
        # Initialize all components
        cb_config = CircuitBreakerConfig(failure_threshold=10)
        cb = CircuitBreaker(cb_config, "integration_perf")
        
        rl_config = RateLimitConfig(max_requests=1000, burst_size=100)
        rate_limiter = TokenBucketRateLimiter(rl_config)
        
        cache_config = CacheConfig(max_size=1000)
        cache = InMemoryCache(cache_config)
        
        benchmark = PerformanceBenchmark()
        operation_counter = 0
        
        async def integrated_operation():
            nonlocal operation_counter
            operation_counter += 1
            
            # Rate limiting check
            allowed, _ = await rate_limiter.is_allowed("integrated_client")
            if not allowed:
                return "rate_limited"
            
            # Cache check
            cache_key = f"data_{operation_counter % 100}"
            cached_value = await cache.get(cache_key)
            if cached_value:
                return cached_value
            
            # Simulated work with circuit breaker protection
            async def do_work():
                await asyncio.sleep(0.001)  # Simulate 1ms work
                return f"result_{operation_counter}"
            
            result = await cb.acall(do_work)
            await cache.set(cache_key, result)
            return result
        
        result = await benchmark.benchmark_async_operation(
            "Full Stack Integration",
            integrated_operation,
            num_operations=10000,
            concurrency=50
        )
        
        # Integrated system should maintain reasonable performance
        assert result.operations_per_second > 500
        assert result.success_rate > 0.90
        assert result.avg_latency_ms < 50
        
        benchmark.print_results()
    
    @pytest.mark.performance
    @pytest.mark.asyncio 
    async def test_stress_test(self):
        """Stress test with high load and error conditions."""
        
        # More aggressive settings for stress test
        cb_config = CircuitBreakerConfig(failure_threshold=3, timeout=0.5)
        cb = CircuitBreaker(cb_config, "stress_test")
        
        rl_config = RateLimitConfig(max_requests=100, burst_size=20)
        rate_limiter = TokenBucketRateLimiter(rl_config)
        
        benchmark = PerformanceBenchmark()
        stress_counter = 0
        
        async def stress_operation():
            nonlocal stress_counter
            stress_counter += 1
            
            # Rate limiting (will cause many rejections)
            allowed, _ = await rate_limiter.is_allowed(f"stress_client_{stress_counter % 10}")
            if not allowed:
                raise Exception("Rate limited")
            
            # Circuit breaker with failures
            async def unreliable_work():
                if stress_counter % 7 == 0:  # ~14% failure rate
                    raise Exception("Simulated failure")
                return f"stress_result_{stress_counter}"
            
            return await cb.acall(unreliable_work)
        
        result = await benchmark.benchmark_async_operation(
            "Stress Test",
            stress_operation,
            num_operations=5000,
            concurrency=100
        )
        
        # Under stress, system should degrade gracefully
        assert result.operations_per_second > 50  # Lower threshold for stress test
        # Success rate will be lower due to rate limiting and failures
        
        benchmark.print_results()


@pytest.mark.performance
def test_memory_leak_detection():
    """Test for memory leaks during extended operation."""
    import gc
    import sys
    
    # Force garbage collection and get baseline
    gc.collect()
    initial_objects = len(gc.get_objects())
    
    # Create and destroy many components
    for iteration in range(100):
        # Circuit breakers
        cb_config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(cb_config, f"memory_test_{iteration}")
        
        # Generate activity
        for i in range(10):
            try:
                cb.call(lambda: "success" if i % 3 != 0 else exec('raise Exception("test")'))
            except:
                pass
        
        # Rate limiters
        rl_config = RateLimitConfig(max_requests=10)
        rate_limiter = TokenBucketRateLimiter(rl_config)
        
        # Caches
        cache_config = CacheConfig(max_size=10)
        cache = InMemoryCache(cache_config)
        
        # Force cleanup
        del cb, rate_limiter, cache
        
        if iteration % 10 == 0:
            gc.collect()
    
    # Final cleanup and measurement
    gc.collect()
    final_objects = len(gc.get_objects())
    
    objects_growth = final_objects - initial_objects
    
    # Allow some growth but detect significant leaks
    assert objects_growth < 1000, f"Potential memory leak detected: {objects_growth} objects created"


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-m", "performance", "--tb=short"])