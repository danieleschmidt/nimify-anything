"""Comprehensive tests for advanced features including circuit breaker, rate limiter, and auto-scaler."""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import asdict

from src.nimify.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerException, 
    CircuitState, circuit_breaker_registry
)
from src.nimify.rate_limiter import (
    TokenBucketRateLimiter, SlidingWindowRateLimiter, AdaptiveRateLimiter,
    MultiTierRateLimiter, RateLimitConfig, RateLimitAlgorithm, RateLimitException
)
from src.nimify.auto_scaler import (
    IntelligentAutoScaler, ScalingConfig, ScalingAction, ScalingDecision, MetricsCollector
)
from src.nimify.caching_system import (
    InMemoryCache, MultiLevelCache, CacheManager, CacheConfig, CacheStrategy
)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        config = CircuitBreakerConfig(failure_threshold=3, timeout=30.0)
        cb = CircuitBreaker(config, "test")
        
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.config.failure_threshold == 3
        assert cb.name == "test"
    
    def test_successful_calls(self):
        """Test successful function calls through circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config, "test")
        
        def successful_function():
            return "success"
        
        result = cb.call(successful_function)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.total_successes == 1
    
    def test_failure_threshold_opens_circuit(self):
        """Test that circuit opens after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=1.0)
        cb = CircuitBreaker(config, "test")
        
        def failing_function():
            raise Exception("Test failure")
        
        # First failure
        with pytest.raises(Exception):
            cb.call(failing_function)
        assert cb.state == CircuitState.CLOSED
        
        # Second failure - should open circuit
        with pytest.raises(Exception):
            cb.call(failing_function)
        assert cb.state == CircuitState.OPEN
        
        # Third call should be rejected
        with pytest.raises(CircuitBreakerException):
            cb.call(failing_function)
    
    def test_circuit_transitions_to_half_open(self):
        """Test circuit transitions to half-open after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=0.1)
        cb = CircuitBreaker(config, "test")
        
        def failing_function():
            raise Exception("Test failure")
        
        # Trigger circuit to open
        with pytest.raises(Exception):
            cb.call(failing_function)
        assert cb.state == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Next call should transition to half-open
        def successful_function():
            return "success"
        
        result = cb.call(successful_function)
        assert result == "success"
        assert cb.state == CircuitState.HALF_OPEN
    
    def test_half_open_to_closed_transition(self):
        """Test transition from half-open to closed after successful calls."""
        config = CircuitBreakerConfig(failure_threshold=1, success_threshold=2, timeout=0.1)
        cb = CircuitBreaker(config, "test")
        
        # Force circuit to open
        cb._transition_to_open()
        time.sleep(0.2)
        
        # Force to half-open
        cb._transition_to_half_open()
        
        def successful_function():
            return "success"
        
        # First success in half-open
        result = cb.call(successful_function)
        assert result == "success"
        assert cb.state == CircuitState.HALF_OPEN
        
        # Second success should close circuit
        result = cb.call(successful_function)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_async_circuit_breaker(self):
        """Test circuit breaker with async functions."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker(config, "async_test")
        
        async def async_function():
            return "async_success"
        
        result = await cb.acall(async_function)
        assert result == "async_success"
    
    def test_slow_call_detection(self):
        """Test slow call detection."""
        config = CircuitBreakerConfig(slow_call_threshold=0.1)
        cb = CircuitBreaker(config, "slow_test")
        
        def slow_function():
            time.sleep(0.2)  # Slower than threshold
            return "slow_result"
        
        result = cb.call(slow_function)
        assert result == "slow_result"
        # Should have recorded partial failure for slow call
        assert cb.failure_count > 0
    
    def test_circuit_breaker_metrics(self):
        """Test circuit breaker metrics collection."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker(config, "metrics_test")
        
        def success_func():
            return "success"
        
        def fail_func():
            raise Exception("failure")
        
        # Generate some activity
        cb.call(success_func)
        try:
            cb.call(fail_func)
        except:
            pass
        cb.call(success_func)
        
        metrics = cb.get_metrics()
        assert metrics["name"] == "metrics_test"
        assert metrics["total_calls"] == 3
        assert metrics["total_successes"] == 2
        assert metrics["total_failures"] == 1
        assert 0 <= metrics["success_rate"] <= 1
    
    def test_circuit_breaker_registry(self):
        """Test circuit breaker registry functionality."""
        config = CircuitBreakerConfig(failure_threshold=3)
        
        # Get or create circuit breaker
        cb1 = circuit_breaker_registry.get_or_create("test_service", config)
        cb2 = circuit_breaker_registry.get_or_create("test_service")
        
        # Should be the same instance
        assert cb1 is cb2
        
        # Test registry metrics
        metrics = circuit_breaker_registry.get_all_metrics()
        assert "test_service" in metrics
        
        # Test reset all
        circuit_breaker_registry.reset_all()
        assert cb1.state == CircuitState.CLOSED
        assert cb1.failure_count == 0


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def test_token_bucket_initialization(self):
        """Test token bucket rate limiter initialization."""
        config = RateLimitConfig(max_requests=100, window_size=60)
        limiter = TokenBucketRateLimiter(config)
        
        assert limiter.config.max_requests == 100
        assert limiter.config.window_size == 60
    
    @pytest.mark.asyncio
    async def test_token_bucket_allows_requests(self):
        """Test that token bucket allows requests within limits."""
        config = RateLimitConfig(max_requests=10, burst_size=10, refill_rate=1.0)
        limiter = TokenBucketRateLimiter(config)
        
        # Should allow requests up to burst size
        for i in range(5):
            allowed, retry_after = await limiter.is_allowed("test_client")
            assert allowed is True
            assert retry_after is None
    
    @pytest.mark.asyncio
    async def test_token_bucket_rejects_excess_requests(self):
        """Test that token bucket rejects excess requests."""
        config = RateLimitConfig(max_requests=5, burst_size=5, refill_rate=0.1)
        limiter = TokenBucketRateLimiter(config)
        
        # Use up all tokens
        for i in range(5):
            allowed, _ = await limiter.is_allowed("test_client")
            assert allowed is True
        
        # Next request should be rejected
        allowed, retry_after = await limiter.is_allowed("test_client")
        assert allowed is False
        assert retry_after is not None
        assert retry_after > 0
    
    @pytest.mark.asyncio
    async def test_sliding_window_rate_limiter(self):
        """Test sliding window rate limiter."""
        config = RateLimitConfig(
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            max_requests=3,
            window_size=1
        )
        limiter = SlidingWindowRateLimiter(config)
        
        # Allow requests within limit
        for i in range(3):
            allowed, _ = await limiter.is_allowed("test_client")
            assert allowed is True
        
        # Reject excess request
        allowed, retry_after = await limiter.is_allowed("test_client")
        assert allowed is False
        assert retry_after is not None
    
    @pytest.mark.asyncio
    async def test_adaptive_rate_limiter(self):
        """Test adaptive rate limiter adjusts limits based on load."""
        config = RateLimitConfig(
            algorithm=RateLimitAlgorithm.ADAPTIVE,
            max_requests=100,
            min_requests=10,
            max_requests_adaptive=200
        )
        limiter = AdaptiveRateLimiter(config)
        
        # Should work with default limit
        allowed, _ = await limiter.is_allowed("test_client")
        assert allowed is True
        
        # Test metrics collection
        metrics = limiter.get_metrics()
        assert "adaptive_current_limit" in metrics
    
    @pytest.mark.asyncio
    async def test_multi_tier_rate_limiter(self):
        """Test multi-tier rate limiter with global and per-client limits."""
        global_config = RateLimitConfig(max_requests=1000, window_size=60)
        client_config = RateLimitConfig(max_requests=10, window_size=60)
        
        limiter = MultiTierRateLimiter(global_config, client_config)
        
        # Should allow requests within both global and client limits
        for i in range(5):
            allowed, retry_after, limit_type = await limiter.is_allowed("client1")
            assert allowed is True
            assert limit_type == "allowed"
        
        # Test penalties
        client_limiter = limiter.client_limiters.get("client1")
        if client_limiter:
            # Exhaust client limit to trigger penalty
            for i in range(10):
                await limiter.is_allowed("client1")
    
    def test_rate_limiter_metrics(self):
        """Test rate limiter metrics collection."""
        config = RateLimitConfig(max_requests=100)
        limiter = TokenBucketRateLimiter(config)
        
        metrics = limiter.get_metrics()
        assert "algorithm" in metrics
        assert "total_requests" in metrics
        assert "total_rejected" in metrics
        assert "overall_rejection_rate" in metrics
        assert "config" in metrics


class TestAutoScaler:
    """Test auto-scaling functionality."""
    
    def test_scaling_config_initialization(self):
        """Test scaling configuration initialization."""
        config = ScalingConfig(min_replicas=2, max_replicas=10)
        
        assert config.min_replicas == 2
        assert config.max_replicas == 10
        assert config.target_cpu_utilization == 70.0
    
    @pytest.mark.asyncio
    async def test_metrics_collector(self):
        """Test metrics collection."""
        config = ScalingConfig()
        collector = MetricsCollector(config)
        
        metrics = await collector.collect_metrics()
        
        # Should have basic system metrics
        assert "timestamp" in metrics
        assert "cpu_utilization" in metrics
        assert "memory_utilization" in metrics
    
    def test_scaling_decision_creation(self):
        """Test scaling decision creation."""
        decision = ScalingDecision(
            action=ScalingAction.SCALE_UP,
            current_replicas=3,
            target_replicas=5,
            confidence=0.8,
            reasoning=["High CPU usage"],
            triggered_by=[],
            timestamp=time.time()
        )
        
        assert decision.action == ScalingAction.SCALE_UP
        assert decision.target_replicas == 5
        assert decision.confidence == 0.8
    
    @pytest.mark.asyncio
    async def test_auto_scaler_initialization(self):
        """Test auto-scaler initialization."""
        config = ScalingConfig(min_replicas=2, max_replicas=10)
        scaler = IntelligentAutoScaler(config, "test-namespace")
        
        assert scaler.current_replicas == config.min_replicas
        assert scaler.namespace == "test-namespace"
        assert scaler.is_running is False
    
    @pytest.mark.asyncio
    async def test_scaling_decision_logic(self):
        """Test scaling decision logic."""
        config = ScalingConfig(
            target_cpu_utilization=50.0,
            target_memory_utilization=60.0
        )
        scaler = IntelligentAutoScaler(config)
        
        # Mock high CPU metrics
        high_cpu_metrics = {
            "cpu_utilization": 80.0,
            "memory_utilization": 40.0,
            "timestamp": time.time()
        }
        
        metrics_summary = {
            "cpu_utilization_avg": 75.0,
            "memory_utilization_avg": 45.0
        }
        
        decision = await scaler._make_scaling_decision(
            high_cpu_metrics, metrics_summary, {}
        )
        
        # Should decide to scale up due to high CPU
        assert decision.action in [ScalingAction.SCALE_UP, ScalingAction.MAINTAIN]
    
    def test_auto_scaler_status(self):
        """Test auto-scaler status reporting."""
        config = ScalingConfig()
        scaler = IntelligentAutoScaler(config)
        
        status = scaler.get_status()
        
        assert "is_running" in status
        assert "current_replicas" in status
        assert "min_replicas" in status
        assert "max_replicas" in status
        assert "recent_decisions" in status
    
    def test_auto_scaler_recommendations(self):
        """Test auto-scaler recommendations."""
        config = ScalingConfig()
        scaler = IntelligentAutoScaler(config)
        
        recommendations = scaler.get_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0


class TestCachingSystem:
    """Test caching system functionality."""
    
    def test_cache_config_initialization(self):
        """Test cache configuration initialization."""
        config = CacheConfig(
            max_size=1000,
            ttl_seconds=3600,
            strategy=CacheStrategy.LRU
        )
        
        assert config.max_size == 1000
        assert config.ttl_seconds == 3600
        assert config.strategy == CacheStrategy.LRU
    
    @pytest.mark.asyncio
    async def test_in_memory_cache_basic_operations(self):
        """Test basic cache operations."""
        config = CacheConfig(max_size=100, ttl_seconds=60)
        cache = InMemoryCache(config)
        
        # Test set and get
        success = await cache.set("test_key", "test_value")
        assert success is True
        
        value = await cache.get("test_key")
        assert value == "test_value"
        
        # Test delete
        success = await cache.delete("test_key")
        assert success is True
        
        value = await cache.get("test_key")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self):
        """Test cache TTL expiration."""
        config = CacheConfig(max_size=100, ttl_seconds=1)
        cache = InMemoryCache(config)
        
        await cache.set("expiring_key", "expiring_value", ttl=1)
        
        # Should be available immediately
        value = await cache.get("expiring_key")
        assert value == "expiring_value"
        
        # Wait for expiration
        await asyncio.sleep(1.5)
        
        value = await cache.get("expiring_key")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        """Test cache eviction when max size is reached."""
        config = CacheConfig(max_size=3, strategy=CacheStrategy.LRU)
        cache = InMemoryCache(config)
        
        # Fill cache to capacity
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        
        # Access key1 to make it more recently used
        await cache.get("key1")
        
        # Add one more item - should evict key2 (least recently used)
        await cache.set("key4", "value4")
        
        # key2 should be evicted
        assert await cache.get("key2") is None
        # Others should still be there
        assert await cache.get("key1") == "value1"
        assert await cache.get("key3") == "value3"
        assert await cache.get("key4") == "value4"
    
    @pytest.mark.asyncio
    async def test_multi_level_cache(self):
        """Test multi-level cache hierarchy."""
        config = CacheConfig(
            enable_multilevel=True,
            l1_size=5,
            l2_size=10
        )
        
        with patch('src.nimify.caching_system.RedisCache') as mock_redis:
            # Mock Redis to avoid dependency
            mock_redis.return_value = AsyncMock()
            
            cache = MultiLevelCache(config)
            
            # Test basic operations
            await cache.set("test_key", "test_value")
            value = await cache.get("test_key")
            
            # Should get value from L1 cache
            assert value == "test_value"
    
    @pytest.mark.asyncio
    async def test_cache_manager_decorator(self):
        """Test cache manager decorator functionality."""
        config = CacheConfig(max_size=100)
        manager = CacheManager(config)
        
        call_count = 0
        
        @manager.cached(ttl=60)
        async def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call should execute function
        result1 = await expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call should use cache
        result2 = await expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increase
        
        # Different parameters should execute function again
        result3 = await expensive_function(10)
        assert result3 == 20
        assert call_count == 2
    
    def test_cache_stats(self):
        """Test cache statistics collection."""
        config = CacheConfig(max_size=100)
        cache = InMemoryCache(config)
        
        stats = cache.get_stats()
        
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "strategy" in stats
        assert "items" in stats
    
    @pytest.mark.asyncio
    async def test_cache_bulk_operations(self):
        """Test cache bulk operations."""
        config = CacheConfig(max_size=100)
        manager = CacheManager(config)
        
        # Test bulk set
        items = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        
        success_count = await manager.bulk_set(items)
        assert success_count == 3
        
        # Test bulk get
        results = await manager.bulk_get(["key1", "key2", "key3", "nonexistent"])
        
        assert len(results) == 3
        assert results["key1"] == "value1"
        assert results["key2"] == "value2"
        assert results["key3"] == "value3"
        assert "nonexistent" not in results


class TestIntegration:
    """Integration tests combining multiple advanced features."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_with_rate_limiter(self):
        """Test circuit breaker and rate limiter working together."""
        # Circuit breaker config
        cb_config = CircuitBreakerConfig(failure_threshold=2, timeout=1.0)
        cb = CircuitBreaker(cb_config, "integration_test")
        
        # Rate limiter config
        rl_config = RateLimitConfig(max_requests=5, window_size=1)
        rate_limiter = TokenBucketRateLimiter(rl_config)
        
        call_count = 0
        
        @cb
        async def protected_function():
            nonlocal call_count
            call_count += 1
            
            # Check rate limit first
            allowed, _ = await rate_limiter.is_allowed("test_client")
            if not allowed:
                raise RateLimitException("Rate limit exceeded")
            
            if call_count <= 3:
                return f"success_{call_count}"
            else:
                raise Exception("Service failure")
        
        # First few calls should succeed
        for i in range(3):
            result = await protected_function()
            assert result == f"success_{i+1}"
        
        # Rate limit should kick in
        with pytest.raises(RateLimitException):
            await protected_function()
    
    @pytest.mark.asyncio
    async def test_auto_scaler_with_caching(self):
        """Test auto-scaler using cached metrics."""
        cache_config = CacheConfig(max_size=100, ttl_seconds=30)
        cache_manager = CacheManager(cache_config)
        
        scaling_config = ScalingConfig(min_replicas=2, max_replicas=10)
        scaler = IntelligentAutoScaler(scaling_config)
        
        # Cache some metrics
        cached_metrics = {
            "cpu_utilization": 75.0,
            "memory_utilization": 60.0,
            "timestamp": time.time()
        }
        
        await cache_manager.set("current_metrics", cached_metrics)
        
        # Retrieve and use cached metrics
        metrics = await cache_manager.get("current_metrics")
        assert metrics is not None
        assert metrics["cpu_utilization"] == 75.0
    
    def test_comprehensive_monitoring(self):
        """Test comprehensive monitoring across all components."""
        # Initialize all components
        cb_config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(cb_config, "monitor_test")
        
        rl_config = RateLimitConfig(max_requests=100)
        rate_limiter = TokenBucketRateLimiter(rl_config)
        
        scaling_config = ScalingConfig()
        scaler = IntelligentAutoScaler(scaling_config)
        
        cache_config = CacheConfig(max_size=100)
        cache = InMemoryCache(cache_config)
        
        # Collect metrics from all components
        all_metrics = {
            "circuit_breaker": cb.get_metrics(),
            "rate_limiter": rate_limiter.get_metrics(),
            "auto_scaler": scaler.get_status(),
            "cache": cache.get_stats()
        }
        
        # Verify all metrics are collected
        assert "circuit_breaker" in all_metrics
        assert "rate_limiter" in all_metrics
        assert "auto_scaler" in all_metrics
        assert "cache" in all_metrics
        
        # Verify structure of each metric set
        assert "state" in all_metrics["circuit_breaker"]
        assert "algorithm" in all_metrics["rate_limiter"]
        assert "current_replicas" in all_metrics["auto_scaler"]
        assert "hit_rate" in all_metrics["cache"]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])