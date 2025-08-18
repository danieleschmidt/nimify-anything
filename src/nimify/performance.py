"""Performance optimization and monitoring utilities."""

import asyncio
import hashlib
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import psutil
from cachetools import LRUCache, TTLCache

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    latency_p50: float
    latency_p95: float  
    latency_p99: float
    throughput_rps: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: float | None = None
    cache_hit_rate: float = 0.0
    concurrent_requests: int = 0


class MetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.request_timestamps = deque(maxlen=window_size)
        self.cache_hits = 0
        self.cache_misses = 0
        self.concurrent_requests = 0
        self._lock = threading.Lock()
    
    def record_request(self, latency_ms: float):
        """Record a request with its latency."""
        with self._lock:
            self.latencies.append(latency_ms)
            self.request_timestamps.append(time.time())
    
    def record_cache_hit(self):
        """Record cache hit."""
        with self._lock:
            self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record cache miss."""
        with self._lock:
            self.cache_misses += 1
    
    def increment_concurrent(self):
        """Increment concurrent request counter."""
        with self._lock:
            self.concurrent_requests += 1
    
    def decrement_concurrent(self):
        """Decrement concurrent request counter."""
        with self._lock:
            self.concurrent_requests = max(0, self.concurrent_requests - 1)
    
    def get_metrics(self) -> PerformanceMetrics:
        """Calculate current performance metrics."""
        with self._lock:
            # Calculate latency percentiles
            if self.latencies:
                sorted_latencies = sorted(self.latencies)
                n = len(sorted_latencies)
                p50 = sorted_latencies[int(n * 0.5)]
                p95 = sorted_latencies[int(n * 0.95)]
                p99 = sorted_latencies[int(n * 0.99)]
            else:
                p50 = p95 = p99 = 0.0
            
            # Calculate throughput (requests per second)
            now = time.time()
            recent_requests = [ts for ts in self.request_timestamps if now - ts <= 60]
            throughput = len(recent_requests) / 60.0
            
            # Calculate cache hit rate
            total_cache_requests = self.cache_hits + self.cache_misses
            cache_hit_rate = self.cache_hits / total_cache_requests if total_cache_requests > 0 else 0.0
            
            # Get system metrics
            memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            cpu_usage = psutil.cpu_percent()
            
            # GPU metrics (if available)
            gpu_usage = self._get_gpu_usage()
            
            return PerformanceMetrics(
                latency_p50=p50,
                latency_p95=p95,
                latency_p99=p99,
                throughput_rps=throughput,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                gpu_usage_percent=gpu_usage,
                cache_hit_rate=cache_hit_rate,
                concurrent_requests=self.concurrent_requests
            )
    
    def _get_gpu_usage(self) -> float | None:
        """Get GPU usage if available."""
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            info = nvml.nvmlDeviceGetUtilizationRates(handle)
            return float(info.gpu)
        except:
            return None


class ModelCache:
    """Intelligent caching for model inputs and outputs."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache = TTLCache(maxsize=max_size, ttl=ttl_seconds)
        self.lru_cache = LRUCache(maxsize=max_size // 2)
        self._lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
    
    def _hash_input(self, input_data: list[list[float]]) -> str:
        """Create hash of input data for caching."""
        # Convert to numpy array for consistent hashing
        arr = np.array(input_data, dtype=np.float32)
        return hashlib.md5(arr.tobytes()).hexdigest()
    
    def get(self, input_data: list[list[float]]) -> list[list[float]] | None:
        """Get cached result for input."""
        key = self._hash_input(input_data)
        
        with self._lock:
            # Try TTL cache first (recent results)
            if key in self.cache:
                self.hit_count += 1
                return self.cache[key]
            
            # Try LRU cache (frequently used results)
            if key in self.lru_cache:
                self.hit_count += 1
                # Promote to TTL cache
                result = self.lru_cache[key]
                self.cache[key] = result
                return result
            
            self.miss_count += 1
            return None
    
    def put(self, input_data: list[list[float]], result: list[list[float]]):
        """Cache result for input."""
        key = self._hash_input(input_data)
        
        with self._lock:
            self.cache[key] = result
            self.lru_cache[key] = result
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def clear(self):
        """Clear all caches."""
        with self._lock:
            self.cache.clear()
            self.lru_cache.clear()


class BatchProcessor:
    """Dynamic batching for improved throughput."""
    
    def __init__(self, max_batch_size: int = 32, max_wait_ms: int = 10):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.pending_requests = []
        self.batch_condition = asyncio.Condition()
        self.processing = False
        
    async def add_request(self, request_data: Any, future: asyncio.Future):
        """Add request to batch queue."""
        async with self.batch_condition:
            self.pending_requests.append((request_data, future))
            
            # Trigger batch processing if conditions are met
            if (len(self.pending_requests) >= self.max_batch_size or 
                (self.pending_requests and not self.processing)):
                self.batch_condition.notify()
    
    async def process_batches(self, inference_func: Callable):
        """Process batches continuously."""
        while True:
            async with self.batch_condition:
                # Wait for requests or timeout
                try:
                    await asyncio.wait_for(
                        self.batch_condition.wait_for(lambda: self.pending_requests),
                        timeout=self.max_wait_ms / 1000.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                if not self.pending_requests:
                    continue
                
                # Extract batch
                batch_size = min(len(self.pending_requests), self.max_batch_size)
                batch = self.pending_requests[:batch_size]
                self.pending_requests = self.pending_requests[batch_size:]
                
                self.processing = True
            
            # Process batch outside of lock
            try:
                await self._process_batch(batch, inference_func)
            finally:
                async with self.batch_condition:
                    self.processing = False
    
    async def _process_batch(self, batch: list[tuple[Any, asyncio.Future]], inference_func: Callable):
        """Process a single batch."""
        if not batch:
            return
        
        try:
            # Combine inputs
            inputs = [item[0] for item in batch]
            futures = [item[1] for item in batch]
            
            # Run inference
            results = await inference_func(inputs)
            
            # Distribute results
            for _i, (future, result) in enumerate(zip(futures, results, strict=False)):
                if not future.done():
                    future.set_result(result)
        
        except Exception as e:
            # Set exception for all futures
            for _, future in batch:
                if not future.done():
                    future.set_exception(e)


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        with self._lock:
            if self.state == "CLOSED":
                return True
            
            if self.state == "OPEN":
                if (time.time() - self.last_failure_time) > self.timeout_seconds:
                    self.state = "HALF_OPEN"
                    return True
                return False
            
            return self.state == "HALF_OPEN"
    
    def record_success(self):
        """Record successful execution."""
        with self._lock:
            self.failure_count = 0
            self.state = "CLOSED"
    
    def record_failure(self):
        """Record failed execution."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class ResourcePool:
    """Pool for expensive resources (e.g., model sessions)."""
    
    def __init__(self, factory: Callable, min_size: int = 1, max_size: int = 10):
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.pool = asyncio.Queue(maxsize=max_size)
        self.created_count = 0
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the pool with minimum resources."""
        for _ in range(self.min_size):
            resource = await self._create_resource()
            await self.pool.put(resource)
    
    async def _create_resource(self):
        """Create a new resource."""
        async with self._lock:
            if self.created_count >= self.max_size:
                raise RuntimeError("Maximum pool size reached")
            
            resource = await self.factory()
            self.created_count += 1
            logger.debug(f"Created resource, pool size: {self.created_count}")
            return resource
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a resource from the pool."""
        try:
            # Try to get existing resource
            resource = await asyncio.wait_for(self.pool.get(), timeout=1.0)
        except asyncio.TimeoutError:
            # Create new resource if pool is empty and we haven't reached max
            if self.created_count < self.max_size:
                resource = await self._create_resource()
            else:
                # Wait longer for a resource
                resource = await self.pool.get()
        
        try:
            yield resource
        finally:
            # Return resource to pool
            try:
                self.pool.put_nowait(resource)
            except asyncio.QueueFull:
                # Pool is full, discard resource
                logger.debug("Pool full, discarding resource")


class AdaptiveScaler:
    """Adaptive scaling based on load metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.scale_up_threshold = 0.8  # CPU/memory threshold to scale up
        self.scale_down_threshold = 0.3  # CPU/memory threshold to scale down
        self.latency_threshold_ms = 200  # Latency threshold to scale up
        
    def should_scale_up(self) -> bool:
        """Determine if scaling up is needed."""
        metrics = self.metrics_collector.get_metrics()
        
        # Scale up conditions
        conditions = [
            metrics.cpu_usage_percent > self.scale_up_threshold * 100,
            metrics.memory_usage_mb > 500,  # Arbitrary threshold
            metrics.latency_p95 > self.latency_threshold_ms,
            metrics.concurrent_requests > 50  # High concurrency
        ]
        
        # Scale up if any 2 conditions are met
        return sum(conditions) >= 2
    
    def should_scale_down(self) -> bool:
        """Determine if scaling down is needed."""
        metrics = self.metrics_collector.get_metrics()
        
        # Scale down conditions (all must be met)
        conditions = [
            metrics.cpu_usage_percent < self.scale_down_threshold * 100,
            metrics.latency_p95 < self.latency_threshold_ms / 2,
            metrics.concurrent_requests < 10
        ]
        
        return all(conditions)
    
    def get_recommended_replicas(self, current_replicas: int) -> int:
        """Get recommended number of replicas."""
        if self.should_scale_up():
            return min(current_replicas * 2, 10)  # Max 10 replicas
        elif self.should_scale_down() and current_replicas > 1:
            return max(current_replicas // 2, 1)  # Min 1 replica
        else:
            return current_replicas


# Global instances
metrics_collector = MetricsCollector()
model_cache = ModelCache()
circuit_breaker = CircuitBreaker()
adaptive_scaler = AdaptiveScaler(metrics_collector)