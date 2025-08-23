"""Advanced performance optimization features for NIM services."""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache replacement strategies."""
    LRU = "lru"
    LFU = "lfu"
    ADAPTIVE = "adaptive"
    TTL = "ttl"


class OptimizationTarget(Enum):
    """Optimization targets."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    COST = "cost"
    BALANCED = "balanced"


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    enable_request_batching: bool = True
    max_batch_size: int = 32
    batch_timeout_ms: int = 10
    enable_response_caching: bool = True
    cache_size_mb: int = 512
    cache_ttl_seconds: int = 3600
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    enable_model_optimization: bool = True
    enable_connection_pooling: bool = True
    connection_pool_size: int = 50
    enable_async_processing: bool = True
    worker_threads: int = 8
    optimization_target: OptimizationTarget = OptimizationTarget.BALANCED


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = None
    size_bytes: int = 0
    ttl: Optional[float] = None


class AdaptiveCache:
    """High-performance adaptive cache with multiple eviction strategies."""
    
    def __init__(self, max_size_mb: int = 512, default_ttl: int = 3600):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        self.default_ttl = default_ttl
        self.entries = {}
        self.access_order = deque()  # For LRU
        self.frequency_counter = defaultdict(int)  # For LFU
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with performance tracking."""
        entry = self.entries.get(key)
        
        if entry is None:
            self.miss_count += 1
            return None
        
        # Check TTL expiration
        if entry.ttl and time.time() > entry.ttl:
            await self.delete(key)
            self.miss_count += 1
            return None
        
        # Update access patterns
        entry.access_count += 1
        entry.last_access = time.time()
        self.frequency_counter[key] += 1
        
        # Update LRU order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        self.hit_count += 1
        return entry.value
    
    async def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put value in cache with intelligent eviction."""
        # Calculate size
        size_bytes = self._estimate_size(value)
        
        # Check if single item is too large
        if size_bytes > self.max_size_bytes:
            logger.warning(f"Item too large for cache: {size_bytes} bytes")
            return False
        
        # Calculate TTL
        ttl_timestamp = None
        if ttl is not None:
            ttl_timestamp = time.time() + ttl
        elif self.default_ttl:
            ttl_timestamp = time.time() + self.default_ttl
        
        # Remove existing entry if present
        if key in self.entries:
            await self.delete(key)
        
        # Ensure space available
        while self.current_size_bytes + size_bytes > self.max_size_bytes:
            evicted = await self._evict_one()
            if not evicted:
                logger.warning("Could not evict entry for new cache item")
                return False
        
        # Create and store entry
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            size_bytes=size_bytes,
            ttl=ttl_timestamp
        )
        
        self.entries[key] = entry
        self.current_size_bytes += size_bytes
        self.access_order.append(key)
        self.frequency_counter[key] = 1
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        entry = self.entries.pop(key, None)
        if entry:
            self.current_size_bytes -= entry.size_bytes
            if key in self.access_order:
                self.access_order.remove(key)
            self.frequency_counter.pop(key, None)
            return True
        return False
    
    async def _evict_one(self) -> bool:
        """Evict one entry using adaptive strategy."""
        if not self.entries:
            return False
        
        # Adaptive eviction strategy
        hit_rate = self.hit_count / max(1, self.hit_count + self.miss_count)
        
        if hit_rate > 0.8:
            # High hit rate - use LRU to evict old items
            victim_key = await self._find_lru_victim()
        elif hit_rate < 0.3:
            # Low hit rate - use LFU to evict less popular items
            victim_key = await self._find_lfu_victim()
        else:
            # Medium hit rate - use TTL-based eviction
            victim_key = await self._find_ttl_victim()
        
        if victim_key:
            await self.delete(victim_key)
            self.eviction_count += 1
            return True
        
        return False
    
    async def _find_lru_victim(self) -> Optional[str]:
        """Find least recently used item."""
        if self.access_order:
            return self.access_order[0]
        return None
    
    async def _find_lfu_victim(self) -> Optional[str]:
        """Find least frequently used item."""
        if self.frequency_counter:
            return min(self.frequency_counter.items(), key=lambda x: x[1])[0]
        return None
    
    async def _find_ttl_victim(self) -> Optional[str]:
        """Find expired or soon-to-expire item."""
        current_time = time.time()
        
        # First, find already expired items
        for key, entry in self.entries.items():
            if entry.ttl and current_time > entry.ttl:
                return key
        
        # Then find items expiring soonest
        candidates = [(key, entry) for key, entry in self.entries.items() if entry.ttl]
        if candidates:
            return min(candidates, key=lambda x: x[1].ttl)[0]
        
        # Fallback to LRU
        return await self._find_lru_victim()
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of cached value in bytes."""
        if isinstance(value, str):
            return len(value.encode('utf-8'))
        elif isinstance(value, (int, float)):
            return 8
        elif isinstance(value, list):
            return sum(self._estimate_size(item) for item in value[:10]) * (len(value) / min(10, len(value)))
        elif isinstance(value, dict):
            sample_items = list(value.items())[:5]
            sample_size = sum(self._estimate_size(k) + self._estimate_size(v) for k, v in sample_items)
            return sample_size * (len(value) / min(5, len(value)))
        else:
            # Default estimate
            return 64
    
    def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        hit_rate = self.hit_count / max(1, self.hit_count + self.miss_count)
        
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "eviction_count": self.eviction_count,
            "current_entries": len(self.entries),
            "current_size_mb": self.current_size_bytes / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "utilization": self.current_size_bytes / self.max_size_bytes,
            "avg_entry_size_bytes": self.current_size_bytes / max(1, len(self.entries))
        }
    
    async def clear(self):
        """Clear all cache entries."""
        self.entries.clear()
        self.access_order.clear()
        self.frequency_counter.clear()
        self.current_size_bytes = 0


class RequestBatcher:
    """Intelligent request batching for improved throughput."""
    
    def __init__(self, max_batch_size: int = 32, timeout_ms: int = 10):
        self.max_batch_size = max_batch_size
        self.timeout_seconds = timeout_ms / 1000.0
        self.pending_requests = []
        self.batch_count = 0
        self.total_requests = 0
        self.total_batches = 0
        
    async def add_request(self, request_data: Any, response_future: asyncio.Future):
        """Add request to current batch."""
        self.pending_requests.append((request_data, response_future))
        self.total_requests += 1
        
        # Trigger batch processing if full or timeout
        if len(self.pending_requests) >= self.max_batch_size:
            await self._process_batch()
        else:
            # Set timeout for partial batch
            asyncio.create_task(self._batch_timeout())
    
    async def _batch_timeout(self):
        """Process partial batch after timeout."""
        await asyncio.sleep(self.timeout_seconds)
        if self.pending_requests:
            await self._process_batch()
    
    async def _process_batch(self):
        """Process current batch of requests."""
        if not self.pending_requests:
            return
        
        batch = self.pending_requests.copy()
        self.pending_requests.clear()
        self.total_batches += 1
        
        try:
            # Extract request data
            request_data = [req[0] for req in batch]
            
            # Process batch (mock implementation)
            batch_results = await self._process_batch_inference(request_data)
            
            # Distribute results to futures
            for i, (_, response_future) in enumerate(batch):
                if i < len(batch_results):
                    response_future.set_result(batch_results[i])
                else:
                    response_future.set_exception(Exception("Batch processing error"))
        
        except Exception as e:
            # Set exception for all futures in batch
            for _, response_future in batch:
                if not response_future.done():
                    response_future.set_exception(e)
    
    async def _process_batch_inference(self, batch_data: list[Any]) -> list[Any]:
        """Process batch inference (mock implementation)."""
        # Simulate batch processing
        await asyncio.sleep(0.01)  # 10ms processing time
        
        # Mock results
        results = []
        for data in batch_data:
            if isinstance(data, list) and data:
                # Mock prediction: multiply by 2
                result = [[x * 2 for x in row] for row in data]
                results.append(result)
            else:
                results.append([])
        
        return results
    
    def get_batch_stats(self) -> dict[str, Any]:
        """Get batching statistics."""
        avg_batch_size = self.total_requests / max(1, self.total_batches)
        
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "avg_batch_size": avg_batch_size,
            "pending_requests": len(self.pending_requests),
            "batch_efficiency": avg_batch_size / self.max_batch_size
        }


class AsyncConnectionPool:
    """High-performance async connection pool for external services."""
    
    def __init__(self, max_connections: int = 50, connection_timeout: int = 30):
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.available_connections = deque()
        self.active_connections = set()
        self.total_created = 0
        self.total_acquired = 0
        self.total_released = 0
        self._lock = asyncio.Lock()
    
    async def acquire_connection(self) -> Any:
        """Acquire connection from pool."""
        async with self._lock:
            # Try to get existing connection
            if self.available_connections:
                conn = self.available_connections.popleft()
                self.active_connections.add(conn)
                self.total_acquired += 1
                return conn
            
            # Create new connection if under limit
            if len(self.active_connections) < self.max_connections:
                conn = await self._create_connection()
                self.active_connections.add(conn)
                self.total_created += 1
                self.total_acquired += 1
                return conn
            
            # Wait for connection to become available
            # (In production, would implement proper waiting queue)
            await asyncio.sleep(0.01)
            return await self.acquire_connection()
    
    async def release_connection(self, connection: Any):
        """Release connection back to pool."""
        async with self._lock:
            if connection in self.active_connections:
                self.active_connections.remove(connection)
                
                # Check if connection is still healthy
                if await self._is_connection_healthy(connection):
                    self.available_connections.append(connection)
                else:
                    await self._close_connection(connection)
                
                self.total_released += 1
    
    async def _create_connection(self) -> Any:
        """Create new connection (mock implementation)."""
        # Mock connection object
        return {
            "id": self.total_created,
            "created_at": time.time(),
            "healthy": True
        }
    
    async def _is_connection_healthy(self, connection: Any) -> bool:
        """Check if connection is healthy (mock implementation)."""
        return connection.get("healthy", True)
    
    async def _close_connection(self, connection: Any):
        """Close connection (mock implementation)."""
        connection["healthy"] = False
    
    def get_pool_stats(self) -> dict[str, Any]:
        """Get connection pool statistics."""
        return {
            "max_connections": self.max_connections,
            "active_connections": len(self.active_connections),
            "available_connections": len(self.available_connections),
            "total_created": self.total_created,
            "total_acquired": self.total_acquired,
            "total_released": self.total_released,
            "pool_utilization": len(self.active_connections) / self.max_connections
        }


class PerformanceOptimizer:
    """Main performance optimization engine."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.cache = AdaptiveCache(config.cache_size_mb, config.cache_ttl_seconds) if config.enable_response_caching else None
        self.batcher = RequestBatcher(config.max_batch_size, config.batch_timeout_ms) if config.enable_request_batching else None
        self.connection_pool = AsyncConnectionPool(config.connection_pool_size) if config.enable_connection_pooling else None
        
        # Performance metrics
        self.optimization_metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "batch_savings": 0,
            "optimized_requests": 0,
            "start_time": time.time()
        }
    
    async def optimize_request(self, request_key: str, request_data: Any, processor: Callable) -> Any:
        """Apply comprehensive optimizations to request processing."""
        
        # Check cache first if enabled
        if self.cache:
            cached_result = await self.cache.get(request_key)
            if cached_result is not None:
                self.optimization_metrics["cache_hits"] += 1
                return cached_result
            else:
                self.optimization_metrics["cache_misses"] += 1
        
        # Process with batching if enabled
        if self.batcher:
            result = await self._process_with_batching(request_data, processor)
        else:
            result = await processor(request_data)
        
        # Cache result if enabled
        if self.cache and result is not None:
            await self.cache.put(request_key, result)
        
        self.optimization_metrics["optimized_requests"] += 1
        return result
    
    async def _process_with_batching(self, request_data: Any, processor: Callable) -> Any:
        """Process request with intelligent batching."""
        # Create future for this request
        response_future = asyncio.Future()
        
        # Add to batch
        await self.batcher.add_request(request_data, response_future)
        
        # Wait for result
        result = await response_future
        self.optimization_metrics["batch_savings"] += 1
        
        return result
    
    async def warmup_cache(self, common_requests: list[tuple[str, Any, Callable]]):
        """Pre-warm cache with common requests."""
        if not self.cache:
            return
        
        logger.info(f"Warming up cache with {len(common_requests)} common requests")
        
        tasks = []
        for request_key, request_data, processor in common_requests:
            task = self._warmup_single_request(request_key, request_data, processor)
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Cache warmup completed")
    
    async def _warmup_single_request(self, request_key: str, request_data: Any, processor: Callable):
        """Warm up single cache entry."""
        try:
            result = await processor(request_data)
            if result is not None:
                await self.cache.put(request_key, result)
        except Exception as e:
            logger.warning(f"Cache warmup failed for {request_key}: {e}")
    
    async def optimize_memory_usage(self):
        """Optimize memory usage across components."""
        optimizations_applied = []
        
        # Optimize cache
        if self.cache:
            cache_stats = self.cache.get_cache_stats()
            if cache_stats["hit_rate"] < 0.3:
                # Low hit rate - reduce cache size
                await self.cache.clear()
                optimizations_applied.append("cleared_low_performance_cache")
            elif cache_stats["utilization"] > 0.9:
                # High utilization - trigger early eviction
                for _ in range(10):  # Evict 10 items
                    await self.cache._evict_one()
                optimizations_applied.append("cache_early_eviction")
        
        # Optimize connection pool
        if self.connection_pool:
            pool_stats = self.connection_pool.get_pool_stats()
            if pool_stats["pool_utilization"] < 0.2:
                # Low utilization - could reduce pool size
                optimizations_applied.append("connection_pool_optimization_candidate")
        
        logger.info(f"Memory optimizations applied: {optimizations_applied}")
        return optimizations_applied
    
    async def auto_tune_performance(self):
        """Automatically tune performance parameters based on usage patterns."""
        current_time = time.time()
        uptime_minutes = (current_time - self.optimization_metrics["start_time"]) / 60
        
        if uptime_minutes < 5:
            return  # Need more data
        
        recommendations = []
        
        # Cache tuning
        if self.cache:
            cache_stats = self.cache.get_cache_stats()
            hit_rate = cache_stats["hit_rate"]
            
            if hit_rate > 0.8 and cache_stats["utilization"] > 0.9:
                recommendations.append("increase_cache_size")
            elif hit_rate < 0.3:
                recommendations.append("decrease_cache_size_or_ttl")
        
        # Batch tuning
        if self.batcher:
            batch_stats = self.batcher.get_batch_stats()
            efficiency = batch_stats["batch_efficiency"]
            
            if efficiency < 0.5:
                recommendations.append("increase_batch_timeout")
            elif efficiency > 0.9 and batch_stats["avg_batch_size"] == self.config.max_batch_size:
                recommendations.append("increase_max_batch_size")
        
        if recommendations:
            logger.info(f"Performance tuning recommendations: {recommendations}")
        
        return recommendations
    
    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "optimization_metrics": self.optimization_metrics.copy(),
            "cache_enabled": self.cache is not None,
            "batching_enabled": self.batcher is not None,
            "connection_pooling_enabled": self.connection_pool is not None,
            "uptime_minutes": (time.time() - self.optimization_metrics["start_time"]) / 60
        }
        
        if self.cache:
            stats["cache_stats"] = self.cache.get_cache_stats()
        
        if self.batcher:
            stats["batch_stats"] = self.batcher.get_batch_stats()
        
        if self.connection_pool:
            stats["connection_pool_stats"] = self.connection_pool.get_pool_stats()
        
        return stats


# Global performance optimizer
global_performance_optimizer: Optional[PerformanceOptimizer] = None


def initialize_performance_optimizer(config: PerformanceConfig = None) -> PerformanceOptimizer:
    """Initialize global performance optimizer."""
    global global_performance_optimizer
    
    config = config or PerformanceConfig()
    global_performance_optimizer = PerformanceOptimizer(config)
    
    return global_performance_optimizer