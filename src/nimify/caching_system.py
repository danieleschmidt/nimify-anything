"""Advanced caching system with multiple strategies and intelligent cache management."""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import threading
from collections import OrderedDict, defaultdict
import weakref

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "least_recently_used"
    LFU = "least_frequently_used"
    TTL = "time_to_live"
    ADAPTIVE = "adaptive"
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"


class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DISK = "l3_disk"


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""
    max_size: int = 1000
    ttl_seconds: int = 3600
    strategy: CacheStrategy = CacheStrategy.LRU
    
    # Memory settings
    max_memory_mb: int = 500
    compression_enabled: bool = True
    compression_threshold: int = 1024  # bytes
    
    # Multi-level settings
    enable_multilevel: bool = True
    l1_size: int = 100
    l2_size: int = 1000
    l3_enabled: bool = False
    
    # Performance settings
    async_write: bool = True
    batch_operations: bool = True
    prefetch_enabled: bool = True
    
    # Adaptive settings
    adaptive_resize: bool = True
    hit_rate_threshold: float = 0.8
    miss_penalty_weight: float = 1.5


@dataclass
class CacheItem:
    """Individual cache item with metadata."""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    size_bytes: int
    ttl: Optional[float] = None
    compressed: bool = False
    
    @property
    def age(self) -> float:
        return time.time() - self.created_at
    
    @property
    def time_since_access(self) -> float:
        return time.time() - self.accessed_at
    
    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)


class CacheStats:
    """Cache statistics and metrics."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.size = 0
        self.memory_usage = 0
        self.lock = threading.RLock()
        
        # Performance metrics
        self.avg_access_time = 0.0
        self.total_access_time = 0.0
        self.access_count = 0
        
        # Hit rate history for adaptive behavior
        self.hit_rate_history = []
        self.last_reset = time.time()
    
    def record_hit(self, access_time: float = 0):
        with self.lock:
            self.hits += 1
            self._update_access_time(access_time)
    
    def record_miss(self, access_time: float = 0):
        with self.lock:
            self.misses += 1
            self._update_access_time(access_time)
    
    def record_eviction(self):
        with self.lock:
            self.evictions += 1
    
    def _update_access_time(self, access_time: float):
        self.access_count += 1
        self.total_access_time += access_time
        self.avg_access_time = self.total_access_time / self.access_count
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate
    
    def get_metrics(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": self.hit_rate,
                "miss_rate": self.miss_rate,
                "size": self.size,
                "memory_usage_mb": self.memory_usage / (1024 * 1024),
                "avg_access_time_ms": self.avg_access_time * 1000,
                "uptime_seconds": time.time() - self.last_reset
            }
    
    def reset(self):
        with self.lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.total_access_time = 0.0
            self.access_count = 0
            self.avg_access_time = 0.0
            self.last_reset = time.time()


class Cache(ABC):
    """Abstract base class for cache implementations."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    async def clear(self):
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass


class InMemoryCache(Cache):
    """High-performance in-memory cache with multiple eviction strategies."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: Dict[str, CacheItem] = {}
        self.access_order = OrderedDict()  # For LRU
        self.frequency_counter = defaultdict(int)  # For LFU
        self.stats = CacheStats()
        self.lock = threading.RLock()
        
        # Compression support
        if config.compression_enabled:
            try:
                import lz4.frame
                self.compressor = lz4.frame
            except ImportError:
                try:
                    import gzip
                    self.compressor = gzip
                except ImportError:
                    logger.warning("No compression library available, disabling compression")
                    self.config.compression_enabled = False
        
        # Background cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task for expired items."""
        async def cleanup_loop():
            while True:
                try:
                    await self._cleanup_expired()
                    await asyncio.sleep(60)  # Cleanup every minute
                except Exception as e:
                    logger.error(f"Cache cleanup failed: {e}")
                    await asyncio.sleep(60)
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        start_time = time.time()
        
        with self.lock:
            if key not in self.cache:
                self.stats.record_miss(time.time() - start_time)
                return None
            
            item = self.cache[key]
            
            # Check expiration
            if item.is_expired:
                del self.cache[key]
                if key in self.access_order:
                    del self.access_order[key]
                self.stats.record_miss(time.time() - start_time)
                return None
            
            # Update access metadata
            item.accessed_at = time.time()
            item.access_count += 1
            self.frequency_counter[key] += 1
            
            # Update access order for LRU
            if key in self.access_order:
                del self.access_order[key]
            self.access_order[key] = None
            
            self.stats.record_hit(time.time() - start_time)
            
            # Decompress if needed
            value = item.value
            if item.compressed and self.config.compression_enabled:
                value = await self._decompress(value)
            
            return value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in cache."""
        try:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Compress if enabled and value is large enough
            compressed = False
            if (self.config.compression_enabled and 
                size_bytes > self.config.compression_threshold):
                value = await self._compress(value)
                compressed = True
                size_bytes = self._calculate_size(value)  # Recalculate after compression
            
            # Create cache item
            item = CacheItem(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                access_count=1,
                size_bytes=size_bytes,
                ttl=ttl or self.config.ttl_seconds,
                compressed=compressed
            )
            
            with self.lock:
                # Check if we need to evict items
                await self._ensure_capacity(size_bytes)
                
                # Remove old item if exists
                if key in self.cache:
                    old_item = self.cache[key]
                    self.stats.memory_usage -= old_item.size_bytes
                
                # Add new item
                self.cache[key] = item
                self.access_order[key] = None
                self.frequency_counter[key] += 1
                
                # Update stats
                self.stats.size = len(self.cache)
                self.stats.memory_usage += size_bytes
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set cache item {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete item from cache."""
        with self.lock:
            if key not in self.cache:
                return False
            
            item = self.cache[key]
            del self.cache[key]
            
            if key in self.access_order:
                del self.access_order[key]
            
            if key in self.frequency_counter:
                del self.frequency_counter[key]
            
            self.stats.size = len(self.cache)
            self.stats.memory_usage -= item.size_bytes
            
            return True
    
    async def clear(self):
        """Clear all items from cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.frequency_counter.clear()
            self.stats.size = 0
            self.stats.memory_usage = 0
    
    async def _ensure_capacity(self, new_item_size: int):
        """Ensure cache has capacity for new item."""
        # Check size limit
        while (len(self.cache) >= self.config.max_size or 
               self.stats.memory_usage + new_item_size > self.config.max_memory_mb * 1024 * 1024):
            
            if not self.cache:
                break
            
            await self._evict_item()
    
    async def _evict_item(self):
        """Evict item based on configured strategy."""
        if not self.cache:
            return
        
        if self.config.strategy == CacheStrategy.LRU:
            # Remove least recently used
            key = next(iter(self.access_order))
        elif self.config.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            key = min(self.frequency_counter.keys(), key=lambda k: self.frequency_counter[k])
        elif self.config.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive strategy based on hit rate
            if self.stats.hit_rate < self.config.hit_rate_threshold:
                # Low hit rate, use LFU to keep popular items
                key = min(self.frequency_counter.keys(), key=lambda k: self.frequency_counter[k])
            else:
                # Good hit rate, use LRU for temporal locality
                key = next(iter(self.access_order))
        else:
            # Default to LRU
            key = next(iter(self.access_order))
        
        await self.delete(key)
        self.stats.record_eviction()
    
    async def _cleanup_expired(self):
        """Remove expired items."""
        current_time = time.time()
        expired_keys = []
        
        with self.lock:
            for key, item in self.cache.items():
                if item.is_expired:
                    expired_keys.append(key)
        
        for key in expired_keys:
            await self.delete(key)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            if isinstance(value, (str, bytes)):
                return len(value.encode() if isinstance(value, str) else value)
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple, dict)):
                return len(pickle.dumps(value))
            else:
                return len(pickle.dumps(value))
        except:
            return 1024  # Default estimate
    
    async def _compress(self, value: Any) -> bytes:
        """Compress value."""
        try:
            serialized = pickle.dumps(value)
            if hasattr(self.compressor, 'compress'):
                return self.compressor.compress(serialized)
            else:
                # For gzip
                import gzip
                return gzip.compress(serialized)
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return pickle.dumps(value)
    
    async def _decompress(self, compressed_value: bytes) -> Any:
        """Decompress value."""
        try:
            if hasattr(self.compressor, 'decompress'):
                serialized = self.compressor.decompress(compressed_value)
            else:
                # For gzip
                import gzip
                serialized = gzip.decompress(compressed_value)
            return pickle.loads(serialized)
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return pickle.loads(compressed_value)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.stats.get_metrics()
        with self.lock:
            stats.update({
                "strategy": self.config.strategy.value,
                "max_size": self.config.max_size,
                "compression_enabled": self.config.compression_enabled,
                "items": len(self.cache),
                "avg_item_size": (self.stats.memory_usage / len(self.cache)) if self.cache else 0
            })
        return stats


class MultiLevelCache(Cache):
    """Multi-level cache hierarchy with L1 memory, L2 Redis, L3 disk."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        
        # Initialize cache levels
        l1_config = CacheConfig(
            max_size=config.l1_size,
            ttl_seconds=config.ttl_seconds,
            strategy=CacheStrategy.LRU,
            compression_enabled=False  # L1 should be fast
        )
        self.l1_cache = InMemoryCache(l1_config)
        
        # L2 Redis cache (if available)
        self.l2_cache = None
        if config.enable_multilevel:
            try:
                self.l2_cache = RedisCache(config)
            except Exception as e:
                logger.warning(f"L2 Redis cache not available: {e}")
        
        # L3 Disk cache (if enabled)
        self.l3_cache = None
        if config.l3_enabled:
            try:
                self.l3_cache = DiskCache(config)
            except Exception as e:
                logger.warning(f"L3 Disk cache not available: {e}")
        
        self.stats = CacheStats()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache hierarchy."""
        start_time = time.time()
        
        # Try L1 first
        value = await self.l1_cache.get(key)
        if value is not None:
            self.stats.record_hit(time.time() - start_time)
            return value
        
        # Try L2 if available
        if self.l2_cache:
            value = await self.l2_cache.get(key)
            if value is not None:
                # Promote to L1
                await self.l1_cache.set(key, value)
                self.stats.record_hit(time.time() - start_time)
                return value
        
        # Try L3 if available
        if self.l3_cache:
            value = await self.l3_cache.get(key)
            if value is not None:
                # Promote to L1 and L2
                await self.l1_cache.set(key, value)
                if self.l2_cache:
                    await self.l2_cache.set(key, value)
                self.stats.record_hit(time.time() - start_time)
                return value
        
        self.stats.record_miss(time.time() - start_time)
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in cache hierarchy."""
        results = []
        
        # Set in all available levels
        results.append(await self.l1_cache.set(key, value, ttl))
        
        if self.l2_cache:
            results.append(await self.l2_cache.set(key, value, ttl))
        
        if self.l3_cache:
            results.append(await self.l3_cache.set(key, value, ttl))
        
        return any(results)
    
    async def delete(self, key: str) -> bool:
        """Delete item from all cache levels."""
        results = []
        
        results.append(await self.l1_cache.delete(key))
        
        if self.l2_cache:
            results.append(await self.l2_cache.delete(key))
        
        if self.l3_cache:
            results.append(await self.l3_cache.delete(key))
        
        return any(results)
    
    async def clear(self):
        """Clear all cache levels."""
        await self.l1_cache.clear()
        
        if self.l2_cache:
            await self.l2_cache.clear()
        
        if self.l3_cache:
            await self.l3_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive stats from all levels."""
        stats = {
            "multi_level": {
                "l1": self.l1_cache.get_stats(),
                "l2": self.l2_cache.get_stats() if self.l2_cache else None,
                "l3": self.l3_cache.get_stats() if self.l3_cache else None
            },
            "overall": self.stats.get_metrics()
        }
        return stats


class RedisCache(Cache):
    """Redis-based distributed cache."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.stats = CacheStats()
        
        try:
            import redis.asyncio as redis
            self.redis = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=False  # Handle bytes for serialization
            )
        except ImportError:
            raise Exception("Redis library not available")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from Redis."""
        start_time = time.time()
        
        try:
            data = await self.redis.get(f"cache:{key}")
            if data is None:
                self.stats.record_miss(time.time() - start_time)
                return None
            
            # Deserialize
            value = pickle.loads(data)
            self.stats.record_hit(time.time() - start_time)
            return value
            
        except Exception as e:
            logger.error(f"Redis get failed for {key}: {e}")
            self.stats.record_miss(time.time() - start_time)
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in Redis."""
        try:
            # Serialize
            data = pickle.dumps(value)
            
            # Set with TTL
            ttl = ttl or self.config.ttl_seconds
            await self.redis.setex(f"cache:{key}", ttl, data)
            return True
            
        except Exception as e:
            logger.error(f"Redis set failed for {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete item from Redis."""
        try:
            result = await self.redis.delete(f"cache:{key}")
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete failed for {key}: {e}")
            return False
    
    async def clear(self):
        """Clear all cache items."""
        try:
            # Delete all keys with cache: prefix
            keys = await self.redis.keys("cache:*")
            if keys:
                await self.redis.delete(*keys)
        except Exception as e:
            logger.error(f"Redis clear failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache stats."""
        return self.stats.get_metrics()


class DiskCache(Cache):
    """Disk-based cache for large objects."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.stats = CacheStats()
        
        import tempfile
        import os
        
        self.cache_dir = os.path.join(tempfile.gettempdir(), "nimify_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from disk."""
        start_time = time.time()
        
        try:
            file_path = self._get_file_path(key)
            
            if not os.path.exists(file_path):
                self.stats.record_miss(time.time() - start_time)
                return None
            
            # Check if file is expired
            if self._is_file_expired(file_path):
                os.remove(file_path)
                self.stats.record_miss(time.time() - start_time)
                return None
            
            # Read and deserialize
            with open(file_path, 'rb') as f:
                value = pickle.load(f)
            
            self.stats.record_hit(time.time() - start_time)
            return value
            
        except Exception as e:
            logger.error(f"Disk cache get failed for {key}: {e}")
            self.stats.record_miss(time.time() - start_time)
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item on disk."""
        try:
            file_path = self._get_file_path(key)
            
            # Serialize and write
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
            
            return True
            
        except Exception as e:
            logger.error(f"Disk cache set failed for {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete item from disk."""
        try:
            file_path = self._get_file_path(key)
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            logger.error(f"Disk cache delete failed for {key}: {e}")
            return False
    
    async def clear(self):
        """Clear all disk cache files."""
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Disk cache clear failed: {e}")
    
    def _get_file_path(self, key: str) -> str:
        """Get file path for cache key."""
        # Hash key for filename
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.cache")
    
    def _is_file_expired(self, file_path: str) -> bool:
        """Check if file is expired based on modification time."""
        try:
            import os
            mtime = os.path.getmtime(file_path)
            return time.time() - mtime > self.config.ttl_seconds
        except:
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get disk cache stats."""
        stats = self.stats.get_metrics()
        
        try:
            import os
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.cache')]
            total_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f)) 
                for f in cache_files
            )
            
            stats.update({
                "disk_files": len(cache_files),
                "disk_size_mb": total_size / (1024 * 1024)
            })
        except:
            pass
        
        return stats


class CacheManager:
    """High-level cache manager with intelligent caching strategies."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        
        # Initialize appropriate cache implementation
        if config.enable_multilevel:
            self.cache = MultiLevelCache(config)
        else:
            self.cache = InMemoryCache(config)
        
        # Cache for function results
        self.function_cache: Dict[str, Any] = {}
        self.cache_decorators: Dict[str, Callable] = {}
        
        logger.info(f"Cache manager initialized with strategy: {config.strategy.value}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        return await self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in cache."""
        return await self.cache.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete item from cache."""
        return await self.cache.delete(key)
    
    async def clear(self):
        """Clear cache."""
        await self.cache.clear()
    
    def cached(self, ttl: Optional[int] = None, key_func: Optional[Callable] = None):
        """Decorator for caching function results."""
        def decorator(func: Callable):
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache result
                await self.set(cache_key, result, ttl)
                return result
            
            def sync_wrapper(*args, **kwargs):
                # For synchronous functions, we need to handle async cache operations
                cache_key = key_func(*args, **kwargs) if key_func else self._generate_cache_key(func.__name__, args, kwargs)
                
                # This would need proper async handling in a real implementation
                result = func(*args, **kwargs)
                
                # Schedule cache set operation
                asyncio.create_task(self.set(cache_key, result, ttl))
                return result
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments."""
        # Create a hash of the function call
        key_data = {
            'function': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    async def warm_up(self, warm_up_data: Dict[str, Any]):
        """Pre-populate cache with frequently accessed data."""
        logger.info(f"Warming up cache with {len(warm_up_data)} items")
        
        for key, value in warm_up_data.items():
            await self.set(key, value)
    
    async def bulk_get(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple items from cache efficiently."""
        results = {}
        
        # For now, use sequential gets
        # In production, this could be optimized with pipeline operations
        for key in keys:
            value = await self.get(key)
            if value is not None:
                results[key] = value
        
        return results
    
    async def bulk_set(self, items: Dict[str, Any], ttl: Optional[int] = None) -> int:
        """Set multiple items in cache efficiently."""
        success_count = 0
        
        for key, value in items.items():
            if await self.set(key, value, ttl):
                success_count += 1
        
        return success_count
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return self.cache.get_stats()
    
    async def optimize(self):
        """Optimize cache performance based on usage patterns."""
        stats = self.get_comprehensive_stats()
        
        # Adaptive optimization based on hit rate
        if self.config.adaptive_resize:
            hit_rate = stats.get('hit_rate', 0)
            
            if hit_rate < self.config.hit_rate_threshold:
                # Low hit rate, consider increasing cache size
                if hasattr(self.cache, 'config'):
                    old_size = self.cache.config.max_size
                    new_size = min(old_size * 1.2, old_size + 1000)  # Increase by 20% or 1000, whichever is smaller
                    self.cache.config.max_size = int(new_size)
                    logger.info(f"Increased cache size from {old_size} to {new_size} due to low hit rate ({hit_rate:.2f})")
            
            elif hit_rate > 0.95:
                # Very high hit rate, might be able to reduce size
                if hasattr(self.cache, 'config'):
                    old_size = self.cache.config.max_size
                    new_size = max(old_size * 0.9, self.config.max_size * 0.5)  # Decrease by 10% but not below 50% of original
                    self.cache.config.max_size = int(new_size)
                    logger.info(f"Decreased cache size from {old_size} to {new_size} due to high hit rate ({hit_rate:.2f})")


# Global cache instance
default_cache_config = CacheConfig(
    max_size=10000,
    ttl_seconds=3600,
    strategy=CacheStrategy.ADAPTIVE,
    enable_multilevel=True,
    compression_enabled=True
)

cache_manager = CacheManager(default_cache_config)