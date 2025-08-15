"""Performance optimization and caching system."""

import time
import asyncio
import threading
import hashlib
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
from functools import wraps
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: float = 3600.0  # 1 hour default
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return time.time() - self.timestamp > self.ttl
    
    def is_stale(self, max_age: float = 1800.0) -> bool:
        """Check if cache entry is stale (30 min default)."""
        return time.time() - self.timestamp > max_age


class IntelligentCache:
    """High-performance cache with intelligent eviction."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: float = 512.0,
        default_ttl: float = 3600.0
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._memory_usage = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = threading.RLock()
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            if isinstance(value, (list, tuple)):
                if value and isinstance(value[0], (int, float)):
                    # Numeric data - use NumPy estimation
                    array = np.array(value)
                    return array.nbytes
            
            # Fallback to JSON serialization size
            return len(json.dumps(value, default=str).encode('utf-8'))
        except:
            # Very rough estimate
            return 1024
    
    def _evict_lru(self):
        """Evict least recently used items."""
        with self._lock:
            while (len(self._cache) >= self.max_size or 
                   self._memory_usage >= self.max_memory_bytes):
                
                if not self._cache:
                    break
                    
                # Remove oldest (LRU) item
                key, entry = self._cache.popitem(last=False)
                self._memory_usage -= entry.size_bytes
                self._evictions += 1
                
                logger.debug(f"Evicted cache entry: {key}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                if entry.is_expired():
                    self._cache.pop(key)
                    self._memory_usage -= entry.size_bytes
                    self._misses += 1
                    return None
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.access_count += 1
                self._hits += 1
                
                return entry.value
            
            self._misses += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in cache."""
        if ttl is None:
            ttl = self.default_ttl
        
        size_bytes = self._estimate_size(value)
        
        # Don't cache items that are too large
        if size_bytes > self.max_memory_bytes * 0.5:
            return False
        
        with self._lock:
            # Update existing entry
            if key in self._cache:
                old_entry = self._cache[key]
                self._memory_usage -= old_entry.size_bytes
            
            # Create new entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes,
                ttl=ttl
            )
            
            self._cache[key] = entry
            self._memory_usage += size_bytes
            
            # Evict if necessary
            self._evict_lru()
            
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "memory_usage_mb": self._memory_usage / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "hit_rate": hit_rate,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions
            }


def cache_key_from_input(input_data: Any) -> str:
    """Generate cache key from input data."""
    # Convert to hashable representation
    if isinstance(input_data, (list, tuple)):
        # For numeric data, create hash from array
        try:
            array = np.array(input_data, dtype=np.float32)
            # Use shape and hash of flattened array
            shape_str = str(array.shape)
            data_hash = hashlib.md5(array.tobytes()).hexdigest()
            return f"input_{shape_str}_{data_hash}"
        except:
            pass
    
    # Fallback to JSON hash
    json_str = json.dumps(input_data, sort_keys=True, default=str)
    return hashlib.md5(json_str.encode()).hexdigest()


class ModelInferenceCache:
    """Specialized cache for model inference."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.cache = IntelligentCache(
            max_size=5000,  # Cache more inference results
            max_memory_mb=1024.0,  # 1GB for inference cache
            default_ttl=7200.0  # 2 hours for inference results
        )
    
    def get_prediction(self, input_data: List[List[float]]) -> Optional[List[List[float]]]:
        """Get cached prediction."""
        cache_key = cache_key_from_input(input_data)
        return self.cache.get(cache_key)
    
    def cache_prediction(
        self,
        input_data: List[List[float]],
        predictions: List[List[float]]
    ) -> bool:
        """Cache prediction result."""
        cache_key = cache_key_from_input(input_data)
        return self.cache.put(cache_key, predictions)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.cache.get_stats()
        stats["model_name"] = self.model_name
        return stats


class PerformanceMonitor:
    """Monitor and optimize performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self._lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a performance metric."""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = {
                    'values': [],
                    'timestamps': [],
                    'labels': []
                }
            
            self.metrics[name]['values'].append(value)
            self.metrics[name]['timestamps'].append(time.time())
            self.metrics[name]['labels'].append(labels or {})
            
            # Keep only recent metrics (last 1000 points)
            if len(self.metrics[name]['values']) > 1000:
                self.metrics[name]['values'] = self.metrics[name]['values'][-1000:]
                self.metrics[name]['timestamps'] = self.metrics[name]['timestamps'][-1000:]
                self.metrics[name]['labels'] = self.metrics[name]['labels'][-1000:]
    
    def get_metric_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get statistics for a metric."""
        with self._lock:
            if name not in self.metrics or not self.metrics[name]['values']:
                return None
            
            values = self.metrics[name]['values']
            recent_values = values[-100:]  # Last 100 measurements
            
            return {
                'count': len(values),
                'latest': values[-1],
                'mean': np.mean(recent_values),
                'median': np.median(recent_values),
                'p95': np.percentile(recent_values, 95),
                'p99': np.percentile(recent_values, 99),
                'min': np.min(recent_values),
                'max': np.max(recent_values)
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {name: self.get_metric_stats(name) for name in self.metrics}


def with_caching(cache: IntelligentCache, ttl: Optional[float] = None):
    """Decorator to add caching to a function."""
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}_{cache_key_from_input((args, kwargs))}"
            
            # Try cache first
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache.put(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def with_performance_monitoring(monitor: PerformanceMonitor):
    """Decorator to add performance monitoring to a function."""
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000  # ms
                
                monitor.record_metric(
                    f"{func.__name__}_latency_ms",
                    execution_time,
                    {"status": "success"}
                )
                
                return result
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000  # ms
                
                monitor.record_metric(
                    f"{func.__name__}_latency_ms",
                    execution_time,
                    {"status": "error", "error_type": type(e).__name__}
                )
                
                raise
        
        return wrapper
    return decorator


# Global instances
global_cache = IntelligentCache(max_size=10000, max_memory_mb=2048.0)
global_monitor = PerformanceMonitor()