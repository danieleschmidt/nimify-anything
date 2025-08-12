"""Advanced performance optimization for bioneuro-olfactory fusion."""

import asyncio
import numpy as np
import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from functools import wraps, lru_cache
from collections import defaultdict
import logging
import psutil
import queue
import weakref
import gc
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    cache_hit_rate: float = 0.0
    throughput_ops_per_sec: float = 0.0
    latency_p95_ms: float = 0.0
    error_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'execution_time': self.execution_time,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'gpu_usage_percent': self.gpu_usage_percent,
            'cache_hit_rate': self.cache_hit_rate,
            'throughput_ops_per_sec': self.throughput_ops_per_sec,
            'latency_p95_ms': self.latency_p95_ms,
            'error_rate': self.error_rate
        }


class AdaptiveCache:
    """Intelligent caching system that adapts to usage patterns."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._access_times = {}
        self._access_counts = defaultdict(int)
        self._lock = threading.RLock()
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self._cache:
                # Check TTL
                access_time = self._access_times.get(key, 0)
                if time.time() - access_time < self.ttl_seconds:
                    self._access_times[key] = time.time()
                    self._access_counts[key] += 1
                    self.hits += 1
                    return self._cache[key]
                else:
                    # Expired
                    self._remove_key(key)
            
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            self._cache[key] = value
            self._access_times[key] = time.time()
            self._access_counts[key] += 1
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return
        
        # Find least recently accessed
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        self._remove_key(lru_key)
    
    def _remove_key(self, key: str) -> None:
        """Remove key from all tracking structures."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self._access_counts.pop(key, None)
    
    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'memory_usage_mb': self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of cache in MB."""
        import sys
        total_size = 0
        
        for key, value in self._cache.items():
            total_size += sys.getsizeof(key) + sys.getsizeof(value)
        
        return total_size / (1024 * 1024)


class ResourceMonitor:
    """Monitors system resources for adaptive optimization."""
    
    def __init__(self, monitor_interval: float = 1.0):
        self.monitor_interval = monitor_interval
        self._monitoring = False
        self._monitor_thread = None
        self._metrics_history = []
        self._max_history = 100
        
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                self._metrics_history.append(metrics)
                
                # Keep history bounded
                if len(self._metrics_history) > self._max_history:
                    self._metrics_history.pop(0)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.monitor_interval)
    
    def _collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # Network I/O
        network = psutil.net_io_counters()
        
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_available_mb': memory_available_mb,
            'disk_percent': disk_percent,
            'network_bytes_sent': network.bytes_sent,
            'network_bytes_recv': network.bytes_recv
        }
        
        # GPU metrics if available
        try:
            gpu_stats = self._get_gpu_metrics()
            if gpu_stats:
                metrics.update(gpu_stats)
        except Exception:
            pass  # GPU monitoring not available
        
        return metrics
    
    def _get_gpu_metrics(self) -> Optional[Dict[str, float]]:
        """Get GPU metrics if available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                return None
            
            # Get metrics for first GPU
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used_mb = mem_info.used / (1024 * 1024)
            memory_total_mb = mem_info.total / (1024 * 1024)
            memory_util = (mem_info.used / mem_info.total) * 100
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            return {
                'gpu_utilization': gpu_util,
                'gpu_memory_used_mb': memory_used_mb,
                'gpu_memory_total_mb': memory_total_mb,
                'gpu_memory_percent': memory_util,
                'gpu_temperature': temp
            }
            
        except (ImportError, Exception):
            return None
    
    def get_current_metrics(self) -> Optional[Dict[str, float]]:
        """Get most recent metrics."""
        if not self._metrics_history:
            return self._collect_metrics()
        return self._metrics_history[-1]
    
    def get_resource_pressure(self) -> Dict[str, str]:
        """Assess current resource pressure levels."""
        current = self.get_current_metrics()
        if not current:
            return {'overall': 'unknown'}
        
        pressure_levels = {}
        
        # CPU pressure
        cpu_usage = current.get('cpu_percent', 0)
        if cpu_usage < 30:
            pressure_levels['cpu'] = 'low'
        elif cpu_usage < 70:
            pressure_levels['cpu'] = 'medium'
        else:
            pressure_levels['cpu'] = 'high'
        
        # Memory pressure
        memory_usage = current.get('memory_percent', 0)
        if memory_usage < 40:
            pressure_levels['memory'] = 'low'
        elif memory_usage < 80:
            pressure_levels['memory'] = 'medium'
        else:
            pressure_levels['memory'] = 'high'
        
        # GPU pressure if available
        gpu_usage = current.get('gpu_utilization', 0)
        if gpu_usage > 0:
            if gpu_usage < 30:
                pressure_levels['gpu'] = 'low'
            elif gpu_usage < 70:
                pressure_levels['gpu'] = 'medium'
            else:
                pressure_levels['gpu'] = 'high'
        
        # Overall pressure
        high_count = sum(1 for level in pressure_levels.values() if level == 'high')
        if high_count >= 2:
            pressure_levels['overall'] = 'high'
        elif high_count == 1 or any(level == 'medium' for level in pressure_levels.values()):
            pressure_levels['overall'] = 'medium'
        else:
            pressure_levels['overall'] = 'low'
        
        return pressure_levels


class ConcurrentProcessor:
    """Handles concurrent processing with adaptive thread/process pools."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) * 2)
        self._thread_pool = None
        self._process_pool = None
        self._current_load = 0
        self._load_lock = threading.Lock()
        
        # Performance tracking
        self.execution_times = []
        self.error_count = 0
        
    @property
    def thread_pool(self) -> ThreadPoolExecutor:
        """Get or create thread pool."""
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._thread_pool
    
    @property
    def process_pool(self) -> ProcessPoolExecutor:
        """Get or create process pool."""
        if self._process_pool is None:
            self._process_pool = ProcessPoolExecutor(max_workers=min(self.max_workers, mp.cpu_count()))
        return self._process_pool
    
    async def run_concurrent_tasks(
        self,
        tasks: List[Callable],
        task_args: List[tuple],
        use_processes: bool = False,
        max_concurrent: Optional[int] = None
    ) -> List[Any]:
        """Run multiple tasks concurrently."""
        
        if not tasks:
            return []
        
        max_concurrent = max_concurrent or len(tasks)
        executor = self.process_pool if use_processes else self.thread_pool
        
        # Execute tasks in batches to control concurrency
        results = []
        for i in range(0, len(tasks), max_concurrent):
            batch_tasks = tasks[i:i + max_concurrent]
            batch_args = task_args[i:i + max_concurrent]
            
            loop = asyncio.get_event_loop()
            futures = []
            
            for task, args in zip(batch_tasks, batch_args):
                future = loop.run_in_executor(executor, task, *args)
                futures.append(future)
            
            batch_results = await asyncio.gather(*futures, return_exceptions=True)
            results.extend(batch_results)
        
        return results
    
    def submit_batch(
        self,
        func: Callable,
        data_batch: List[Any],
        use_processes: bool = False
    ) -> List[Any]:
        """Submit batch processing job."""
        
        with self._load_lock:
            self._current_load += len(data_batch)
        
        try:
            executor = self.process_pool if use_processes else self.thread_pool
            
            start_time = time.time()
            futures = [executor.submit(func, item) for item in data_batch]
            results = [future.result() for future in futures]
            
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            
            # Keep execution time history bounded
            if len(self.execution_times) > 100:
                self.execution_times.pop(0)
            
            return results
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Batch processing error: {e}")
            raise
        finally:
            with self._load_lock:
                self._current_load -= len(data_batch)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_execution_time = np.mean(self.execution_times) if self.execution_times else 0.0
        
        return {
            'current_load': self._current_load,
            'max_workers': self.max_workers,
            'avg_execution_time': avg_execution_time,
            'error_count': self.error_count,
            'total_executions': len(self.execution_times)
        }
    
    def shutdown(self) -> None:
        """Shutdown thread and process pools."""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None
        
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
            self._process_pool = None


class BioneuroOptimizer:
    """Main optimization system for bioneuro-olfactory fusion."""
    
    def __init__(self):
        self.cache = AdaptiveCache(max_size=10000, ttl_seconds=7200)  # 2 hour TTL
        self.resource_monitor = ResourceMonitor(monitor_interval=2.0)
        self.processor = ConcurrentProcessor()
        
        # Optimization strategies
        self.optimization_strategies = {
            'neural_processing': self._optimize_neural_processing,
            'olfactory_analysis': self._optimize_olfactory_analysis,
            'fusion_computation': self._optimize_fusion_computation
        }
        
        # Performance tracking
        self.performance_history = []
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
    
    def optimize_neural_processing(
        self,
        neural_data: np.ndarray,
        config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], PerformanceMetrics]:
        """Optimize neural signal processing."""
        
        cache_key = self._generate_cache_key('neural', neural_data, config)
        cached_result = self.cache.get(cache_key)
        
        if cached_result is not None:
            logger.debug(f"Cache hit for neural processing: {cache_key}")
            return cached_result
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_usage()
        
        try:
            # Apply optimization strategy
            optimized_result = self.optimization_strategies['neural_processing'](neural_data, config)
            
            # Cache result
            self.cache.set(cache_key, optimized_result)
            
            # Track performance
            execution_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            cpu_usage = self._get_cpu_usage() - start_cpu
            
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                cache_hit_rate=self.cache.get_stats()['hit_rate']
            )
            
            self.performance_history.append(metrics)
            
            return optimized_result, metrics
            
        except Exception as e:
            logger.error(f"Neural processing optimization failed: {e}")
            raise
    
    def optimize_olfactory_analysis(
        self,
        molecule_data: Dict[str, Any],
        concentration: float,
        config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], PerformanceMetrics]:
        """Optimize olfactory stimulus analysis."""
        
        cache_key = self._generate_cache_key('olfactory', molecule_data, {'concentration': concentration, **config})
        cached_result = self.cache.get(cache_key)
        
        if cached_result is not None:
            logger.debug(f"Cache hit for olfactory analysis: {cache_key}")
            return cached_result
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            # Apply optimization strategy
            optimized_result = self.optimization_strategies['olfactory_analysis'](
                molecule_data, concentration, config
            )
            
            # Cache result
            self.cache.set(cache_key, optimized_result)
            
            # Track performance
            execution_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=0.0,  # Will be updated by monitor
                cache_hit_rate=self.cache.get_stats()['hit_rate']
            )
            
            return optimized_result, metrics
            
        except Exception as e:
            logger.error(f"Olfactory analysis optimization failed: {e}")
            raise
    
    def optimize_fusion_processing(
        self,
        neural_features: Dict[str, Any],
        olfactory_features: Dict[str, Any],
        fusion_config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], PerformanceMetrics]:
        """Optimize multi-modal fusion processing."""
        
        cache_key = self._generate_cache_key('fusion', neural_features, olfactory_features, fusion_config)
        cached_result = self.cache.get(cache_key)
        
        if cached_result is not None:
            logger.debug(f"Cache hit for fusion processing: {cache_key}")
            return cached_result
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            # Apply optimization strategy
            optimized_result = self.optimization_strategies['fusion_computation'](
                neural_features, olfactory_features, fusion_config
            )
            
            # Cache result
            self.cache.set(cache_key, optimized_result)
            
            # Track performance
            execution_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=0.0,
                cache_hit_rate=self.cache.get_stats()['hit_rate']
            )
            
            return optimized_result, metrics
            
        except Exception as e:
            logger.error(f"Fusion processing optimization failed: {e}")
            raise
    
    def _optimize_neural_processing(self, neural_data: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
        """Internal neural processing optimization."""
        from .neural_processor import NeuralSignalProcessor
        from .core import NeuralConfig, NeuralSignalType
        
        # Create optimized neural config
        signal_type = NeuralSignalType(config.get('signal_type', 'eeg'))
        neural_config = NeuralConfig(
            signal_type=signal_type,
            sampling_rate=config.get('sampling_rate', 1000),
            channels=neural_data.shape[0] if len(neural_data.shape) > 1 else 1,
            time_window=config.get('time_window', 2.0),
            artifact_removal=config.get('artifact_removal', True)
        )
        
        # Optimize processing based on resource pressure
        pressure = self.resource_monitor.get_resource_pressure()
        
        if pressure['overall'] == 'high':
            # Use simplified processing for high resource pressure
            neural_config.preprocessing_filters = ['bandpass']  # Minimal preprocessing
            logger.info("Using simplified neural processing due to high resource pressure")
        
        processor = NeuralSignalProcessor(neural_config)
        return processor.process(neural_data)
    
    def _optimize_olfactory_analysis(
        self,
        molecule_data: Dict[str, Any],
        concentration: float,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Internal olfactory analysis optimization."""
        from .olfactory_analyzer import OlfactoryAnalyzer
        from .core import OlfactoryConfig
        
        # Create optimized olfactory config
        olfactory_config = OlfactoryConfig(
            molecule_types=config.get('molecule_types', []),
            concentration_range=config.get('concentration_range', (0.001, 10.0)),
            stimulus_duration=config.get('stimulus_duration', 3.0)
        )
        
        analyzer = OlfactoryAnalyzer(olfactory_config)
        return analyzer.analyze(molecule_data, concentration)
    
    def _optimize_fusion_computation(
        self,
        neural_features: Dict[str, Any],
        olfactory_features: Dict[str, Any],
        fusion_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Internal fusion computation optimization."""
        from .fusion_engine import MultiModalFusionEngine, FusionStrategy
        
        # Select fusion strategy based on resource pressure
        pressure = self.resource_monitor.get_resource_pressure()
        
        if pressure['overall'] == 'high':
            fusion_strategy = FusionStrategy.EARLY_FUSION  # Simpler, faster
            logger.info("Using early fusion strategy due to high resource pressure")
        else:
            fusion_strategy = FusionStrategy(fusion_config.get('strategy', 'attention_fusion'))
        
        fusion_engine = MultiModalFusionEngine(fusion_strategy)
        return fusion_engine.fuse(
            neural_features,
            olfactory_features,
            temporal_alignment=fusion_config.get('temporal_alignment', True)
        )
    
    def _generate_cache_key(self, *args) -> str:
        """Generate cache key from arguments."""
        import hashlib
        
        # Convert args to string representation
        key_parts = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                # Use array shape, dtype, and hash of first/last elements
                shape_str = str(arg.shape)
                dtype_str = str(arg.dtype)
                if arg.size > 0:
                    sample_hash = hash((arg.flat[0], arg.flat[-1] if arg.size > 1 else arg.flat[0]))
                else:
                    sample_hash = 0
                key_parts.append(f"{shape_str}_{dtype_str}_{sample_hash}")
            elif isinstance(arg, dict):
                # Sort dict keys for consistent hashing
                sorted_items = sorted(arg.items())
                key_parts.append(str(sorted_items))
            else:
                key_parts.append(str(arg))
        
        # Create hash of combined key parts
        key_string = "_".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        import psutil
        return psutil.cpu_percent(interval=0.1)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        cache_stats = self.cache.get_stats()
        processor_stats = self.processor.get_performance_stats()
        resource_metrics = self.resource_monitor.get_current_metrics()
        resource_pressure = self.resource_monitor.get_resource_pressure()
        
        # Performance trends
        if self.performance_history:
            recent_metrics = self.performance_history[-10:]  # Last 10 operations
            avg_execution_time = np.mean([m.execution_time for m in recent_metrics])
            avg_memory_usage = np.mean([m.memory_usage_mb for m in recent_metrics])
            avg_cache_hit_rate = np.mean([m.cache_hit_rate for m in recent_metrics])
        else:
            avg_execution_time = 0.0
            avg_memory_usage = 0.0
            avg_cache_hit_rate = 0.0
        
        return {
            'cache_stats': cache_stats,
            'processor_stats': processor_stats,
            'resource_metrics': resource_metrics,
            'resource_pressure': resource_pressure,
            'performance_trends': {
                'avg_execution_time': avg_execution_time,
                'avg_memory_usage_mb': avg_memory_usage,
                'avg_cache_hit_rate': avg_cache_hit_rate,
                'total_operations': len(self.performance_history)
            },
            'optimization_recommendations': self._get_optimization_recommendations()
        }
    
    def _get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on current state."""
        recommendations = []
        
        cache_stats = self.cache.get_stats()
        resource_pressure = self.resource_monitor.get_resource_pressure()
        
        # Cache recommendations
        if cache_stats['hit_rate'] < 0.5:
            recommendations.append("Consider increasing cache size or TTL for better hit rate")
        
        if cache_stats['memory_usage_mb'] > 500:  # 500MB cache
            recommendations.append("Cache memory usage is high, consider reducing cache size")
        
        # Resource recommendations
        if resource_pressure['overall'] == 'high':
            recommendations.append("High resource pressure detected - consider scaling horizontally")
        
        if resource_pressure.get('memory', 'low') == 'high':
            recommendations.append("High memory pressure - enable aggressive caching strategies")
        
        if resource_pressure.get('cpu', 'low') == 'high':
            recommendations.append("High CPU usage - consider process-based parallelization")
        
        # Performance recommendations
        if self.performance_history:
            recent_times = [m.execution_time for m in self.performance_history[-10:]]
            if np.mean(recent_times) > 5.0:  # More than 5 seconds average
                recommendations.append("Long execution times detected - review algorithm complexity")
        
        return recommendations if recommendations else ["System performance is optimal"]
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.resource_monitor.stop_monitoring()
        self.processor.shutdown()
        self.cache.clear()
        gc.collect()


# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            logger.info(
                f"Performance: {func.__name__} - "
                f"Time: {execution_time:.3f}s, "
                f"Memory: {memory_delta:+.2f}MB, "
                f"Success: {success}"
            )
            
            if not success:
                logger.error(f"Function {func.__name__} failed with error: {error}")
        
        return result
    
    return wrapper


# Global optimizer instance
global_optimizer = BioneuroOptimizer()


@contextmanager
def optimization_context(cache_enabled: bool = True, monitoring_enabled: bool = True):
    """Context manager for optimization settings."""
    
    # Store original settings
    original_cache_enabled = hasattr(global_optimizer, '_cache_enabled')
    
    try:
        global_optimizer._cache_enabled = cache_enabled
        global_optimizer._monitoring_enabled = monitoring_enabled
        
        yield global_optimizer
        
    finally:
        # Restore original settings
        if not original_cache_enabled:
            delattr(global_optimizer, '_cache_enabled')