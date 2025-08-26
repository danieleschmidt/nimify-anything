"""Advanced scaling and performance optimization system."""

import asyncio
import json
import logging
import math
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import threading
import uuid

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil


@dataclass
class ScalingPolicy:
    """Configuration for auto-scaling policies."""
    name: str
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    target_gpu_utilization: float = 80.0
    target_latency_ms: float = 100.0
    scale_up_threshold: float = 0.8  # Scale up when above 80% of target
    scale_down_threshold: float = 0.3  # Scale down when below 30% of target
    scale_up_cooldown: int = 300  # 5 minutes
    scale_down_cooldown: int = 600  # 10 minutes
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceProfile:
    """Performance characteristics of the service."""
    avg_request_duration: float
    p95_request_duration: float
    p99_request_duration: float
    throughput_rps: float
    error_rate: float
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: Optional[float] = None
    batch_efficiency: float = 1.0
    cache_hit_rate: float = 0.0


class ScalingDecisionEngine:
    """Advanced scaling decision engine using multiple metrics."""
    
    def __init__(self, policy: ScalingPolicy, logger: Optional[logging.Logger] = None):
        self.policy = policy
        self.logger = logger or logging.getLogger(__name__)
        
        # State tracking
        self.current_replicas = policy.min_replicas
        self.last_scale_up = 0
        self.last_scale_down = 0
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.scaling_events: List[Dict[str, Any]] = []
        
        # Prediction models
        self.load_predictor = LoadPredictor()
        self.resource_predictor = ResourcePredictor()
    
    def should_scale(self, current_metrics: Dict[str, float]) -> Tuple[str, int, str]:
        """
        Determine if scaling is needed.
        Returns: (action, target_replicas, reason)
        """
        # Record metrics
        for metric, value in current_metrics.items():
            self.metric_history[metric].append({
                'timestamp': time.time(),
                'value': value
            })
        
        # Get current conditions
        current_time = time.time()
        cpu_util = current_metrics.get('cpu_utilization', 0)
        memory_util = current_metrics.get('memory_utilization', 0)
        gpu_util = current_metrics.get('gpu_utilization', 0)
        latency = current_metrics.get('avg_latency_ms', 0)
        error_rate = current_metrics.get('error_rate', 0)
        
        # Calculate scaling signals
        scaling_signals = self._calculate_scaling_signals(current_metrics)
        
        # Predict future load
        predicted_load = self.load_predictor.predict_load(
            list(self.metric_history['cpu_utilization'])
        )
        
        # Check scale-up conditions
        scale_up_needed = self._should_scale_up(
            scaling_signals, current_time, predicted_load
        )
        
        if scale_up_needed:
            target_replicas = min(
                self.current_replicas + self._calculate_scale_increment(scaling_signals),
                self.policy.max_replicas
            )
            
            reason = self._generate_scale_reason("scale_up", scaling_signals)
            return "scale_up", target_replicas, reason
        
        # Check scale-down conditions
        scale_down_needed = self._should_scale_down(
            scaling_signals, current_time, predicted_load
        )
        
        if scale_down_needed:
            target_replicas = max(
                self.current_replicas - self._calculate_scale_decrement(scaling_signals),
                self.policy.min_replicas
            )
            
            reason = self._generate_scale_reason("scale_down", scaling_signals)
            return "scale_down", target_replicas, reason
        
        return "no_change", self.current_replicas, "No scaling needed"
    
    def _calculate_scaling_signals(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate normalized scaling signals from metrics."""
        signals = {}
        
        # CPU signal
        cpu_util = metrics.get('cpu_utilization', 0)
        signals['cpu'] = cpu_util / self.policy.target_cpu_utilization
        
        # Memory signal
        memory_util = metrics.get('memory_utilization', 0)
        signals['memory'] = memory_util / self.policy.target_memory_utilization
        
        # GPU signal (if available)
        gpu_util = metrics.get('gpu_utilization', 0)
        if gpu_util > 0:
            signals['gpu'] = gpu_util / self.policy.target_gpu_utilization
        
        # Latency signal (inverted - higher latency = higher signal)
        latency = metrics.get('avg_latency_ms', 0)
        if latency > 0:
            signals['latency'] = latency / self.policy.target_latency_ms
        
        # Error rate signal
        error_rate = metrics.get('error_rate', 0)
        # Error rate above 1% starts contributing to scaling signal
        signals['error_rate'] = max(0, error_rate - 1.0) * 10
        
        # Queue depth signal
        queue_depth = metrics.get('queue_depth', 0)
        signals['queue'] = queue_depth / 10  # Scale at queue depth 10
        
        # Custom metrics
        for metric_name, target_value in self.policy.custom_metrics.items():
            if metric_name in metrics:
                signals[metric_name] = metrics[metric_name] / target_value
        
        return signals
    
    def _should_scale_up(
        self, 
        signals: Dict[str, float], 
        current_time: float, 
        predicted_load: float
    ) -> bool:
        """Determine if scale-up is needed."""
        # Check cooldown
        if current_time - self.last_scale_up < self.policy.scale_up_cooldown:
            return False
        
        # Check if already at maximum
        if self.current_replicas >= self.policy.max_replicas:
            return False
        
        # Check any signal above scale-up threshold
        high_signals = [
            signal for signal in signals.values() 
            if signal > self.policy.scale_up_threshold
        ]
        
        # Need at least one strong signal or multiple moderate signals
        strong_signal = any(signal > 1.0 for signal in signals.values())
        multiple_moderate = sum(1 for signal in signals.values() if signal > 0.6) >= 2
        
        # Consider predicted load
        predicted_scale_up = predicted_load > 1.2  # 20% increase predicted
        
        return strong_signal or multiple_moderate or predicted_scale_up
    
    def _should_scale_down(
        self, 
        signals: Dict[str, float], 
        current_time: float, 
        predicted_load: float
    ) -> bool:
        """Determine if scale-down is needed."""
        # Check cooldown
        if current_time - self.last_scale_down < self.policy.scale_down_cooldown:
            return False
        
        # Check if already at minimum
        if self.current_replicas <= self.policy.min_replicas:
            return False
        
        # All signals should be below scale-down threshold
        low_signals = [
            signal for signal in signals.values() 
            if signal < self.policy.scale_down_threshold
        ]
        
        all_low = len(low_signals) == len(signals)
        
        # Consider predicted load
        predicted_scale_down = predicted_load < 0.5  # 50% decrease predicted
        
        # Be conservative with scale-down
        consistent_low = self._is_consistently_low(signals)
        
        return all_low and (predicted_scale_down or consistent_low)
    
    def _is_consistently_low(self, current_signals: Dict[str, float]) -> bool:
        """Check if signals have been consistently low."""
        if len(self.metric_history['cpu_utilization']) < 10:
            return False
        
        # Check last 10 measurements
        recent_cpu = [
            entry['value'] for entry in list(self.metric_history['cpu_utilization'])[-10:]
        ]
        
        avg_cpu_signal = np.mean(recent_cpu) / self.policy.target_cpu_utilization
        return avg_cpu_signal < self.policy.scale_down_threshold
    
    def _calculate_scale_increment(self, signals: Dict[str, float]) -> int:
        """Calculate how many replicas to add."""
        max_signal = max(signals.values()) if signals else 0
        
        if max_signal > 2.0:  # Critical load
            return max(2, self.current_replicas // 2)
        elif max_signal > 1.5:  # High load
            return max(1, self.current_replicas // 4)
        else:  # Moderate load
            return 1
    
    def _calculate_scale_decrement(self, signals: Dict[str, float]) -> int:
        """Calculate how many replicas to remove."""
        max_signal = max(signals.values()) if signals else 0
        
        # Be conservative with scale-down
        if max_signal < 0.1:  # Very low load
            return max(1, self.current_replicas // 4)
        else:
            return 1
    
    def _generate_scale_reason(self, action: str, signals: Dict[str, float]) -> str:
        """Generate human-readable scaling reason."""
        high_signals = [
            f"{name}: {value:.2f}" 
            for name, value in signals.items() 
            if value > 0.8
        ]
        
        if high_signals:
            return f"{action} due to high: {', '.join(high_signals)}"
        else:
            avg_signal = np.mean(list(signals.values()))
            return f"{action} due to average signal: {avg_signal:.2f}"
    
    def record_scaling_event(self, action: str, old_replicas: int, new_replicas: int, reason: str):
        """Record a scaling event."""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'old_replicas': old_replicas,
            'new_replicas': new_replicas,
            'reason': reason
        }
        
        self.scaling_events.append(event)
        self.current_replicas = new_replicas
        
        if action == "scale_up":
            self.last_scale_up = time.time()
        elif action == "scale_down":
            self.last_scale_down = time.time()
        
        self.logger.info(f"Scaling event: {reason} ({old_replicas} -> {new_replicas})")


class LoadPredictor:
    """Predicts future load based on historical patterns."""
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.seasonal_patterns: Dict[str, List[float]] = {}
        self.trend_data: deque = deque(maxlen=1000)
    
    def predict_load(self, historical_data: List[Dict[str, Any]]) -> float:
        """Predict future load based on historical patterns."""
        if not historical_data:
            return 1.0
        
        # Extract values and timestamps
        values = [entry['value'] for entry in historical_data[-self.window_size:]]
        
        if len(values) < 5:
            return 1.0
        
        # Simple trend analysis
        recent_avg = np.mean(values[-5:])
        older_avg = np.mean(values[-10:-5]) if len(values) >= 10 else recent_avg
        
        trend_factor = recent_avg / max(older_avg, 1.0)
        
        # Seasonal pattern detection
        seasonal_factor = self._detect_seasonal_pattern(values)
        
        # Combine predictions
        prediction = trend_factor * seasonal_factor
        
        return max(0.1, min(5.0, prediction))  # Clamp between 0.1 and 5.0
    
    def _detect_seasonal_pattern(self, values: List[float]) -> float:
        """Detect seasonal patterns in load data."""
        if len(values) < 20:
            return 1.0
        
        # Simple pattern detection - look for cycles
        autocorrelations = []
        for lag in range(1, min(10, len(values) // 2)):
            if len(values) > lag:
                correlation = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                if not np.isnan(correlation):
                    autocorrelations.append(correlation)
        
        if autocorrelations:
            max_correlation = max(autocorrelations)
            return 1.0 + (max_correlation * 0.2)  # Small seasonal adjustment
        
        return 1.0


class ResourcePredictor:
    """Predicts resource requirements based on workload characteristics."""
    
    def __init__(self):
        self.resource_profiles: Dict[str, Dict[str, float]] = {}
        self.learning_enabled = True
    
    def predict_resources(
        self, 
        workload: Dict[str, Any], 
        target_replicas: int
    ) -> Dict[str, float]:
        """Predict resource requirements for given workload."""
        base_cpu = self._predict_cpu_usage(workload, target_replicas)
        base_memory = self._predict_memory_usage(workload, target_replicas)
        base_gpu = self._predict_gpu_usage(workload, target_replicas)
        
        return {
            'cpu_cores': base_cpu,
            'memory_gb': base_memory,
            'gpu_memory_gb': base_gpu,
            'network_mbps': self._predict_network_usage(workload),
            'storage_iops': self._predict_storage_usage(workload)
        }
    
    def _predict_cpu_usage(self, workload: Dict[str, Any], replicas: int) -> float:
        """Predict CPU usage based on workload characteristics."""
        base_cpu = 0.5  # Base CPU per replica
        
        # Adjust based on model complexity
        model_size = workload.get('model_size_mb', 100)
        complexity_factor = 1.0 + (model_size / 1000.0) * 0.1  # +10% per GB
        
        # Adjust based on batch size
        batch_size = workload.get('avg_batch_size', 1)
        batch_factor = 1.0 + math.log(max(1, batch_size)) * 0.1
        
        # Adjust based on throughput
        target_rps = workload.get('target_rps', 10)
        throughput_factor = 1.0 + (target_rps / 100.0) * 0.2
        
        per_replica_cpu = base_cpu * complexity_factor * batch_factor * throughput_factor
        return per_replica_cpu
    
    def _predict_memory_usage(self, workload: Dict[str, Any], replicas: int) -> float:
        """Predict memory usage based on workload characteristics."""
        base_memory = 1.0  # Base memory per replica in GB
        
        # Model memory requirements
        model_size = workload.get('model_size_mb', 100) / 1024.0  # Convert to GB
        model_memory = model_size * 1.5  # Model + overhead
        
        # Batch processing memory
        batch_size = workload.get('avg_batch_size', 1)
        input_size = workload.get('avg_input_size_kb', 10) / 1024.0 / 1024.0  # Convert to GB
        batch_memory = batch_size * input_size * 2  # Input + processing
        
        # Cache memory
        cache_size = workload.get('cache_size_mb', 100) / 1024.0
        
        per_replica_memory = base_memory + model_memory + batch_memory + cache_size
        return per_replica_memory
    
    def _predict_gpu_usage(self, workload: Dict[str, Any], replicas: int) -> float:
        """Predict GPU memory usage."""
        if not workload.get('uses_gpu', False):
            return 0.0
        
        base_gpu_memory = 2.0  # Base GPU memory in GB
        
        # Model GPU requirements
        model_size = workload.get('model_size_mb', 100) / 1024.0
        gpu_model_memory = model_size * 2.0  # Model on GPU with overhead
        
        # Batch processing GPU memory
        batch_size = workload.get('avg_batch_size', 1)
        gpu_batch_memory = batch_size * 0.1  # Estimate
        
        return base_gpu_memory + gpu_model_memory + gpu_batch_memory
    
    def _predict_network_usage(self, workload: Dict[str, Any]) -> float:
        """Predict network bandwidth requirements."""
        target_rps = workload.get('target_rps', 10)
        avg_request_size = workload.get('avg_request_size_kb', 10) / 1024.0  # MB
        avg_response_size = workload.get('avg_response_size_kb', 5) / 1024.0  # MB
        
        network_mbps = target_rps * (avg_request_size + avg_response_size) * 8  # Convert to Mbps
        return max(10, network_mbps * 1.2)  # 20% overhead, minimum 10 Mbps
    
    def _predict_storage_usage(self, workload: Dict[str, Any]) -> float:
        """Predict storage IOPS requirements."""
        target_rps = workload.get('target_rps', 10)
        
        # Logging and monitoring IOPS
        logging_iops = target_rps * 0.1
        
        # Model loading IOPS
        model_loading_iops = 50  # Periodic model reloading
        
        return max(100, logging_iops + model_loading_iops)


class PerformanceOptimizer:
    """Optimizes service performance through various techniques."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.optimizations: List[Callable] = []
        self.optimization_results: Dict[str, Any] = {}
        self.performance_baseline: Optional[PerformanceProfile] = None
        
        # Register built-in optimizations
        self._register_builtin_optimizations()
    
    def _register_builtin_optimizations(self):
        """Register built-in optimization techniques."""
        self.optimizations.extend([
            self._optimize_batch_processing,
            self._optimize_memory_usage,
            self._optimize_cpu_usage,
            self._optimize_caching,
            self._optimize_threading,
            self._optimize_network_io
        ])
    
    async def optimize_service(
        self, 
        current_profile: PerformanceProfile,
        target_improvements: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Run comprehensive service optimization."""
        target_improvements = target_improvements or {
            'latency_improvement': 0.2,  # 20% improvement
            'throughput_improvement': 0.3,  # 30% improvement
            'resource_efficiency': 0.15  # 15% improvement
        }
        
        self.performance_baseline = current_profile
        optimization_results = []
        
        self.logger.info("Starting performance optimization...")
        
        # Run optimizations concurrently where possible
        optimization_tasks = []
        for optimization in self.optimizations:
            task = asyncio.create_task(
                self._run_optimization_safely(optimization, current_profile)
            )
            optimization_tasks.append((optimization.__name__, task))
        
        # Collect results
        for opt_name, task in optimization_tasks:
            try:
                result = await task
                if result:
                    optimization_results.append({
                        'optimization': opt_name,
                        'result': result,
                        'timestamp': datetime.utcnow().isoformat()
                    })
            except Exception as e:
                self.logger.error(f"Optimization {opt_name} failed: {e}")
        
        # Calculate overall improvement
        overall_improvement = self._calculate_overall_improvement(
            optimization_results, target_improvements
        )
        
        return {
            'baseline_performance': current_profile.__dict__,
            'optimizations_applied': optimization_results,
            'overall_improvement': overall_improvement,
            'recommendations': self._generate_recommendations(optimization_results)
        }
    
    async def _run_optimization_safely(
        self, 
        optimization: Callable, 
        profile: PerformanceProfile
    ) -> Optional[Dict[str, Any]]:
        """Run optimization with error handling."""
        try:
            if asyncio.iscoroutinefunction(optimization):
                return await optimization(profile)
            else:
                return optimization(profile)
        except Exception as e:
            self.logger.error(f"Optimization {optimization.__name__} failed: {e}")
            return None
    
    async def _optimize_batch_processing(self, profile: PerformanceProfile) -> Dict[str, Any]:
        """Optimize batch processing parameters."""
        current_efficiency = profile.batch_efficiency
        
        # Calculate optimal batch size based on current performance
        if profile.avg_request_duration > 0:
            optimal_batch_size = min(
                64,  # Maximum reasonable batch size
                max(1, int(100 / profile.avg_request_duration))  # Target 100ms per request
            )
        else:
            optimal_batch_size = 8  # Default
        
        improvements = {
            'recommended_batch_size': optimal_batch_size,
            'expected_efficiency_gain': min(0.3, (optimal_batch_size / 8) * 0.1),
            'implementation': 'dynamic_batching'
        }
        
        return improvements
    
    async def _optimize_memory_usage(self, profile: PerformanceProfile) -> Dict[str, Any]:
        """Optimize memory usage patterns."""
        memory_util = profile.memory_utilization
        
        improvements = {}
        
        if memory_util > 80:
            improvements.update({
                'memory_pressure': 'high',
                'recommendations': [
                    'Enable memory pooling',
                    'Implement request queuing',
                    'Add memory-based circuit breaker'
                ],
                'expected_improvement': 0.15
            })
        elif memory_util < 30:
            improvements.update({
                'memory_utilization': 'low',
                'recommendations': [
                    'Increase cache size',
                    'Enable model preloading',
                    'Increase batch buffer size'
                ],
                'expected_improvement': 0.1
            })
        
        return improvements
    
    async def _optimize_cpu_usage(self, profile: PerformanceProfile) -> Dict[str, Any]:
        """Optimize CPU usage patterns."""
        cpu_util = profile.cpu_utilization
        
        improvements = {}
        
        if cpu_util > 85:
            improvements.update({
                'cpu_pressure': 'high',
                'recommendations': [
                    'Enable CPU affinity',
                    'Implement request prioritization',
                    'Add CPU-based load balancing'
                ],
                'expected_improvement': 0.2
            })
        
        # Threading optimization
        optimal_threads = min(os.cpu_count(), max(2, int(cpu_util / 20)))
        improvements['recommended_threads'] = optimal_threads
        
        return improvements
    
    async def _optimize_caching(self, profile: PerformanceProfile) -> Dict[str, Any]:
        """Optimize caching strategies."""
        cache_hit_rate = profile.cache_hit_rate
        
        improvements = {}
        
        if cache_hit_rate < 0.3:
            improvements.update({
                'cache_efficiency': 'low',
                'recommendations': [
                    'Increase cache size',
                    'Implement smarter cache eviction',
                    'Add request deduplication'
                ],
                'expected_latency_improvement': 0.25,
                'expected_throughput_improvement': 0.15
            })
        elif cache_hit_rate > 0.8:
            improvements.update({
                'cache_efficiency': 'excellent',
                'recommendations': [
                    'Cache size appears optimal',
                    'Consider read-through caching',
                    'Implement cache warming'
                ]
            })
        
        return improvements
    
    async def _optimize_threading(self, profile: PerformanceProfile) -> Dict[str, Any]:
        """Optimize threading configuration."""
        # Calculate optimal thread pool size
        cpu_cores = os.cpu_count()
        io_bound_ratio = 0.3  # Estimate based on typical workload
        
        optimal_threads = int(cpu_cores * (1 + io_bound_ratio))
        
        return {
            'recommended_thread_pool_size': optimal_threads,
            'recommended_async_workers': min(cpu_cores * 2, 16),
            'implementation': 'thread_pool_optimization',
            'expected_improvement': 0.1
        }
    
    async def _optimize_network_io(self, profile: PerformanceProfile) -> Dict[str, Any]:
        """Optimize network I/O performance."""
        improvements = {
            'recommendations': [
                'Enable HTTP/2',
                'Implement connection pooling',
                'Add response compression',
                'Enable keep-alive connections'
            ],
            'expected_latency_improvement': 0.15,
            'implementation_priority': 'high'
        }
        
        return improvements
    
    def _calculate_overall_improvement(
        self, 
        optimization_results: List[Dict[str, Any]],
        target_improvements: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate overall expected improvement."""
        total_latency_improvement = 0
        total_throughput_improvement = 0
        total_resource_improvement = 0
        
        for result in optimization_results:
            opt_result = result.get('result', {})
            total_latency_improvement += opt_result.get('expected_latency_improvement', 0)
            total_throughput_improvement += opt_result.get('expected_throughput_improvement', 0)
            total_resource_improvement += opt_result.get('expected_improvement', 0)
        
        # Cap improvements to realistic values
        total_latency_improvement = min(0.5, total_latency_improvement)
        total_throughput_improvement = min(0.8, total_throughput_improvement)
        total_resource_improvement = min(0.3, total_resource_improvement)
        
        return {
            'latency_improvement': total_latency_improvement,
            'throughput_improvement': total_throughput_improvement,
            'resource_efficiency_improvement': total_resource_improvement,
            'meets_targets': {
                'latency': total_latency_improvement >= target_improvements.get('latency_improvement', 0),
                'throughput': total_throughput_improvement >= target_improvements.get('throughput_improvement', 0),
                'resource_efficiency': total_resource_improvement >= target_improvements.get('resource_efficiency', 0)
            }
        }
    
    def _generate_recommendations(
        self, 
        optimization_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate prioritized recommendations."""
        all_recommendations = []
        
        for result in optimization_results:
            opt_result = result.get('result', {})
            recommendations = opt_result.get('recommendations', [])
            
            for rec in recommendations:
                all_recommendations.append({
                    'recommendation': rec,
                    'source_optimization': result['optimization'],
                    'expected_impact': opt_result.get('expected_improvement', 0),
                    'priority': opt_result.get('implementation_priority', 'medium')
                })
        
        # Sort by expected impact
        all_recommendations.sort(
            key=lambda x: x['expected_impact'], 
            reverse=True
        )
        
        return all_recommendations[:10]  # Top 10 recommendations


class GlobalLoadBalancer:
    """Intelligent global load balancing system."""
    
    def __init__(self, regions: List[str], logger: Optional[logging.Logger] = None):
        self.regions = regions
        self.logger = logger or logging.getLogger(__name__)
        
        # Region health and performance tracking
        self.region_health: Dict[str, Dict[str, Any]] = {
            region: {
                'healthy': True,
                'latency': 0.0,
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'active_connections': 0,
                'last_update': time.time()
            }
            for region in regions
        }
        
        # Traffic distribution weights
        self.traffic_weights: Dict[str, float] = {
            region: 1.0 for region in regions
        }
        
        # Load balancing strategies
        self.strategies = {
            'round_robin': self._round_robin_strategy,
            'weighted': self._weighted_strategy,
            'latency_based': self._latency_based_strategy,
            'resource_based': self._resource_based_strategy,
            'geographic': self._geographic_strategy
        }
        
        self.current_strategy = 'weighted'
        self.strategy_history: List[Dict[str, Any]] = []
    
    def update_region_status(
        self, 
        region: str, 
        health_data: Dict[str, Any]
    ):
        """Update health status for a region."""
        if region in self.region_health:
            self.region_health[region].update(health_data)
            self.region_health[region]['last_update'] = time.time()
            
            # Auto-adjust traffic weights based on health
            self._auto_adjust_weights()
    
    def route_request(
        self, 
        client_location: Optional[str] = None,
        request_type: str = "default"
    ) -> str:
        """Route request to optimal region."""
        strategy_func = self.strategies.get(self.current_strategy)
        if not strategy_func:
            strategy_func = self.strategies['weighted']
        
        selected_region = strategy_func(client_location, request_type)
        
        # Record routing decision
        self.strategy_history.append({
            'timestamp': time.time(),
            'strategy': self.current_strategy,
            'selected_region': selected_region,
            'client_location': client_location,
            'request_type': request_type
        })
        
        return selected_region
    
    def _round_robin_strategy(
        self, 
        client_location: Optional[str], 
        request_type: str
    ) -> str:
        """Simple round-robin routing."""
        healthy_regions = [
            region for region, health in self.region_health.items()
            if health['healthy']
        ]
        
        if not healthy_regions:
            return self.regions[0]  # Fallback
        
        # Simple round-robin based on request count
        request_count = len(self.strategy_history)
        return healthy_regions[request_count % len(healthy_regions)]
    
    def _weighted_strategy(
        self, 
        client_location: Optional[str], 
        request_type: str
    ) -> str:
        """Weighted routing based on traffic weights."""
        healthy_regions = [
            region for region, health in self.region_health.items()
            if health['healthy']
        ]
        
        if not healthy_regions:
            return self.regions[0]
        
        # Calculate weighted selection
        weights = [self.traffic_weights.get(region, 0.1) for region in healthy_regions]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return healthy_regions[0]
        
        # Weighted random selection
        import random
        rand_val = random.random() * total_weight
        
        cumulative = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            if rand_val <= cumulative:
                return healthy_regions[i]
        
        return healthy_regions[-1]
    
    def _latency_based_strategy(
        self, 
        client_location: Optional[str], 
        request_type: str
    ) -> str:
        """Route to region with lowest latency."""
        healthy_regions = [
            region for region, health in self.region_health.items()
            if health['healthy']
        ]
        
        if not healthy_regions:
            return self.regions[0]
        
        # Find region with lowest latency
        best_region = min(
            healthy_regions,
            key=lambda r: self.region_health[r]['latency']
        )
        
        return best_region
    
    def _resource_based_strategy(
        self, 
        client_location: Optional[str], 
        request_type: str
    ) -> str:
        """Route to region with most available resources."""
        healthy_regions = [
            region for region, health in self.region_health.items()
            if health['healthy']
        ]
        
        if not healthy_regions:
            return self.regions[0]
        
        # Calculate resource availability score
        def resource_score(region: str) -> float:
            health = self.region_health[region]
            cpu_available = 100 - health.get('cpu_usage', 100)
            memory_available = 100 - health.get('memory_usage', 100)
            connection_load = health.get('active_connections', 1000)
            
            # Higher score = more available resources
            score = (cpu_available + memory_available) / 2 - (connection_load / 100)
            return score
        
        best_region = max(healthy_regions, key=resource_score)
        return best_region
    
    def _geographic_strategy(
        self, 
        client_location: Optional[str], 
        request_type: str
    ) -> str:
        """Route based on geographic proximity."""
        if not client_location:
            return self._weighted_strategy(client_location, request_type)
        
        # Simple geographic mapping (in practice, use proper geo-location)
        geo_mapping = {
            'us': ['us-east-1', 'us-west-2'],
            'eu': ['eu-west-1', 'eu-central-1'],
            'asia': ['ap-northeast-1', 'ap-southeast-1']
        }
        
        # Find closest regions
        for geo_region, regions in geo_mapping.items():
            if geo_region in client_location.lower():
                healthy_regions = [
                    region for region in regions
                    if region in self.regions and self.region_health[region]['healthy']
                ]
                
                if healthy_regions:
                    # Use resource-based selection within the geographic region
                    return self._resource_based_strategy(client_location, request_type)
        
        # Fallback to weighted strategy
        return self._weighted_strategy(client_location, request_type)
    
    def _auto_adjust_weights(self):
        """Automatically adjust traffic weights based on region health."""
        total_healthy = sum(
            1 for health in self.region_health.values()
            if health['healthy']
        )
        
        if total_healthy == 0:
            return
        
        for region, health in self.region_health.items():
            if not health['healthy']:
                self.traffic_weights[region] = 0.0
                continue
            
            # Calculate weight based on performance
            cpu_factor = max(0.1, (100 - health.get('cpu_usage', 50)) / 100)
            memory_factor = max(0.1, (100 - health.get('memory_usage', 50)) / 100)
            latency_factor = max(0.1, 1.0 / max(0.01, health.get('latency', 0.1)))
            
            # Normalize latency factor
            latency_factor = min(2.0, latency_factor)
            
            weight = cpu_factor * memory_factor * latency_factor
            self.traffic_weights[region] = weight
        
        # Normalize weights
        total_weight = sum(self.traffic_weights.values())
        if total_weight > 0:
            for region in self.traffic_weights:
                self.traffic_weights[region] /= total_weight
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        recent_history = [
            entry for entry in self.strategy_history
            if time.time() - entry['timestamp'] <= 3600  # Last hour
        ]
        
        region_counts = defaultdict(int)
        for entry in recent_history:
            region_counts[entry['selected_region']] += 1
        
        return {
            'current_strategy': self.current_strategy,
            'traffic_weights': dict(self.traffic_weights),
            'region_health': dict(self.region_health),
            'hourly_distribution': dict(region_counts),
            'total_requests': len(recent_history)
        }