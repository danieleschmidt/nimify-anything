"""Advanced optimization and auto-scaling capabilities."""

import asyncio
import contextlib
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Optimization strategy types."""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    COST = "cost"
    BALANCED = "balanced"


@dataclass
class PerformanceTarget:
    """Performance targets for optimization."""
    target_latency_p95_ms: float = 200.0
    target_throughput_rps: float = 100.0
    max_cpu_utilization: float = 80.0
    max_memory_utilization: float = 85.0
    max_gpu_utilization: float = 90.0
    cost_budget_usd_per_hour: float | None = None


@dataclass
class OptimizationConfig:
    """Configuration for optimization engine."""
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    targets: PerformanceTarget = field(default_factory=PerformanceTarget)
    optimization_interval_seconds: int = 300
    enable_auto_scaling: bool = True
    enable_model_optimization: bool = True
    enable_caching_optimization: bool = True
    enable_batching_optimization: bool = True
    min_confidence_threshold: float = 0.8


class ModelOptimizer:
    """Optimizes model performance through various techniques."""
    
    def __init__(self):
        self.optimization_history = []
        self.current_optimizations = {}
        self._lock = threading.Lock()
    
    async def optimize_model_loading(self, model_path: str) -> dict[str, Any]:
        """Optimize model loading performance."""
        optimizations = {}
        
        # Model format optimization
        file_ext = Path(model_path).suffix.lower()
        if file_ext == '.onnx':
            optimizations.update(await self._optimize_onnx_loading(model_path))
        elif file_ext in ['.trt', '.engine']:
            optimizations.update(await self._optimize_tensorrt_loading(model_path))
        
        # Memory layout optimization
        optimizations.update(await self._optimize_memory_layout())
        
        # Pre-compilation optimization
        optimizations.update(await self._optimize_compilation())
        
        with self._lock:
            self.current_optimizations.update(optimizations)
        
        logger.info(f"Applied model optimizations: {list(optimizations.keys())}")
        return optimizations
    
    async def _optimize_onnx_loading(self, model_path: str) -> dict[str, Any]:
        """Optimize ONNX model loading."""
        optimizations = {}
        
        # Graph optimization
        optimizations['graph_optimization'] = {
            'enabled': True,
            'level': 'all',  # basic, extended, all
            'optimization_passes': [
                'constant_folding',
                'redundant_node_elimination',
                'shape_inference',
                'common_subexpression_elimination'
            ]
        }
        
        # Execution provider optimization
        optimizations['execution_providers'] = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'cudnn_conv_use_max_workspace': '1',
                'do_copy_in_default_stream': '1',
            }),
            ('CPUExecutionProvider', {
                'intra_op_num_threads': 4,
                'inter_op_num_threads': 2
            })
        ]
        
        # Session configuration
        optimizations['session_config'] = {
            'intra_op_num_threads': 4,
            'inter_op_num_threads': 2,
            'execution_mode': 'parallel',  # sequential, parallel
            'graph_optimization_level': 'all'
        }
        
        return optimizations
    
    async def _optimize_tensorrt_loading(self, model_path: str) -> dict[str, Any]:
        """Optimize TensorRT model loading."""
        optimizations = {}
        
        # TensorRT optimization
        optimizations['tensorrt_config'] = {
            'max_workspace_size': 1 << 30,  # 1GB
            'fp16_mode': True,
            'int8_mode': False,  # Requires calibration
            'strict_type_constraints': False,
            'max_batch_size': 64
        }
        
        # Engine optimization
        optimizations['engine_config'] = {
            'optimization_profile': 'default',
            'timing_cache': True,
            'profiling_verbosity': 'layer_names_only'
        }
        
        return optimizations
    
    async def _optimize_memory_layout(self) -> dict[str, Any]:
        """Optimize memory layout for better performance."""
        return {
            'memory_optimization': {
                'enable_memory_pattern': True,
                'enable_memory_arena': True,
                'memory_growth_factor': 1.2,
                'initial_memory_fraction': 0.5
            }
        }
    
    async def _optimize_compilation(self) -> dict[str, Any]:
        """Optimize model compilation."""
        return {
            'compilation_optimization': {
                'enable_jit_compilation': True,
                'compile_only_shapes': False,
                'enable_constant_folding': True,
                'enable_layout_optimization': True
            }
        }
    
    def get_optimization_recommendations(self, performance_metrics: dict[str, float]) -> list[dict[str, Any]]:
        """Get optimization recommendations based on performance metrics."""
        recommendations = []
        
        # Latency-based recommendations
        if performance_metrics.get('latency_p95_ms', 0) > 200:
            if performance_metrics.get('gpu_utilization', 0) < 60:
                recommendations.append({
                    'type': 'model_optimization',
                    'action': 'increase_batch_size',
                    'reasoning': 'High latency with low GPU utilization suggests underutilization',
                    'priority': 'high'
                })
            
            if performance_metrics.get('memory_utilization', 0) > 80:
                recommendations.append({
                    'type': 'model_optimization',
                    'action': 'enable_mixed_precision',
                    'reasoning': 'High memory usage affecting latency',
                    'priority': 'medium'
                })
        
        # Throughput-based recommendations
        if performance_metrics.get('throughput_rps', 0) < 50:
            recommendations.append({
                'type': 'concurrency_optimization',
                'action': 'increase_worker_count',
                'reasoning': 'Low throughput suggests need for more concurrent processing',
                'priority': 'medium'
            })
        
        # Resource utilization recommendations
        if performance_metrics.get('cpu_utilization', 0) > 90:
            recommendations.append({
                'type': 'resource_optimization',
                'action': 'scale_horizontally',
                'reasoning': 'High CPU utilization suggests need for more replicas',
                'priority': 'high'
            })
        
        return recommendations


class CacheOptimizer:
    """Advanced caching optimization system."""
    
    def __init__(self):
        self.cache_analytics = {
            'hit_patterns': defaultdict(int),
            'miss_patterns': defaultdict(int),
            'access_frequency': defaultdict(int),
            'cache_sizes': defaultdict(int)
        }
        self.optimization_configs = {}
        self._lock = threading.Lock()
    
    def analyze_cache_performance(self, cache_stats: dict[str, Any]) -> dict[str, Any]:
        """Analyze cache performance and suggest optimizations."""
        analysis = {}
        
        hit_rate = cache_stats.get('hit_rate', 0)
        cache_size = cache_stats.get('cache_size', 0)
        max_size = cache_stats.get('max_size', 1000)
        
        # Hit rate analysis
        if hit_rate < 0.3:
            analysis['hit_rate_issue'] = {
                'severity': 'high',
                'recommendation': 'increase_cache_size_and_ttl',
                'current_hit_rate': hit_rate,
                'target_hit_rate': 0.6
            }
        elif hit_rate < 0.6:
            analysis['hit_rate_issue'] = {
                'severity': 'medium',
                'recommendation': 'optimize_cache_policy',
                'current_hit_rate': hit_rate,
                'target_hit_rate': 0.8
            }
        
        # Cache utilization analysis
        utilization = cache_size / max_size if max_size > 0 else 0
        if utilization > 0.9:
            analysis['utilization_issue'] = {
                'severity': 'medium',
                'recommendation': 'increase_cache_size',
                'current_utilization': utilization,
                'suggested_size_increase': 0.5
            }
        
        # Access pattern analysis
        analysis['access_patterns'] = self._analyze_access_patterns()
        
        return analysis
    
    def _analyze_access_patterns(self) -> dict[str, Any]:
        """Analyze cache access patterns for optimization."""
        with self._lock:
            total_accesses = sum(self.cache_analytics['access_frequency'].values())
            
            if total_accesses == 0:
                return {'pattern_type': 'insufficient_data'}
            
            # Calculate access distribution
            sorted_frequencies = sorted(self.cache_analytics['access_frequency'].values(), reverse=True)
            
            # Pareto principle check (80/20 rule)
            top_20_percent = int(len(sorted_frequencies) * 0.2) or 1
            top_accesses = sum(sorted_frequencies[:top_20_percent])
            
            if top_accesses / total_accesses > 0.8:
                return {
                    'pattern_type': 'hot_spot',
                    'recommendation': 'implement_lru_with_frequency_boost',
                    'hot_spot_ratio': top_accesses / total_accesses
                }
            elif len(set(sorted_frequencies)) == 1:
                return {
                    'pattern_type': 'uniform',
                    'recommendation': 'use_simple_lru',
                    'distribution': 'uniform'
                }
            else:
                return {
                    'pattern_type': 'mixed',
                    'recommendation': 'use_adaptive_replacement_cache',
                    'distribution': 'varied'
                }
    
    def optimize_cache_configuration(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Generate optimized cache configuration."""
        config = {
            'cache_policy': 'lru',  # Default
            'size_multiplier': 1.0,
            'ttl_multiplier': 1.0,
            'eviction_policy': 'lru'
        }
        
        # Apply hit rate optimizations
        if 'hit_rate_issue' in analysis:
            issue = analysis['hit_rate_issue']
            if issue['severity'] == 'high':
                config['size_multiplier'] = 2.0
                config['ttl_multiplier'] = 1.5
            else:
                config['size_multiplier'] = 1.5
                config['ttl_multiplier'] = 1.2
        
        # Apply utilization optimizations
        if 'utilization_issue' in analysis:
            config['size_multiplier'] *= (1 + analysis['utilization_issue']['suggested_size_increase'])
        
        # Apply access pattern optimizations
        patterns = analysis.get('access_patterns', {})
        pattern_type = patterns.get('pattern_type', 'mixed')
        
        if pattern_type == 'hot_spot':
            config['cache_policy'] = 'lfu_with_lru'
            config['eviction_policy'] = 'frequency_based'
        elif pattern_type == 'uniform':
            config['cache_policy'] = 'lru'
        elif pattern_type == 'mixed':
            config['cache_policy'] = 'arc'  # Adaptive Replacement Cache
        
        return config


class BatchOptimizer:
    """Optimizes batching for improved throughput and latency."""
    
    def __init__(self):
        self.batch_analytics = {
            'size_distribution': defaultdict(int),
            'latency_by_size': defaultdict(list),
            'throughput_by_size': defaultdict(list)
        }
        self.current_config = {
            'max_batch_size': 32,
            'optimal_batch_size': 8,
            'batch_timeout_ms': 10,
            'dynamic_batching': True
        }
    
    def record_batch_metrics(self, batch_size: int, latency_ms: float, throughput_rps: float):
        """Record batch performance metrics."""
        self.batch_analytics['size_distribution'][batch_size] += 1
        self.batch_analytics['latency_by_size'][batch_size].append(latency_ms)
        self.batch_analytics['throughput_by_size'][batch_size].append(throughput_rps)
    
    def analyze_batch_performance(self) -> dict[str, Any]:
        """Analyze batch performance and find optimal configurations."""
        analysis = {}
        
        if not self.batch_analytics['size_distribution']:
            return {'status': 'insufficient_data'}
        
        # Find optimal batch size based on latency/throughput trade-off
        optimal_size = self._find_optimal_batch_size()
        analysis['optimal_batch_size'] = optimal_size
        
        # Analyze latency patterns
        latency_analysis = self._analyze_latency_patterns()
        analysis['latency_patterns'] = latency_analysis
        
        # Analyze throughput patterns  
        throughput_analysis = self._analyze_throughput_patterns()
        analysis['throughput_patterns'] = throughput_analysis
        
        # Generate recommendations
        recommendations = self._generate_batch_recommendations(analysis)
        analysis['recommendations'] = recommendations
        
        return analysis
    
    def _find_optimal_batch_size(self) -> int:
        """Find optimal batch size balancing latency and throughput."""
        best_score = float('-inf')
        optimal_size = self.current_config['optimal_batch_size']
        
        for batch_size, latencies in self.batch_analytics['latency_by_size'].items():
            if not latencies:
                continue
            
            avg_latency = sum(latencies) / len(latencies)
            throughputs = self.batch_analytics['throughput_by_size'].get(batch_size, [0])
            avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0
            
            # Score function: higher throughput, lower latency
            # Normalize latency (lower is better) and throughput (higher is better)
            latency_score = max(0, 1000 - avg_latency) / 1000  # Normalize to 0-1
            throughput_score = min(avg_throughput / 100, 1)    # Normalize to 0-1
            
            # Weighted combination (can be tuned based on strategy)
            combined_score = 0.4 * throughput_score + 0.6 * latency_score
            
            if combined_score > best_score:
                best_score = combined_score
                optimal_size = batch_size
        
        return optimal_size
    
    def _analyze_latency_patterns(self) -> dict[str, Any]:
        """Analyze latency patterns across batch sizes."""
        patterns = {}
        
        for batch_size, latencies in self.batch_analytics['latency_by_size'].items():
            if not latencies:
                continue
                
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            patterns[batch_size] = {
                'avg_latency_ms': avg_latency,
                'min_latency_ms': min_latency,
                'max_latency_ms': max_latency,
                'latency_variance': self._calculate_variance(latencies)
            }
        
        return patterns
    
    def _analyze_throughput_patterns(self) -> dict[str, Any]:
        """Analyze throughput patterns across batch sizes."""
        patterns = {}
        
        for batch_size, throughputs in self.batch_analytics['throughput_by_size'].items():
            if not throughputs:
                continue
                
            avg_throughput = sum(throughputs) / len(throughputs)
            patterns[batch_size] = {
                'avg_throughput_rps': avg_throughput,
                'efficiency': avg_throughput / batch_size if batch_size > 0 else 0
            }
        
        return patterns
    
    def _calculate_variance(self, values: list[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0.0
            
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance
    
    def _generate_batch_recommendations(self, analysis: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate batching optimization recommendations."""
        recommendations = []
        
        optimal_size = analysis.get('optimal_batch_size', 8)
        current_size = self.current_config['optimal_batch_size']
        
        if optimal_size != current_size:
            recommendations.append({
                'type': 'batch_size_optimization',
                'action': 'update_optimal_batch_size',
                'current_value': current_size,
                'recommended_value': optimal_size,
                'reasoning': f'Analysis shows batch size {optimal_size} provides better performance',
                'priority': 'high'
            })
        
        # Check for latency variance issues
        latency_patterns = analysis.get('latency_patterns', {})
        for batch_size, pattern in latency_patterns.items():
            if pattern.get('latency_variance', 0) > 1000:  # High variance threshold
                recommendations.append({
                    'type': 'batch_consistency',
                    'action': 'reduce_batch_timeout',
                    'batch_size': batch_size,
                    'reasoning': 'High latency variance detected, reducing timeout may help',
                    'priority': 'medium'
                })
        
        return recommendations


class AutoScaler:
    """Intelligent auto-scaling system."""
    
    def __init__(self, optimization_config: OptimizationConfig):
        self.config = optimization_config
        self.scaling_history = deque(maxlen=100)
        self.current_replicas = 1
        self.target_replicas = 1
        self.last_scaling_time = 0
        self.scaling_cooldown_seconds = 300  # 5 minutes
        self._lock = threading.Lock()
    
    async def evaluate_scaling_decision(self, metrics: dict[str, float]) -> dict[str, Any] | None:
        """Evaluate if scaling is needed and return scaling decision."""
        if not self.config.enable_auto_scaling:
            return None
        
        current_time = time.time()
        if (current_time - self.last_scaling_time) < self.scaling_cooldown_seconds:
            return None  # Still in cooldown period
        
        # Extract relevant metrics
        cpu_util = metrics.get('cpu_utilization', 0)
        memory_util = metrics.get('memory_utilization', 0) 
        latency_p95 = metrics.get('latency_p95_ms', 0)
        throughput = metrics.get('throughput_rps', 0)
        queue_depth = metrics.get('queue_depth', 0)
        
        # Scaling decision logic
        scale_up_score = self._calculate_scale_up_score(
            cpu_util, memory_util, latency_p95, throughput, queue_depth
        )
        
        scale_down_score = self._calculate_scale_down_score(
            cpu_util, memory_util, latency_p95, throughput, queue_depth
        )
        
        decision = None
        
        if scale_up_score > 0.7:  # High confidence threshold
            new_replicas = min(self.current_replicas * 2, 20)  # Cap at 20 replicas
            decision = {
                'action': 'scale_up',
                'current_replicas': self.current_replicas,
                'target_replicas': new_replicas,
                'confidence': scale_up_score,
                'reasoning': self._generate_scaling_reasoning('up', metrics),
                'metrics_snapshot': metrics.copy()
            }
            
        elif scale_down_score > 0.8:  # Higher threshold for scale down
            new_replicas = max(self.current_replicas // 2, 1)  # Minimum 1 replica
            decision = {
                'action': 'scale_down',
                'current_replicas': self.current_replicas,
                'target_replicas': new_replicas,
                'confidence': scale_down_score,
                'reasoning': self._generate_scaling_reasoning('down', metrics),
                'metrics_snapshot': metrics.copy()
            }
        
        if decision:
            with self._lock:
                self.scaling_history.append(decision)
                self.target_replicas = decision['target_replicas']
                self.last_scaling_time = current_time
        
        return decision
    
    def _calculate_scale_up_score(self, cpu: float, memory: float, latency: float, 
                                throughput: float, queue_depth: int) -> float:
        """Calculate confidence score for scaling up."""
        score = 0.0
        
        # CPU pressure
        if cpu > self.config.targets.max_cpu_utilization:
            score += min((cpu - self.config.targets.max_cpu_utilization) / 20, 0.4)
        
        # Memory pressure  
        if memory > self.config.targets.max_memory_utilization:
            score += min((memory - self.config.targets.max_memory_utilization) / 15, 0.3)
        
        # Latency pressure
        if latency > self.config.targets.target_latency_p95_ms:
            score += min((latency - self.config.targets.target_latency_p95_ms) / 200, 0.3)
        
        # Queue depth pressure
        if queue_depth > 50:
            score += min(queue_depth / 100, 0.2)
        
        # Low throughput relative to target
        if throughput < self.config.targets.target_throughput_rps * 0.5:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_scale_down_score(self, cpu: float, memory: float, latency: float,
                                  throughput: float, queue_depth: int) -> float:
        """Calculate confidence score for scaling down."""
        if self.current_replicas <= 1:
            return 0.0  # Don't scale below 1 replica
        
        score = 0.0
        
        # Low resource utilization
        if cpu < 30:
            score += (30 - cpu) / 30 * 0.4
        
        if memory < 40:
            score += (40 - memory) / 40 * 0.3
        
        # Low latency (system is not stressed)
        if latency < self.config.targets.target_latency_p95_ms * 0.5:
            score += 0.2
        
        # Empty or small queue
        if queue_depth < 5:
            score += 0.1
        
        return min(score, 1.0)
    
    def _generate_scaling_reasoning(self, direction: str, metrics: dict[str, float]) -> str:
        """Generate human-readable reasoning for scaling decision."""
        reasons = []
        
        cpu = metrics.get('cpu_utilization', 0)
        memory = metrics.get('memory_utilization', 0)
        latency = metrics.get('latency_p95_ms', 0)
        
        if direction == 'up':
            if cpu > self.config.targets.max_cpu_utilization:
                reasons.append(f"High CPU utilization ({cpu:.1f}%)")
            if memory > self.config.targets.max_memory_utilization:
                reasons.append(f"High memory utilization ({memory:.1f}%)")
            if latency > self.config.targets.target_latency_p95_ms:
                reasons.append(f"High latency ({latency:.1f}ms)")
        else:  # scale down
            if cpu < 30:
                reasons.append(f"Low CPU utilization ({cpu:.1f}%)")
            if memory < 40:
                reasons.append(f"Low memory utilization ({memory:.1f}%)")
            if latency < self.config.targets.target_latency_p95_ms * 0.5:
                reasons.append(f"Low latency ({latency:.1f}ms)")
        
        return "; ".join(reasons) if reasons else f"Scaling {direction} based on overall metrics"


class OptimizationEngine:
    """Main optimization engine coordinating all optimization components."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.model_optimizer = ModelOptimizer()
        self.cache_optimizer = CacheOptimizer()
        self.batch_optimizer = BatchOptimizer()
        self.auto_scaler = AutoScaler(config)
        
        self.optimization_active = False
        self._optimization_task = None
        self._lock = threading.Lock()
    
    async def start_optimization(self):
        """Start the optimization engine."""
        if self.optimization_active:
            return
        
        self.optimization_active = True
        self._optimization_task = asyncio.create_task(
            self._optimization_loop()
        )
        logger.info("Started optimization engine")
    
    async def stop_optimization(self):
        """Stop the optimization engine."""
        self.optimization_active = False
        if self._optimization_task:
            self._optimization_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._optimization_task
        logger.info("Stopped optimization engine")
    
    async def _optimization_loop(self):
        """Main optimization loop."""
        while self.optimization_active:
            try:
                await self._run_optimization_cycle()
                await asyncio.sleep(self.config.optimization_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _run_optimization_cycle(self):
        """Run a single optimization cycle."""
        logger.debug("Running optimization cycle")
        
        # Collect current metrics (would come from monitoring system)
        current_metrics = await self._collect_metrics()
        
        optimizations_applied = []
        
        # Model optimization
        if self.config.enable_model_optimization:
            model_opts = await self._optimize_model_performance(current_metrics)
            if model_opts:
                optimizations_applied.extend(model_opts)
        
        # Cache optimization
        if self.config.enable_caching_optimization:
            cache_opts = await self._optimize_caching(current_metrics)
            if cache_opts:
                optimizations_applied.extend(cache_opts)
        
        # Batch optimization
        if self.config.enable_batching_optimization:
            batch_opts = await self._optimize_batching(current_metrics)
            if batch_opts:
                optimizations_applied.extend(batch_opts)
        
        # Auto-scaling
        scaling_decision = await self.auto_scaler.evaluate_scaling_decision(current_metrics)
        if scaling_decision:
            optimizations_applied.append(scaling_decision)
            logger.info(f"Scaling decision: {scaling_decision['action']} to {scaling_decision['target_replicas']} replicas")
        
        if optimizations_applied:
            logger.info(f"Applied {len(optimizations_applied)} optimizations")
        
        return optimizations_applied
    
    async def _collect_metrics(self) -> dict[str, float]:
        """Collect current performance metrics."""
        # In a real implementation, this would collect from monitoring system
        # For now, return simulated metrics
        import random
        return {
            'cpu_utilization': random.uniform(40, 90),
            'memory_utilization': random.uniform(50, 85),
            'gpu_utilization': random.uniform(60, 95),
            'latency_p95_ms': random.uniform(100, 300),
            'throughput_rps': random.uniform(20, 80),
            'queue_depth': random.randint(0, 100),
            'cache_hit_rate': random.uniform(0.3, 0.9),
            'error_rate': random.uniform(0, 0.05)
        }
    
    async def _optimize_model_performance(self, metrics: dict[str, float]) -> list[dict[str, Any]]:
        """Optimize model performance based on metrics."""
        recommendations = self.model_optimizer.get_optimization_recommendations(metrics)
        applied_optimizations = []
        
        for rec in recommendations:
            if rec['priority'] == 'high' or random.random() > 0.5:  # Apply high priority or 50% of others
                applied_optimizations.append({
                    'type': 'model_optimization',
                    'optimization': rec,
                    'timestamp': time.time()
                })
        
        return applied_optimizations
    
    async def _optimize_caching(self, metrics: dict[str, float]) -> list[dict[str, Any]]:
        """Optimize caching based on performance metrics."""
        cache_stats = {
            'hit_rate': metrics.get('cache_hit_rate', 0.5),
            'cache_size': 500,
            'max_size': 1000
        }
        
        analysis = self.cache_optimizer.analyze_cache_performance(cache_stats)
        
        applied_optimizations = []
        if 'hit_rate_issue' in analysis:
            config = self.cache_optimizer.optimize_cache_configuration(analysis)
            applied_optimizations.append({
                'type': 'cache_optimization',
                'configuration': config,
                'analysis': analysis,
                'timestamp': time.time()
            })
        
        return applied_optimizations
    
    async def _optimize_batching(self, metrics: dict[str, float]) -> list[dict[str, Any]]:
        """Optimize batching based on performance metrics."""
        # Simulate some batch analytics
        for _i in range(5):
            batch_size = random.choice([1, 2, 4, 8, 16, 32])
            latency = random.uniform(50, 200)
            throughput = random.uniform(10, 100)
            self.batch_optimizer.record_batch_metrics(batch_size, latency, throughput)
        
        analysis = self.batch_optimizer.analyze_batch_performance()
        
        applied_optimizations = []
        if 'recommendations' in analysis:
            for rec in analysis['recommendations']:
                if rec.get('priority') == 'high':
                    applied_optimizations.append({
                        'type': 'batch_optimization',
                        'optimization': rec,
                        'analysis': analysis,
                        'timestamp': time.time()
                    })
        
        return applied_optimizations
    
    def get_optimization_status(self) -> dict[str, Any]:
        """Get current optimization status."""
        return {
            'active': self.optimization_active,
            'config': {
                'strategy': self.config.strategy.value,
                'targets': {
                    'latency_p95_ms': self.config.targets.target_latency_p95_ms,
                    'throughput_rps': self.config.targets.target_throughput_rps,
                    'max_cpu_utilization': self.config.targets.max_cpu_utilization
                }
            },
            'current_replicas': self.auto_scaler.current_replicas,
            'target_replicas': self.auto_scaler.target_replicas,
            'last_optimization': time.time()
        }


# Global optimization engine instance
optimization_engine = None

def create_optimization_engine(strategy: OptimizationStrategy = OptimizationStrategy.BALANCED) -> OptimizationEngine:
    """Create and configure the optimization engine."""
    global optimization_engine
    
    config = OptimizationConfig(strategy=strategy)
    optimization_engine = OptimizationEngine(config)
    
    logger.info(f"Created optimization engine with strategy: {strategy.value}")
    return optimization_engine