"""Intelligent scaling and resource optimization for NIM services."""

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np
import psutil

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Possible scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    OPTIMIZE = "optimize"


class ResourceMetric(Enum):
    """Resource metrics for scaling decisions."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    GPU_USAGE = "gpu_usage"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"


@dataclass
class ScalingConfig:
    """Configuration for intelligent scaling."""
    min_replicas: int = 1
    max_replicas: int = 20
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    target_gpu_utilization: float = 85.0
    target_response_time_ms: float = 100.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    scale_up_cooldown_seconds: int = 300
    scale_down_cooldown_seconds: int = 600
    prediction_window_minutes: int = 5
    enable_predictive_scaling: bool = True
    enable_cost_optimization: bool = True


@dataclass 
class ResourceUsage:
    """Current resource usage metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_percent: float = 0.0
    network_io_mbps: float = 0.0
    disk_io_mbps: float = 0.0
    queue_length: int = 0
    response_time_ms: float = 0.0
    throughput_rps: float = 0.0
    error_rate_percent: float = 0.0


@dataclass
class ScalingRecommendation:
    """Scaling recommendation with reasoning."""
    action: ScalingAction
    target_replicas: int
    current_replicas: int
    confidence: float
    reasoning: str
    expected_improvement: dict[str, float]
    cost_impact: float
    urgency: int  # 1-5 scale


class PredictiveModel:
    """Simple time series prediction for resource usage."""
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.history = []
        
    def add_measurement(self, usage: ResourceUsage):
        """Add new measurement to history."""
        self.history.append(usage)
        
        # Keep only recent measurements
        if len(self.history) > self.window_size:
            self.history.pop(0)
    
    def predict_usage(self, minutes_ahead: int = 5) -> Optional[ResourceUsage]:
        """Predict resource usage for specified minutes ahead."""
        if len(self.history) < 10:
            return None
            
        # Extract time series for each metric
        timestamps = [h.timestamp for h in self.history]
        cpu_usage = [h.cpu_percent for h in self.history]
        memory_usage = [h.memory_percent for h in self.history]
        response_times = [h.response_time_ms for h in self.history]
        
        try:
            # Simple linear trend prediction
            future_timestamp = timestamps[-1] + (minutes_ahead * 60)
            
            predicted_cpu = self._predict_linear_trend(timestamps, cpu_usage, future_timestamp)
            predicted_memory = self._predict_linear_trend(timestamps, memory_usage, future_timestamp)
            predicted_response_time = self._predict_linear_trend(timestamps, response_times, future_timestamp)
            
            # Apply bounds
            predicted_cpu = max(0, min(100, predicted_cpu))
            predicted_memory = max(0, min(100, predicted_memory))
            predicted_response_time = max(0, predicted_response_time)
            
            return ResourceUsage(
                timestamp=future_timestamp,
                cpu_percent=predicted_cpu,
                memory_percent=predicted_memory,
                response_time_ms=predicted_response_time,
                gpu_percent=0.0,  # Simplified
                throughput_rps=0.0,
                error_rate_percent=0.0
            )
            
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            return None
    
    def _predict_linear_trend(self, x_values: list[float], y_values: list[float], future_x: float) -> float:
        """Simple linear regression prediction."""
        n = len(x_values)
        if n < 2:
            return y_values[-1] if y_values else 0.0
        
        # Calculate linear regression coefficients
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        numerator = sum((x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return y_mean
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        return slope * future_x + intercept
    
    def get_trend_analysis(self) -> dict[str, Any]:
        """Analyze trends in resource usage."""
        if len(self.history) < 10:
            return {"trend": "insufficient_data"}
        
        recent_cpu = [h.cpu_percent for h in self.history[-10:]]
        older_cpu = [h.cpu_percent for h in self.history[-20:-10]] if len(self.history) >= 20 else recent_cpu
        
        cpu_trend = "increasing" if np.mean(recent_cpu) > np.mean(older_cpu) * 1.1 else "decreasing" if np.mean(recent_cpu) < np.mean(older_cpu) * 0.9 else "stable"
        
        return {
            "cpu_trend": cpu_trend,
            "current_avg_cpu": np.mean(recent_cpu),
            "previous_avg_cpu": np.mean(older_cpu),
            "volatility": np.std(recent_cpu),
            "data_points": len(self.history)
        }


class ResourceMonitor:
    """Monitors system resources and collects metrics."""
    
    def __init__(self):
        self.last_measurement_time = 0
        self.request_queue_length = 0
        self.recent_response_times = []
        self.recent_throughput = []
        self.recent_error_count = 0
        self.total_requests = 0
        
    async def collect_metrics(self) -> ResourceUsage:
        """Collect current resource usage metrics."""
        current_time = time.time()
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Network I/O
        network_io = psutil.net_io_counters()
        network_mbps = 0.0
        if hasattr(self, '_last_bytes_sent'):
            time_diff = current_time - self.last_measurement_time
            if time_diff > 0:
                bytes_diff = (network_io.bytes_sent + network_io.bytes_recv) - self._last_bytes_sent
                network_mbps = (bytes_diff / time_diff) / (1024 * 1024)  # MB/s
        
        self._last_bytes_sent = network_io.bytes_sent + network_io.bytes_recv
        self.last_measurement_time = current_time
        
        # Application metrics
        avg_response_time = np.mean(self.recent_response_times) if self.recent_response_times else 0.0
        current_throughput = len(self.recent_throughput)
        error_rate = (self.recent_error_count / max(1, self.total_requests)) * 100
        
        # Clean up old metrics (keep last 60 seconds)
        cutoff_time = current_time - 60
        self.recent_response_times = [rt for rt in self.recent_response_times if rt > cutoff_time]
        self.recent_throughput = [t for t in self.recent_throughput if t > cutoff_time]
        
        return ResourceUsage(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_percent=self._get_gpu_usage(),
            network_io_mbps=network_mbps,
            queue_length=self.request_queue_length,
            response_time_ms=avg_response_time,
            throughput_rps=current_throughput,
            error_rate_percent=error_rate
        )
    
    def _get_gpu_usage(self) -> float:
        """Get GPU utilization (mock implementation)."""
        try:
            # In production, use nvidia-ml-py or similar
            # return pynvml.nvmlDeviceGetUtilizationRates(device).gpu
            return 0.0  # Mock value
        except Exception:
            return 0.0
    
    def record_request_start(self):
        """Record start of request processing."""
        self.request_queue_length += 1
        self.total_requests += 1
    
    def record_request_end(self, response_time_ms: float, is_error: bool = False):
        """Record end of request processing."""
        self.request_queue_length = max(0, self.request_queue_length - 1)
        self.recent_response_times.append(response_time_ms)
        self.recent_throughput.append(time.time())
        
        if is_error:
            self.recent_error_count += 1


class IntelligentScaler:
    """AI-powered scaling decision engine."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.monitor = ResourceMonitor()
        self.predictor = PredictiveModel()
        self.last_scale_action_time = 0
        self.current_replicas = 1
        self.scaling_history = []
        
    async def analyze_and_recommend(self) -> ScalingRecommendation:
        """Analyze current state and recommend scaling action."""
        # Collect current metrics
        current_usage = await self.monitor.collect_metrics()
        self.predictor.add_measurement(current_usage)
        
        # Get trend analysis
        trend_analysis = self.predictor.get_trend_analysis()
        
        # Check cooldown periods
        current_time = time.time()
        time_since_last_action = current_time - self.last_scale_action_time
        
        # Analyze current resource pressure
        resource_pressure = self._calculate_resource_pressure(current_usage)
        
        # Predictive analysis if enabled
        predicted_usage = None
        if self.config.enable_predictive_scaling:
            predicted_usage = self.predictor.predict_usage(self.config.prediction_window_minutes)
        
        # Generate recommendation
        recommendation = self._generate_scaling_recommendation(
            current_usage,
            predicted_usage,
            resource_pressure,
            trend_analysis,
            time_since_last_action
        )
        
        return recommendation
    
    def _calculate_resource_pressure(self, usage: ResourceUsage) -> dict[str, float]:
        """Calculate pressure score for each resource type."""
        pressure = {}
        
        # CPU pressure
        if usage.cpu_percent > self.config.target_cpu_utilization:
            pressure["cpu"] = (usage.cpu_percent - self.config.target_cpu_utilization) / (100 - self.config.target_cpu_utilization)
        else:
            pressure["cpu"] = 0.0
        
        # Memory pressure
        if usage.memory_percent > self.config.target_memory_utilization:
            pressure["memory"] = (usage.memory_percent - self.config.target_memory_utilization) / (100 - self.config.target_memory_utilization)
        else:
            pressure["memory"] = 0.0
        
        # Response time pressure
        if usage.response_time_ms > self.config.target_response_time_ms:
            pressure["response_time"] = min(1.0, usage.response_time_ms / (self.config.target_response_time_ms * 3))
        else:
            pressure["response_time"] = 0.0
        
        # Queue pressure
        if usage.queue_length > 10:
            pressure["queue"] = min(1.0, usage.queue_length / 50.0)
        else:
            pressure["queue"] = 0.0
        
        # Error rate pressure
        if usage.error_rate_percent > 1.0:
            pressure["error_rate"] = min(1.0, usage.error_rate_percent / 10.0)
        else:
            pressure["error_rate"] = 0.0
        
        return pressure
    
    def _generate_scaling_recommendation(
        self,
        current_usage: ResourceUsage,
        predicted_usage: Optional[ResourceUsage],
        pressure: dict[str, float],
        trend_analysis: dict[str, Any],
        time_since_last_action: float
    ) -> ScalingRecommendation:
        """Generate intelligent scaling recommendation."""
        
        max_pressure = max(pressure.values()) if pressure else 0.0
        avg_pressure = sum(pressure.values()) / len(pressure) if pressure else 0.0
        
        # Determine base action
        if max_pressure > 0.7 or (predicted_usage and predicted_usage.cpu_percent > 85):
            base_action = ScalingAction.SCALE_UP
            urgency = 5 if max_pressure > 0.9 else 4 if max_pressure > 0.8 else 3
        elif avg_pressure < 0.2 and self.current_replicas > self.config.min_replicas:
            base_action = ScalingAction.SCALE_DOWN
            urgency = 2 if time_since_last_action > self.config.scale_down_cooldown_seconds else 1
        elif max_pressure > 0.4:
            base_action = ScalingAction.OPTIMIZE
            urgency = 2
        else:
            base_action = ScalingAction.MAINTAIN
            urgency = 1
        
        # Apply cooldown constraints
        if base_action == ScalingAction.SCALE_UP and time_since_last_action < self.config.scale_up_cooldown_seconds:
            base_action = ScalingAction.MAINTAIN
            urgency = 1
        elif base_action == ScalingAction.SCALE_DOWN and time_since_last_action < self.config.scale_down_cooldown_seconds:
            base_action = ScalingAction.MAINTAIN
            urgency = 1
        
        # Calculate target replicas
        target_replicas = self._calculate_target_replicas(base_action, current_usage, pressure)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(base_action, pressure, trend_analysis, current_usage)
        
        # Calculate confidence
        confidence = self._calculate_confidence(pressure, trend_analysis, predicted_usage)
        
        # Estimate improvement
        expected_improvement = self._estimate_improvement(base_action, target_replicas, current_usage)
        
        # Calculate cost impact
        cost_impact = self._calculate_cost_impact(target_replicas, self.current_replicas)
        
        return ScalingRecommendation(
            action=base_action,
            target_replicas=target_replicas,
            current_replicas=self.current_replicas,
            confidence=confidence,
            reasoning=reasoning,
            expected_improvement=expected_improvement,
            cost_impact=cost_impact,
            urgency=urgency
        )
    
    def _calculate_target_replicas(
        self,
        action: ScalingAction,
        usage: ResourceUsage,
        pressure: dict[str, float]
    ) -> int:
        """Calculate optimal number of target replicas."""
        if action == ScalingAction.MAINTAIN or action == ScalingAction.OPTIMIZE:
            return self.current_replicas
        
        max_pressure = max(pressure.values()) if pressure else 0.0
        
        if action == ScalingAction.SCALE_UP:
            # Calculate scaling factor based on pressure
            if max_pressure > 0.9:
                scale_factor = 2.0  # Double replicas for extreme pressure
            elif max_pressure > 0.8:
                scale_factor = 1.5  # 50% increase for high pressure
            else:
                scale_factor = 1.25  # 25% increase for moderate pressure
            
            target = min(self.config.max_replicas, math.ceil(self.current_replicas * scale_factor))
        
        else:  # SCALE_DOWN
            # Conservative scale down
            target = max(self.config.min_replicas, math.floor(self.current_replicas * 0.75))
        
        return target
    
    def _generate_reasoning(
        self,
        action: ScalingAction,
        pressure: dict[str, float],
        trend_analysis: dict[str, Any],
        usage: ResourceUsage
    ) -> str:
        """Generate human-readable reasoning for the recommendation."""
        if action == ScalingAction.SCALE_UP:
            high_pressure_resources = [k for k, v in pressure.items() if v > 0.6]
            return f"High resource pressure detected: {', '.join(high_pressure_resources)}. CPU: {usage.cpu_percent:.1f}%, Response time: {usage.response_time_ms:.1f}ms"
        
        elif action == ScalingAction.SCALE_DOWN:
            return f"Low resource utilization: CPU {usage.cpu_percent:.1f}%, Memory {usage.memory_percent:.1f}%. Trend: {trend_analysis.get('cpu_trend', 'unknown')}"
        
        elif action == ScalingAction.OPTIMIZE:
            return f"Moderate pressure detected. Consider resource optimization before scaling. Queue: {usage.queue_length}, Error rate: {usage.error_rate_percent:.2f}%"
        
        else:
            return f"System operating within normal parameters. CPU: {usage.cpu_percent:.1f}%, Response time: {usage.response_time_ms:.1f}ms"
    
    def _calculate_confidence(
        self,
        pressure: dict[str, float],
        trend_analysis: dict[str, Any],
        predicted_usage: Optional[ResourceUsage]
    ) -> float:
        """Calculate confidence score for the recommendation."""
        confidence = 0.5  # Base confidence
        
        # Higher confidence with more data points
        data_points = trend_analysis.get("data_points", 0)
        confidence += min(0.3, data_points / 100.0)
        
        # Higher confidence with clear pressure signals
        max_pressure = max(pressure.values()) if pressure else 0.0
        if max_pressure > 0.8 or max_pressure < 0.2:
            confidence += 0.2  # Clear signal
        
        # Higher confidence with predictive data
        if predicted_usage:
            confidence += 0.1
        
        # Lower confidence with high volatility
        volatility = trend_analysis.get("volatility", 0)
        if volatility > 20:
            confidence -= 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def _estimate_improvement(self, action: ScalingAction, target_replicas: int, usage: ResourceUsage) -> dict[str, float]:
        """Estimate expected improvements from scaling action."""
        if action == ScalingAction.SCALE_UP and target_replicas > self.current_replicas:
            scale_factor = self.current_replicas / target_replicas
            return {
                "cpu_reduction_percent": (usage.cpu_percent * (1 - scale_factor)),
                "response_time_improvement_ms": usage.response_time_ms * 0.3,  # Estimated improvement
                "queue_reduction": usage.queue_length * 0.5
            }
        elif action == ScalingAction.SCALE_DOWN:
            return {
                "cost_savings_percent": ((self.current_replicas - target_replicas) / self.current_replicas) * 100,
                "resource_efficiency_improvement": 15.0
            }
        else:
            return {"status": "optimization_only"}
    
    def _calculate_cost_impact(self, target_replicas: int, current_replicas: int) -> float:
        """Calculate relative cost impact of scaling change."""
        if target_replicas == current_replicas:
            return 0.0
        
        # Simplified cost model (in practice, would use actual cost data)
        cost_per_replica_per_hour = 1.0  # $1/hour per replica
        replica_change = target_replicas - current_replicas
        
        return replica_change * cost_per_replica_per_hour
    
    async def execute_scaling_recommendation(self, recommendation: ScalingRecommendation) -> bool:
        """Execute the scaling recommendation."""
        if recommendation.action == ScalingAction.MAINTAIN:
            logger.info("No scaling action needed")
            return True
        
        try:
            if recommendation.action == ScalingAction.SCALE_UP:
                success = await self._scale_up(recommendation.target_replicas)
            elif recommendation.action == ScalingAction.SCALE_DOWN:
                success = await self._scale_down(recommendation.target_replicas)
            elif recommendation.action == ScalingAction.OPTIMIZE:
                success = await self._optimize_resources()
            else:
                success = True
            
            if success:
                self.last_scale_action_time = time.time()
                self.current_replicas = recommendation.target_replicas
                
                # Record scaling action
                self.scaling_history.append({
                    "timestamp": time.time(),
                    "action": recommendation.action.value,
                    "from_replicas": recommendation.current_replicas,
                    "to_replicas": recommendation.target_replicas,
                    "reasoning": recommendation.reasoning,
                    "confidence": recommendation.confidence
                })
                
                # Keep history manageable
                if len(self.scaling_history) > 100:
                    self.scaling_history = self.scaling_history[-50:]
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to execute scaling recommendation: {e}")
            return False
    
    async def _scale_up(self, target_replicas: int) -> bool:
        """Scale up to target replicas."""
        logger.info(f"Scaling up from {self.current_replicas} to {target_replicas} replicas")
        
        # In production, this would call Kubernetes API, cloud provider APIs, etc.
        await asyncio.sleep(0.1)  # Simulate API call
        
        return True
    
    async def _scale_down(self, target_replicas: int) -> bool:
        """Scale down to target replicas."""
        logger.info(f"Scaling down from {self.current_replicas} to {target_replicas} replicas")
        
        # In production, would gracefully terminate instances
        await asyncio.sleep(0.1)  # Simulate API call
        
        return True
    
    async def _optimize_resources(self) -> bool:
        """Optimize resource allocation without changing replica count."""
        logger.info("Optimizing resource allocation")
        
        # In production, might adjust:
        # - JVM heap sizes
        # - Connection pool sizes
        # - Worker thread counts
        # - Cache sizes
        
        return True
    
    def get_scaling_status(self) -> dict[str, Any]:
        """Get current scaling status and history."""
        return {
            "current_replicas": self.current_replicas,
            "min_replicas": self.config.min_replicas,
            "max_replicas": self.config.max_replicas,
            "last_action_time": self.last_scale_action_time,
            "recent_actions": self.scaling_history[-10:],  # Last 10 actions
            "predictive_scaling_enabled": self.config.enable_predictive_scaling,
            "cost_optimization_enabled": self.config.enable_cost_optimization
        }


class AutoScalingOrchestrator:
    """Main orchestrator for automatic scaling operations."""
    
    def __init__(self, config: ScalingConfig):
        self.scaler = IntelligentScaler(config)
        self.running = False
        self.check_interval_seconds = 30
        
    async def start_auto_scaling(self):
        """Start automatic scaling loop."""
        self.running = True
        logger.info("Starting intelligent auto-scaling orchestrator")
        
        while self.running:
            try:
                # Analyze and get recommendation
                recommendation = await self.scaler.analyze_and_recommend()
                
                # Log recommendation
                logger.info(
                    f"Scaling recommendation: {recommendation.action.value} "
                    f"(confidence: {recommendation.confidence:.2f}, urgency: {recommendation.urgency}/5)"
                )
                logger.debug(f"Reasoning: {recommendation.reasoning}")
                
                # Execute if urgent and confident
                if recommendation.urgency >= 4 and recommendation.confidence >= 0.7:
                    success = await self.scaler.execute_scaling_recommendation(recommendation)
                    logger.info(f"Scaling action executed: {success}")
                elif recommendation.urgency >= 3 and recommendation.confidence >= 0.8:
                    success = await self.scaler.execute_scaling_recommendation(recommendation)
                    logger.info(f"High-confidence scaling action executed: {success}")
                
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
            
            # Wait for next check
            await asyncio.sleep(self.check_interval_seconds)
    
    def stop_auto_scaling(self):
        """Stop automatic scaling."""
        self.running = False
        logger.info("Stopped auto-scaling orchestrator")
    
    async def manual_scale_check(self) -> ScalingRecommendation:
        """Manually trigger scaling analysis."""
        return await self.scaler.analyze_and_recommend()


# Global auto-scaling orchestrator
global_auto_scaler: Optional[AutoScalingOrchestrator] = None


def initialize_auto_scaling(config: ScalingConfig = None) -> AutoScalingOrchestrator:
    """Initialize global auto-scaling orchestrator."""
    global global_auto_scaler
    
    config = config or ScalingConfig()
    global_auto_scaler = AutoScalingOrchestrator(config)
    
    return global_auto_scaler