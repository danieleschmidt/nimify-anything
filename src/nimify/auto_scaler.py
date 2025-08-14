"""Intelligent auto-scaling system for dynamic resource management."""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import deque
import math

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down" 
    MAINTAIN = "maintain"


class ScalingTrigger(Enum):
    """Scaling trigger types."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_LENGTH = "queue_length"
    GPU_UTILIZATION = "gpu_utilization"
    CUSTOM_METRIC = "custom_metric"


@dataclass
class ScalingMetric:
    """Individual scaling metric."""
    name: str
    current_value: float
    threshold_up: float
    threshold_down: float
    weight: float = 1.0
    enabled: bool = True


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling behavior."""
    
    # Basic scaling parameters
    min_replicas: int = 2
    max_replicas: int = 20
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    target_response_time_ms: float = 500.0
    
    # Advanced parameters
    scale_up_cooldown: int = 300    # seconds
    scale_down_cooldown: int = 600  # seconds
    scale_up_threshold: float = 2.0  # multiplier for aggressive scaling
    scale_down_threshold: float = 0.5
    
    # Predictive scaling
    enable_predictive: bool = True
    prediction_window: int = 900    # seconds
    seasonal_patterns: bool = True
    
    # Custom metrics
    custom_metrics: List[ScalingMetric] = None
    
    # Safety settings
    max_scale_up_rate: float = 0.5   # max 50% increase per scaling event
    max_scale_down_rate: float = 0.3 # max 30% decrease per scaling event
    emergency_scale_threshold: float = 95.0  # emergency scaling at 95% util
    
    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = []


@dataclass 
class ScalingDecision:
    """Scaling decision with reasoning."""
    action: ScalingAction
    current_replicas: int
    target_replicas: int
    confidence: float
    reasoning: List[str]
    triggered_by: List[ScalingTrigger]
    timestamp: float
    cooldown_remaining: float = 0


class MetricsCollector:
    """Collects and processes metrics for scaling decisions."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.metrics_history: deque = deque(maxlen=1000)
        self.lock = threading.RLock()
        
    async def collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        try:
            import psutil
            
            # Basic system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Network metrics
            network = psutil.net_io_counters()
            
            metrics = {
                "cpu_utilization": cpu_percent,
                "memory_utilization": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "network_bytes_per_sec": getattr(self, '_last_network_bytes', 0),
                "timestamp": time.time()
            }
            
            # Calculate network rate
            if hasattr(self, '_last_network_bytes') and hasattr(self, '_last_network_time'):
                time_diff = metrics["timestamp"] - self._last_network_time
                if time_diff > 0:
                    bytes_diff = network.bytes_sent + network.bytes_recv - self._last_network_bytes
                    metrics["network_bytes_per_sec"] = bytes_diff / time_diff
            
            self._last_network_bytes = network.bytes_sent + network.bytes_recv
            self._last_network_time = metrics["timestamp"]
            
            # GPU metrics if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                metrics.update({
                    "gpu_utilization": gpu_util.gpu,
                    "gpu_memory_utilization": (gpu_memory.used / gpu_memory.total) * 100
                })
            except:
                pass  # GPU not available
            
            # Application-specific metrics (would be injected from monitoring system)
            metrics.update(await self._collect_application_metrics())
            
            # Store in history
            with self.lock:
                self.metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return {}
    
    async def _collect_application_metrics(self) -> Dict[str, float]:
        """Collect application-specific metrics."""
        # In a real implementation, this would integrate with:
        # - Prometheus metrics
        # - Application performance counters
        # - Load balancer metrics
        # - Database connection pools
        
        # Simulated application metrics
        return {
            "request_rate_per_sec": 10.0,
            "average_response_time_ms": 150.0,
            "active_connections": 50,
            "queue_length": 5,
            "error_rate": 0.01
        }
    
    def get_metrics_summary(self, window_seconds: int = 300) -> Dict[str, Any]:
        """Get aggregated metrics over a time window."""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        with self.lock:
            recent_metrics = [
                m for m in self.metrics_history 
                if m.get("timestamp", 0) >= cutoff_time
            ]
        
        if not recent_metrics:
            return {}
        
        # Calculate aggregations
        summary = {}
        metric_keys = set()
        for metrics in recent_metrics:
            metric_keys.update(metrics.keys())
        
        for key in metric_keys:
            if key == "timestamp":
                continue
                
            values = [m.get(key, 0) for m in recent_metrics if key in m]
            if values:
                summary[f"{key}_avg"] = sum(values) / len(values)
                summary[f"{key}_max"] = max(values)
                summary[f"{key}_min"] = min(values)
                summary[f"{key}_current"] = values[-1] if values else 0
        
        summary["sample_count"] = len(recent_metrics)
        summary["window_seconds"] = window_seconds
        
        return summary


class PredictiveScaler:
    """Predictive scaling based on historical patterns and trends."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.historical_data: deque = deque(maxlen=10000)  # Store more history for prediction
        self.seasonal_patterns: Dict[str, List[float]] = {}
        self.trend_analysis: Dict[str, float] = {}
        
    def record_metrics(self, metrics: Dict[str, float]):
        """Record metrics for predictive analysis."""
        self.historical_data.append({
            **metrics,
            "hour_of_day": time.localtime().tm_hour,
            "day_of_week": time.localtime().tm_wday
        })
        
        # Update seasonal patterns periodically
        if len(self.historical_data) % 100 == 0:
            self._update_seasonal_patterns()
    
    def _update_seasonal_patterns(self):
        """Update seasonal patterns from historical data."""
        try:
            # Group by hour of day
            hourly_patterns = {}
            for hour in range(24):
                hour_data = [
                    m for m in self.historical_data 
                    if m.get("hour_of_day") == hour
                ]
                if len(hour_data) >= 5:  # Need sufficient samples
                    hourly_patterns[hour] = {
                        "cpu_avg": sum(m.get("cpu_utilization", 0) for m in hour_data) / len(hour_data),
                        "memory_avg": sum(m.get("memory_utilization", 0) for m in hour_data) / len(hour_data),
                        "request_rate_avg": sum(m.get("request_rate_per_sec", 0) for m in hour_data) / len(hour_data)
                    }
            
            self.seasonal_patterns["hourly"] = hourly_patterns
            
            # Group by day of week
            daily_patterns = {}
            for day in range(7):
                day_data = [
                    m for m in self.historical_data 
                    if m.get("day_of_week") == day
                ]
                if len(day_data) >= 20:  # Need sufficient samples
                    daily_patterns[day] = {
                        "cpu_avg": sum(m.get("cpu_utilization", 0) for m in day_data) / len(day_data),
                        "memory_avg": sum(m.get("memory_utilization", 0) for m in day_data) / len(day_data),
                        "request_rate_avg": sum(m.get("request_rate_per_sec", 0) for m in day_data) / len(day_data)
                    }
            
            self.seasonal_patterns["daily"] = daily_patterns
            
        except Exception as e:
            logger.error(f"Failed to update seasonal patterns: {e}")
    
    def predict_load(self, minutes_ahead: int = 15) -> Dict[str, float]:
        """Predict system load for the specified time ahead."""
        if not self.config.enable_predictive or len(self.historical_data) < 50:
            return {}
        
        try:
            current_time = time.localtime()
            future_time = time.localtime(time.time() + minutes_ahead * 60)
            
            predictions = {}
            
            # Use seasonal patterns if available
            if "hourly" in self.seasonal_patterns:
                future_hour = future_time.tm_hour
                if future_hour in self.seasonal_patterns["hourly"]:
                    hour_pattern = self.seasonal_patterns["hourly"][future_hour]
                    predictions.update({
                        f"predicted_{k}": v for k, v in hour_pattern.items()
                    })
            
            # Simple trend analysis
            if len(self.historical_data) >= 10:
                recent_data = list(self.historical_data)[-10:]
                
                # Calculate trends for key metrics
                for metric in ["cpu_utilization", "memory_utilization", "request_rate_per_sec"]:
                    values = [m.get(metric, 0) for m in recent_data]
                    if len(values) >= 5:
                        # Simple linear trend
                        x = list(range(len(values)))
                        trend = self._calculate_linear_trend(x, values)
                        
                        current_value = values[-1]
                        predicted_value = current_value + (trend * minutes_ahead)
                        predictions[f"predicted_{metric}"] = max(0, predicted_value)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to predict load: {e}")
            return {}
    
    def _calculate_linear_trend(self, x: List[float], y: List[float]) -> float:
        """Calculate linear trend slope."""
        n = len(x)
        if n < 2:
            return 0
        
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0


class IntelligentAutoScaler:
    """Main auto-scaling engine with intelligent decision making."""
    
    def __init__(self, config: ScalingConfig, namespace: str = "default"):
        self.config = config
        self.namespace = namespace
        self.metrics_collector = MetricsCollector(config)
        self.predictive_scaler = PredictiveScaler(config)
        
        self.current_replicas = config.min_replicas
        self.last_scale_time = 0
        self.last_scale_action = ScalingAction.MAINTAIN
        self.scaling_history: deque = deque(maxlen=100)
        
        self.is_running = False
        self.lock = threading.RLock()
        
        logger.info(f"Auto-scaler initialized for namespace '{namespace}' with config: {config}")
    
    async def start(self, interval_seconds: int = 30):
        """Start the auto-scaling loop."""
        self.is_running = True
        logger.info(f"Starting auto-scaler with {interval_seconds}s interval")
        
        while self.is_running:
            try:
                await self._scaling_iteration()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Auto-scaler iteration failed: {e}")
                await asyncio.sleep(interval_seconds)
    
    def stop(self):
        """Stop the auto-scaling loop."""
        self.is_running = False
        logger.info("Auto-scaler stopped")
    
    async def _scaling_iteration(self):
        """Single iteration of scaling decision logic."""
        # Collect current metrics
        current_metrics = await self.metrics_collector.collect_metrics()
        if not current_metrics:
            return
        
        # Record for predictive analysis
        self.predictive_scaler.record_metrics(current_metrics)
        
        # Get aggregated metrics
        metrics_summary = self.metrics_collector.get_metrics_summary()
        
        # Get predictions
        predictions = self.predictive_scaler.predict_load()
        
        # Make scaling decision
        decision = await self._make_scaling_decision(current_metrics, metrics_summary, predictions)
        
        # Execute scaling if needed
        if decision.action != ScalingAction.MAINTAIN:
            await self._execute_scaling_decision(decision)
        
        # Record decision
        with self.lock:
            self.scaling_history.append(decision)
    
    async def _make_scaling_decision(
        self, 
        current_metrics: Dict[str, float],
        metrics_summary: Dict[str, Any],
        predictions: Dict[str, float]
    ) -> ScalingDecision:
        """Make intelligent scaling decision based on multiple factors."""
        
        current_time = time.time()
        reasoning = []
        triggered_by = []
        confidence = 0.0
        
        # Check cooldown periods
        time_since_last_scale = current_time - self.last_scale_time
        scale_up_cooldown_remaining = max(0, self.config.scale_up_cooldown - time_since_last_scale)
        scale_down_cooldown_remaining = max(0, self.config.scale_down_cooldown - time_since_last_scale)
        
        if self.last_scale_action == ScalingAction.SCALE_UP and scale_up_cooldown_remaining > 0:
            return ScalingDecision(
                action=ScalingAction.MAINTAIN,
                current_replicas=self.current_replicas,
                target_replicas=self.current_replicas,
                confidence=1.0,
                reasoning=[f"Scale-up cooldown active ({scale_up_cooldown_remaining:.0f}s remaining)"],
                triggered_by=[],
                timestamp=current_time,
                cooldown_remaining=scale_up_cooldown_remaining
            )
        
        if self.last_scale_action == ScalingAction.SCALE_DOWN and scale_down_cooldown_remaining > 0:
            return ScalingDecision(
                action=ScalingAction.MAINTAIN,
                current_replicas=self.current_replicas,
                target_replicas=self.current_replicas,
                confidence=1.0,
                reasoning=[f"Scale-down cooldown active ({scale_down_cooldown_remaining:.0f}s remaining)"],
                triggered_by=[],
                timestamp=current_time,
                cooldown_remaining=scale_down_cooldown_remaining
            )
        
        # Collect scaling signals
        scale_up_signals = []
        scale_down_signals = []
        
        # CPU utilization
        cpu_current = current_metrics.get("cpu_utilization", 0)
        cpu_avg = metrics_summary.get("cpu_utilization_avg", 0)
        
        if cpu_avg > self.config.target_cpu_utilization * 1.2:  # 20% above target
            scale_up_signals.append(("CPU", cpu_avg, 0.8))
            triggered_by.append(ScalingTrigger.CPU_UTILIZATION)
        elif cpu_avg < self.config.target_cpu_utilization * 0.5:  # 50% below target
            scale_down_signals.append(("CPU", cpu_avg, 0.6))
            triggered_by.append(ScalingTrigger.CPU_UTILIZATION)
        
        # Memory utilization
        memory_current = current_metrics.get("memory_utilization", 0)
        memory_avg = metrics_summary.get("memory_utilization_avg", 0)
        
        if memory_avg > self.config.target_memory_utilization * 1.1:  # 10% above target
            scale_up_signals.append(("Memory", memory_avg, 0.9))
            triggered_by.append(ScalingTrigger.MEMORY_UTILIZATION)
        elif memory_avg < self.config.target_memory_utilization * 0.4:  # 60% below target
            scale_down_signals.append(("Memory", memory_avg, 0.5))
        
        # Response time
        response_time = current_metrics.get("average_response_time_ms", 0)
        if response_time > self.config.target_response_time_ms * 1.5:
            scale_up_signals.append(("Response Time", response_time, 0.7))
            triggered_by.append(ScalingTrigger.RESPONSE_TIME)
        
        # GPU utilization (if available)
        gpu_util = current_metrics.get("gpu_utilization", 0)
        if gpu_util > 85:
            scale_up_signals.append(("GPU", gpu_util, 0.9))
            triggered_by.append(ScalingTrigger.GPU_UTILIZATION)
        elif gpu_util < 20:
            scale_down_signals.append(("GPU", gpu_util, 0.4))
        
        # Queue length
        queue_length = current_metrics.get("queue_length", 0)
        if queue_length > 10:
            scale_up_signals.append(("Queue", queue_length, 0.8))
            triggered_by.append(ScalingTrigger.QUEUE_LENGTH)
        
        # Emergency scaling
        emergency_conditions = []
        if cpu_current > self.config.emergency_scale_threshold:
            emergency_conditions.append(f"CPU: {cpu_current:.1f}%")
        if memory_current > self.config.emergency_scale_threshold:
            emergency_conditions.append(f"Memory: {memory_current:.1f}%")
        
        if emergency_conditions:
            # Emergency scale up regardless of cooldown
            target_replicas = min(
                self.config.max_replicas,
                int(self.current_replicas * (1 + self.config.max_scale_up_rate))
            )
            
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                current_replicas=self.current_replicas,
                target_replicas=target_replicas,
                confidence=1.0,
                reasoning=[f"Emergency scaling triggered: {', '.join(emergency_conditions)}"],
                triggered_by=[ScalingTrigger.CPU_UTILIZATION, ScalingTrigger.MEMORY_UTILIZATION],
                timestamp=current_time
            )
        
        # Predictive scaling
        if predictions:
            predicted_cpu = predictions.get("predicted_cpu_utilization", 0)
            predicted_memory = predictions.get("predicted_memory_utilization", 0)
            
            if predicted_cpu > self.config.target_cpu_utilization * 1.3:
                scale_up_signals.append(("Predicted CPU", predicted_cpu, 0.6))
                reasoning.append(f"Predicted CPU spike: {predicted_cpu:.1f}%")
            
            if predicted_memory > self.config.target_memory_utilization * 1.2:
                scale_up_signals.append(("Predicted Memory", predicted_memory, 0.6))
                reasoning.append(f"Predicted memory spike: {predicted_memory:.1f}%")
        
        # Calculate overall scaling signal
        if scale_up_signals:
            # Weighted average confidence for scale up
            total_weight = sum(signal[2] for signal in scale_up_signals)
            confidence = total_weight / len(scale_up_signals)
            
            # Calculate target replicas
            max_signal_strength = max(signal[1] / 100 for signal in scale_up_signals)  # Normalize to 0-1
            scale_factor = 1 + (max_signal_strength * self.config.max_scale_up_rate)
            target_replicas = min(
                self.config.max_replicas,
                max(self.current_replicas + 1, int(self.current_replicas * scale_factor))
            )
            
            reasoning.extend([f"{signal[0]}: {signal[1]:.1f}" for signal in scale_up_signals])
            
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                current_replicas=self.current_replicas,
                target_replicas=target_replicas,
                confidence=confidence,
                reasoning=reasoning,
                triggered_by=triggered_by,
                timestamp=current_time
            )
        
        elif scale_down_signals:
            # Only scale down if we have strong confidence and multiple signals
            if len(scale_down_signals) >= 2:
                total_weight = sum(signal[2] for signal in scale_down_signals)
                confidence = total_weight / len(scale_down_signals)
                
                # Conservative scale down
                target_replicas = max(
                    self.config.min_replicas,
                    int(self.current_replicas * (1 - self.config.max_scale_down_rate))
                )
                
                reasoning.extend([f"{signal[0]}: {signal[1]:.1f}" for signal in scale_down_signals])
                
                return ScalingDecision(
                    action=ScalingAction.SCALE_DOWN,
                    current_replicas=self.current_replicas,
                    target_replicas=target_replicas,
                    confidence=confidence,
                    reasoning=reasoning,
                    triggered_by=triggered_by,
                    timestamp=current_time
                )
        
        # No scaling needed
        return ScalingDecision(
            action=ScalingAction.MAINTAIN,
            current_replicas=self.current_replicas,
            target_replicas=self.current_replicas,
            confidence=1.0,
            reasoning=["All metrics within acceptable ranges"],
            triggered_by=[],
            timestamp=current_time
        )
    
    async def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute the scaling decision."""
        try:
            # In a real implementation, this would:
            # 1. Update Kubernetes HPA or Deployment replicas
            # 2. Update load balancer configuration
            # 3. Notify monitoring systems
            # 4. Update service mesh configuration
            
            logger.info(
                f"Scaling {decision.action.value}: {decision.current_replicas} -> {decision.target_replicas} "
                f"(confidence: {decision.confidence:.2f}, reasons: {', '.join(decision.reasoning)})"
            )
            
            # Simulate scaling API call
            await self._update_replica_count(decision.target_replicas)
            
            # Update internal state
            with self.lock:
                self.current_replicas = decision.target_replicas
                self.last_scale_time = decision.timestamp
                self.last_scale_action = decision.action
            
            # Log scaling event
            self._log_scaling_event(decision)
            
        except Exception as e:
            logger.error(f"Failed to execute scaling decision: {e}")
    
    async def _update_replica_count(self, target_replicas: int):
        """Update the actual replica count in the deployment system."""
        # Placeholder for actual scaling implementation
        # In production, this would use Kubernetes API, AWS Auto Scaling, etc.
        
        logger.info(f"Updating replica count to {target_replicas}")
        
        # Simulate API call delay
        await asyncio.sleep(0.1)
    
    def _log_scaling_event(self, decision: ScalingDecision):
        """Log scaling event for audit and analysis."""
        event_data = {
            "timestamp": decision.timestamp,
            "action": decision.action.value,
            "current_replicas": decision.current_replicas,
            "target_replicas": decision.target_replicas,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "triggered_by": [trigger.value for trigger in decision.triggered_by],
            "namespace": self.namespace
        }
        
        # In production, send to logging/monitoring system
        logger.info(f"Scaling event: {json.dumps(event_data)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current auto-scaler status."""
        with self.lock:
            recent_decisions = list(self.scaling_history)[-10:]
            
            return {
                "is_running": self.is_running,
                "current_replicas": self.current_replicas,
                "min_replicas": self.config.min_replicas,
                "max_replicas": self.config.max_replicas,
                "last_scale_action": self.last_scale_action.value,
                "last_scale_time": self.last_scale_time,
                "cooldown_remaining": max(0, 
                    (self.last_scale_time + self.config.scale_up_cooldown) - time.time()
                    if self.last_scale_action == ScalingAction.SCALE_UP else
                    (self.last_scale_time + self.config.scale_down_cooldown) - time.time()
                ),
                "recent_decisions": [asdict(decision) for decision in recent_decisions],
                "metrics_history_size": len(self.metrics_collector.metrics_history),
                "predictive_enabled": self.config.enable_predictive,
                "seasonal_patterns_available": bool(self.predictive_scaler.seasonal_patterns)
            }
    
    def get_recommendations(self) -> List[str]:
        """Get scaling recommendations based on current state and history."""
        recommendations = []
        
        with self.lock:
            if len(self.scaling_history) >= 5:
                recent_actions = [d.action for d in list(self.scaling_history)[-5:]]
                
                # Check for oscillation
                if len(set(recent_actions)) > 1:
                    scale_ups = recent_actions.count(ScalingAction.SCALE_UP)
                    scale_downs = recent_actions.count(ScalingAction.SCALE_DOWN)
                    
                    if scale_ups > 0 and scale_downs > 0:
                        recommendations.append(
                            "Detected scaling oscillation - consider adjusting thresholds or cooldown periods"
                        )
                
                # Check for consistent scale-ups
                if recent_actions.count(ScalingAction.SCALE_UP) >= 3:
                    recommendations.append(
                        "Frequent scale-ups detected - consider increasing baseline replicas or reviewing resource limits"
                    )
                
                # Check if hitting limits
                if self.current_replicas >= self.config.max_replicas * 0.9:
                    recommendations.append(
                        "Approaching maximum replica limit - consider increasing max_replicas or optimizing resource usage"
                    )
                
                if self.current_replicas <= self.config.min_replicas:
                    recommendations.append(
                        "At minimum replica count - monitor for adequate capacity during traffic spikes"
                    )
        
        if not recommendations:
            recommendations.append("Auto-scaling operating normally")
        
        return recommendations