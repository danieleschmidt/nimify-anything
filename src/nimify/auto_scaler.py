"""Auto-scaling system based on load and performance metrics."""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
import json

import numpy as np

from .performance_optimizer import PerformanceMonitor

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "scale_up"
    DOWN = "scale_down"
    STABLE = "stable"


class ResourceType(Enum):
    """Resource types for scaling."""
    REPLICAS = "replicas"
    CPU = "cpu"
    MEMORY = "memory" 
    GPU = "gpu"


@dataclass
class ScalingRule:
    """Rule for scaling decisions."""
    metric_name: str
    threshold_up: float
    threshold_down: float
    resource_type: ResourceType
    scale_factor: float = 1.5
    cooldown_seconds: float = 300.0  # 5 minutes
    min_value: int = 1
    max_value: int = 100
    
    def should_scale_up(self, metric_value: float) -> bool:
        """Check if should scale up."""
        return metric_value > self.threshold_up
    
    def should_scale_down(self, metric_value: float) -> bool:
        """Check if should scale down."""
        return metric_value < self.threshold_down


@dataclass
class ScalingDecision:
    """Scaling decision with context."""
    direction: ScalingDirection
    resource_type: ResourceType
    current_value: int
    target_value: int
    reason: str
    confidence: float
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Collects and aggregates metrics for scaling decisions."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {}
        self.lock = threading.Lock()
    
    def record_metric(self, name: str, value: float):
        """Record a metric value."""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
            
            self.metrics[name].append(value)
            
            # Keep only recent values
            if len(self.metrics[name]) > self.window_size:
                self.metrics[name] = self.metrics[name][-self.window_size:]
    
    def get_metric_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get statistics for a metric."""
        with self.lock:
            if name not in self.metrics or not self.metrics[name]:
                return None
            
            values = self.metrics[name]
            recent_values = values[-min(50, len(values)):]  # Last 50 measurements
            
            return {
                'current': values[-1],
                'mean': np.mean(recent_values),
                'p95': np.percentile(recent_values, 95),
                'trend': self._calculate_trend(values[-10:])
            }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend of values (positive = increasing)."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        n = len(values)
        x = np.arange(n)
        y = np.array(values)
        
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x * x) - np.sum(x) * np.sum(x))
        return float(slope)


class AutoScaler:
    """Intelligent auto-scaling system."""
    
    def __init__(self):
        self.rules: List[ScalingRule] = []
        self.metrics_collector = MetricsCollector()
        self.current_resources = {
            ResourceType.REPLICAS: 2,
            ResourceType.CPU: 1000,  # millicores
            ResourceType.MEMORY: 2048,  # MB
            ResourceType.GPU: 1
        }
        self.last_scaling_time = {}
        self.scaling_history = []
        self.lock = threading.Lock()
        
        # Add default scaling rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default scaling rules."""
        # CPU-based scaling
        self.add_rule(ScalingRule(
            metric_name="cpu_utilization",
            threshold_up=70.0,
            threshold_down=30.0,
            resource_type=ResourceType.REPLICAS,
            scale_factor=2.0,
            cooldown_seconds=300.0,
            min_value=1,
            max_value=50
        ))
        
        # Latency-based scaling
        self.add_rule(ScalingRule(
            metric_name="p95_latency_ms",
            threshold_up=200.0,
            threshold_down=50.0,
            resource_type=ResourceType.REPLICAS,
            scale_factor=1.5,
            cooldown_seconds=180.0,
            min_value=1,
            max_value=20
        ))
        
        # Request rate based scaling
        self.add_rule(ScalingRule(
            metric_name="requests_per_second",
            threshold_up=100.0,
            threshold_down=20.0,
            resource_type=ResourceType.REPLICAS,
            scale_factor=2.0,
            cooldown_seconds=240.0,
            min_value=1,
            max_value=100
        ))
    
    def add_rule(self, rule: ScalingRule):
        """Add a scaling rule."""
        with self.lock:
            self.rules.append(rule)
            logger.info(f"Added scaling rule: {rule}")
    
    def record_metric(self, name: str, value: float):
        """Record a metric for scaling decisions."""
        self.metrics_collector.record_metric(name, value)
    
    def evaluate_scaling(self) -> List[ScalingDecision]:
        """Evaluate if scaling is needed."""
        decisions = []
        current_time = time.time()
        
        with self.lock:
            for rule in self.rules:
                # Check cooldown
                last_scaling = self.last_scaling_time.get(f"{rule.resource_type.value}_{rule.metric_name}", 0)
                if current_time - last_scaling < rule.cooldown_seconds:
                    continue
                
                # Get metric stats
                stats = self.metrics_collector.get_metric_stats(rule.metric_name)
                if not stats:
                    continue
                
                current_value = self.current_resources[rule.resource_type]
                target_value = current_value
                direction = ScalingDirection.STABLE
                reason = ""
                confidence = 0.0
                
                # Check scaling conditions
                if rule.should_scale_up(stats['mean']):
                    # Scale up
                    if stats['trend'] > 0:  # Trend is increasing
                        confidence = min(1.0, (stats['mean'] - rule.threshold_up) / rule.threshold_up)
                        target_value = min(
                            rule.max_value,
                            max(rule.min_value, int(current_value * rule.scale_factor))
                        )
                        direction = ScalingDirection.UP
                        reason = f"{rule.metric_name} is {stats['mean']:.2f} (threshold: {rule.threshold_up})"
                
                elif rule.should_scale_down(stats['mean']):
                    # Scale down
                    if stats['trend'] <= 0:  # Trend is stable or decreasing
                        confidence = min(1.0, (rule.threshold_down - stats['mean']) / rule.threshold_down)
                        target_value = max(
                            rule.min_value,
                            min(rule.max_value, int(current_value / rule.scale_factor))
                        )
                        direction = ScalingDirection.DOWN
                        reason = f"{rule.metric_name} is {stats['mean']:.2f} (threshold: {rule.threshold_down})"
                
                # Create decision if scaling is needed
                if direction != ScalingDirection.STABLE and target_value != current_value:
                    decision = ScalingDecision(
                        direction=direction,
                        resource_type=rule.resource_type,
                        current_value=current_value,
                        target_value=target_value,
                        reason=reason,
                        confidence=confidence,
                        metrics={rule.metric_name: stats['mean']}
                    )
                    decisions.append(decision)
        
        return decisions
    
    def apply_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Apply scaling decision."""
        try:
            with self.lock:
                # Update resource value
                old_value = self.current_resources[decision.resource_type]
                self.current_resources[decision.resource_type] = decision.target_value
                
                # Record scaling action
                self.last_scaling_time[f"{decision.resource_type.value}"] = time.time()
                self.scaling_history.append(decision)
                
                # Keep only recent history (last 100 scaling actions)
                if len(self.scaling_history) > 100:
                    self.scaling_history = self.scaling_history[-100:]
                
                logger.info(
                    f"Scaled {decision.resource_type.value}: "
                    f"{old_value} -> {decision.target_value} "
                    f"({decision.reason})"
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to apply scaling decision: {e}")
            return False
    
    def get_current_resources(self) -> Dict[ResourceType, int]:
        """Get current resource allocation."""
        with self.lock:
            return self.current_resources.copy()
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics."""
        with self.lock:
            total_scalings = len(self.scaling_history)
            scale_ups = sum(1 for d in self.scaling_history if d.direction == ScalingDirection.UP)
            scale_downs = sum(1 for d in self.scaling_history if d.direction == ScalingDirection.DOWN)
            
            recent_decisions = self.scaling_history[-10:]  # Last 10 decisions
            avg_confidence = np.mean([d.confidence for d in recent_decisions]) if recent_decisions else 0.0
            
            return {
                "current_resources": {k.value: v for k, v in self.current_resources.items()},
                "total_scalings": total_scalings,
                "scale_ups": scale_ups,
                "scale_downs": scale_downs,
                "avg_confidence": avg_confidence,
                "rules_count": len(self.rules)
            }


# Global auto-scaler instance
global_auto_scaler = AutoScaler()