"""Advanced monitoring and metrics collection system."""

import asyncio
import json
import logging
import psutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
import threading
import uuid

from prometheus_client import Counter, Gauge, Histogram, Summary, Info, generate_latest
import numpy as np


@dataclass
class MetricDefinition:
    """Definition of a custom metric."""
    name: str
    metric_type: str  # counter, gauge, histogram, summary
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms


class MetricsCollector:
    """Advanced metrics collection and aggregation system."""
    
    def __init__(self, service_name: str = "nimify", collection_interval: int = 10):
        self.service_name = service_name
        self.collection_interval = collection_interval
        self.logger = logging.getLogger(__name__)
        
        # Built-in metrics
        self._setup_builtin_metrics()
        
        # Custom metrics registry
        self.custom_metrics: Dict[str, Any] = {}
        
        # Metric history for analysis
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Active requests tracking
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        
        # Performance baselines
        self.performance_baselines: Dict[str, float] = {}
        
        # Start background collection
        self.collection_thread = threading.Thread(
            target=self._run_collection_loop, 
            daemon=True
        )
        self.collection_thread.start()
        
        # Service state
        self.service_start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
    
    def _setup_builtin_metrics(self):
        """Setup built-in Prometheus metrics."""
        # Request metrics
        self.request_total = Counter(
            f'{self.service_name}_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.request_duration = Histogram(
            f'{self.service_name}_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.request_size = Histogram(
            f'{self.service_name}_request_size_bytes',
            'Request size in bytes',
            ['method', 'endpoint'],
            buckets=[100, 1000, 10000, 100000, 1000000]
        )
        
        self.response_size = Histogram(
            f'{self.service_name}_response_size_bytes',
            'Response size in bytes',
            ['method', 'endpoint'],
            buckets=[100, 1000, 10000, 100000, 1000000]
        )
        
        # Inference metrics
        self.inference_duration = Histogram(
            f'{self.service_name}_inference_duration_seconds',
            'Model inference duration in seconds',
            ['model_name', 'model_version'],
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        self.batch_size = Histogram(
            f'{self.service_name}_batch_size',
            'Batch sizes processed',
            ['model_name'],
            buckets=[1, 2, 4, 8, 16, 32, 64, 128]
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            f'{self.service_name}_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.memory_usage = Gauge(
            f'{self.service_name}_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.gpu_utilization = Gauge(
            f'{self.service_name}_gpu_utilization_percent',
            'GPU utilization percentage'
        )
        
        self.gpu_memory_usage = Gauge(
            f'{self.service_name}_gpu_memory_usage_bytes',
            'GPU memory usage in bytes'
        )
        
        # Service metrics
        self.active_connections = Gauge(
            f'{self.service_name}_active_connections',
            'Number of active connections'
        )
        
        self.uptime_seconds = Gauge(
            f'{self.service_name}_uptime_seconds',
            'Service uptime in seconds'
        )
        
        # Error metrics
        self.error_total = Counter(
            f'{self.service_name}_errors_total',
            'Total number of errors',
            ['error_type', 'error_category', 'severity']
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            f'{self.service_name}_cache_hits_total',
            'Total cache hits'
        )
        
        self.cache_misses = Counter(
            f'{self.service_name}_cache_misses_total',
            'Total cache misses'
        )
        
        self.cache_size = Gauge(
            f'{self.service_name}_cache_size',
            'Current cache size'
        )
        
        # Rate limiting metrics
        self.rate_limit_hits = Counter(
            f'{self.service_name}_rate_limit_hits_total',
            'Rate limit violations',
            ['client_id']
        )
        
        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            f'{self.service_name}_circuit_breaker_open',
            'Circuit breaker state (1=open, 0=closed)',
            ['service']
        )
        
        # Business metrics
        self.throughput = Gauge(
            f'{self.service_name}_throughput_requests_per_second',
            'Current throughput in requests per second'
        )
        
        self.error_rate = Gauge(
            f'{self.service_name}_error_rate_percent',
            'Current error rate percentage'
        )
    
    def _run_collection_loop(self):
        """Background loop for collecting system metrics."""
        while True:
            try:
                self._collect_system_metrics()
                self._update_derived_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent()
            self.cpu_usage.set(cpu_percent)
            self._record_metric_history('cpu_usage', cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.used)
            self._record_metric_history('memory_usage', memory.used)
            
            # Disk metrics (if needed)
            disk = psutil.disk_usage('/')
            self._record_metric_history('disk_usage', disk.percent)
            
            # Network metrics (if needed)
            network = psutil.net_io_counters()
            self._record_metric_history('network_bytes_sent', network.bytes_sent)
            self._record_metric_history('network_bytes_recv', network.bytes_recv)
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            self._record_metric_history('process_memory_rss', process_memory.rss)
            
            # Update uptime
            uptime = time.time() - self.service_start_time
            self.uptime_seconds.set(uptime)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
        
        # GPU metrics (if available)
        self._collect_gpu_metrics()
    
    def _collect_gpu_metrics(self):
        """Collect GPU metrics if available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.gpu_utilization.set(utilization.gpu)
                
                # GPU memory
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.gpu_memory_usage.set(memory_info.used)
                
                self._record_metric_history(f'gpu_{i}_utilization', utilization.gpu)
                self._record_metric_history(f'gpu_{i}_memory_used', memory_info.used)
                
        except ImportError:
            # pynvml not available
            pass
        except Exception as e:
            self.logger.debug(f"GPU metrics collection failed: {e}")
    
    def _update_derived_metrics(self):
        """Update derived metrics based on collected data."""
        try:
            # Calculate throughput
            current_time = time.time()
            window_size = 60  # 1 minute window
            
            # Get requests in the last minute
            recent_requests = [
                timestamp for timestamp in self.metric_history.get('request_timestamps', [])
                if current_time - timestamp <= window_size
            ]
            
            throughput = len(recent_requests) / window_size
            self.throughput.set(throughput)
            
            # Calculate error rate
            if self.total_requests > 0:
                error_rate = (self.failed_requests / self.total_requests) * 100
                self.error_rate.set(error_rate)
            
        except Exception as e:
            self.logger.error(f"Error updating derived metrics: {e}")
    
    def _record_metric_history(self, metric_name: str, value: Union[float, int]):
        """Record metric value in history for analysis."""
        timestamp = time.time()
        self.metric_history[metric_name].append({
            'timestamp': timestamp,
            'value': value
        })
    
    def record_request_start(self, request_id: str, method: str, endpoint: str, size_bytes: int = 0):
        """Record the start of a request."""
        self.active_requests[request_id] = {
            'method': method,
            'endpoint': endpoint,
            'start_time': time.time(),
            'size_bytes': size_bytes
        }
        
        self.total_requests += 1
        self.active_connections.set(len(self.active_requests))
        
        # Record request size
        if size_bytes > 0:
            self.request_size.labels(method=method, endpoint=endpoint).observe(size_bytes)
        
        # Record timestamp for throughput calculation
        self._record_metric_history('request_timestamps', time.time())
    
    def record_request_complete(
        self, 
        request_id: str, 
        status_code: int, 
        response_size_bytes: int = 0
    ):
        """Record the completion of a request."""
        if request_id not in self.active_requests:
            return
        
        request_info = self.active_requests.pop(request_id)
        duration = time.time() - request_info['start_time']
        
        method = request_info['method']
        endpoint = request_info['endpoint']
        
        # Record metrics
        self.request_total.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=str(status_code)
        ).inc()
        
        self.request_duration.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
        
        if response_size_bytes > 0:
            self.response_size.labels(
                method=method, 
                endpoint=endpoint
            ).observe(response_size_bytes)
        
        # Update success/failure counts
        if 200 <= status_code < 400:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.active_connections.set(len(self.active_requests))
    
    def record_inference(
        self, 
        model_name: str, 
        model_version: str, 
        duration_seconds: float, 
        batch_size: int
    ):
        """Record model inference metrics."""
        self.inference_duration.labels(
            model_name=model_name, 
            model_version=model_version
        ).observe(duration_seconds)
        
        self.batch_size.labels(model_name=model_name).observe(batch_size)
        
        # Record in history for analysis
        self._record_metric_history('inference_duration', duration_seconds)
        self._record_metric_history('batch_size', batch_size)
    
    def record_error(
        self, 
        error_type: str, 
        error_category: str = "unknown", 
        severity: str = "medium"
    ):
        """Record error occurrence."""
        self.error_total.labels(
            error_type=error_type,
            error_category=error_category,
            severity=severity
        ).inc()
        
        self._record_metric_history('errors', 1)
    
    def record_cache_hit(self):
        """Record cache hit."""
        self.cache_hits.inc()
        self._record_metric_history('cache_hits', 1)
    
    def record_cache_miss(self):
        """Record cache miss."""
        self.cache_misses.inc()
        self._record_metric_history('cache_misses', 1)
    
    def set_cache_size(self, size: int):
        """Set current cache size."""
        self.cache_size.set(size)
    
    def record_rate_limit_hit(self, client_id: str = "unknown"):
        """Record rate limit violation."""
        self.rate_limit_hits.labels(client_id=client_id).inc()
    
    def set_circuit_breaker_state(self, service_name: str, is_open: bool):
        """Set circuit breaker state."""
        self.circuit_breaker_state.labels(service=service_name).set(1 if is_open else 0)
    
    def get_active_requests(self) -> int:
        """Get number of active requests."""
        return len(self.active_requests)
    
    def register_custom_metric(self, definition: MetricDefinition):
        """Register a custom metric."""
        if definition.metric_type == "counter":
            metric = Counter(
                f'{self.service_name}_{definition.name}',
                definition.description,
                definition.labels
            )
        elif definition.metric_type == "gauge":
            metric = Gauge(
                f'{self.service_name}_{definition.name}',
                definition.description,
                definition.labels
            )
        elif definition.metric_type == "histogram":
            metric = Histogram(
                f'{self.service_name}_{definition.name}',
                definition.description,
                definition.labels,
                buckets=definition.buckets
            )
        elif definition.metric_type == "summary":
            metric = Summary(
                f'{self.service_name}_{definition.name}',
                definition.description,
                definition.labels
            )
        else:
            raise ValueError(f"Unsupported metric type: {definition.metric_type}")
        
        self.custom_metrics[definition.name] = metric
        return metric
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "service_info": {
                "name": self.service_name,
                "uptime_seconds": time.time() - self.service_start_time,
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "active_requests": len(self.active_requests)
            },
            "performance": {
                "avg_response_time": self._calculate_average_response_time(),
                "current_throughput": self._calculate_current_throughput(),
                "error_rate_percent": (self.failed_requests / max(1, self.total_requests)) * 100
            },
            "system": {
                "cpu_usage": self._get_latest_metric('cpu_usage'),
                "memory_usage_mb": self._get_latest_metric('memory_usage') / (1024 * 1024) if self._get_latest_metric('memory_usage') else None,
                "active_connections": len(self.active_requests)
            }
        }
    
    def _calculate_average_response_time(self) -> Optional[float]:
        """Calculate average response time from history."""
        durations = [
            entry['value'] for entry in self.metric_history.get('request_duration', [])
            if time.time() - entry['timestamp'] <= 300  # Last 5 minutes
        ]
        
        return np.mean(durations) if durations else None
    
    def _calculate_current_throughput(self) -> float:
        """Calculate current throughput (requests per second)."""
        current_time = time.time()
        recent_requests = [
            entry['timestamp'] for entry in self.metric_history.get('request_timestamps', [])
            if current_time - entry <= 60  # Last minute
        ]
        
        return len(recent_requests) / 60.0
    
    def _get_latest_metric(self, metric_name: str) -> Optional[float]:
        """Get the latest value of a metric."""
        history = self.metric_history.get(metric_name, [])
        return history[-1]['value'] if history else None
    
    def export_prometheus_metrics(self) -> str:
        """Export all metrics in Prometheus format."""
        return generate_latest()


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.health_checks: Dict[str, Callable] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.logger = logging.getLogger(__name__)
    
    def add_check(self, name: str, check_function: Callable):
        """Add a health check function."""
        self.health_checks[name] = check_function
    
    def remove_check(self, name: str):
        """Remove a health check."""
        if name in self.health_checks:
            del self.health_checks[name]
    
    async def run_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if name not in self.health_checks:
            return {"status": "error", "message": f"Health check '{name}' not found"}
        
        start_time = time.time()
        
        try:
            check_function = self.health_checks[name]
            
            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(check_function):
                result = await check_function()
            else:
                result = check_function()
            
            duration = time.time() - start_time
            
            # Ensure result is a dict with required fields
            if not isinstance(result, dict):
                result = {"status": "healthy", "value": result}
            
            result.update({
                "check_name": name,
                "duration_seconds": duration,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Record in history
            self.health_history[name].append(result)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            error_result = {
                "check_name": name,
                "status": "error",
                "error": str(e),
                "duration_seconds": duration,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.health_history[name].append(error_result)
            self.logger.error(f"Health check '{name}' failed: {e}")
            
            return error_result
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        
        # Run all checks concurrently
        tasks = [
            self.run_check(name) for name in self.health_checks.keys()
        ]
        
        if tasks:
            check_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(check_results):
                check_name = list(self.health_checks.keys())[i]
                
                if isinstance(result, Exception):
                    results[check_name] = {
                        "status": "error",
                        "error": str(result),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                else:
                    results[check_name] = result
        
        return results
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall health status."""
        if not self.health_checks:
            return {
                "status": "unknown",
                "message": "No health checks configured",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Get latest results for each check
        latest_results = {}
        overall_status = "healthy"
        
        for name, history in self.health_history.items():
            if history:
                latest_result = history[-1]
                latest_results[name] = latest_result
                
                # Determine overall status
                if latest_result.get("status") == "error":
                    overall_status = "unhealthy"
                elif latest_result.get("status") == "warning" and overall_status != "unhealthy":
                    overall_status = "degraded"
        
        return {
            "status": overall_status,
            "checks": latest_results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_health_history(self, name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get health check history for a specific check."""
        history = self.health_history.get(name, [])
        return list(history)[-limit:] if limit > 0 else list(history)


class AlertManager:
    """Alert management system based on metrics and health checks."""
    
    def __init__(self, metrics_collector: MetricsCollector, health_checker: HealthChecker):
        self.metrics_collector = metrics_collector
        self.health_checker = health_checker
        self.alert_rules: List[Dict[str, Any]] = []
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        
        # Setup default alert rules
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        default_rules = [
            {
                "name": "high_error_rate",
                "condition": lambda: self._get_error_rate() > 5.0,
                "severity": "warning",
                "message": "Error rate above 5%"
            },
            {
                "name": "critical_error_rate",
                "condition": lambda: self._get_error_rate() > 20.0,
                "severity": "critical",
                "message": "Error rate above 20%"
            },
            {
                "name": "high_memory_usage",
                "condition": lambda: self._get_memory_usage_percent() > 85.0,
                "severity": "warning",
                "message": "Memory usage above 85%"
            },
            {
                "name": "critical_memory_usage",
                "condition": lambda: self._get_memory_usage_percent() > 95.0,
                "severity": "critical",
                "message": "Memory usage above 95%"
            },
            {
                "name": "high_response_time",
                "condition": lambda: self._get_avg_response_time() > 2.0,
                "severity": "warning",
                "message": "Average response time above 2 seconds"
            },
            {
                "name": "service_unhealthy",
                "condition": lambda: self.health_checker.get_overall_health()["status"] == "unhealthy",
                "severity": "critical",
                "message": "Service health checks failing"
            }
        ]
        
        self.alert_rules.extend(default_rules)
    
    def add_alert_rule(
        self, 
        name: str, 
        condition: Callable[[], bool], 
        severity: str = "warning",
        message: str = ""
    ):
        """Add a custom alert rule."""
        rule = {
            "name": name,
            "condition": condition,
            "severity": severity,
            "message": message or f"Alert: {name}"
        }
        
        self.alert_rules.append(rule)
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check all alert conditions."""
        new_alerts = []
        resolved_alerts = []
        
        for rule in self.alert_rules:
            rule_name = rule["name"]
            
            try:
                condition_met = rule["condition"]()
                
                if condition_met:
                    # Alert condition is met
                    if rule_name not in self.active_alerts:
                        # New alert
                        alert = {
                            "name": rule_name,
                            "severity": rule["severity"],
                            "message": rule["message"],
                            "started_at": datetime.utcnow().isoformat(),
                            "acknowledged": False
                        }
                        
                        self.active_alerts[rule_name] = alert
                        new_alerts.append(alert)
                        
                        self.logger.warning(f"Alert triggered: {rule_name} - {rule['message']}")
                
                else:
                    # Alert condition is not met
                    if rule_name in self.active_alerts:
                        # Alert resolved
                        resolved_alert = self.active_alerts.pop(rule_name)
                        resolved_alert["resolved_at"] = datetime.utcnow().isoformat()
                        
                        resolved_alerts.append(resolved_alert)
                        self.alert_history.append(resolved_alert)
                        
                        self.logger.info(f"Alert resolved: {rule_name}")
            
            except Exception as e:
                self.logger.error(f"Error checking alert rule '{rule_name}': {e}")
        
        return {
            "new_alerts": new_alerts,
            "resolved_alerts": resolved_alerts,
            "active_alerts": list(self.active_alerts.values())
        }
    
    def acknowledge_alert(self, alert_name: str, acknowledged_by: str = "system"):
        """Acknowledge an active alert."""
        if alert_name in self.active_alerts:
            self.active_alerts[alert_name]["acknowledged"] = True
            self.active_alerts[alert_name]["acknowledged_by"] = acknowledged_by
            self.active_alerts[alert_name]["acknowledged_at"] = datetime.utcnow().isoformat()
            
            self.logger.info(f"Alert acknowledged: {alert_name} by {acknowledged_by}")
            return True
        
        return False
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get alert history."""
        return self.alert_history[-limit:] if limit > 0 else self.alert_history
    
    def _get_error_rate(self) -> float:
        """Get current error rate percentage."""
        total = self.metrics_collector.total_requests
        failed = self.metrics_collector.failed_requests
        return (failed / max(1, total)) * 100
    
    def _get_memory_usage_percent(self) -> float:
        """Get current memory usage percentage."""
        try:
            return psutil.virtual_memory().percent
        except Exception:
            return 0.0
    
    def _get_avg_response_time(self) -> float:
        """Get average response time."""
        return self.metrics_collector._calculate_average_response_time() or 0.0


# Global instances
global_metrics_collector = MetricsCollector()
global_health_checker = HealthChecker()
global_alert_manager = AlertManager(global_metrics_collector, global_health_checker)