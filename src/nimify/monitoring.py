"""Advanced monitoring and observability system."""

import asyncio
import contextlib
import json
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: float
    tags: dict[str, str]
    unit: str = ""


@dataclass
class Alert:
    """Alert configuration and state."""
    name: str
    condition: Callable[[dict[str, float]], bool]
    message: str
    severity: str = "warning"  # info, warning, critical
    cooldown_seconds: int = 300
    last_triggered: float = 0
    active: bool = False


class MetricsCollector:
    """Advanced metrics collection and aggregation."""
    
    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self.metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.counters: dict[str, float] = defaultdict(float)
        self.gauges: dict[str, float] = defaultdict(float)
        self.histograms: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.RLock()
    
    def record_counter(self, name: str, value: float = 1.0, tags: dict[str, str] = None):
        """Record a counter metric (monotonically increasing)."""
        with self._lock:
            self.counters[name] += value
            self._record_point(name, value, tags, "count")
    
    def record_gauge(self, name: str, value: float, tags: dict[str, str] = None):
        """Record a gauge metric (point-in-time value)."""
        with self._lock:
            self.gauges[name] = value
            self._record_point(name, value, tags, "gauge")
    
    def record_histogram(self, name: str, value: float, tags: dict[str, str] = None):
        """Record a histogram metric (for latency, sizes, etc.)."""
        with self._lock:
            self.histograms[name].append(value)
            # Keep only recent values to prevent memory growth
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
            self._record_point(name, value, tags, "histogram")
    
    def _record_point(self, name: str, value: float, tags: dict[str, str], unit: str):
        """Record a metric point."""
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            unit=unit
        )
        self.metrics[name].append(point)
    
    def get_counter(self, name: str) -> float:
        """Get current counter value."""
        with self._lock:
            return self.counters.get(name, 0.0)
    
    def get_gauge(self, name: str) -> float:
        """Get current gauge value."""
        with self._lock:
            return self.gauges.get(name, 0.0)
    
    def get_histogram_stats(self, name: str) -> dict[str, float]:
        """Get histogram statistics."""
        with self._lock:
            values = self.histograms.get(name, [])
            if not values:
                return {}
            
            sorted_values = sorted(values)
            n = len(sorted_values)
            
            return {
                'count': n,
                'sum': sum(sorted_values),
                'min': sorted_values[0],
                'max': sorted_values[-1],
                'mean': sum(sorted_values) / n,
                'p50': sorted_values[int(0.5 * n)],
                'p90': sorted_values[int(0.9 * n)],
                'p95': sorted_values[int(0.95 * n)],
                'p99': sorted_values[int(0.99 * n)]
            }
    
    def get_recent_metrics(self, name: str, duration_seconds: int = 60) -> list[MetricPoint]:
        """Get recent metrics within duration."""
        with self._lock:
            now = time.time()
            cutoff = now - duration_seconds
            points = self.metrics.get(name, deque())
            return [p for p in points if p.timestamp >= cutoff]
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self._lock:
            # Export counters
            for name, value in self.counters.items():
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {value}")
            
            # Export gauges
            for name, value in self.gauges.items():
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")
            
            # Export histograms
            for name, values in self.histograms.items():
                if values:
                    stats = self.get_histogram_stats(name)
                    lines.append(f"# TYPE {name} histogram")
                    lines.append(f"{name}_count {stats['count']}")
                    lines.append(f"{name}_sum {stats['sum']}")
                    for quantile in [0.5, 0.9, 0.95, 0.99]:
                        lines.append(f"{name}_quantile{{quantile=\"{quantile}\"}} {stats[f'p{int(quantile*100)}']}")
        
        return "\\n".join(lines)


class SystemMonitor:
    """Monitor system resources and health."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.monitoring_active = False
        self._monitor_task = None
    
    async def start_monitoring(self, interval_seconds: int = 30):
        """Start system monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval_seconds))
        logger.info("Started system monitoring")
    
    async def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        if self._monitor_task:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task
        logger.info("Stopped system monitoring")
    
    async def _monitor_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.record_gauge("system_cpu_usage_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics_collector.record_gauge("system_memory_usage_percent", memory.percent)
            self.metrics_collector.record_gauge("system_memory_available_bytes", memory.available)
            self.metrics_collector.record_gauge("system_memory_used_bytes", memory.used)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.metrics_collector.record_gauge("system_disk_usage_percent", 
                                              (disk.used / disk.total) * 100)
            self.metrics_collector.record_gauge("system_disk_free_bytes", disk.free)
            
            # Network metrics (if available)
            try:
                net_io = psutil.net_io_counters()
                self.metrics_collector.record_counter("system_network_bytes_sent", 
                                                    net_io.bytes_sent)
                self.metrics_collector.record_counter("system_network_bytes_recv", 
                                                    net_io.bytes_recv)
            except Exception:
                pass  # Network metrics not available
            
            # Process metrics
            process = psutil.Process()
            self.metrics_collector.record_gauge("process_memory_bytes", 
                                              process.memory_info().rss)
            self.metrics_collector.record_gauge("process_cpu_percent", 
                                              process.cpu_percent())
            self.metrics_collector.record_gauge("process_num_threads", 
                                              process.num_threads())
            
            # GPU metrics (if available)
            await self._collect_gpu_metrics()
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_gpu_metrics(self):
        """Collect GPU metrics if NVIDIA GPU is available."""
        try:
            # This would require nvidia-ml-py package
            # For demonstration, we'll just log that GPU monitoring is not available
            logger.debug("GPU metrics collection not implemented (requires nvidia-ml-py)")
        except Exception as e:
            logger.debug(f"GPU metrics not available: {e}")


class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts: dict[str, Alert] = {}
        self.alert_history: list[dict[str, Any]] = []
        self.max_history = 1000
        self._check_task = None
        self.checking_active = False
    
    def add_alert(self, alert: Alert):
        """Add an alert configuration."""
        self.alerts[alert.name] = alert
        logger.info(f"Added alert: {alert.name}")
    
    def remove_alert(self, name: str):
        """Remove an alert configuration."""
        if name in self.alerts:
            del self.alerts[name]
            logger.info(f"Removed alert: {name}")
    
    async def start_alert_checking(self, interval_seconds: int = 60):
        """Start alert checking."""
        if self.checking_active:
            return
        
        self.checking_active = True
        self._check_task = asyncio.create_task(self._check_loop(interval_seconds))
        logger.info("Started alert checking")
    
    async def stop_alert_checking(self):
        """Stop alert checking."""
        self.checking_active = False
        if self._check_task:
            self._check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._check_task
        logger.info("Stopped alert checking")
    
    async def _check_loop(self, interval_seconds: int):
        """Main alert checking loop."""
        while self.checking_active:
            try:
                await self._check_alerts()
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert checking loop: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def _check_alerts(self):
        """Check all configured alerts."""
        now = time.time()
        
        # Get current metric values
        current_metrics = {}
        for name in self.metrics_collector.counters:
            current_metrics[name] = self.metrics_collector.get_counter(name)
        for name in self.metrics_collector.gauges:
            current_metrics[name] = self.metrics_collector.get_gauge(name)
        
        for alert_name, alert in self.alerts.items():
            try:
                # Check cooldown period
                if alert.active and (now - alert.last_triggered) < alert.cooldown_seconds:
                    continue
                
                # Evaluate alert condition
                should_trigger = alert.condition(current_metrics)
                
                if should_trigger and not alert.active:
                    # Trigger alert
                    await self._trigger_alert(alert, current_metrics)
                elif not should_trigger and alert.active:
                    # Resolve alert
                    await self._resolve_alert(alert)
                    
            except Exception as e:
                logger.error(f"Error checking alert {alert_name}: {e}")
    
    async def _trigger_alert(self, alert: Alert, metrics: dict[str, float]):
        """Trigger an alert."""
        alert.active = True
        alert.last_triggered = time.time()
        
        alert_event = {
            'name': alert.name,
            'message': alert.message,
            'severity': alert.severity,
            'timestamp': time.time(),
            'status': 'triggered',
            'metrics': metrics.copy()
        }
        
        self.alert_history.append(alert_event)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)
        
        logger.warning(f"ALERT TRIGGERED: {alert.name} - {alert.message}")
        
        # Here you would integrate with notification systems
        await self._send_notification(alert_event)
    
    async def _resolve_alert(self, alert: Alert):
        """Resolve an alert."""
        alert.active = False
        
        alert_event = {
            'name': alert.name,
            'message': f"Alert resolved: {alert.message}",
            'severity': 'info',
            'timestamp': time.time(),
            'status': 'resolved'
        }
        
        self.alert_history.append(alert_event)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)
        
        logger.info(f"ALERT RESOLVED: {alert.name}")
        
        await self._send_notification(alert_event)
    
    async def _send_notification(self, alert_event: dict[str, Any]):
        """Send alert notification (placeholder for integration)."""
        # In a real implementation, this would send notifications via:
        # - Email
        # - Slack/Teams
        # - PagerDuty
        # - Webhook
        logger.info(f"NOTIFICATION: {alert_event['status'].upper()} - {alert_event['name']}")
    
    def get_active_alerts(self) -> list[dict[str, Any]]:
        """Get all active alerts."""
        active = []
        for alert in self.alerts.values():
            if alert.active:
                active.append({
                    'name': alert.name,
                    'message': alert.message,
                    'severity': alert.severity,
                    'triggered_at': alert.last_triggered
                })
        return active
    
    def get_alert_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent alert history."""
        return self.alert_history[-limit:]


class DashboardGenerator:
    """Generate monitoring dashboard configurations."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
    
    def generate_grafana_dashboard(self) -> dict[str, Any]:
        """Generate Grafana dashboard JSON."""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Nimify Service Monitoring",
                "tags": ["nimify", "ml", "monitoring"],
                "timezone": "browser",
                "panels": [
                    self._create_requests_panel(),
                    self._create_latency_panel(),
                    self._create_system_panel(),
                    self._create_errors_panel()
                ],
                "time": {"from": "now-1h", "to": "now"},
                "timepicker": {},
                "templating": {"list": []},
                "annotations": {"list": []},
                "refresh": "5s",
                "schemaVersion": 30,
                "version": 1
            }
        }
        return dashboard
    
    def _create_requests_panel(self) -> dict[str, Any]:
        """Create requests per second panel."""
        return {
            "id": 1,
            "title": "Requests per Second",
            "type": "graph",
            "targets": [
                {
                    "expr": "rate(nim_requests_total[5m])",
                    "legendFormat": "{{method}} {{endpoint}}"
                }
            ],
            "yAxes": [
                {"label": "Requests/sec", "min": 0}
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
        }
    
    def _create_latency_panel(self) -> dict[str, Any]:
        """Create latency percentiles panel."""
        return {
            "id": 2,
            "title": "Response Latency",
            "type": "graph",
            "targets": [
                {
                    "expr": "histogram_quantile(0.50, rate(nim_request_duration_seconds_bucket[5m]))",
                    "legendFormat": "p50"
                },
                {
                    "expr": "histogram_quantile(0.95, rate(nim_request_duration_seconds_bucket[5m]))",
                    "legendFormat": "p95"
                },
                {
                    "expr": "histogram_quantile(0.99, rate(nim_request_duration_seconds_bucket[5m]))",
                    "legendFormat": "p99"
                }
            ],
            "yAxes": [
                {"label": "Latency (seconds)", "min": 0}
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
        }
    
    def _create_system_panel(self) -> dict[str, Any]:
        """Create system metrics panel."""
        return {
            "id": 3,
            "title": "System Resources",
            "type": "graph",
            "targets": [
                {
                    "expr": "system_cpu_usage_percent",
                    "legendFormat": "CPU Usage %"
                },
                {
                    "expr": "system_memory_usage_percent",
                    "legendFormat": "Memory Usage %"
                }
            ],
            "yAxes": [
                {"label": "Percentage", "min": 0, "max": 100}
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
        }
    
    def _create_errors_panel(self) -> dict[str, Any]:
        """Create error rate panel."""
        return {
            "id": 4,
            "title": "Error Rate",
            "type": "graph",
            "targets": [
                {
                    "expr": "rate(nim_errors_total[5m])",
                    "legendFormat": "{{error_type}}"
                }
            ],
            "yAxes": [
                {"label": "Errors/sec", "min": 0}
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
        }
    
    def save_dashboard_config(self, output_path: Path):
        """Save dashboard configuration to file."""
        dashboard = self.generate_grafana_dashboard()
        with open(output_path, 'w') as f:
            json.dump(dashboard, f, indent=2)
        logger.info(f"Saved dashboard configuration to {output_path}")


# Default alert configurations
def create_default_alerts() -> list[Alert]:
    """Create default alert configurations."""
    alerts = [
        Alert(
            name="high_cpu_usage",
            condition=lambda m: m.get("system_cpu_usage_percent", 0) > 80,
            message="CPU usage is above 80%",
            severity="warning",
            cooldown_seconds=300
        ),
        Alert(
            name="high_memory_usage",
            condition=lambda m: m.get("system_memory_usage_percent", 0) > 90,
            message="Memory usage is above 90%",
            severity="critical",
            cooldown_seconds=180
        ),
        Alert(
            name="high_error_rate",
            condition=lambda m: m.get("nim_errors_total", 0) > 10,
            message="Error rate is elevated (>10 errors)",
            severity="warning",
            cooldown_seconds=300
        ),
        Alert(
            name="service_unavailable",
            condition=lambda m: m.get("nim_requests_total", 1) == 0,
            message="Service appears to be unavailable (no requests)",
            severity="critical",
            cooldown_seconds=600
        )
    ]
    return alerts


# Global monitoring instances
global_metrics_collector = MetricsCollector()
system_monitor = SystemMonitor(global_metrics_collector)
alert_manager = AlertManager(global_metrics_collector)
dashboard_generator = DashboardGenerator(global_metrics_collector)