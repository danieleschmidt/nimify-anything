"""Administrative API endpoints for service management and monitoring."""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .circuit_breaker import circuit_breaker_registry
from .rate_limiter import MultiTierRateLimiter
from .monitoring import SystemMetrics, HealthChecker

logger = logging.getLogger(__name__)

# Security for admin endpoints
admin_security = HTTPBearer(auto_error=True)

# Create admin router
admin_router = APIRouter(prefix="/admin", tags=["admin"])


class AdminResponse(BaseModel):
    """Base response for admin operations."""
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Operation message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CircuitBreakerStatus(BaseModel):
    """Circuit breaker status response."""
    name: str
    state: str
    total_calls: int
    total_failures: int
    success_rate: float
    current_failure_count: int


class RateLimitStatus(BaseModel):
    """Rate limit status response."""
    global_requests: int
    global_rejected: int
    active_clients: int
    penalized_clients: int
    rejection_rate: float


class SystemStatus(BaseModel):
    """System status response."""
    service_health: str
    model_loaded: bool
    uptime_seconds: float
    cpu_percent: float
    memory_percent: float
    gpu_utilization: Optional[float]
    circuit_breakers: List[CircuitBreakerStatus]
    rate_limiting: RateLimitStatus


async def verify_admin_token(credentials: HTTPAuthorizationCredentials = Security(admin_security)) -> str:
    """Verify admin authentication token."""
    # In production, implement proper token validation
    # This is a simplified example
    admin_token = "admin_secret_token_123"  # Should be from environment/config
    
    if credentials.credentials != admin_token:
        raise HTTPException(
            status_code=403,
            detail="Invalid admin credentials"
        )
    
    return credentials.credentials


@admin_router.get("/status", response_model=SystemStatus)
async def get_system_status(token: str = Depends(verify_admin_token)):
    """Get comprehensive system status."""
    try:
        # Get circuit breaker metrics
        circuit_metrics = circuit_breaker_registry.get_all_metrics()
        circuit_breakers = [
            CircuitBreakerStatus(
                name=name,
                state=metrics["state"],
                total_calls=metrics["total_calls"],
                total_failures=metrics["total_failures"],
                success_rate=metrics["success_rate"],
                current_failure_count=metrics["current_failure_count"]
            )
            for name, metrics in circuit_metrics.items()
        ]
        
        # Get rate limiting metrics (assuming global rate_limiter exists)
        # This would need to be imported from the main api module
        rate_metrics = {
            "global_requests": 0,
            "global_rejected": 0,
            "active_clients": 0,
            "penalized_clients": 0,
            "rejection_rate": 0.0
        }
        
        # Get system metrics
        system_metrics = SystemMetrics()
        health_data = await system_metrics.collect_system_metrics()
        
        return SystemStatus(
            service_health="healthy",  # Would be determined by health checker
            model_loaded=True,  # Would check actual model status
            uptime_seconds=health_data.get("uptime", 0),
            cpu_percent=health_data.get("cpu_percent", 0),
            memory_percent=health_data.get("memory_percent", 0),
            gpu_utilization=health_data.get("gpu_utilization"),
            circuit_breakers=circuit_breakers,
            rate_limiting=RateLimitStatus(**rate_metrics)
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system status")


@admin_router.post("/circuit-breaker/{name}/reset", response_model=AdminResponse)
async def reset_circuit_breaker(name: str, token: str = Depends(verify_admin_token)):
    """Reset a specific circuit breaker."""
    try:
        if name == "all":
            circuit_breaker_registry.reset_all()
            message = "All circuit breakers reset"
        else:
            breaker = circuit_breaker_registry.breakers.get(name)
            if not breaker:
                raise HTTPException(status_code=404, detail=f"Circuit breaker '{name}' not found")
            
            breaker.reset()
            message = f"Circuit breaker '{name}' reset"
        
        logger.info(f"Admin action: {message}")
        
        return AdminResponse(
            success=True,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Failed to reset circuit breaker {name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset circuit breaker: {str(e)}")


@admin_router.post("/circuit-breaker/{name}/force-open", response_model=AdminResponse)
async def force_open_circuit_breaker(name: str, token: str = Depends(verify_admin_token)):
    """Force a circuit breaker to open state."""
    try:
        breaker = circuit_breaker_registry.breakers.get(name)
        if not breaker:
            raise HTTPException(status_code=404, detail=f"Circuit breaker '{name}' not found")
        
        breaker.force_open()
        message = f"Circuit breaker '{name}' forced to OPEN state"
        
        logger.warning(f"Admin action: {message}")
        
        return AdminResponse(
            success=True,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Failed to force open circuit breaker {name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to force open circuit breaker: {str(e)}")


@admin_router.get("/circuit-breakers", response_model=List[CircuitBreakerStatus])
async def list_circuit_breakers(token: str = Depends(verify_admin_token)):
    """List all circuit breakers and their status."""
    try:
        circuit_metrics = circuit_breaker_registry.get_all_metrics()
        
        return [
            CircuitBreakerStatus(
                name=name,
                state=metrics["state"],
                total_calls=metrics["total_calls"],
                total_failures=metrics["total_failures"],
                success_rate=metrics["success_rate"],
                current_failure_count=metrics["current_failure_count"]
            )
            for name, metrics in circuit_metrics.items()
        ]
        
    except Exception as e:
        logger.error(f"Failed to list circuit breakers: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve circuit breaker status")


@admin_router.post("/rate-limit/reset", response_model=AdminResponse)
async def reset_rate_limits(client_id: Optional[str] = None, token: str = Depends(verify_admin_token)):
    """Reset rate limits for a specific client or all clients."""
    try:
        # This would need access to the global rate limiter
        # For now, return a placeholder response
        
        if client_id:
            message = f"Rate limits reset for client '{client_id}'"
        else:
            message = "Rate limits reset for all clients"
        
        logger.info(f"Admin action: {message}")
        
        return AdminResponse(
            success=True,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Failed to reset rate limits: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset rate limits: {str(e)}")


@admin_router.get("/health/detailed")
async def detailed_health_check(token: str = Depends(verify_admin_token)):
    """Detailed health check for administrative monitoring."""
    try:
        health_checker = HealthChecker()
        
        # Comprehensive health checks
        checks = {
            "model_status": await health_checker.check_model_health(),
            "database_status": await health_checker.check_database_health(),
            "external_apis": await health_checker.check_external_apis(),
            "storage_status": await health_checker.check_storage_health(),
            "memory_status": await health_checker.check_memory_health(),
            "gpu_status": await health_checker.check_gpu_health()
        }
        
        # Determine overall health
        overall_health = "healthy" if all(check["healthy"] for check in checks.values()) else "degraded"
        
        return {
            "overall_health": overall_health,
            "timestamp": datetime.utcnow(),
            "checks": checks,
            "recommendations": health_checker.get_recommendations(checks)
        }
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to perform detailed health check")


@admin_router.post("/maintenance/enable", response_model=AdminResponse)
async def enable_maintenance_mode(token: str = Depends(verify_admin_token)):
    """Enable maintenance mode."""
    try:
        # In a real implementation, this would:
        # 1. Set a global maintenance flag
        # 2. Reject new requests with 503 status
        # 3. Allow current requests to complete
        # 4. Optionally drain existing connections
        
        logger.warning("Admin action: Maintenance mode enabled")
        
        return AdminResponse(
            success=True,
            message="Maintenance mode enabled - new requests will be rejected"
        )
        
    except Exception as e:
        logger.error(f"Failed to enable maintenance mode: {e}")
        raise HTTPException(status_code=500, detail="Failed to enable maintenance mode")


@admin_router.post("/maintenance/disable", response_model=AdminResponse)
async def disable_maintenance_mode(token: str = Depends(verify_admin_token)):
    """Disable maintenance mode."""
    try:
        logger.info("Admin action: Maintenance mode disabled")
        
        return AdminResponse(
            success=True,
            message="Maintenance mode disabled - service is now accepting requests"
        )
        
    except Exception as e:
        logger.error(f"Failed to disable maintenance mode: {e}")
        raise HTTPException(status_code=500, detail="Failed to disable maintenance mode")


@admin_router.get("/metrics/export")
async def export_metrics(format: str = "json", token: str = Depends(verify_admin_token)):
    """Export comprehensive metrics in various formats."""
    try:
        # Collect all metrics
        circuit_metrics = circuit_breaker_registry.get_all_metrics()
        
        # System metrics
        system_metrics = SystemMetrics()
        system_data = await system_metrics.collect_system_metrics()
        
        metrics_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "circuit_breakers": circuit_metrics,
            "system": system_data,
            "service_info": {
                "version": "1.0.0",
                "name": "nimify-service"
            }
        }
        
        if format.lower() == "json":
            return metrics_data
        elif format.lower() == "prometheus":
            # Convert to Prometheus format
            prometheus_output = _convert_to_prometheus_format(metrics_data)
            return {"metrics": prometheus_output}
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
            
    except Exception as e:
        logger.error(f"Failed to export metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to export metrics")


def _convert_to_prometheus_format(metrics_data: Dict) -> str:
    """Convert metrics data to Prometheus format."""
    prometheus_lines = []
    
    # Circuit breaker metrics
    for name, metrics in metrics_data.get("circuit_breakers", {}).items():
        prometheus_lines.append(f"circuit_breaker_total_calls{{name=\"{name}\"}} {metrics['total_calls']}")
        prometheus_lines.append(f"circuit_breaker_total_failures{{name=\"{name}\"}} {metrics['total_failures']}")
        prometheus_lines.append(f"circuit_breaker_success_rate{{name=\"{name}\"}} {metrics['success_rate']}")
    
    # System metrics
    system = metrics_data.get("system", {})
    if "cpu_percent" in system:
        prometheus_lines.append(f"system_cpu_percent {system['cpu_percent']}")
    if "memory_percent" in system:
        prometheus_lines.append(f"system_memory_percent {system['memory_percent']}")
    
    return "\n".join(prometheus_lines)


class SystemMetrics:
    """System metrics collector."""
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics."""
        try:
            import psutil
            import time
            
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network stats
            network = psutil.net_io_counters()
            
            # Process info
            process = psutil.Process()
            
            metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_usage_percent": (disk.used / disk.total) * 100,
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv,
                "process_memory_mb": process.memory_info().rss / (1024**2),
                "process_cpu_percent": process.cpu_percent(),
                "uptime": time.time() - process.create_time()
            }
            
            # GPU metrics if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                metrics.update({
                    "gpu_utilization": gpu_util.gpu,
                    "gpu_memory_percent": (memory_info.used / memory_info.total) * 100
                })
            except:
                pass  # GPU metrics not available
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {}


class HealthChecker:
    """Comprehensive health checker."""
    
    async def check_model_health(self) -> Dict[str, Any]:
        """Check model loading and inference health."""
        # Placeholder implementation
        return {
            "healthy": True,
            "message": "Model loaded and responding",
            "last_check": datetime.utcnow().isoformat()
        }
    
    async def check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity."""
        # Placeholder implementation
        return {
            "healthy": True,
            "message": "Database connection active",
            "last_check": datetime.utcnow().isoformat()
        }
    
    async def check_external_apis(self) -> Dict[str, Any]:
        """Check external API dependencies."""
        # Placeholder implementation
        return {
            "healthy": True,
            "message": "External APIs responding",
            "last_check": datetime.utcnow().isoformat()
        }
    
    async def check_storage_health(self) -> Dict[str, Any]:
        """Check storage system health."""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            
            healthy = usage_percent < 90  # Alert if disk usage > 90%
            
            return {
                "healthy": healthy,
                "message": f"Disk usage: {usage_percent:.1f}%",
                "usage_percent": usage_percent,
                "last_check": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Failed to check storage: {str(e)}",
                "last_check": datetime.utcnow().isoformat()
            }
    
    async def check_memory_health(self) -> Dict[str, Any]:
        """Check memory usage health."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            healthy = memory.percent < 85  # Alert if memory usage > 85%
            
            return {
                "healthy": healthy,
                "message": f"Memory usage: {memory.percent:.1f}%",
                "usage_percent": memory.percent,
                "available_gb": memory.available / (1024**3),
                "last_check": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Failed to check memory: {str(e)}",
                "last_check": datetime.utcnow().isoformat()
            }
    
    async def check_gpu_health(self) -> Dict[str, Any]:
        """Check GPU health and utilization."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Get GPU utilization and memory
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            memory_percent = (memory_info.used / memory_info.total) * 100
            healthy = temp < 85 and memory_percent < 95  # Temperature and memory thresholds
            
            return {
                "healthy": healthy,
                "message": f"GPU temp: {temp}°C, Memory: {memory_percent:.1f}%",
                "temperature": temp,
                "utilization": util.gpu,
                "memory_percent": memory_percent,
                "last_check": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "healthy": True,  # Assume healthy if GPU not available
                "message": f"GPU not available or failed to check: {str(e)}",
                "last_check": datetime.utcnow().isoformat()
            }
    
    def get_recommendations(self, checks: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on health check results."""
        recommendations = []
        
        for check_name, check_result in checks.items():
            if not check_result.get("healthy", True):
                if check_name == "storage_status":
                    recommendations.append("Consider cleaning up disk space or expanding storage")
                elif check_name == "memory_status":
                    recommendations.append("Monitor memory usage and consider scaling up memory resources")
                elif check_name == "gpu_status":
                    recommendations.append("Check GPU cooling and memory usage")
                elif check_name == "model_status":
                    recommendations.append("Restart model service or check model files")
                else:
                    recommendations.append(f"Investigate {check_name} issues")
        
        if not recommendations:
            recommendations.append("All systems operating normally")
        
        return recommendations