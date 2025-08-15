"""Optimized FastAPI application with caching, performance monitoring and auto-scaling."""

import time
import uuid
import asyncio
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import numpy as np

from .logging_config import setup_logging, log_performance_metric
from .error_handling import global_error_handler, ModelError
from .circuit_breaker import ModelInferenceCircuitBreaker, CircuitBreakerException
from .performance_optimizer import (
    ModelInferenceCache, IntelligentCache, global_monitor,
    with_caching, with_performance_monitoring
)
from .auto_scaler import global_auto_scaler

# Set up logging
logger = setup_logging("nim-service-optimized", log_level="INFO")

# Enhanced metrics
REQUEST_COUNT = Counter('nim_optimized_request_count_total', 'Total requests', ['method', 'endpoint', 'status'])
CACHE_HITS = Counter('nim_cache_hits_total', 'Cache hits', ['cache_type'])
CACHE_MISSES = Counter('nim_cache_misses_total', 'Cache misses', ['cache_type'])
INFERENCE_DURATION = Histogram('nim_optimized_inference_duration_seconds', 'Model inference duration')
AUTO_SCALING_EVENTS = Counter('nim_autoscaling_events_total', 'Auto-scaling events', ['direction'])


class OptimizedPredictionRequest(BaseModel):
    """Optimized request model with caching hints."""
    input: List[List[float]] = Field(..., description="Input data for inference", min_length=1, max_length=64)
    cache_ttl: Optional[int] = Field(None, description="Cache TTL in seconds", ge=0, le=7200)
    priority: Optional[str] = Field("normal", description="Request priority", pattern="^(low|normal|high)$")


class OptimizedPredictionResponse(BaseModel):
    """Optimized response model with performance metadata."""
    predictions: List[List[float]] = Field(..., description="Model predictions")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    request_id: str = Field(..., description="Unique request identifier")
    cache_hit: bool = Field(..., description="Whether result was served from cache")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    scaling_status: Dict[str, Any] = Field(..., description="Auto-scaling status")


class OptimizedModelLoader:
    """Model loader with caching and performance optimization."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session: Optional[Any] = None
        self.circuit_breaker = ModelInferenceCircuitBreaker("optimized_model")
        self.inference_cache = ModelInferenceCache("optimized_model")
        self.warmup_completed = False
    
    async def load_model(self):
        """Load model with performance optimizations."""
        try:
            # Mock model loading
            self.session = {"mock": True, "optimized": True}
            logger.info(f"Optimized model loaded: {self.model_path}")
            
            # Warm up the model
            await self._warmup_model()
            
        except Exception as e:
            raise ModelError(f"Failed to load optimized model: {str(e)}")
    
    async def _warmup_model(self):
        """Warm up the model with dummy inference."""
        if self.warmup_completed:
            return
            
        try:
            # Warm up with dummy data
            warmup_data = [[1.0, 2.0, 3.0]]
            await self._raw_predict(warmup_data, skip_cache=True)
            self.warmup_completed = True
            logger.info("Model warmup completed")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    @with_performance_monitoring(global_monitor)
    async def predict(
        self,
        input_data: List[List[float]],
        cache_ttl: Optional[int] = None,
        priority: str = "normal"
    ) -> tuple[List[List[float]], bool]:
        """Optimized prediction with caching."""
        if not self.session:
            raise ModelError("Model not loaded")
        
        # Check cache first
        cache_hit = False
        cached_predictions = self.inference_cache.get_prediction(input_data)
        
        if cached_predictions is not None:
            CACHE_HITS.labels(cache_type="inference").inc()
            cache_hit = True
            
            # Record cache hit metrics
            global_monitor.record_metric("cache_hit_rate", 1.0)
            log_performance_metric("inference_cache_hit", 1, "boolean")
            
            return cached_predictions, cache_hit
        
        CACHE_MISSES.labels(cache_type="inference").inc()
        global_monitor.record_metric("cache_hit_rate", 0.0)
        
        # Perform inference
        predictions = await self._raw_predict(input_data, priority=priority)
        
        # Cache result
        cache_ttl = cache_ttl or 3600  # 1 hour default
        self.inference_cache.cache_prediction(input_data, predictions)
        
        return predictions, cache_hit
    
    async def _raw_predict(
        self,
        input_data: List[List[float]],
        skip_cache: bool = False,
        priority: str = "normal"
    ) -> List[List[float]]:
        """Raw inference without caching."""
        
        def _do_inference():
            # Mock inference with some processing time based on priority
            processing_delay = {
                "low": 0.05,
                "normal": 0.02,  
                "high": 0.01
            }.get(priority, 0.02)
            
            time.sleep(processing_delay)  # Simulate processing
            
            # Mock prediction
            input_array = np.array(input_data, dtype=np.float32)
            predictions = (input_array * 2.1 + 0.5).tolist()  # More complex mock
            return predictions
        
        # Use circuit breaker for protection
        start_time = time.time()
        predictions = self.circuit_breaker.predict(_do_inference)
        inference_time = (time.time() - start_time) * 1000
        
        # Record performance metrics
        INFERENCE_DURATION.observe(inference_time / 1000)
        global_monitor.record_metric("raw_inference_time_ms", inference_time)
        
        # Record auto-scaling metrics
        global_auto_scaler.record_metric("p95_latency_ms", inference_time)
        global_auto_scaler.record_metric("cpu_utilization", min(100, inference_time / 5))  # Mock CPU usage
        
        return predictions
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        cache_stats = self.inference_cache.get_stats()
        circuit_stats = self.circuit_breaker.get_status()
        
        return {
            "cache": cache_stats,
            "circuit_breaker": circuit_stats,
            "warmup_completed": self.warmup_completed
        }


# Global state
model_loader: Optional[OptimizedModelLoader] = None
service_start_time = time.time()
request_counter = 0


async def get_request_id() -> str:
    """Generate unique request ID."""
    return str(uuid.uuid4())


async def performance_middleware(request: Request, call_next):
    """Performance monitoring middleware."""
    global request_counter
    request_counter += 1
    
    start_time = time.time()
    
    # Record request rate for auto-scaling
    global_auto_scaler.record_metric("requests_per_second", 1.0)  # Simplified
    
    response = await call_next(request)
    
    # Record performance metrics
    duration = (time.time() - start_time) * 1000
    global_monitor.record_metric("request_duration_ms", duration)
    
    # Auto-scaling evaluation (every 10 requests)
    if request_counter % 10 == 0:
        scaling_decisions = global_auto_scaler.evaluate_scaling()
        for decision in scaling_decisions:
            AUTO_SCALING_EVENTS.labels(direction=decision.direction.value).inc()
            global_auto_scaler.apply_scaling_decision(decision)
    
    return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optimized application lifespan manager."""
    global model_loader, service_start_time
    
    # Startup
    service_start_time = time.time()
    import os
    model_path = os.getenv("MODEL_PATH", "/models/model.onnx")
    model_loader = OptimizedModelLoader(model_path)
    
    try:
        await model_loader.load_model()
        logger.info("Optimized NIM service started successfully")
    except Exception as e:
        logger.error(f"Failed to start optimized service: {e}")
    
    yield
    
    # Shutdown
    logger.info("Optimized NIM service shutting down")


# Create optimized FastAPI app
app = FastAPI(
    title="Optimized NIM Service API",
    description="High-performance NVIDIA NIM microservice with caching, monitoring, and auto-scaling",
    version="3.0.0",
    lifespan=lifespan
)

# Add performance middleware
app.middleware("http")(performance_middleware)


@app.post("/v1/predict", response_model=OptimizedPredictionResponse)
async def optimized_predict(
    request: OptimizedPredictionRequest,
    req: Request,
    request_id: str = Depends(get_request_id)
):
    """Optimized prediction endpoint with caching and performance monitoring."""
    REQUEST_COUNT.labels(method="POST", endpoint="/v1/predict", status="start").inc()
    
    if not model_loader or not model_loader.session:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        overall_start = time.time()
        
        # Perform optimized inference
        predictions, cache_hit = await model_loader.predict(
            input_data=request.input,
            cache_ttl=request.cache_ttl,
            priority=request.priority
        )
        
        total_time = (time.time() - overall_start) * 1000
        
        # Get performance metrics
        perf_stats = global_monitor.get_all_stats()
        performance_metrics = {
            "total_time_ms": total_time,
            "cache_hit_rate": perf_stats.get("cache_hit_rate", {}).get("latest", 0.0) if perf_stats.get("cache_hit_rate") else 0.0,
        }
        
        # Get scaling status
        scaling_status = global_auto_scaler.get_scaling_stats()
        
        REQUEST_COUNT.labels(method="POST", endpoint="/v1/predict", status="success").inc()
        
        return OptimizedPredictionResponse(
            predictions=predictions,
            inference_time_ms=total_time,
            request_id=request_id,
            cache_hit=cache_hit,
            performance_metrics=performance_metrics,
            scaling_status=scaling_status
        )
        
    except CircuitBreakerException:
        REQUEST_COUNT.labels(method="POST", endpoint="/v1/predict", status="circuit_open").inc()
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service temporarily unavailable (circuit breaker open)",
                "request_id": request_id,
                "retry_after": 30
            }
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/v1/predict", status="error").inc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Internal server error: {str(e)}",
                "request_id": request_id
            }
        )


@app.get("/health")
async def optimized_health_check():
    """Comprehensive health check with performance data."""
    model_loaded = model_loader is not None and model_loader.session is not None
    uptime = time.time() - service_start_time
    
    health_data = {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "version": "3.0.0",
        "uptime_seconds": uptime,
        "performance": {},
        "scaling": {}
    }
    
    if model_loader:
        health_data["performance"] = model_loader.get_performance_stats()
    
    health_data["scaling"] = global_auto_scaler.get_scaling_stats()
    
    return health_data


@app.get("/metrics")
async def metrics():
    """Enhanced Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/performance")
async def performance_stats():
    """Detailed performance statistics."""
    stats = {
        "monitor": global_monitor.get_all_stats(),
        "scaling": global_auto_scaler.get_scaling_stats()
    }
    
    if model_loader:
        stats["model"] = model_loader.get_performance_stats()
    
    return stats


@app.post("/admin/clear-cache")
async def clear_cache():
    """Admin endpoint to clear inference cache."""
    if model_loader:
        model_loader.inference_cache.cache.clear()
    
    return {"message": "Cache cleared successfully"}


@app.post("/admin/warmup")
async def manual_warmup():
    """Admin endpoint to manually warmup the model."""
    if model_loader:
        await model_loader._warmup_model()
    
    return {"message": "Model warmup completed"}


@app.get("/")
async def root():
    """Root endpoint with optimized service info."""
    return {
        "service": "Optimized NVIDIA NIM Microservice",
        "version": "3.0.0",
        "features": [
            "Intelligent Caching",
            "Auto-scaling",
            "Performance Monitoring", 
            "Circuit Breaker Protection",
            "Priority-based Processing",
            "Model Warmup"
        ],
        "performance": {
            "cache_enabled": True,
            "auto_scaling_enabled": True,
            "monitoring_enabled": True
        },
        "endpoints": {
            "predict": "/v1/predict",
            "health": "/health",
            "metrics": "/metrics",
            "performance": "/performance",
            "admin": {
                "clear_cache": "/admin/clear-cache",
                "warmup": "/admin/warmup"
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)