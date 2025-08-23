"""Optimized FastAPI application with advanced performance features and scaling."""

import asyncio
import hashlib
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from .intelligent_scaling import ScalingConfig, initialize_auto_scaling, global_auto_scaler
from .logging_config import setup_logging
from .performance_optimization import PerformanceConfig, initialize_performance_optimizer, global_performance_optimizer

# Set up logging
logger = setup_logging("nim-service-optimized", log_level="INFO", enable_audit=True)

# Enhanced metrics
REQUEST_COUNT = Counter('nim_optimized_request_count_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('nim_optimized_request_duration_seconds', 'Request duration')
INFERENCE_DURATION = Histogram('nim_optimized_inference_duration_seconds', 'Model inference duration')
CACHE_HITS = Counter('nim_optimized_cache_hits_total', 'Cache hits')
CACHE_MISSES = Counter('nim_optimized_cache_misses_total', 'Cache misses')
BATCH_SIZE = Histogram('nim_optimized_batch_size', 'Batch sizes processed')
CONCURRENT_REQUESTS = Histogram('nim_optimized_concurrent_requests', 'Concurrent requests')


class PredictionRequest(BaseModel):
    """Optimized request model with caching hints."""
    input: list[list[float]] = Field(..., description="Input data for inference", min_length=1, max_length=128)
    use_cache: bool = Field(True, description="Whether to use caching for this request")
    priority: int = Field(1, description="Request priority (1=low, 5=high)")
    timeout_ms: Optional[int] = Field(None, description="Request timeout in milliseconds")


class PredictionResponse(BaseModel):
    """Optimized response model with performance metadata."""
    predictions: list[list[float]] = Field(..., description="Model predictions")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    request_id: str = Field(..., description="Unique request identifier")
    served_from_cache: bool = Field(False, description="Whether result was served from cache")
    batch_size: int = Field(1, description="Batch size used for processing")
    optimization_applied: list[str] = Field([], description="Optimizations applied")
    performance_score: float = Field(0.0, description="Performance score (0-100)")


class HealthResponse(BaseModel):
    """Comprehensive health response with performance metrics."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version") 
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    performance_stats: dict[str, Any] = Field(..., description="Performance statistics")
    scaling_status: dict[str, Any] = Field(..., description="Auto-scaling status")
    optimization_recommendations: list[str] = Field([], description="Performance recommendations")


class OptimizedModelLoader:
    """Model loader with advanced optimization features."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session: Any | None = None
        self.model_loaded = False
        self.load_time = 0.0
        self.prediction_count = 0
        self.total_inference_time = 0.0
        
    async def load_model(self):
        """Load model with optimization."""
        start_time = time.time()
        
        try:
            # Mock model loading with optimization
            await asyncio.sleep(0.1)  # Simulate loading time
            self.session = {"optimized": True, "loaded_at": time.time()}
            self.model_loaded = True
            self.load_time = time.time() - start_time
            
            logger.info(f"Optimized model loaded in {self.load_time:.3f}s: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    async def predict(self, input_data: list[list[float]], use_optimization: bool = True) -> tuple[list[list[float]], dict[str, Any]]:
        """Run optimized inference."""
        if not self.model_loaded:
            raise ValueError("Model not loaded")
        
        start_time = time.time()
        optimizations_applied = []
        
        try:
            input_array = np.array(input_data, dtype=np.float32)
            
            # Apply optimizations based on input characteristics
            if use_optimization:
                if input_array.size > 1000:
                    # Large input - use chunked processing
                    predictions = await self._chunked_inference(input_array)
                    optimizations_applied.append("chunked_processing")
                elif len(input_data) > 10:
                    # Batch input - use vectorized operations
                    predictions = await self._vectorized_inference(input_array)
                    optimizations_applied.append("vectorized_operations")
                else:
                    # Standard processing
                    predictions = await self._standard_inference(input_array)
            else:
                predictions = await self._standard_inference(input_array)
            
            inference_time = time.time() - start_time
            
            # Update statistics
            self.prediction_count += 1
            self.total_inference_time += inference_time
            
            metadata = {
                "optimizations_applied": optimizations_applied,
                "inference_time_ms": inference_time * 1000,
                "input_shape": input_array.shape,
                "prediction_count": self.prediction_count
            }
            
            INFERENCE_DURATION.observe(inference_time)
            return predictions, metadata
            
        except Exception as e:
            logger.error(f"Optimized inference failed: {e}")
            raise
    
    async def _standard_inference(self, input_array: np.ndarray) -> list[list[float]]:
        """Standard inference processing."""
        await asyncio.sleep(0.005)  # Simulate processing
        return (input_array * 2.0).tolist()
    
    async def _vectorized_inference(self, input_array: np.ndarray) -> list[list[float]]:
        """Optimized vectorized inference."""
        await asyncio.sleep(0.003)  # Faster processing
        # Vectorized operations for better performance
        result = np.multiply(input_array, 2.0, dtype=np.float32)
        return result.tolist()
    
    async def _chunked_inference(self, input_array: np.ndarray) -> list[list[float]]:
        """Chunked processing for large inputs."""
        chunk_size = 100
        results = []
        
        for i in range(0, len(input_array), chunk_size):
            chunk = input_array[i:i + chunk_size]
            await asyncio.sleep(0.001)  # Simulate chunk processing
            chunk_result = chunk * 2.0
            results.extend(chunk_result.tolist())
        
        return results
    
    def get_model_stats(self) -> dict[str, Any]:
        """Get model performance statistics."""
        avg_inference_time = (self.total_inference_time / max(1, self.prediction_count)) * 1000
        
        return {
            "model_loaded": self.model_loaded,
            "load_time_seconds": self.load_time,
            "prediction_count": self.prediction_count,
            "avg_inference_time_ms": avg_inference_time,
            "total_inference_time_seconds": self.total_inference_time
        }


# Global state
model_loader: OptimizedModelLoader | None = None
service_start_time = time.time()
auto_scaler = None
performance_optimizer = None


def generate_cache_key(input_data: list[list[float]]) -> str:
    """Generate cache key for request data."""
    # Create deterministic hash of input data
    input_str = str(sorted([sorted(row) for row in input_data]))
    return hashlib.md5(input_str.encode()).hexdigest()


async def get_request_id() -> str:
    """Generate unique request ID."""
    return str(uuid.uuid4())


async def track_concurrent_requests(request: Request):
    """Track concurrent request metrics."""
    # Simple tracking using request state
    if not hasattr(request.app.state, 'concurrent_count'):
        request.app.state.concurrent_count = 0
    
    request.app.state.concurrent_count += 1
    CONCURRENT_REQUESTS.observe(request.app.state.concurrent_count)
    
    try:
        yield
    finally:
        request.app.state.concurrent_count = max(0, request.app.state.concurrent_count - 1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan with performance optimization."""
    global model_loader, service_start_time, auto_scaler, performance_optimizer
    
    # Startup
    service_start_time = time.time()
    logger.info("Starting optimized NIM service...")
    
    # Initialize performance optimizer
    perf_config = PerformanceConfig(
        enable_request_batching=True,
        enable_response_caching=True,
        cache_size_mb=1024,
        max_batch_size=64
    )
    performance_optimizer = initialize_performance_optimizer(perf_config)
    
    # Initialize auto-scaler
    scaling_config = ScalingConfig(
        min_replicas=2,
        max_replicas=50,
        enable_predictive_scaling=True,
        enable_cost_optimization=True
    )
    auto_scaler = initialize_auto_scaling(scaling_config)
    
    # Load model
    import os
    model_path = os.getenv("MODEL_PATH", "/models/model.onnx")
    model_loader = OptimizedModelLoader(model_path)
    
    try:
        await model_loader.load_model()
        
        # Start auto-scaling in background
        if global_auto_scaler:
            asyncio.create_task(global_auto_scaler.start_auto_scaling())
        
        logger.info("Optimized NIM service started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
    
    # Cache warmup with common patterns
    if performance_optimizer and performance_optimizer.cache:
        common_requests = [
            ("warmup_1", [[1.0, 2.0, 3.0]], lambda x: model_loader.predict(x)[0]),
            ("warmup_2", [[0.5, 1.5, 2.5]], lambda x: model_loader.predict(x)[0]),
        ]
        await performance_optimizer.warmup_cache(common_requests)
    
    yield
    
    # Shutdown
    logger.info("Shutting down optimized NIM service...")
    if global_auto_scaler:
        global_auto_scaler.stop_auto_scaling()


# Create optimized FastAPI app
app = FastAPI(
    title="Optimized NIM Service API",
    description="High-performance NVIDIA NIM microservice with auto-scaling and intelligent optimization",
    version="3.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


@app.post("/v1/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    req: Request,
    background_tasks: BackgroundTasks,
    request_id: str = Depends(get_request_id),
    _: Any = Depends(track_concurrent_requests)
):
    """Optimized prediction endpoint with caching, batching, and scaling."""
    REQUEST_COUNT.labels(method="POST", endpoint="/v1/predict", status="start").inc()
    
    if not model_loader or not model_loader.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Model not loaded",
                "request_id": request_id,
                "timestamp": time.time()
            }
        )
    
    start_time = time.time()
    served_from_cache = False
    optimizations_applied = []
    
    try:
        with REQUEST_DURATION.time():
            # Generate cache key
            cache_key = generate_cache_key(request.input) if request.use_cache else None
            
            # Try performance optimization
            if performance_optimizer and cache_key:
                
                async def inference_processor(input_data):
                    predictions, metadata = await model_loader.predict(input_data, use_optimization=True)
                    return predictions, metadata
                
                result = await performance_optimizer.optimize_request(
                    cache_key, 
                    request.input,
                    lambda data: inference_processor(data)
                )
                
                if isinstance(result, tuple):
                    predictions, inference_metadata = result
                else:
                    predictions = result
                    inference_metadata = {"optimizations_applied": []}
                
                # Check if served from cache
                if performance_optimizer.cache:
                    cache_stats = performance_optimizer.cache.get_cache_stats()
                    served_from_cache = cache_stats["hit_count"] > CACHE_HITS._value._value
                    if served_from_cache:
                        CACHE_HITS.inc()
                        optimizations_applied.append("cache_hit")
                    else:
                        CACHE_MISSES.inc()
                
                optimizations_applied.extend(inference_metadata.get("optimizations_applied", []))
            else:
                # Direct inference without optimization
                predictions, inference_metadata = await model_loader.predict(request.input)
                optimizations_applied.extend(inference_metadata.get("optimizations_applied", []))
            
            # Calculate performance metrics
            total_time_ms = (time.time() - start_time) * 1000
            batch_size = len(request.input)
            
            BATCH_SIZE.observe(batch_size)
            
            # Calculate performance score
            performance_score = min(100.0, max(0.0, 100 - total_time_ms / 10))  # 100 for <10ms, 0 for >1000ms
            
            # Background tasks for optimization
            background_tasks.add_task(
                optimize_performance_background,
                total_time_ms,
                batch_size,
                optimizations_applied
            )
            
            REQUEST_COUNT.labels(method="POST", endpoint="/v1/predict", status="success").inc()
            
            return PredictionResponse(
                predictions=predictions,
                inference_time_ms=total_time_ms,
                request_id=request_id,
                served_from_cache=served_from_cache,
                batch_size=batch_size,
                optimization_applied=optimizations_applied,
                performance_score=performance_score
            )
    
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/v1/predict", status="error").inc()
        logger.error(f"Optimized prediction error: {e}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Optimized inference failed",
                "request_id": request_id,
                "timestamp": time.time()
            }
        )


async def optimize_performance_background(
    response_time_ms: float,
    batch_size: int,
    optimizations_applied: list[str]
):
    """Background task for performance optimization."""
    if not performance_optimizer:
        return
    
    try:
        # Auto-tune performance based on recent metrics
        recommendations = await performance_optimizer.auto_tune_performance()
        
        if recommendations:
            logger.info(f"Performance recommendations: {recommendations}")
        
        # Periodic memory optimization
        if len(optimizations_applied) > 5:  # High optimization activity
            await performance_optimizer.optimize_memory_usage()
        
    except Exception as e:
        logger.warning(f"Background performance optimization failed: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check with optimization metrics."""
    uptime = time.time() - service_start_time
    
    # Get performance stats
    perf_stats = {}
    if performance_optimizer:
        perf_stats = performance_optimizer.get_performance_stats()
    
    # Get scaling status
    scaling_status = {}
    if global_auto_scaler:
        scaling_status = global_auto_scaler.scaler.get_scaling_status()
    
    # Get optimization recommendations
    recommendations = []
    if performance_optimizer and uptime > 300:  # After 5 minutes
        try:
            recommendations = await performance_optimizer.auto_tune_performance()
        except Exception:
            pass
    
    # Determine overall status
    status = "healthy"
    if not model_loader or not model_loader.model_loaded:
        status = "degraded"
    elif perf_stats.get("cache_stats", {}).get("hit_rate", 0) < 0.2 and uptime > 600:
        status = "suboptimal"
    
    return HealthResponse(
        status=status,
        version="3.0.0",
        uptime_seconds=uptime,
        performance_stats=perf_stats,
        scaling_status=scaling_status,
        optimization_recommendations=recommendations or []
    )


@app.get("/metrics")
async def metrics():
    """Enhanced Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/performance/stats")
async def performance_stats():
    """Detailed performance statistics endpoint."""
    if not performance_optimizer:
        return {"error": "Performance optimizer not initialized"}
    
    stats = performance_optimizer.get_performance_stats()
    
    if model_loader:
        stats["model_stats"] = model_loader.get_model_stats()
    
    return stats


@app.post("/admin/optimize")
async def manual_optimization():
    """Manual performance optimization trigger."""
    if not performance_optimizer:
        return {"error": "Performance optimizer not initialized"}
    
    optimizations = await performance_optimizer.optimize_memory_usage()
    recommendations = await performance_optimizer.auto_tune_performance()
    
    return {
        "optimizations_applied": optimizations,
        "recommendations": recommendations,
        "timestamp": time.time()
    }


@app.get("/admin/scaling")
async def scaling_status():
    """Get current auto-scaling status."""
    if not global_auto_scaler:
        return {"error": "Auto-scaler not initialized"}
    
    return global_auto_scaler.scaler.get_scaling_status()


@app.post("/admin/scaling/analyze")
async def analyze_scaling():
    """Manually trigger scaling analysis."""
    if not global_auto_scaler:
        return {"error": "Auto-scaler not initialized"}
    
    recommendation = await global_auto_scaler.manual_scale_check()
    
    return {
        "action": recommendation.action.value,
        "target_replicas": recommendation.target_replicas,
        "current_replicas": recommendation.current_replicas,
        "confidence": recommendation.confidence,
        "reasoning": recommendation.reasoning,
        "expected_improvement": recommendation.expected_improvement,
        "cost_impact": recommendation.cost_impact,
        "urgency": recommendation.urgency
    }


@app.get("/")
async def root():
    """Root endpoint with optimization features."""
    return {
        "service": "Optimized NVIDIA NIM Microservice",
        "version": "3.0.0",
        "features": [
            "Intelligent Request Caching",
            "Dynamic Request Batching",
            "Auto-Scaling with Predictive Analytics",
            "Performance Optimization Engine",
            "Memory Management",
            "Connection Pooling",
            "Real-time Performance Tuning"
        ],
        "optimization_status": {
            "performance_optimizer": performance_optimizer is not None,
            "auto_scaler": global_auto_scaler is not None,
            "uptime_minutes": (time.time() - service_start_time) / 60
        },
        "endpoints": {
            "predict": "/v1/predict",
            "health": "/health",
            "metrics": "/metrics",
            "performance_stats": "/performance/stats",
            "admin_optimize": "/admin/optimize",
            "admin_scaling": "/admin/scaling",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)