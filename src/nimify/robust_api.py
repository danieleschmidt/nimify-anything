"""Robust FastAPI application with error handling and circuit breaker."""

import time
import uuid
import logging
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import numpy as np

from .logging_config import setup_logging, log_api_request, log_security_event, log_performance_metric
from .error_handling import (
    global_error_handler, with_error_handling, ValidationError as ValidError,
    ModelError, SecurityError, InfrastructureError, ErrorSeverity
)
from .circuit_breaker import (
    ModelInferenceCircuitBreaker, CircuitBreakerException,
    global_circuit_breakers
)


# Set up logging
logger = setup_logging("nim-service-robust", log_level="INFO", enable_audit=True)

# Metrics
REQUEST_COUNT = Counter('nim_robust_request_count_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('nim_robust_request_duration_seconds', 'Request duration')
INFERENCE_DURATION = Histogram('nim_robust_inference_duration_seconds', 'Model inference duration')
ERROR_COUNT = Counter('nim_robust_error_count_total', 'Total errors', ['error_type', 'endpoint'])


class PredictionRequest(BaseModel):
    """Enhanced request model with validation."""
    input: List[List[float]] = Field(..., description="Input data for inference", min_length=1, max_length=64)


class PredictionResponse(BaseModel):
    """Enhanced response model with metadata."""
    predictions: List[List[float]] = Field(..., description="Model predictions")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    request_id: str = Field(..., description="Unique request identifier")
    model_version: str = Field("1.0.0", description="Model version")
    circuit_breaker_status: str = Field(..., description="Circuit breaker status")


class HealthResponse(BaseModel):
    """Comprehensive health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    circuit_breakers: Dict[str, str] = Field(..., description="Circuit breaker states")
    error_counts: Dict[str, int] = Field(..., description="Error statistics")


class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type")
    request_id: str = Field(..., description="Request identifier")
    timestamp: float = Field(..., description="Error timestamp")
    severity: str = Field(..., description="Error severity")
    recovery_suggestions: Optional[str] = Field(None, description="Recovery suggestions")


class RobustModelLoader:
    """Model loader with circuit breaker and error handling."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session: Optional[Any] = None
        self.input_name: Optional[str] = None
        self.output_names: Optional[List[str]] = None
        self.circuit_breaker = ModelInferenceCircuitBreaker("primary_model")
        self.fallback_available = False
    
    @with_error_handling(global_error_handler)
    async def load_model(self):
        """Load model with error handling."""
        try:
            # Mock model loading (replace with actual ONNX loading)
            self.session = {"mock": True}  # Placeholder
            self.input_name = "input"
            self.output_names = ["output"]
            
            logger.info(f"Model loaded successfully: {self.model_path}")
            
        except Exception as e:
            raise ModelError(
                f"Failed to load model {self.model_path}: {str(e)}",
                details={"model_path": self.model_path},
                recovery_suggestions="Check model file format and permissions"
            )
    
    async def predict(self, input_data: List[List[float]]) -> List[List[float]]:
        """Run inference with circuit breaker protection."""
        if not self.session:
            raise ModelError("Model not loaded")
        
        def _do_inference():
            """Internal inference function."""
            # Mock inference (replace with actual model inference)
            input_array = np.array(input_data, dtype=np.float32)
            predictions = (input_array * 2).tolist()  # Simple mock
            return predictions
        
        try:
            start_time = time.time()
            predictions = self.circuit_breaker.predict(_do_inference)
            inference_time = (time.time() - start_time) * 1000
            
            INFERENCE_DURATION.observe(inference_time / 1000)
            log_performance_metric("inference_time", inference_time, "ms")
            
            return predictions
            
        except CircuitBreakerException:
            # Circuit breaker is open
            if self.fallback_available:
                logger.warning("Using fallback prediction due to circuit breaker")
                return await self._fallback_predict(input_data)
            else:
                raise
        except Exception as e:
            raise ModelError(
                f"Inference failed: {str(e)}",
                severity=ErrorSeverity.HIGH,
                details={"input_shape": np.array(input_data).shape if input_data else None},
                recovery_suggestions="Check input format and model compatibility"
            )
    
    async def _fallback_predict(self, input_data: List[List[float]]) -> List[List[float]]:
        """Fallback prediction when circuit breaker is open."""
        # Simple fallback: return zeros or last known good prediction
        input_array = np.array(input_data, dtype=np.float32)
        return np.zeros_like(input_array).tolist()


# Global state
model_loader: Optional[RobustModelLoader] = None
service_start_time = time.time()


async def get_request_id() -> str:
    """Generate unique request ID."""
    return str(uuid.uuid4())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan manager."""
    global model_loader, service_start_time
    
    # Startup
    service_start_time = time.time()
    import os
    model_path = os.getenv("MODEL_PATH", "/models/model.onnx")
    model_loader = RobustModelLoader(model_path)
    
    try:
        await model_loader.load_model()
        logger.info("Robust NIM service started successfully")
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        # Continue for health check endpoint
    
    yield
    
    # Shutdown
    logger.info("Robust NIM service shutting down")
    # Reset circuit breakers for clean restart
    global_circuit_breakers.reset_all()


# Create FastAPI app
app = FastAPI(
    title="Robust NIM Service API",
    description="Production-ready NVIDIA NIM microservice with comprehensive error handling",
    version="2.0.0",
    lifespan=lifespan
)


@app.post("/v1/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    req: Request,
    request_id: str = Depends(get_request_id)
):
    """Enhanced prediction endpoint with comprehensive error handling."""
    REQUEST_COUNT.labels(method="POST", endpoint="/v1/predict", status="start").inc()
    
    client_ip = req.client.host if req.client else "unknown"
    
    if not model_loader or not model_loader.session:
        ERROR_COUNT.labels(error_type="model_not_loaded", endpoint="/v1/predict").inc()
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model not loaded",
                "error_type": "ModelError",
                "request_id": request_id,
                "timestamp": time.time(),
                "severity": "HIGH"
            }
        )
    
    try:
        with REQUEST_DURATION.time():
            start_time = time.time()
            
            # Enhanced input validation
            if len(request.input) > 64:
                raise HTTPException(status_code=422, detail="Batch size too large (max: 64)")
            
            # Run inference with circuit breaker protection
            try:
                predictions = await model_loader.predict(request.input)
                inference_time_ms = (time.time() - start_time) * 1000
                
            except CircuitBreakerException as e:
                ERROR_COUNT.labels(error_type="circuit_breaker_open", endpoint="/v1/predict").inc()
                
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "Service temporarily unavailable due to repeated failures",
                        "error_type": "CircuitBreakerException",
                        "request_id": request_id,
                        "timestamp": time.time(),
                        "severity": "HIGH",
                        "recovery_suggestions": "Wait 30 seconds before retrying"
                    },
                    headers={"Retry-After": "30"}
                )
        
        REQUEST_COUNT.labels(method="POST", endpoint="/v1/predict", status="success").inc()
        
        return PredictionResponse(
            predictions=predictions,
            inference_time_ms=inference_time_ms,
            request_id=request_id,
            circuit_breaker_status=model_loader.circuit_breaker.get_status()["state"]
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        ERROR_COUNT.labels(error_type="inference_error", endpoint="/v1/predict").inc()
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error during inference",
                "error_type": type(e).__name__,
                "request_id": request_id,
                "timestamp": time.time(),
                "severity": "HIGH"
            }
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check with system status."""
    model_loaded = model_loader is not None and model_loader.session is not None
    uptime = time.time() - service_start_time
    
    # Get circuit breaker states
    cb_status = global_circuit_breakers.get_all_status()
    cb_states = {name: status["state"] for name, status in cb_status.items()}
    
    # Get error statistics
    error_stats = global_error_handler.get_error_stats()
    
    # Determine overall status
    status = "healthy"
    if not model_loaded:
        status = "degraded"
    elif any(state == "open" for state in cb_states.values()):
        status = "degraded"
    elif sum(error_stats.values()) > 100:  # Threshold for too many errors
        status = "degraded"
    
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        version="2.0.0",
        uptime_seconds=uptime,
        circuit_breakers=cb_states,
        error_counts=error_stats
    )


@app.get("/metrics")
async def metrics():
    """Enhanced Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/debug/circuit-breakers")
async def debug_circuit_breakers():
    """Debug endpoint for circuit breaker status."""
    return global_circuit_breakers.get_all_status()


@app.post("/admin/reset-circuit-breakers")
async def reset_circuit_breakers():
    """Admin endpoint to reset all circuit breakers."""
    global_circuit_breakers.reset_all()
    return {"message": "All circuit breakers reset to closed state"}


@app.get("/")
async def root():
    """Root endpoint with enhanced service info."""
    return {
        "service": "Robust NVIDIA NIM Microservice",
        "version": "2.0.0",
        "features": [
            "Circuit Breaker Protection",
            "Comprehensive Error Handling", 
            "Security Monitoring",
            "Performance Metrics",
            "Automatic Recovery"
        ],
        "endpoints": {
            "predict": "/v1/predict",
            "health": "/health",
            "metrics": "/metrics",
            "debug": "/debug/circuit-breakers",
            "admin": "/admin/reset-circuit-breakers",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)