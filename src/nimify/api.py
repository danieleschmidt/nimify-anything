"""FastAPI application for NIM service runtime."""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import numpy as np

from .validation import RequestValidator, ValidationError
from .logging_config import setup_logging, log_api_request, log_security_event, log_performance_metric


# Metrics
REQUEST_COUNT = Counter('nim_request_count_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('nim_request_duration_seconds', 'Request duration')
INFERENCE_DURATION = Histogram('nim_inference_duration_seconds', 'Model inference duration')
ERROR_COUNT = Counter('nim_error_count_total', 'Total errors', ['error_type', 'endpoint'])
CONCURRENT_REQUESTS = Histogram('nim_concurrent_requests', 'Concurrent requests')

# Set up logging
setup_logging("nim-service", log_level="INFO", enable_audit=True)
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """Request model for predictions with validation."""
    input: List[List[float]] = Field(..., description="Input data for inference", min_items=1, max_items=64)
    
    class Config:
        schema_extra = {
            "example": {
                "input": [[1.0, 2.0, 3.0]]
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[List[float]] = Field(..., description="Model predictions")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    request_id: str = Field(..., description="Unique request identifier")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    request_id: str = Field(..., description="Request identifier")
    timestamp: str = Field(..., description="Error timestamp")


class ModelLoader:
    """Handles model loading and inference."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_names: Optional[List[str]] = None
    
    async def load_model(self):
        """Load the ONNX model asynchronously."""
        try:
            # Create ONNX Runtime session with GPU if available
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            logger.info(f"Model loaded: {self.model_path}")
            logger.info(f"Input: {self.input_name}")
            logger.info(f"Outputs: {self.output_names}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_path}: {e}")
            raise
    
    async def predict(self, input_data: List[List[float]]) -> List[List[float]]:
        """Run inference on input data."""
        if not self.session:
            raise ValueError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Convert input to numpy array
            input_array = np.array(input_data, dtype=np.float32)
            
            # Run inference
            outputs = self.session.run(
                self.output_names,
                {self.input_name: input_array}
            )
            
            # Convert outputs to lists
            predictions = [output.tolist() for output in outputs]
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            INFERENCE_DURATION.observe(inference_time / 1000)
            
            return predictions[0] if len(predictions) == 1 else predictions
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise


# Global state
model_loader: Optional[ModelLoader] = None
service_start_time = time.time()

# Security
security = HTTPBearer(auto_error=False)


async def get_request_id() -> str:
    """Generate unique request ID."""
    return str(uuid.uuid4())


async def log_request_middleware(request: Request, call_next):
    """Middleware to log all requests."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        
        # Log API request
        log_api_request(
            method=request.method,
            endpoint=str(request.url.path),
            status_code=response.status_code,
            duration_ms=duration_ms,
            ip_address=client_ip,
            user_agent=request.headers.get("user-agent"),
            request_id=request_id
        )
        
        return response
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        
        # Log error
        log_api_request(
            method=request.method,
            endpoint=str(request.url.path),
            status_code=500,
            duration_ms=duration_ms,
            ip_address=client_ip,
            user_agent=request.headers.get("user-agent"),
            request_id=request_id
        )
        
        # Log security event for errors
        log_security_event(
            event_type="api_error",
            message=f"API error: {str(e)}",
            ip_address=client_ip,
            request_id=request_id
        )
        
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global model_loader, service_start_time
    
    # Startup
    service_start_time = time.time()
    import os
    model_path = os.getenv("MODEL_PATH", "/models/model.onnx")
    model_loader = ModelLoader(model_path)
    
    try:
        await model_loader.load_model()
        logger.info("NIM service started successfully")
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        # Continue anyway for health check endpoint
    
    yield
    
    # Shutdown
    logger.info("NIM service shutting down")


app = FastAPI(
    title="NIM Service API",
    description="Auto-generated NVIDIA NIM microservice with security and monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# Add security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # Configure based on deployment
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on security requirements
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

# Add request logging middleware
app.middleware("http")(log_request_middleware)


@app.post("/v1/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, req: Request, request_id: str = Depends(get_request_id)):
    """Run model inference with comprehensive error handling."""
    REQUEST_COUNT.labels(method="POST", endpoint="/v1/predict", status="start").inc()
    
    # Rate limiting check (basic)
    client_ip = req.client.host if req.client else "unknown"
    
    if not model_loader or not model_loader.session:
        ERROR_COUNT.labels(error_type="model_not_loaded", endpoint="/v1/predict").inc()
        REQUEST_COUNT.labels(method="POST", endpoint="/v1/predict", status="error").inc()
        
        log_security_event(
            event_type="service_unavailable",
            message="Model not loaded for prediction request",
            ip_address=client_ip,
            request_id=request_id
        )
        
        raise HTTPException(
            status_code=503, 
            detail={
                "error": "Model not loaded",
                "request_id": request_id,
                "timestamp": time.time()
            }
        )
    
    try:
        # Validate request (Pydantic will handle this automatically)
        with REQUEST_DURATION.time():
            start_time = time.time()
            
            # Additional runtime validation
            if len(request.input) > 64:  # Max batch size
                raise HTTPException(status_code=422, detail="Batch size too large (max: 64)")
            
            if not all(isinstance(row, list) and all(isinstance(val, (int, float)) for val in row) for row in request.input):
                raise HTTPException(status_code=422, detail="Invalid input format: expected list of lists of numbers")
            
            # Run inference with circuit breaker protection
            try:
                predictions = await model_loader.predict(request.input)
                inference_time_ms = (time.time() - start_time) * 1000
            
            except CircuitBreakerException as e:
                ERROR_COUNT.labels(error_type="circuit_breaker_open", endpoint="/v1/predict").inc()
                REQUEST_COUNT.labels(method="POST", endpoint="/v1/predict", status="circuit_open").inc()
                
                log_security_event(
                    event_type="circuit_breaker_open",
                    message=f"Circuit breaker open for inference: {str(e)}",
                    ip_address=client_ip,
                    request_id=request_id
                )
                
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "Service temporarily unavailable (circuit breaker open)",
                        "request_id": request_id,
                        "timestamp": time.time(),
                        "retry_after": 30  # Suggest retry after 30 seconds
                    },
                    headers={"Retry-After": "30"}
                )
        
        # Log performance metrics
        log_performance_metric("inference_time", inference_time_ms, "ms")
        log_performance_metric("batch_size", len(request.input), "count")
        
        REQUEST_COUNT.labels(method="POST", endpoint="/v1/predict", status="success").inc()
        
        return PredictionResponse(
            predictions=predictions,
            inference_time_ms=inference_time_ms,
            request_id=request_id
        )
        
    except ValidationError as e:
        ERROR_COUNT.labels(error_type="validation_error", endpoint="/v1/predict").inc()
        REQUEST_COUNT.labels(method="POST", endpoint="/v1/predict", status="error").inc()
        
        log_security_event(
            event_type="validation_failure",
            message=f"Input validation failed: {str(e)}",
            ip_address=client_ip,
            request_id=request_id
        )
        
        raise HTTPException(
            status_code=422,
            detail={
                "error": f"Validation error: {str(e)}",
                "request_id": request_id,
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        ERROR_COUNT.labels(error_type="inference_error", endpoint="/v1/predict").inc()
        REQUEST_COUNT.labels(method="POST", endpoint="/v1/predict", status="error").inc()
        
        logger.error(f"Prediction error: {e}", extra={"request_id": request_id})
        
        log_security_event(
            event_type="inference_error",
            message=f"Inference failed: {str(e)}",
            ip_address=client_ip,
            request_id=request_id
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "request_id": request_id,
                "timestamp": time.time()
            }
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Service health check with detailed status."""
    model_loaded = model_loader is not None and model_loader.session is not None
    uptime = time.time() - service_start_time
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version="1.0.0",
        uptime_seconds=uptime
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "NVIDIA NIM Microservice", 
        "version": "1.0.0",
        "endpoints": {
            "predict": "/v1/predict",
            "health": "/health", 
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }