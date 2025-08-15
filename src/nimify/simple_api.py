"""Simple FastAPI application for basic NIM service functionality."""

import time
import uuid
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import numpy as np

from .logging_config import setup_logging

# Set up logging
setup_logging("nim-service-simple", log_level="INFO")


# Simple metrics
REQUEST_COUNT = Counter('nim_request_count_total', 'Total requests', ['endpoint', 'status'])
REQUEST_DURATION = Histogram('nim_request_duration_seconds', 'Request duration')


class PredictionRequest(BaseModel):
    """Simple request model for predictions."""
    input: List[List[float]] = Field(..., description="Input data for inference")


class PredictionResponse(BaseModel):
    """Simple response model for predictions."""
    predictions: List[List[float]] = Field(..., description="Model predictions")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")


# Create FastAPI app
app = FastAPI(
    title="NIM Service API (Simple)",
    description="Basic NVIDIA NIM microservice for model inference",
    version="1.0.0"
)


@app.post("/v1/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Simple prediction endpoint with mock inference."""
    REQUEST_COUNT.labels(endpoint="/v1/predict", status="start").inc()
    
    start_time = time.time()
    
    try:
        # Mock inference (replace with actual model inference)
        input_data = np.array(request.input, dtype=np.float32)
        
        # Simple mock prediction: return input * 2
        predictions = (input_data * 2).tolist()
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        REQUEST_COUNT.labels(endpoint="/v1/predict", status="success").inc()
        REQUEST_DURATION.observe(inference_time_ms / 1000)
        
        return PredictionResponse(
            predictions=predictions,
            inference_time_ms=inference_time_ms
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/v1/predict", status="error").inc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Simple health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    """Root endpoint with basic service info."""
    return {
        "service": "NVIDIA NIM Microservice (Simple)",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/v1/predict",
            "health": "/health",
            "metrics": "/metrics"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)