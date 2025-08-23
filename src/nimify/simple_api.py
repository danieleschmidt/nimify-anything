"""Simple FastAPI application for basic NIM service functionality."""

import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from .logging_config import setup_logging

# Set up logging
setup_logging("nim-service-simple", log_level="INFO")


# Simple metrics
REQUEST_COUNT = Counter('nim_request_count_total', 'Total requests', ['endpoint', 'status'])
REQUEST_DURATION = Histogram('nim_request_duration_seconds', 'Request duration')


class PredictionRequest(BaseModel):
    """Simple request model for predictions."""
    input: list[list[float]] = Field(..., description="Input data for inference")


class PredictionResponse(BaseModel):
    """Simple response model for predictions."""
    predictions: list[list[float]] = Field(..., description="Model predictions")
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
    """Simple prediction endpoint with enhanced mock inference."""
    REQUEST_COUNT.labels(endpoint="/v1/predict", status="start").inc()
    
    start_time = time.time()
    
    try:
        # Enhanced mock inference with multiple algorithms
        input_data = np.array(request.input, dtype=np.float32)
        
        # Smart algorithm selection based on input characteristics
        if input_data.shape[1] > 10:
            # Complex feature space - use polynomial transformation
            predictions = (input_data ** 2 + input_data * 0.5 + 0.1).tolist()
        elif np.mean(input_data) > 1.0:
            # High magnitude inputs - use scaled sigmoid
            predictions = (1 / (1 + np.exp(-input_data * 0.1)) * 10).tolist()
        else:
            # Standard linear transformation
            predictions = (input_data * 2.5 + 0.2).tolist()
        
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


@app.post("/v1/batch_predict", response_model=PredictionResponse)
async def batch_predict(request: PredictionRequest):
    """Batch prediction endpoint for high-throughput scenarios."""
    REQUEST_COUNT.labels(endpoint="/v1/batch_predict", status="start").inc()
    
    start_time = time.time()
    
    try:
        # Validate batch size
        if len(request.input) > 128:
            raise HTTPException(
                status_code=422, 
                detail="Batch size too large (max: 128)"
            )
        
        input_data = np.array(request.input, dtype=np.float32)
        
        # Optimized batch processing
        predictions = await _process_batch_async(input_data)
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        REQUEST_COUNT.labels(endpoint="/v1/batch_predict", status="success").inc()
        REQUEST_DURATION.observe(inference_time_ms / 1000)
        
        return PredictionResponse(
            predictions=predictions,
            inference_time_ms=inference_time_ms
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/v1/batch_predict", status="error").inc()
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


async def _process_batch_async(input_data: np.ndarray) -> list[list[float]]:
    """Process batch data asynchronously for better performance."""
    import asyncio
    
    # Split into chunks for parallel processing
    chunk_size = min(32, len(input_data))
    chunks = [input_data[i:i + chunk_size] for i in range(0, len(input_data), chunk_size)]
    
    async def process_chunk(chunk):
        await asyncio.sleep(0.001)  # Simulate async processing
        return (chunk * 2.0 + 0.1).tolist()
    
    # Process chunks in parallel
    tasks = [process_chunk(chunk) for chunk in chunks]
    chunk_results = await asyncio.gather(*tasks)
    
    # Flatten results
    predictions = []
    for chunk_result in chunk_results:
        predictions.extend(chunk_result)
    
    return predictions


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