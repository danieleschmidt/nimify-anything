# Performance Optimization Guide

## Overview

This guide covers best practices for optimizing Nimify service performance across different deployment scenarios.

## Model Optimization

### TensorRT Optimization

For NVIDIA GPUs, TensorRT provides significant performance improvements:

```python
from nimify import TensorRTOptimizer

# Optimize ONNX model for TensorRT
optimizer = TensorRTOptimizer()
optimized_model = optimizer.optimize(
    "model.onnx",
    precision="fp16",  # Use mixed precision
    max_batch_size=32,
    max_workspace_size="4GB"
)

# Deploy optimized model
service = nimifier.wrap_model(optimized_model)
```

### ONNX Runtime Optimization

```python
from nimify import ONNXOptimizer

optimizer = ONNXOptimizer()
optimized_model = optimizer.optimize(
    "model.onnx",
    optimization_level="all",
    enable_cpu_mem_arena=True,
    enable_mem_pattern=True
)
```

## Batch Processing

### Dynamic Batching

Enable dynamic batching to improve throughput:

```python
config = ModelConfig(
    name="my-model",
    dynamic_batching=True,
    max_batch_size=32,
    preferred_batch_sizes=[4, 8, 16, 32],
    max_queue_delay_microseconds=1000
)
```

### Batch Size Guidelines

| Model Type | Recommended Batch Size | GPU Memory |
|------------|----------------------|------------|
| Small CNN (ResNet-18) | 64-128 | 8GB |
| Large CNN (ResNet-152) | 16-32 | 8GB |
| Transformer (BERT-base) | 8-16 | 8GB |
| Transformer (GPT-3.5) | 1-4 | 24GB |

## Memory Management

### GPU Memory Optimization

```python
config = ModelConfig(
    gpu_memory_fraction=0.8,  # Use 80% of GPU memory
    allow_memory_growth=True,  # Grow memory as needed
    memory_limit="6GB"         # Hard limit
)
```

### CPU Memory Optimization

```python
# Limit CPU memory usage
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

config = ModelConfig(
    cpu_cores=4,
    max_workers=2
)
```

## Caching Strategies

### Model Caching

```python
from nimify.cache import ModelCache

# Enable model caching
cache = ModelCache(
    backend="redis",
    host="localhost",
    port=6379,
    ttl=3600  # 1 hour
)

config = ModelConfig(
    model_cache=cache
)
```

### Result Caching

```python
from nimify.cache import ResultCache

# Cache inference results
result_cache = ResultCache(
    backend="memcached",
    servers=["localhost:11211"],
    expiry=300  # 5 minutes
)

service = nimifier.wrap_model(
    "model.onnx",
    result_cache=result_cache
)
```

## Networking Optimization

### Connection Pooling

```python
from nimify.client import NimifyClient

client = NimifyClient(
    base_url="http://localhost:8000",
    max_connections=100,
    max_keepalive_connections=20,
    keepalive_expiry=30
)
```

### Load Balancing

```yaml
# nginx.conf
upstream nimify_backend {
    least_conn;
    server nimify-1:8000 weight=3;
    server nimify-2:8000 weight=3;
    server nimify-3:8000 weight=2;
}

server {
    listen 80;
    location / {
        proxy_pass http://nimify_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring and Profiling

### Performance Metrics

```python
from nimify.metrics import PerformanceMonitor

monitor = PerformanceMonitor()

# Track key metrics
@monitor.track_latency("inference_time")
@monitor.track_throughput("requests_per_second")
def predict(data):
    return model.predict(data)
```

### Profiling Tools

```bash
# Profile with py-spy
py-spy record -o profile.svg -- python -m nimify serve model.onnx

# Profile with cProfile
python -m cProfile -o profile.stats -m nimify serve model.onnx

# Analyze with snakeviz
snakeviz profile.stats
```

### NVIDIA Profiling

```bash
# Profile GPU usage with nsight
nsys profile --trace=cuda,nvtx python -m nimify serve model.onnx

# Profile with NVIDIA Triton metrics
curl http://localhost:8002/metrics
```

## Deployment Optimization

### Container Optimization

```dockerfile
# Multi-stage build for smaller images
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as runtime

# Use specific Python version
FROM python:3.10-slim as builder

# Install only required packages
RUN pip install --no-cache-dir nimify-anything[minimal]

# Copy only necessary files
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY model.onnx /app/model.onnx

# Set optimal environment variables
ENV PYTHONPATH=/app
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_CACHE_DISABLE=0
```

### Kubernetes Optimization

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nimify-service
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: nimify
        image: nimify:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        env:
        - name: NIMIFY_BATCH_SIZE
          value: "32"
        - name: NIMIFY_WORKERS
          value: "1"
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
```

### Autoscaling Configuration

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nimify-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nimify-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: nim_request_duration_seconds_p95
      target:
        type: AverageValue
        averageValue: "100m"  # 100ms
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

## Performance Benchmarking

### Load Testing

```python
import asyncio
import aiohttp
import time

async def benchmark_service(url, num_requests=1000, concurrency=50):
    """Benchmark service performance"""
    
    async def make_request(session):
        async with session.post(f"{url}/v1/predict", 
                               json={"inputs": {"data": [[1.0, 2.0, 3.0]]}}) as response:
            return await response.json()
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_request():
            async with semaphore:
                return await make_request(session)
        
        tasks = [bounded_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Requests: {num_requests}")
    print(f"Duration: {duration:.2f}s")
    print(f"RPS: {num_requests/duration:.2f}")
    print(f"Avg Latency: {duration/num_requests*1000:.2f}ms")

# Run benchmark
asyncio.run(benchmark_service("http://localhost:8000"))
```

### Performance Testing Framework

```python
from nimify.testing import PerformanceTest

class ModelPerformanceTest(PerformanceTest):
    def setUp(self):
        self.service = nimifier.wrap_model("model.onnx")
        self.test_data = self.load_test_data()
    
    def test_latency_p95_under_100ms(self):
        """Test that P95 latency is under 100ms"""
        results = self.run_load_test(
            requests=1000,
            concurrency=10
        )
        assert results.latency_p95 < 0.1
    
    def test_throughput_over_100_rps(self):
        """Test that throughput is over 100 RPS"""
        results = self.run_load_test(
            duration=60,  # 1 minute
            concurrency=20
        )
        assert results.requests_per_second > 100
    
    def test_memory_usage_stable(self):
        """Test that memory usage remains stable"""
        initial_memory = self.get_memory_usage()
        
        self.run_load_test(
            requests=10000,
            concurrency=50
        )
        
        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be < 10%
        assert memory_increase / initial_memory < 0.1
```

## Optimization Checklist

### Model Level
- [ ] Convert to TensorRT for NVIDIA GPUs
- [ ] Use mixed precision (FP16)
- [ ] Optimize model architecture
- [ ] Remove unused layers
- [ ] Quantize model weights

### Service Level
- [ ] Enable dynamic batching
- [ ] Tune batch sizes
- [ ] Configure memory limits
- [ ] Enable result caching
- [ ] Optimize preprocessing pipeline

### Infrastructure Level
- [ ] Use GPU-optimized containers
- [ ] Configure resource limits
- [ ] Set up horizontal autoscaling
- [ ] Implement load balancing
- [ ] Enable monitoring

### Network Level
- [ ] Use connection pooling
- [ ] Enable HTTP/2
- [ ] Configure keep-alive
- [ ] Implement request compression
- [ ] Set up CDN for static assets

## Troubleshooting Performance Issues

### High Latency

1. **Check GPU utilization**: `nvidia-smi`
2. **Profile inference pipeline**: Use NVTX markers
3. **Analyze batch sizes**: Monitor batch size distribution
4. **Check memory pressure**: Monitor GPU/CPU memory usage

### Low Throughput

1. **Increase batch size**: If memory allows
2. **Enable dynamic batching**: For variable input sizes
3. **Scale horizontally**: Add more replicas
4. **Optimize model**: Use TensorRT or quantization

### Memory Issues

1. **Monitor memory leaks**: Use memory profiling tools
2. **Adjust batch sizes**: Reduce if OOM errors occur
3. **Enable memory pooling**: Reuse allocated memory
4. **Clear model cache**: Periodically clear cached models

### GPU Utilization

1. **Check model compatibility**: Ensure GPU optimization
2. **Optimize data transfer**: Minimize CPU-GPU transfers
3. **Use CUDA streams**: For parallel execution
4. **Profile kernel execution**: Identify bottlenecks