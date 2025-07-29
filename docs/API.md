# Nimify Anything API Documentation

## Overview

The Nimify Anything API provides programmatic access to create, configure, and deploy NVIDIA NIM services from various model formats.

## Core Classes

### `Nimifier`

Main class for wrapping models into NIM services.

```python
from nimify import Nimifier, ModelConfig

# Initialize with configuration
config = ModelConfig(
    name="my-model",
    max_batch_size=32,
    dynamic_batching=True
)

nimifier = Nimifier(config)
```

#### Methods

##### `wrap_model(model_path, **kwargs)`

Wraps a model file into a NIM service.

**Parameters:**
- `model_path` (str): Path to the model file (ONNX, TensorRT, etc.)
- `input_schema` (dict, optional): Input tensor specifications
- `output_schema` (dict, optional): Output tensor specifications
- `preprocessing` (list, optional): Preprocessing pipeline steps
- `postprocessing` (list, optional): Postprocessing pipeline steps

**Returns:**
- `NIMService`: Configured NIM service instance

**Example:**
```python
service = nimifier.wrap_model(
    "model.onnx",
    input_schema={"input": "float32[?,3,224,224]"},
    output_schema={"predictions": "float32[?,1000]"}
)
```

##### `create_ensemble(models, pipeline_type="sequential")`

Creates a multi-model ensemble service.

**Parameters:**
- `models` (dict): Dictionary mapping step names to model paths
- `pipeline_type` (str): "sequential" or "parallel"

**Returns:**
- `EnsembleService`: Multi-model service instance

### `ModelConfig`

Configuration class for model deployment settings.

```python
config = ModelConfig(
    name="sentiment-analyzer",
    max_batch_size=64,
    dynamic_batching=True,
    preferred_batch_sizes=[8, 16, 32, 64],
    max_queue_delay_microseconds=100,
    gpu_memory_fraction=0.8
)
```

#### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Required | Service name |
| `max_batch_size` | int | 8 | Maximum batch size |
| `dynamic_batching` | bool | False | Enable dynamic batching |
| `preferred_batch_sizes` | list | None | Preferred batch sizes for optimization |
| `max_queue_delay_microseconds` | int | 0 | Maximum queueing delay |
| `gpu_memory_fraction` | float | 1.0 | GPU memory fraction to use |
| `cpu_cores` | int | None | CPU cores to allocate |
| `model_warmup` | bool | True | Enable model warmup |

### `NIMService`

Represents a configured NIM service ready for deployment.

#### Methods

##### `generate_openapi(output_path)`

Generates OpenAPI specification for the service.

**Parameters:**
- `output_path` (str): Path to save the OpenAPI JSON/YAML file

##### `generate_helm_chart(output_dir)`

Generates Kubernetes Helm chart for deployment.

**Parameters:**
- `output_dir` (str): Directory to save the Helm chart

##### `build_container(tag, registry=None)`

Builds Docker container image for the service.

**Parameters:**
- `tag` (str): Container image tag
- `registry` (str, optional): Container registry URL

##### `deploy(target="local", **kwargs)`

Deploys the service to specified target.

**Parameters:**
- `target` (str): Deployment target ("local", "kubernetes", "docker-compose")
- `**kwargs`: Target-specific deployment options

## CLI Reference

### `nimify create`

Creates a new NIM service from a model file.

```bash
nimify create model.onnx --name my-service --port 8080
```

**Options:**
- `--name` (str): Service name
- `--port` (int): Service port (default: 8000)
- `--max-batch-size` (int): Maximum batch size
- `--dynamic-batching`: Enable dynamic batching
- `--gpu-memory` (str): GPU memory limit (e.g., "4GB")
- `--input-shapes` (str): Input tensor shapes
- `--preprocessing` (str): Preprocessing pipeline
- `--postprocessing` (str): Postprocessing pipeline

### `nimify deploy`

Deploys a service to a target environment.

```bash
nimify deploy my-service --target kubernetes --replicas 3
```

**Options:**
- `--target` (str): Deployment target
- `--replicas` (int): Number of replicas
- `--autoscale`: Enable autoscaling
- `--namespace` (str): Kubernetes namespace
- `--wait`: Wait for deployment to complete

### `nimify build`

Builds container image for a service.

```bash
nimify build my-service --tag my-registry/my-service:v1
```

**Options:**
- `--tag` (str): Container image tag
- `--push`: Push to registry after building
- `--optimize`: Enable build optimizations
- `--no-cache`: Disable build cache

### `nimify helm`

Generates Helm charts for Kubernetes deployment.

```bash
nimify helm create my-service --output-dir ./charts
```

**Options:**
- `--output-dir` (str): Output directory for charts
- `--values` (str): Custom values file
- `--namespace` (str): Target namespace

## REST API Endpoints

When deployed, NIM services expose the following REST API endpoints:

### Health Checks

- `GET /health` - Health check endpoint
- `GET /ready` - Readiness probe endpoint
- `GET /metrics` - Prometheus metrics endpoint

### Model Inference

- `POST /v1/predict` - Single prediction
- `POST /v1/batch` - Batch predictions
- `GET /v1/models` - List available models
- `GET /v1/models/{model_name}` - Model metadata

### Example Request

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "input": [[1.0, 2.0, 3.0]]
    }
  }'
```

### Example Response

```json
{
  "outputs": {
    "predictions": [[0.1, 0.8, 0.1]]
  },
  "model_name": "my-model",
  "model_version": "1",
  "request_id": "abc123"
}
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Input tensor shape mismatch",
    "details": {
      "expected": "[?, 3, 224, 224]",
      "received": "[?, 224, 224, 3]"
    }
  },
  "request_id": "abc123"
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `INVALID_INPUT` | Input validation failed |
| `MODEL_ERROR` | Model inference error |
| `RESOURCE_EXHAUSTED` | Resource limits exceeded |
| `INTERNAL_ERROR` | Internal server error |
| `SERVICE_UNAVAILABLE` | Service temporarily unavailable |

## Configuration Files

### Service Configuration

```yaml
# nimify-config.yaml
service:
  name: my-model
  version: "1.0"
  
model:
  path: "./model.onnx"
  format: "onnx"
  
runtime:
  max_batch_size: 32
  dynamic_batching: true
  gpu_memory_fraction: 0.8
  
server:
  port: 8000
  host: "0.0.0.0"
  workers: 1
```

### Preprocessing Configuration

```yaml
# preprocessing.yaml
pipeline:
  - name: "resize"
    params:
      size: [224, 224]
  - name: "normalize"
    params:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  - name: "to_tensor"
```

## Monitoring and Metrics

### Prometheus Metrics

Nimify services automatically expose Prometheus metrics:

- `nim_request_duration_seconds` - Request latency histogram
- `nim_request_count_total` - Total request count
- `nim_batch_size_histogram` - Batch size distribution
- `nim_gpu_utilization_percent` - GPU utilization percentage
- `nim_model_loading_time_seconds` - Model loading time
- `nim_queue_size` - Current queue size

### Custom Metrics

```python
from nimify.metrics import Counter, Histogram

# Custom metrics
custom_counter = Counter('my_custom_counter', 'Description')
custom_histogram = Histogram('my_latency', 'Custom latency metric')

# Use in service
custom_counter.inc()
with custom_histogram.time():
    # Your code here
    pass
```

## Advanced Features

### Custom Preprocessors

```python
from nimify import Preprocessor

@Preprocessor.register("custom_normalize")
def custom_normalize(input_data, mean=0.0, std=1.0):
    """Custom normalization preprocessor"""
    return (input_data - mean) / std

# Use in configuration
service = nimifier.wrap_model(
    "model.onnx",
    preprocessing=["custom_normalize"]
)
```

### A/B Testing

```python
from nimify import ABTestConfig

# Configure A/B test
ab_config = ABTestConfig(
    variants={
        "control": "model_v1.onnx",
        "treatment": "model_v2.onnx"
    },
    traffic_split={"control": 0.8, "treatment": 0.2}
)

service = nimifier.create_ab_test("my-experiment", ab_config)
```

### Model Validation

```python
from nimify import ModelValidator

validator = ModelValidator()
results = validator.validate(
    model_path="model.onnx",
    test_data="test_samples.json",
    checks=["output_shape", "latency", "throughput"]
)

if not results.passed:
    print(f"Validation failed: {results.errors}")
```

## SDK Integration

### Python SDK

```python
import nimify

# Initialize client
client = nimify.Client("http://localhost:8000")

# Make predictions
result = client.predict({
    "input": [[1.0, 2.0, 3.0]]
})
print(result["predictions"])
```

### JavaScript SDK

```javascript
import { NimifyClient } from 'nimify-js';

const client = new NimifyClient('http://localhost:8000');

const result = await client.predict({
  input: [[1.0, 2.0, 3.0]]
});
console.log(result.predictions);
```

## Best Practices

### Performance Optimization

1. **Batch Size Tuning**: Start with small batches and increase based on GPU memory
2. **Dynamic Batching**: Enable for variable input sizes
3. **Model Optimization**: Use TensorRT for NVIDIA GPUs
4. **Resource Allocation**: Monitor GPU/CPU utilization

### Security

1. **Input Validation**: Always validate input data
2. **Authentication**: Use API keys or OAuth for production
3. **Rate Limiting**: Implement rate limiting for public APIs
4. **Monitoring**: Monitor for suspicious activity

### Deployment

1. **Health Checks**: Configure proper health and readiness probes
2. **Resource Limits**: Set appropriate CPU/memory limits
3. **Autoscaling**: Configure HPA based on metrics
4. **Logging**: Enable structured logging for observability