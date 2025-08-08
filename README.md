# Nimify Anything

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA-NIM-green.svg)](https://developer.nvidia.com/nim)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://hub.docker.com/r/nimify/anything)
[![CI/CD](https://img.shields.io/badge/CI-GitHub%20Actions-green.svg)](https://github.com/features/actions)

CLI that wraps any ONNX or TensorRT engine into an NVIDIA NIM microservice with auto-generated OpenAPI + Prometheus metrics. Turn any model into a production-ready API in seconds.

## 🚀 Overview

NVIDIA NIM (NVIDIA Inference Microservices) added open MoE models in July 2025, but developers still hand-roll deployment configs. Nimify automates the entire process:

- **One-command deployment** from model file to production API
- **Auto-generated OpenAPI** with type-safe clients
- **Built-in monitoring** via Prometheus/Grafana
- **Smart autoscaling** based on latency and GPU utilization
- **Helm charts** for Kubernetes deployment

## ⚡ Quick Demo

```bash
# Transform any ONNX model into a NIM service
nimify create my-model.onnx --name my-service --port 8080

# Deploy to Kubernetes with autoscaling
nimify deploy my-service --replicas 3 --autoscale

# Access your API
curl http://localhost:8080/v1/predict -d '{"input": [1, 2, 3]}'
```

<p align="center">
  <img src="docs/images/nimify_demo.gif" width="800" alt="Nimify Demo">
</p


## 📋 Requirements

```bash
# Core dependencies
python>=3.10
onnx>=1.16.0
onnxruntime-gpu>=1.18.0
tensorrt>=10.0.0
tritonclient>=2.45.0
nvidia-pyindex

# API & Infrastructure
fastapi>=0.110.0
uvicorn>=0.30.0
pydantic>=2.0.0
prometheus-client>=0.20.0

# Deployment tools
docker>=24.0.0
kubernetes>=29.0.0
helm>=3.14.0
```

## 🛠️ Installation

### From PyPI

```bash
pip install nimify-anything
```

### From Source

```bash
git clone https://github.com/yourusername/nimify-anything.git
cd nimify-anything
pip install -e .
```

### Verify Installation

```bash
# Check version and dependencies
nimify --version
nimify doctor
```

## 🚦 Usage Examples

### Basic Model Wrapping

```bash
# ONNX model
nimify create model.onnx --name my-classifier

# TensorRT engine
nimify create model.trt --name my-detector --input-shapes "images:3,224,224"

# Hugging Face model
nimify create "facebook/bart-large-mnli" --source huggingface
```

### Advanced Configuration

```bash
# Create with custom settings
nimify create model.onnx \
    --name sentiment-analyzer \
    --port 8080 \
    --max-batch-size 32 \
    --dynamic-batching \
    --gpu-memory 4GB \
    --metrics-port 9090
```

### Python API

```python
from nimify import Nimifier, ModelConfig

# Configure model
config = ModelConfig(
    name="my-model",
    max_batch_size=64,
    dynamic_batching=True,
    preferred_batch_sizes=[8, 16, 32, 64],
    max_queue_delay_microseconds=100
)

# Create NIM service
nim = Nimifier(config)
service = nim.wrap_model(
    "model.onnx",
    input_schema={"input": "float32[?,3,224,224]"},
    output_schema={"predictions": "float32[?,1000]"}
)

# Generate artifacts
service.generate_openapi("openapi.json")
service.generate_helm_chart("./helm/my-model")
service.build_container("myregistry/my-model:latest")
```

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Model File    │────▶│   Nimifier   │────▶│  NIM Service    │
│ (ONNX/TRT/HF)   │     │   Engine     │     │  (Container)    │
└─────────────────┘     └──────────────┘     └─────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Model Analysis  │     │   Triton     │     │   Kubernetes    │
│                 │     │   Config     │     │   Deployment    │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

## 🎯 Features

### Auto-Generated OpenAPI

```yaml
# Generated openapi.yaml
openapi: 3.0.0
info:
  title: my-model NIM API
  version: 1.0.0
paths:
  /v1/predict:
    post:
      summary: Run inference
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                input:
                  type: array
                  items:
                    type: number
      responses:
        200:
          description: Successful prediction
          content:
            application/json:
              schema:
                type: object
                properties:
                  predictions:
                    type: array
```

### Prometheus Metrics

```python
# Automatically exposed metrics:
# - nim_request_duration_seconds
# - nim_request_count_total
# - nim_batch_size_histogram
# - nim_gpu_utilization_percent
# - nim_model_loading_time_seconds
# - nim_queue_size
```

### Smart Autoscaling

```yaml
# Generated HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: my-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-model
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: gpu
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: nim_request_duration_seconds_p99
      target:
        type: AverageValue
        averageValue: "100m"  # 100ms
```

## 🐳 Container Building

### Automatic Dockerfile Generation

```dockerfile
# Auto-generated Dockerfile
FROM nvcr.io/nvidia/tritonserver:24.06-py3

# Install NIM runtime
RUN pip install nvidia-nim-runtime

# Copy model and config
COPY model_repository/ /models/
COPY nim_config.pbtxt /models/my-model/config.pbtxt

# Expose ports
EXPOSE 8000 8001 8002

# Launch Triton with NIM
CMD ["tritonserver", "--model-repository=/models", "--nim-mode"]
```

### Build and Push

```bash
# Build optimized container
nimify build my-model --optimize --tag myregistry/my-model:v1

# Push to registry
nimify push myregistry/my-model:v1

# Or use GitHub Actions
nimify generate-ci --platform github
```

## ☸️ Kubernetes Deployment

### Helm Chart Generation

```bash
# Generate production-ready Helm chart
nimify helm create my-model --values prod-values.yaml

# Deploy to Kubernetes
helm install my-model ./my-model-chart \
  --namespace nim \
  --set image.tag=v1 \
  --set autoscaling.enabled=true
```

### Generated Resources

```yaml
# values.yaml
replicaCount: 3

image:
  repository: myregistry/my-model
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: LoadBalancer
  port: 80
  targetPort: 8000

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetGPUUtilizationPercentage: 80
  targetLatencyMilliseconds: 100

resources:
  limits:
    nvidia.com/gpu: 1
    memory: 16Gi
  requests:
    nvidia.com/gpu: 1
    memory: 8Gi

monitoring:
  prometheus:
    enabled: true
    port: 9090
  grafana:
    enabled: true
    dashboards:
      - nim-overview
      - gpu-metrics
```

## 📊 Monitoring Dashboard

### Grafana Integration

```bash
# Deploy Grafana dashboard
nimify grafana deploy --model my-model

# Access dashboard
kubectl port-forward svc/grafana 3000:3000
```

### Pre-built Dashboards
- Request latency P50/P95/P99
- Throughput (requests/sec)
- GPU utilization and memory
- Batch size distribution
- Queue depth and wait times
- Error rates and types

## 🔧 Advanced Features

### Multi-Model Serving

```bash
# Create ensemble service
nimify ensemble create \
  --name multi-stage-pipeline \
  --models preprocessor:preprocess.onnx \
           detector:yolov8.trt \
           classifier:resnet50.onnx \
  --pipeline sequential
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
    traffic_split={"control": 0.8, "treatment": 0.2},
    metrics=["latency", "accuracy"]
)

nimify.create_ab_test("my-experiment", ab_config)
```

### Custom Preprocessors

```python
from nimify import Preprocessor

@Preprocessor.register("image_normalize")
def normalize_image(input_data):
    """Custom preprocessing logic"""
    return (input_data - 127.5) / 127.5

# Use in configuration
nimify create model.onnx \
  --preprocessor image_normalize \
  --postprocessor argmax
```

## 🧪 Testing & Validation

### Load Testing

```bash
# Run built-in load test
nimify loadtest my-model \
  --concurrent-users 100 \
  --duration 5m \
  --rps 1000

# Generate report
nimify loadtest report --output performance.html
```

### Model Validation

```python
from nimify import ModelValidator

validator = ModelValidator()

# Validate model serving
results = validator.validate(
    model_path="model.onnx",
    test_data="test_samples.json",
    checks=["output_shape", "latency", "throughput"]
)

assert results.passed, f"Validation failed: {results.errors}"
```

## 🔄 CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/nimify.yml
name: Build and Deploy NIM

on:
  push:
    branches: [main]

jobs:
  nimify:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Install Nimify
      run: pip install nimify-anything
    
    - name: Build NIM service
      run: |
        nimify create model.onnx --name my-model
        nimify build my-model --tag ${{ github.sha }}
    
    - name: Deploy to Kubernetes
      run: |
        nimify deploy my-model \
          --image my-model:${{ github.sha }} \
          --wait
```

### GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - build
  - deploy

build-nim:
  stage: build
  script:
    - nimify create $MODEL_PATH --name $SERVICE_NAME
    - nimify build $SERVICE_NAME --tag $CI_COMMIT_SHA
    - nimify push $REGISTRY/$SERVICE_NAME:$CI_COMMIT_SHA

deploy-nim:
  stage: deploy
  script:
    - nimify deploy $SERVICE_NAME --image $REGISTRY/$SERVICE_NAME:$CI_COMMIT_SHA
```

## 🎯 Real-World Examples

### Computer Vision Pipeline

```bash
# Object detection service
nimify create yolov8.onnx \
  --name object-detector \
  --input-type image \
  --output-type bounding-boxes \
  --preprocessing resize,normalize \
  --postprocessing nms

# Deploy with GPU optimization
nimify deploy object-detector \
  --gpu-memory 8GB \
  --tensorrt-optimization aggressive
```

### NLP Service

```bash
# Text classification
nimify create bert-sentiment.onnx \
  --name sentiment-analyzer \
  --input-type text \
  --tokenizer bert-base-uncased \
  --max-sequence-length 512
```

### Time Series Prediction

```bash
# Financial forecasting
nimify create lstm-forecast.onnx \
  --name stock-predictor \
  --input-shape "sequence:30,features:5" \
  --output-shape "predictions:5" \
  --streaming-mode true
```

## 🤝 Contributing

We welcome contributions! Priority areas:
- Additional model format support
- Custom metric collectors
- Cloud provider integrations
- Performance optimizations
- Documentation improvements

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 Citation

```bibtex
@software{nimify_anything,
  title={Nimify Anything: Automated NVIDIA NIM Service Generation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/nimify-anything}
}
```

## 🏆 Acknowledgments

- NVIDIA for the NIM framework
- The Triton Inference Server team
- Contributors to the Kubernetes ecosystem

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

## 🔗 Resources

- [Documentation](https://nimify.readthedocs.io)
- [Example Repository](https://github.com/nimify/examples)
- [Video Tutorials](https://youtube.com/@nimify)
- [Discord Community](https://discord.gg/nimify)

## 📧 Contact

- **GitHub Issues**: Bug reports and features
- **Email**: nimify@yourdomain.com
- **Twitter**: [@NimifyAnything](https://twitter.com/nimifyanything)# Nimify Anything

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA-NIM-green.svg)](https://developer.nvidia.com/nim)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://hub.docker.com/r/nimify/anything)
[![CI/CD](https://img.shields.io/badge/CI-GitHub%20Actions-green.svg)](https://github.com/features/actions)

CLI that wraps any ONNX or TensorRT engine into an NVIDIA NIM microservice with auto-generated OpenAPI + Prometheus metrics. Turn any model into a production-ready API in seconds.

## 🚀 Overview

NVIDIA NIM (NVIDIA Inference Microservices) added open MoE models in July 2025, but developers still hand-roll deployment configs. Nimify automates the entire process:

- **One-command deployment** from model file to production API
- **Auto-generated OpenAPI** with type-safe clients
- **Built-in monitoring** via Prometheus/Grafana
- **Smart autoscaling** based on latency and GPU utilization
- **Helm charts** for Kubernetes deployment

## ⚡ Quick Demo

```bash
# Transform any ONNX model into a NIM service
nimify create my-model.onnx --name my-service --port 8080

# Deploy to Kubernetes with autoscaling
nimify deploy my-service --replicas 3 --autoscale

# Access your API
curl http://localhost:8080/v1/predict -d '{"input": [1, 2, 3]}'
```

<p align="center">
  <img src="docs/images/nimify_demo.gif" width="800" alt="Nimify Demo">
</p>
