# 🚀 Nimify Anything - Production Deployment Guide

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA-NIM-green.svg)](https://developer.nvidia.com/nim)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://hub.docker.com/r/nimify/anything)

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)  
- [Local Development](#local-development)
- [Container Deployment](#container-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Production Configuration](#production-configuration)
- [Monitoring & Observability](#monitoring--observability)
- [Security Considerations](#security-considerations)
- [Scaling & Performance](#scaling--performance)
- [Troubleshooting](#troubleshooting)

## ⚡ Quick Start

Deploy any ONNX model as a production-ready NIM service in 3 commands:

```bash
# 1. Create NIM service
nimify create my-model.onnx --name my-service

# 2. Build optimized container
nimify build my-service --optimize

# 3. Deploy to Kubernetes
helm install my-service ./my-service-chart/
```

## 🔧 Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with CUDA 12+ (for GPU acceleration)
- **Memory**: Minimum 8GB RAM (16GB+ recommended)
- **Storage**: 20GB+ free space

### Software Dependencies

```bash
# Core dependencies
docker >= 24.0.0
kubernetes >= 1.28.0
helm >= 3.14.0
kubectl >= 1.28.0

# Optional but recommended
nvidia-docker2        # For GPU support
prometheus           # For monitoring
grafana             # For dashboards
```

### NVIDIA Setup

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 🏠 Local Development

### Installation

```bash
# Clone repository
git clone https://github.com/nimify/nimify-anything.git
cd nimify-anything

# Install with development dependencies
pip install -e ".[dev]"

# Verify installation
nimify doctor
```

### Quick Test

```bash
# Download sample ONNX model
wget https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v1-7.onnx

# Create service
nimify create resnet50-v1-7.onnx --name resnet-classifier --port 8080

# Start development server (if implemented)
uvicorn src.nimify.api:app --host 0.0.0.0 --port 8080
```

## 🐳 Container Deployment

### Build Production Container

```bash
# Build optimized container
nimify create my-model.onnx --name my-service
nimify build my-service --optimize --tag my-registry/my-service:v1.0

# Security scan
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/trivy image my-registry/my-service:v1.0
```

### Docker Compose Deployment

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  nim-service:
    image: my-registry/my-service:v1.0
    ports:
      - "8000:8000"      # API
      - "9090:9090"      # Metrics
    environment:
      - LOG_LEVEL=INFO
      - METRICS_ENABLED=true
    volumes:
      - models:/models:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/dashboards:/var/lib/grafana/dashboards

volumes:
  models:
  prometheus-data:
  grafana-data:
```

### Run with Docker Compose

```bash
# Deploy full stack
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f nim-service
```

## ☸️ Kubernetes Deployment

### Prerequisites

```bash
# Verify cluster access
kubectl cluster-info

# Install NVIDIA GPU Operator (for GPU nodes)
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm install --wait --generate-name nvidia/gpu-operator
```

### Basic Deployment

```bash
# Generate Kubernetes manifests
nimify create my-model.onnx --name my-service
kubectl apply -f my-service-chart/templates/

# Or use Helm (recommended)
helm install my-service ./my-service-chart/ \
    --namespace nim-services \
    --create-namespace
```

### Production Helm Values

```yaml
# production-values.yaml
replicaCount: 3

image:
  repository: my-registry/my-service
  tag: v1.0.0
  pullPolicy: IfNotPresent

service:
  type: LoadBalancer
  port: 80
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "1000"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  hosts:
    - host: my-service.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: my-service-tls
      hosts:
        - my-service.example.com

resources:
  requests:
    cpu: 500m
    memory: 2Gi
    nvidia.com/gpu: 1
  limits:
    cpu: 2000m
    memory: 8Gi
    nvidia.com/gpu: 1

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector:
  accelerator: nvidia-tesla-gpu

tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule

affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchExpressions:
              - key: app
                operator: In
                values: [my-service]
          topologyKey: kubernetes.io/hostname

persistence:
  enabled: true
  size: 50Gi
  storageClass: fast-ssd

monitoring:
  prometheus:
    enabled: true
    serviceMonitor:
      enabled: true
      interval: 30s

security:
  networkPolicies:
    enabled: true
  podSecurityPolicy:
    enabled: true
  serviceAccount:
    create: true
    automountServiceAccountToken: false
```

### Deploy with Production Configuration

```bash
helm upgrade --install my-service ./my-service-chart/ \
    --namespace nim-services \
    --create-namespace \
    --values production-values.yaml \
    --timeout 10m \
    --wait
```

## ⚙️ Production Configuration

### Environment Variables

```bash
# Core configuration
MODEL_PATH=/models/model.onnx
SERVICE_NAME=my-service
LOG_LEVEL=INFO

# Performance tuning
MAX_BATCH_SIZE=32
INFERENCE_TIMEOUT=30
WORKER_THREADS=4
CACHE_SIZE=1000

# Security
API_KEY_REQUIRED=true
RATE_LIMIT_RPS=100
CORS_ORIGINS=https://my-app.com

# Monitoring
METRICS_ENABLED=true
TRACING_ENABLED=true
HEALTH_CHECK_INTERVAL=30
```

### Resource Limits

```yaml
# Kubernetes resource configuration
resources:
  requests:
    cpu: "500m"           # 0.5 CPU cores minimum
    memory: "2Gi"         # 2GB RAM minimum
    nvidia.com/gpu: "1"   # 1 GPU minimum
  limits:
    cpu: "4000m"          # 4 CPU cores maximum
    memory: "16Gi"        # 16GB RAM maximum
    nvidia.com/gpu: "1"   # 1 GPU maximum
```

### Storage Configuration

```yaml
# Persistent storage for models
persistence:
  enabled: true
  accessMode: ReadWriteOnce
  size: 100Gi
  storageClass: fast-ssd  # Use SSD storage class
  mountPath: /models
```

## 📊 Monitoring & Observability

### Prometheus Metrics

Nimify automatically exposes Prometheus metrics on `/metrics`:

```yaml
# Key metrics exposed
nim_request_count_total          # Total requests
nim_request_duration_seconds     # Request latency
nim_inference_duration_seconds   # Model inference time
nim_batch_size_histogram        # Batch size distribution
nim_gpu_utilization_percent     # GPU utilization
nim_error_count_total           # Error counts
nim_concurrent_requests         # Active requests
```

### Grafana Dashboard

Import the pre-built dashboard:

```bash
# Import dashboard from monitoring/dashboards/
kubectl create configmap grafana-dashboard \
    --from-file=monitoring/dashboards/nimify-overview.json
```

### Health Checks

Multiple health check endpoints:

```bash
# Basic health
curl http://service:8000/health

# Detailed status
curl http://service:8000/health | jq '.'
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "uptime_seconds": 3600.5
}
```

### Distributed Tracing

Enable distributed tracing with OpenTelemetry:

```yaml
# In values.yaml
monitoring:
  tracing:
    enabled: true
    jaeger:
      endpoint: http://jaeger:14268/api/traces
    sampling_rate: 0.1
```

## 🔒 Security Considerations

### Container Security

```dockerfile
# Multi-stage builds with security hardening
FROM nvidia/triton:24.06-py3 as base
RUN groupadd -r nimuser && useradd -r -g nimuser nimuser
USER nimuser

# Security scanning in CI/CD
RUN trivy image --security-checks vuln my-service:latest
```

### Network Security

```yaml
# Network policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: my-service-netpol
spec:
  podSelector:
    matchLabels:
      app: my-service
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8000
  egress:
    - ports:
        - protocol: TCP
          port: 443  # HTTPS only
```

### API Security

```yaml
# API security configuration
security:
  authentication:
    enabled: true
    method: api-key  # or jwt, oauth2
  rate_limiting:
    enabled: true
    requests_per_second: 100
    burst_size: 200
  input_validation:
    enabled: true
    max_payload_size: 10MB
    sanitization: strict
```

### Secrets Management

```bash
# Store sensitive data in Kubernetes secrets
kubectl create secret generic my-service-secrets \
    --from-literal=api-key=super-secret-key \
    --from-literal=db-password=secret-password

# Reference in deployment
env:
  - name: API_KEY
    valueFrom:
      secretKeyRef:
        name: my-service-secrets
        key: api-key
```

## 📈 Scaling & Performance

### Horizontal Pod Autoscaling

```yaml
# HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: my-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-service
  minReplicas: 3
  maxReplicas: 50
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    - type: Resource
      resource:
        name: nvidia.com/gpu
        target:
          type: Utilization
          averageUtilization: 80
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
          value: 100
          periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 50
          periodSeconds: 60
```

### Vertical Pod Autoscaling

```yaml
# VPA for automatic resource adjustment
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: my-service-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-service
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
      - containerName: my-service
        maxAllowed:
          cpu: "4"
          memory: "16Gi"
        minAllowed:
          cpu: "100m"
          memory: "1Gi"
```

### Performance Tuning

```bash
# Run performance benchmarks
python scripts/performance-benchmark.py \
    --endpoint http://my-service.example.com \
    --users 100 \
    --duration 300 \
    --stress

# Monitor key metrics
kubectl top nodes
kubectl top pods -n nim-services
```

### Cluster Autoscaling

```yaml
# Cluster autoscaler for node scaling
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-status
  namespace: kube-system
data:
  nodes.min: "3"
  nodes.max: "100"
  node-groups: |
    gpu-nodes:3:20
    cpu-nodes:3:50
```

## 🔧 Troubleshooting

### Common Issues

#### 1. GPU Not Available

```bash
# Check GPU resources
kubectl describe nodes | grep nvidia.com/gpu

# Verify GPU operator
kubectl get pods -n gpu-operator-resources

# Check container GPU access
kubectl exec -it my-service-pod -- nvidia-smi
```

#### 2. Out of Memory (OOM)

```bash
# Check memory usage
kubectl top pods -n nim-services
kubectl describe pod my-service-pod | grep -A 10 "Events:"

# Increase memory limits
helm upgrade my-service ./my-service-chart/ \
    --set resources.limits.memory=32Gi
```

#### 3. High Latency

```bash
# Check performance metrics
curl http://my-service:9090/metrics | grep nim_request_duration

# Enable performance profiling
kubectl port-forward svc/my-service 8000:8000
curl http://localhost:8000/health
```

#### 4. Load Balancer Issues

```bash
# Check service status
kubectl get svc my-service -o wide
kubectl describe svc my-service

# Verify ingress configuration
kubectl get ingress my-service -o yaml
```

### Debugging Commands

```bash
# View detailed pod information
kubectl describe pod my-service-pod

# Access pod logs
kubectl logs -f my-service-pod --previous

# Execute commands in pod
kubectl exec -it my-service-pod -- /bin/bash

# Port forwarding for local access
kubectl port-forward svc/my-service 8000:8000

# Check resource usage
kubectl top pods -n nim-services --sort-by=cpu
kubectl top nodes --sort-by=cpu
```

### Log Analysis

```bash
# Centralized logging with ELK/EFK stack
kubectl logs -l app=my-service --tail=1000 | grep ERROR

# Structured logging queries (if using structured logs)
kubectl logs -l app=my-service | jq '.level == "ERROR"'
```

## 🎯 Performance Targets

### SLA Targets

- **Availability**: 99.95% uptime
- **Latency**: P95 < 200ms, P99 < 500ms
- **Throughput**: 1000+ requests/second per replica
- **Error Rate**: < 0.1% of requests

### Capacity Planning

```yaml
# Sizing guidelines per service type
small_model:    # < 100MB
  cpu: "500m"
  memory: "2Gi"
  replicas: 3-10

medium_model:   # 100MB - 1GB
  cpu: "1000m"
  memory: "4Gi"
  replicas: 2-8

large_model:    # > 1GB
  cpu: "2000m"
  memory: "8Gi"
  replicas: 1-5
```

## 📞 Support

### Getting Help

- **Documentation**: [https://nimify.readthedocs.io](https://nimify.readthedocs.io)
- **GitHub Issues**: [https://github.com/nimify/issues](https://github.com/nimify/issues)
- **Discord**: [https://discord.gg/nimify](https://discord.gg/nimify)

### Enterprise Support

For production deployments requiring SLA guarantees:
- **Email**: enterprise@nimify.ai
- **24/7 Support**: Available with Enterprise Plan
- **Professional Services**: Architecture reviews, optimization, training

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- NVIDIA for the NIM framework and Triton Inference Server
- The Kubernetes and Helm communities
- All contributors and early adopters

---

**Ready to deploy? Start with our [Quick Start Guide](#quick-start) or jump to [Kubernetes Deployment](#kubernetes-deployment) for production setups.**