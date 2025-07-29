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
