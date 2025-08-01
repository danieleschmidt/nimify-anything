# Container Build Guide

This guide covers building and deploying Nimify Anything containers for different environments.

## Quick Start

```bash
# Build development container
./scripts/build-container.sh --dev --tag dev-latest

# Build production container
./scripts/build-container.sh --prod --tag v1.0.0 --push

# Multi-platform build
./scripts/build-container.sh --platforms linux/amd64,linux/arm64 --push
```

## Container Variants

### Production Container (`Dockerfile.production`)

Optimized for production deployments:
- Multi-stage build for minimal size
- Non-root user for security
- Health checks and graceful shutdown
- Comprehensive logging and metrics
- Security hardening

```bash
# Build production image
docker build -f Dockerfile.production --target production -t nimify:prod .

# Run production container
docker run -d \
  --name nimify-prod \
  -p 8000:8000 \
  -p 9090:9090 \
  -v /path/to/models:/models:ro \
  -e NIMIFY_ENV=production \
  -e NGC_API_KEY=$NGC_API_KEY \
  nimify:prod
```

### Development Container (`Dockerfile`)

Includes development tools and debugging capabilities:
- Development dependencies included
- Source code mounted for live reload
- Debugger support (port 5678)
- Extended logging and diagnostics

```bash
# Build development image
docker build -f Dockerfile.production --target development -t nimify:dev .

# Run with debugging
docker run -it \
  --name nimify-dev \
  -p 8000:8000 \
  -p 5678:5678 \
  -v $(pwd)/src:/app/src:rw \
  nimify:dev
```

## Build Script Features

The `scripts/build-container.sh` script provides:

### Multi-Platform Support
```bash
# Build for multiple architectures
./scripts/build-container.sh \
  --platforms linux/amd64,linux/arm64 \
  --tag multi-arch-latest \
  --push
```

### Build Cache Optimization
```bash
# Use registry cache
./scripts/build-container.sh \
  --cache \
  --registry myregistry.com/nimify \
  --tag cached-build

# Disable cache for clean build
./scripts/build-container.sh --no-cache --tag clean-build
```

### Security Scanning
Automatic security scanning with Trivy (if installed):
```bash
# Install Trivy
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -

# Build with security scan
./scripts/build-container.sh --tag secure-build
```

### SBOM Generation
Software Bill of Materials with Syft (if installed):
```bash
# Install Syft
curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -

# Build with SBOM
./scripts/build-container.sh --tag sbom-build
```

## Build Arguments

### Standard Build Args
- `BUILD_DATE`: Build timestamp (automatic)
- `VCS_REF`: Git commit hash (automatic)
- `VERSION`: Version tag (from git)
- `PYTHON_VERSION`: Python version (default: 3.10)

### Custom Build Args
```bash
# Custom Python version
./scripts/build-container.sh \
  --tag python311 \
  --build-arg PYTHON_VERSION=3.11

# Development build with extra packages
./scripts/build-container.sh \
  --dev \
  --build-arg EXTRA_PACKAGES="vim htop"
```

## Registry Configuration

### Local Registry
```bash
# Start local registry
docker run -d -p 5000:5000 --name registry registry:2

# Build and push to local registry
./scripts/build-container.sh \
  --registry localhost:5000 \
  --tag local-build \
  --push
```

### Docker Hub
```bash
# Login to Docker Hub
docker login

# Build and push
./scripts/build-container.sh \
  --registry docker.io/yourusername \
  --tag v1.0.0 \
  --push
```

### Private Registry
```bash
# Login to private registry
docker login myregistry.com

# Build and push
./scripts/build-container.sh \
  --registry myregistry.com/nimify \
  --tag enterprise-v1.0.0 \
  --push
```

## Docker Compose Usage

### Development Stack
```bash
# Start development environment
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Services included:
# - nimify (with debugger)
# - postgres-dev
# - local-registry
# - minio (S3-compatible storage)
# - jaeger (tracing)
```

### Production Stack
```bash
# Start production services
docker-compose --profile inference --profile monitoring up -d

# Services included:
# - triton (NVIDIA Triton server)
# - prometheus
# - grafana
# - redis (optional caching)
```

### Custom Profiles
```bash
# Development with database
docker-compose --profile dev --profile db up -d

# Full monitoring stack
docker-compose --profile monitoring --profile tracing up -d

# Storage testing
docker-compose --profile dev --profile storage up -d
```

## Container Configuration

### Environment Variables
```bash
# Core configuration
NIMIFY_ENV=production              # Environment mode
NIMIFY_LOG_LEVEL=INFO             # Logging level
NIMIFY_MODEL_CACHE=/cache/models  # Model storage path

# NIM configuration
NGC_API_KEY=your_api_key          # NVIDIA NGC API key
NIM_REGISTRY=nvcr.io/nim          # NIM container registry

# Kubernetes configuration
KUBECONFIG_CONTEXT=prod-cluster   # Kubernetes context
NIMIFY_NAMESPACE=nimify-system    # Default namespace

# Monitoring
ENABLE_METRICS=true               # Enable metrics endpoint
NIMIFY_METRICS_PORT=9090          # Metrics port
OTEL_EXPORTER_OTLP_ENDPOINT=...   # OpenTelemetry endpoint
```

### Volume Mounts
```bash
# Model storage
-v /data/models:/models:ro

# Configuration
-v /etc/nimify:/app/config:ro

# Cache (persistent)
-v nimify-cache:/cache

# Logs
-v /var/log/nimify:/app/logs
```

### Network Configuration
```bash
# Standard ports
-p 8000:8000    # HTTP API
-p 9090:9090    # Metrics
-p 5678:5678    # Debugger (dev only)

# Custom network
docker network create nimify-net
docker run --network nimify-net ...
```

## Troubleshooting

### Build Issues

#### Out of Memory
```bash
# Increase Docker memory limit
# Or build with smaller batch sizes
docker build --memory=4g ...
```

#### Permission Errors
```bash
# Check file ownership
ls -la Dockerfile

# Fix permissions
chmod 644 Dockerfile
chmod +x scripts/*.sh
```

#### Network Issues
```bash
# Use different DNS
docker build --dns=8.8.8.8 ...

# Use build context from URL
docker build https://github.com/user/repo.git#main
```

### Runtime Issues

#### Container Won't Start
```bash
# Check logs
docker logs nimify-container

# Run interactively
docker run -it --entrypoint=/bin/bash nimify:latest

# Check health
docker exec nimify-container nimify doctor
```

#### GPU Not Available
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.2-runtime-ubuntu22.04 nvidia-smi

# Verify GPU in container
docker run --rm --gpus all nimify:latest nvidia-smi
```

#### Memory Issues
```bash
# Monitor memory usage
docker stats nimify-container

# Limit memory
docker run --memory=2g nimify:latest

# Check for memory leaks
docker exec nimify-container cat /proc/meminfo
```

## Best Practices

### Security
1. **Non-root user**: Always run as non-root in production
2. **Minimal base image**: Use distroless or alpine variants
3. **Security scanning**: Scan images for vulnerabilities
4. **Secrets management**: Never embed secrets in images
5. **Network policies**: Use least-privilege networking

### Performance
1. **Multi-stage builds**: Minimize final image size
2. **Layer caching**: Optimize Dockerfile layer order
3. **Build cache**: Use registry cache for CI/CD
4. **Resource limits**: Set appropriate CPU/memory limits
5. **Health checks**: Implement proper health endpoints

### Monitoring
1. **Structured logging**: Use JSON logging in production
2. **Metrics collection**: Export Prometheus metrics
3. **Distributed tracing**: Implement OpenTelemetry
4. **Error tracking**: Integrate with error reporting
5. **Performance monitoring**: Track key performance metrics

### CI/CD Integration
1. **Automated builds**: Trigger on code changes
2. **Testing**: Run tests in containers
3. **Security scanning**: Automated vulnerability scanning
4. **SBOM generation**: Generate software bill of materials
5. **Registry management**: Clean up old images regularly