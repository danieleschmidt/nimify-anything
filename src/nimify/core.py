"""Core classes for NVIDIA NIM microservice creation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for model service creation."""
    
    name: str
    max_batch_size: int = 32
    dynamic_batching: bool = True
    preferred_batch_sizes: list[int] = None
    max_queue_delay_microseconds: int = 100
    gpu_memory: str = "auto"
    
    def __post_init__(self):
        if self.preferred_batch_sizes is None:
            self.preferred_batch_sizes = [1, 4, 8, 16, self.max_batch_size]


class Nimifier:
    """Main class for converting models to NIM services."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def wrap_model(
        self,
        model_path: str,
        input_schema: dict[str, str],
        output_schema: dict[str, str]
    ) -> 'NIMService':
        """Wrap a model file into a NIM service."""
        return NIMService(
            config=self.config,
            model_path=model_path,
            input_schema=input_schema,
            output_schema=output_schema
        )


class NIMService:
    """Represents a NIM service instance."""
    
    def __init__(
        self,
        config: ModelConfig,
        model_path: str,
        input_schema: dict[str, str],
        output_schema: dict[str, str]
    ):
        self.config = config
        self.model_path = model_path
        self.input_schema = input_schema
        self.output_schema = output_schema
    
    def _schema_to_openapi(self, schema: dict[str, str]) -> dict[str, Any]:
        """Convert internal schema format to OpenAPI schema."""
        properties = {}
        
        for field_name, field_type in schema.items():
            if "float32" in field_type:
                properties[field_name] = {
                    "type": "array",
                    "items": {"type": "number", "format": "float"},
                    "description": f"Tensor with shape {field_type}"
                }
            elif "int" in field_type:
                properties[field_name] = {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": f"Tensor with shape {field_type}"
                }
            else:
                properties[field_name] = {
                    "type": "array",
                    "description": f"Tensor with shape {field_type}"
                }
        
        return {
            "type": "object",
            "properties": properties,
            "required": list(schema.keys())
        }
    
    def generate_openapi(self, output_path: str) -> None:
        """Generate OpenAPI specification."""
        from pathlib import Path
        
        # Generate OpenAPI spec based on input/output schemas
        openapi_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": f"{self.config.name} NIM API",
                "version": "1.0.0",
                "description": f"Auto-generated API for {self.config.name} model service"
            },
            "paths": {
                "/v1/predict": {
                    "post": {
                        "summary": "Run inference",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": self._schema_to_openapi(self.input_schema)
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Successful prediction",
                                "content": {
                                    "application/json": {
                                        "schema": self._schema_to_openapi(self.output_schema)
                                    }
                                }
                            }
                        }
                    }
                },
                "/health": {
                    "get": {
                        "summary": "Health check",
                        "responses": {
                            "200": {
                                "description": "Service is healthy"
                            }
                        }
                    }
                }
            }
        }
        
        # Write to file
        Path(output_path).write_text(json.dumps(openapi_spec, indent=2))
    
    def generate_helm_chart(self, output_dir: str) -> None:
        """Generate Helm chart for Kubernetes deployment."""
        from pathlib import Path
        
        chart_dir = Path(output_dir)
        chart_dir.mkdir(parents=True, exist_ok=True)
        
        # Chart.yaml
        chart_yaml = f"""apiVersion: v2
name: {self.config.name}
description: Helm chart for {self.config.name} NIM service
type: application
version: 1.0.0
appVersion: "1.0.0"
"""
        (chart_dir / "Chart.yaml").write_text(chart_yaml)
        
        # values.yaml
        values_yaml = f"""replicaCount: 2

image:
  repository: {self.config.name}
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
"""
        (chart_dir / "values.yaml").write_text(values_yaml)
        
        # Create templates directory
        templates_dir = chart_dir / "templates"
        templates_dir.mkdir(exist_ok=True)
        
        # deployment.yaml template
        deployment_yaml = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{{{ include "{self.config.name}.fullname" . }}}}
  labels:
    {{{{- include "{self.config.name}.labels" . | nindent 4 }}}}
spec:
  replicas: {{{{ .Values.replicaCount }}}}
  selector:
    matchLabels:
      {{{{- include "{self.config.name}.selectorLabels" . | nindent 6 }}}}
  template:
    metadata:
      labels:
        {{{{- include "{self.config.name}.selectorLabels" . | nindent 8 }}}}
    spec:
      containers:
      - name: {self.config.name}
        image: "{{{{ .Values.image.repository }}}}:{{{{ .Values.image.tag }}}}"
        ports:
        - containerPort: 8000
        resources:
          {{{{- toYaml .Values.resources | nindent 12 }}}}
"""
        (templates_dir / "deployment.yaml").write_text(deployment_yaml)
    
    def build_container(self, image_tag: str, optimize: bool = True) -> None:
        """Build optimized container image."""
        import subprocess
        from pathlib import Path
        
        # Generate optimized Dockerfile
        dockerfile_content = self._generate_optimized_dockerfile(optimize)
        
        dockerfile_path = Path("Dockerfile.nim")
        dockerfile_path.write_text(dockerfile_content)
        
        # Build Docker image with optimization flags
        build_args = [
            "docker", "build", "-f", str(dockerfile_path),
            "-t", image_tag
        ]
        
        if optimize:
            # Add BuildKit optimizations
            build_args.extend([
                "--build-arg", "BUILDKIT_INLINE_CACHE=1",
                "--cache-from", image_tag,
            ])
        
        build_args.append(".")
        
        # Build Docker image
        try:
            env = {"DOCKER_BUILDKIT": "1"} if optimize else {}
            subprocess.run(
                build_args,
                check=True, 
                capture_output=True, 
                text=True,
                env={**__import__('os').environ, **env}
            )
            print(f"Successfully built container: {image_tag}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to build container: {e.stderr}")
        finally:
            # Clean up temporary Dockerfile
            if dockerfile_path.exists():
                dockerfile_path.unlink()
    
    def _generate_optimized_dockerfile(self, optimize: bool = True) -> str:
        """Generate optimized Dockerfile with security hardening."""
        base_stage = """# Multi-stage build for optimization
FROM nvcr.io/nvidia/tritonserver:24.06-py3 as base

# Security: Create non-root user
RUN groupadd -r nimuser && useradd -r -g nimuser nimuser

# Install system dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    prometheus-client \
    pydantic \
    numpy \
    cachetools \
    psutil

# Copy requirements and install Python packages
COPY requirements.txt* /tmp/
RUN if [ -f /tmp/requirements.txt ]; then \
        pip install --no-cache-dir -r /tmp/requirements.txt; \
    fi

"""
        
        if optimize:
            build_stage = """
# Build stage for optimizations
FROM base as builder

# Copy application source
COPY src/ /app/src/

# Compile Python files for faster startup
RUN python -m compileall /app/src/

"""
            
            runtime_stage = f"""
# Runtime stage - minimal and secure
FROM nvcr.io/nvidia/tritonserver:24.06-py3-min as runtime

# Copy user from base
COPY --from=base /etc/passwd /etc/group /etc/

# Copy Python packages
COPY --from=base /usr/local/lib/python3.* /usr/local/lib/python3.*
COPY --from=base /usr/local/bin /usr/local/bin

# Copy application
COPY --from=builder /app/ /app/

# Copy model
COPY {self.model_path} /models/{self.config.name}/

"""
        else:
            runtime_stage = f"""
# Copy model and application code
COPY {self.model_path} /models/{self.config.name}/
COPY src/ /app/src/

"""
        
        final_stage = """
# Set working directory
WORKDIR /app

# Security hardening
RUN chown -R nimuser:nimuser /app /models

# Switch to non-root user
USER nimuser

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Add labels for metadata
LABEL maintainer="nimify" \
      version="1.0.0" \
      description="NVIDIA NIM microservice" \
      org.opencontainers.image.source="https://github.com/nimify/nimify-anything"

# Start command with proper signal handling
CMD ["python", "-m", "uvicorn", "src.nimify.api:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--access-log"]
"""
        
        if optimize:
            return base_stage + build_stage + runtime_stage + final_stage
        else:
            return base_stage + runtime_stage + final_stage
    
    def deploy_to_kubernetes(self, replicas: int = 3, namespace: str = "default") -> Path:
        """Deploy service to Kubernetes with advanced configuration."""
        from .deployment import DeploymentConfig, DeploymentOrchestrator
        
        # Create deployment configuration
        deployment_config = DeploymentConfig(
            service_name=self.config.name,
            image_name=self.config.name,
            image_tag="latest",
            replicas=replicas,
            namespace=namespace,
            
            # Resource optimization
            cpu_request="100m",
            cpu_limit="2000m", 
            memory_request="512Mi",
            memory_limit="4Gi",
            gpu_limit=1,
            
            # Performance settings
            min_replicas=2,
            max_replicas=20,
            target_cpu_utilization=70,
            
            # Security settings
            enable_network_policies=True,
            enable_pod_security_policy=True,
            
            # Environment variables
            environment_variables={
                "MODEL_PATH": f"/models/{self.config.name}",
                "LOG_LEVEL": "INFO",
                "METRICS_ENABLED": "true"
            }
        )
        
        # Generate deployment package
        orchestrator = DeploymentOrchestrator(deployment_config)
        output_dir = Path.cwd() / "deployments"
        
        return orchestrator.generate_deployment_package(output_dir)