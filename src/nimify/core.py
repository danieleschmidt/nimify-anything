"""Core classes for NVIDIA NIM microservice creation."""

import json
import os
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import uuid


@dataclass
class ModelConfig:
    """Enhanced configuration for model service creation."""
    
    name: str
    max_batch_size: int = 32
    dynamic_batching: bool = True
    preferred_batch_sizes: Optional[List[int]] = None
    max_queue_delay_microseconds: int = 100
    gpu_memory: str = "auto"
    
    # Enhanced configuration options
    model_format: str = "auto"
    optimization_level: str = "standard"  # minimal, standard, aggressive
    enable_metrics: bool = True
    enable_health_checks: bool = True
    request_timeout: int = 30
    concurrent_requests: int = 100
    
    # Auto-scaling configuration
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    target_gpu_utilization: int = 80
    
    # Security configuration
    enable_auth: bool = False
    enable_tls: bool = False
    rate_limit_requests: int = 1000
    
    # Metadata
    version: str = "1.0.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.preferred_batch_sizes is None:
            self.preferred_batch_sizes = [1, 4, 8, 16, self.max_batch_size]
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if not self.description:
            self.description = f"NVIDIA NIM service for {self.name}"


class Nimifier:
    """Enhanced main class for converting models to NIM services."""
    
    def __init__(self, config: ModelConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or self._setup_logger()
        self.session_id = str(uuid.uuid4())[:8]
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for nimifier operations."""
        logger = logging.getLogger(f"nimify.{self.config.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def wrap_model(
        self,
        model_path: str,
        input_schema: Dict[str, str],
        output_schema: Dict[str, str],
        preprocessing_config: Optional[Dict[str, Any]] = None,
        postprocessing_config: Optional[Dict[str, Any]] = None
    ) -> 'NIMService':
        """Wrap a model file into a NIM service with enhanced validation."""
        start_time = time.time()
        self.logger.info(f"Starting model wrapping for {model_path}")
        
        # Validate model path
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Detect model format if auto
        if self.config.model_format == "auto":
            detected_format = self._detect_model_format(model_path_obj)
            self.logger.info(f"Detected model format: {detected_format}")
        
        # Create NIM service
        service = NIMService(
            config=self.config,
            model_path=model_path,
            input_schema=input_schema,
            output_schema=output_schema,
            preprocessing_config=preprocessing_config or {},
            postprocessing_config=postprocessing_config or {},
            logger=self.logger
        )
        
        duration = time.time() - start_time
        self.logger.info(f"Model wrapping completed in {duration:.2f}s")
        
        return service
    
    def _detect_model_format(self, model_path: Path) -> str:
        """Auto-detect model format from file extension and content."""
        ext = model_path.suffix.lower()
        
        format_map = {
            '.onnx': 'onnx',
            '.trt': 'tensorrt',
            '.engine': 'tensorrt',
            '.plan': 'tensorrt',
            '.pb': 'tensorflow',
            '.savedmodel': 'tensorflow',
            '.pth': 'pytorch',
            '.pt': 'pytorch',
            '.torchscript': 'pytorch'
        }
        
        return format_map.get(ext, 'unknown')


class NIMService:
    """Enhanced NIM service instance with comprehensive features."""
    
    def __init__(
        self,
        config: ModelConfig,
        model_path: str,
        input_schema: Dict[str, str],
        output_schema: Dict[str, str],
        preprocessing_config: Optional[Dict[str, Any]] = None,
        postprocessing_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.model_path = model_path
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.preprocessing_config = preprocessing_config or {}
        self.postprocessing_config = postprocessing_config or {}
        self.logger = logger or logging.getLogger(f"nimify.service.{config.name}")
        
        # Service metadata
        self.service_id = str(uuid.uuid4())
        self.created_at = datetime.utcnow()
        self.status = "initialized"
        
        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
    
    def _schema_to_openapi(self, schema: Dict[str, str]) -> Dict[str, Any]:
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
        """Generate comprehensive OpenAPI specification."""
        from pathlib import Path
        
        self.logger.info(f"Generating OpenAPI specification: {output_path}")
        
        # Generate enhanced OpenAPI spec
        openapi_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": f"{self.config.name} NIM API",
                "version": self.config.version,
                "description": self.config.description,
                "contact": {
                    "name": "Nimify Support",
                    "url": "https://github.com/nimify/nimify-anything"
                },
                "license": {
                    "name": "MIT",
                    "url": "https://opensource.org/licenses/MIT"
                }
            },
            "servers": [
                {
                    "url": "http://localhost:8000",
                    "description": "Development server"
                },
                {
                    "url": "https://api.example.com",
                    "description": "Production server"
                }
            ],
            "paths": {
                "/v1/predict": {
                    "post": {
                        "summary": "Run model inference",
                        "description": "Execute prediction using the wrapped model",
                        "tags": ["Inference"],
                        "requestBody": {
                            "required": True,
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
                            },
                            "400": {
                                "description": "Invalid input data",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "error": {"type": "string"},
                                                "details": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            },
                            "500": {
                                "description": "Internal server error",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "error": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/health": {
                    "get": {
                        "summary": "Health check endpoint",
                        "description": "Check service health and readiness",
                        "tags": ["Health"],
                        "responses": {
                            "200": {
                                "description": "Service is healthy",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "status": {"type": "string", "enum": ["healthy"]},
                                                "timestamp": {"type": "string", "format": "date-time"},
                                                "version": {"type": "string"},
                                                "uptime": {"type": "number"}
                                            }
                                        }
                                    }
                                }
                            },
                            "503": {
                                "description": "Service unavailable"
                            }
                        }
                    }
                },
                "/metrics": {
                    "get": {
                        "summary": "Prometheus metrics",
                        "description": "Get service metrics in Prometheus format",
                        "tags": ["Monitoring"],
                        "responses": {
                            "200": {
                                "description": "Metrics data",
                                "content": {
                                    "text/plain": {
                                        "schema": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                },
                "/ready": {
                    "get": {
                        "summary": "Readiness probe",
                        "description": "Check if service is ready to accept requests",
                        "tags": ["Health"],
                        "responses": {
                            "200": {
                                "description": "Service is ready"
                            },
                            "503": {
                                "description": "Service not ready"
                            }
                        }
                    }
                }
            }
        }
        
        # Add components section for reusable schemas
        openapi_spec["components"] = {
            "schemas": {
                "ErrorResponse": {
                    "type": "object",
                    "properties": {
                        "error": {"type": "string"},
                        "details": {"type": "string"},
                        "timestamp": {"type": "string", "format": "date-time"}
                    }
                },
                "HealthResponse": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "timestamp": {"type": "string", "format": "date-time"},
                        "version": {"type": "string"},
                        "uptime": {"type": "number"}
                    }
                }
            }
        }
        
        # Write to file with proper formatting
        try:
            Path(output_path).write_text(json.dumps(openapi_spec, indent=2, ensure_ascii=False))
            self.logger.info(f"OpenAPI specification generated successfully: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to generate OpenAPI spec: {e}")
            raise
    
    def generate_helm_chart(self, output_dir: str) -> None:
        """Generate enhanced Helm chart for Kubernetes deployment."""
        from pathlib import Path
        
        chart_dir = Path(output_dir)
        chart_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Generating Helm chart in: {output_dir}")
        
        # Enhanced Chart.yaml with metadata
        chart_yaml = f"""apiVersion: v2
name: {self.config.name}
description: {self.config.description}
type: application
version: {self.config.version}
appVersion: "{self.config.version}"

# Chart metadata
home: https://github.com/nimify/nimify-anything
sources:
  - https://github.com/nimify/nimify-anything
maintainers:
  - name: Nimify Team
    email: support@nimify.com
    url: https://nimify.com

# Keywords for discoverability
keywords:
  - nvidia
  - nim
  - ai
  - ml
  - inference
  - microservice

# Annotations
annotations:
  category: AI/ML
  licenses: MIT
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
    
    def create_model_ensemble(self, models: dict[str, 'NIMService'], ensemble_strategy: str = "sequential") -> 'ModelEnsemble':
        """Create ensemble from multiple NIM services for advanced use cases."""
        from .ensemble import ModelEnsemble, EnsembleConfig
        
        ensemble_config = EnsembleConfig(
            name=f"{self.config.name}_ensemble",
            strategy=ensemble_strategy,
            models=models,
            routing_rules=self._generate_routing_rules(models, ensemble_strategy)
        )
        
        return ModelEnsemble(ensemble_config)
    
    def _generate_routing_rules(self, models: dict[str, 'NIMService'], strategy: str) -> dict:
        """Generate intelligent routing rules for ensemble."""
        if strategy == "sequential":
            return {"type": "sequential", "order": list(models.keys())}
        elif strategy == "parallel":
            return {"type": "parallel", "aggregation": "weighted_average"}
        elif strategy == "conditional":
            return {"type": "conditional", "rules": self._create_conditional_rules(models)}
        else:
            return {"type": "load_balance", "weights": {name: 1.0 for name in models.keys()}}
    
    def _create_conditional_rules(self, models: dict[str, 'NIMService']) -> list:
        """Create conditional routing rules based on input characteristics."""
        rules = []
        model_names = list(models.keys())
        
        for i, model_name in enumerate(model_names):
            condition = f"input_size < {(i + 1) * 100}" if i < len(model_names) - 1 else "default"
            rules.append({"condition": condition, "target": model_name})
        
        return rules