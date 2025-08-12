"""Core classes for Bioneuro-Olfactory Fusion system."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum


class NeuralSignalType(Enum):
    """Types of neural signals that can be processed."""
    EEG = "electroencephalogram"
    fMRI = "functional_magnetic_resonance"
    MEG = "magnetoencephalography"
    EPHYS = "electrophysiology"
    CALCIUM = "calcium_imaging"


class OlfactoryMoleculeType(Enum):
    """Categories of olfactory molecules."""
    ALDEHYDE = "aldehyde"
    ESTER = "ester"
    KETONE = "ketone"
    ALCOHOL = "alcohol"
    TERPENE = "terpene"
    AROMATIC = "aromatic"


@dataclass
class NeuralConfig:
    """Configuration for neural signal processing."""
    
    signal_type: NeuralSignalType
    sampling_rate: int = 1000  # Hz
    channels: int = 64
    time_window: float = 2.0  # seconds
    preprocessing_filters: List[str] = None
    artifact_removal: bool = True
    
    def __post_init__(self):
        if self.preprocessing_filters is None:
            self.preprocessing_filters = ["bandpass", "notch", "baseline"]


@dataclass
class OlfactoryConfig:
    """Configuration for olfactory stimulus analysis."""
    
    molecule_types: List[OlfactoryMoleculeType]
    concentration_range: Tuple[float, float] = (0.001, 10.0)  # ppm
    molecular_descriptors: List[str] = None
    stimulus_duration: float = 3.0  # seconds
    inter_stimulus_interval: float = 10.0  # seconds
    
    def __post_init__(self):
        if self.molecular_descriptors is None:
            self.molecular_descriptors = [
                "molecular_weight", "vapor_pressure", "polarity",
                "functional_groups", "carbon_chain_length"
            ]


class BioneuroFusion:
    """Main class for bioneuro-olfactory fusion analysis."""
    
    def __init__(self, neural_config: NeuralConfig, olfactory_config: OlfactoryConfig):
        self.neural_config = neural_config
        self.olfactory_config = olfactory_config
        self.fusion_models = {}
    
    def process_neural_data(
        self, 
        neural_data: np.ndarray,
        timestamps: np.ndarray = None
    ) -> Dict[str, Any]:
        """Process neural signal data."""
        from .neural_processor import NeuralSignalProcessor
        
        processor = NeuralSignalProcessor(self.neural_config)
        return processor.process(neural_data, timestamps)
    
    def analyze_olfactory_stimulus(
        self,
        molecule_data: Dict[str, Any],
        concentration: float
    ) -> Dict[str, Any]:
        """Analyze olfactory stimulus properties."""
        from .olfactory_analyzer import OlfactoryAnalyzer
        
        analyzer = OlfactoryAnalyzer(self.olfactory_config)
        return analyzer.analyze(molecule_data, concentration)
    
    def fuse_modalities(
        self,
        neural_features: Dict[str, Any],
        olfactory_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fuse neural and olfactory data modalities."""
        from .fusion_engine import MultiModalFusionEngine
        
        fusion_engine = MultiModalFusionEngine()
        return fusion_engine.fuse(neural_features, olfactory_features)


class BioneuroService:
    """Represents a bioneuro-olfactory fusion service instance."""
    
    def __init__(
        self, 
        neural_config: NeuralConfig,
        olfactory_config: OlfactoryConfig,
        fusion_model_path: Optional[str] = None
    ):
        self.neural_config = neural_config
        self.olfactory_config = olfactory_config
        self.fusion_model_path = fusion_model_path
        self.fusion_engine = BioneuroFusion(neural_config, olfactory_config)
    
    def _generate_bioneuro_schema(self) -> Dict[str, Any]:
        """Generate OpenAPI schema for bioneuro-olfactory fusion."""
        neural_schema = {
            "neural_data": {
                "type": "array",
                "items": {"type": "number"},
                "description": f"{self.neural_config.signal_type.value} signal data"
            },
            "timestamps": {
                "type": "array", 
                "items": {"type": "number"},
                "description": "Temporal timestamps for neural data"
            }
        }
        
        olfactory_schema = {
            "molecule_data": {
                "type": "object",
                "description": "Chemical properties of olfactory stimulus"
            },
            "concentration": {
                "type": "number",
                "minimum": self.olfactory_config.concentration_range[0],
                "maximum": self.olfactory_config.concentration_range[1],
                "description": "Stimulus concentration in ppm"
            }
        }
        
        return {
            "type": "object",
            "properties": {**neural_schema, **olfactory_schema},
            "required": ["neural_data", "molecule_data", "concentration"]
        }
    
    def generate_openapi(self, output_path: str) -> None:
        """Generate OpenAPI specification."""
        from pathlib import Path
        import json
        
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
        import shutil
        
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
            result = subprocess.run(
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