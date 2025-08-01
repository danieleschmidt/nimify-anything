"""Core classes for Nimify functionality."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class ModelConfig:
    """Configuration for a NIM service."""
    
    name: str
    max_batch_size: int = 32
    dynamic_batching: bool = True
    preferred_batch_sizes: Optional[List[int]] = None
    max_queue_delay_microseconds: int = 100
    
    def __post_init__(self):
        if self.preferred_batch_sizes is None:
            self.preferred_batch_sizes = [1, 4, 8, 16, 32]


class Nimifier:
    """Main class for creating NIM services from models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def wrap_model(
        self, 
        model_path: str, 
        input_schema: Dict[str, str], 
        output_schema: Dict[str, str]
    ) -> "NIMService":
        """Wrap a model into a NIM service."""
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
        input_schema: Dict[str, str], 
        output_schema: Dict[str, str]
    ):
        self.config = config
        self.model_path = model_path
        self.input_schema = input_schema
        self.output_schema = output_schema
    
    def _schema_to_openapi(self, schema: Dict[str, str]) -> Dict[str, Any]:
        """Convert internal schema to OpenAPI schema format."""
        properties = {}
        for name, type_info in schema.items():
            if "float32" in type_info:
                properties[name] = {
                    "type": "array",
                    "items": {"type": "number", "format": "float"}
                }
            elif "int" in type_info:
                properties[name] = {
                    "type": "array", 
                    "items": {"type": "integer"}
                }
            else:
                properties[name] = {"type": "array"}
        
        return {
            "type": "object",
            "properties": properties,
            "required": list(schema.keys())
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
    
    def build_container(self, image_tag: str) -> None:
        """Build container image."""
        import subprocess
        from pathlib import Path
        
        # Generate Dockerfile
        dockerfile_content = f"""FROM nvcr.io/nvidia/tritonserver:24.06-py3

# Install Python dependencies
RUN pip install fastapi uvicorn prometheus-client

# Copy model and application code
COPY {self.model_path} /models/{self.config.name}/
COPY src/ /app/src/

# Set working directory
WORKDIR /app

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["uvicorn", "src.nimify.api:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        dockerfile_path = Path("Dockerfile.nim")
        dockerfile_path.write_text(dockerfile_content)
        
        # Build Docker image
        try:
            result = subprocess.run([
                "docker", "build", "-f", str(dockerfile_path), 
                "-t", image_tag, "."
            ], check=True, capture_output=True, text=True)
            print(f"Successfully built container: {image_tag}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to build container: {e.stderr}")
        finally:
            # Clean up temporary Dockerfile
            if dockerfile_path.exists():
                dockerfile_path.unlink()