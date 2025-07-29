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
    
    def generate_openapi(self, output_path: str) -> None:
        """Generate OpenAPI specification."""
        # Implementation placeholder
        pass
    
    def generate_helm_chart(self, output_dir: str) -> None:
        """Generate Helm chart for Kubernetes deployment."""
        # Implementation placeholder
        pass
    
    def build_container(self, image_tag: str) -> None:
        """Build container image."""
        # Implementation placeholder
        pass