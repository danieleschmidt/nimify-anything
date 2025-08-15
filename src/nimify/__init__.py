"""Nimify Anything: CLI that wraps any ONNX or TensorRT engine into an NVIDIA NIM microservice."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

from .core import Nimifier, ModelConfig, NIMService

__all__ = [
    "Nimifier", 
    "ModelConfig", 
    "NIMService",
    "__version__"
]