"""Nimify Anything: CLI for wrapping models into NVIDIA NIM services."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

from .core import Nimifier, ModelConfig

__all__ = ["Nimifier", "ModelConfig", "__version__"]