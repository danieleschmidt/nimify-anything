"""Bioneuro-Olfactory Fusion: AI system for neural-olfactory data analysis."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

from .core import BioneuroFusion, NeuralConfig, OlfactoryConfig
from .neural_processor import NeuralSignalProcessor
from .olfactory_analyzer import OlfactoryAnalyzer
from .fusion_engine import MultiModalFusionEngine

__all__ = [
    "BioneuroFusion", 
    "NeuralConfig", 
    "OlfactoryConfig",
    "NeuralSignalProcessor",
    "OlfactoryAnalyzer", 
    "MultiModalFusionEngine",
    "__version__"
]