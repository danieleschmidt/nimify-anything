"""Model analysis utilities for automatic schema detection."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json

logger = logging.getLogger(__name__)


class ModelAnalyzer:
    """Analyzes model files to extract metadata and generate schemas."""
    
    @staticmethod
    def analyze_onnx_model(model_path: str) -> Dict:
        """Analyze ONNX model and extract schema information."""
        try:
            import onnx
            
            model = onnx.load(model_path)
            
            # Extract input information
            inputs = {}
            for input_tensor in model.graph.input:
                name = input_tensor.name
                shape = []
                for dim in input_tensor.type.tensor_type.shape.dim:
                    if dim.dim_value:
                        shape.append(dim.dim_value)
                    elif dim.dim_param:
                        shape.append(f"?")  # Dynamic dimension
                    else:
                        shape.append("?")
                
                # Get data type
                dtype_map = {
                    1: "float32",
                    2: "uint8", 
                    3: "int8",
                    6: "int32",
                    7: "int64",
                    9: "bool",
                    10: "float16",
                    11: "double"
                }
                dtype = dtype_map.get(input_tensor.type.tensor_type.elem_type, "float32")
                
                inputs[name] = f"{dtype}[{','.join(map(str, shape))}]"
            
            # Extract output information  
            outputs = {}
            for output_tensor in model.graph.output:
                name = output_tensor.name
                shape = []
                for dim in output_tensor.type.tensor_type.shape.dim:
                    if dim.dim_value:
                        shape.append(dim.dim_value)
                    elif dim.dim_param:
                        shape.append("?")  # Dynamic dimension
                    else:
                        shape.append("?")
                
                dtype_map = {
                    1: "float32",
                    2: "uint8",
                    3: "int8", 
                    6: "int32",
                    7: "int64",
                    9: "bool",
                    10: "float16",
                    11: "double"
                }
                dtype = dtype_map.get(output_tensor.type.tensor_type.elem_type, "float32")
                
                outputs[name] = f"{dtype}[{','.join(map(str, shape))}]"
            
            return {
                "format": "onnx",
                "inputs": inputs,
                "outputs": outputs,
                "model_name": Path(model_path).stem
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze ONNX model: {e}")
            return ModelAnalyzer._get_default_schema("onnx", model_path)
    
    @staticmethod
    def analyze_tensorrt_model(model_path: str) -> Dict:
        """Analyze TensorRT model (limited without TensorRT runtime)."""
        logger.warning("TensorRT analysis requires TensorRT runtime - using default schema")
        return ModelAnalyzer._get_default_schema("tensorrt", model_path)
    
    @staticmethod
    def _get_default_schema(format_type: str, model_path: str) -> Dict:
        """Generate default schema based on model type."""
        model_name = Path(model_path).stem
        
        if format_type == "onnx":
            return {
                "format": "onnx",
                "inputs": {"input": "float32[?,3,224,224]"},
                "outputs": {"output": "float32[?,1000]"},
                "model_name": model_name
            }
        elif format_type == "tensorrt":
            return {
                "format": "tensorrt", 
                "inputs": {"images": "float32[?,3,224,224]"},
                "outputs": {"detections": "float32[?,4]"},
                "model_name": model_name
            }
        else:
            return {
                "format": "unknown",
                "inputs": {"input": "float32[?]"},
                "outputs": {"output": "float32[?]"},
                "model_name": model_name
            }
    
    @staticmethod
    def detect_model_type(model_path: str) -> str:
        """Detect model format from file extension."""
        suffix = Path(model_path).suffix.lower()
        
        if suffix == ".onnx":
            return "onnx"
        elif suffix in [".trt", ".engine", ".plan"]:
            return "tensorrt"
        elif suffix in [".pb", ".savedmodel"]:
            return "tensorflow"
        elif suffix in [".pt", ".pth"]:
            return "pytorch"
        else:
            return "unknown"
    
    @classmethod
    def analyze_model(cls, model_path: str) -> Dict:
        """Main analysis function - routes to appropriate analyzer."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_type = cls.detect_model_type(model_path)
        
        if model_type == "onnx":
            return cls.analyze_onnx_model(model_path)
        elif model_type == "tensorrt":
            return cls.analyze_tensorrt_model(model_path)
        else:
            logger.warning(f"Unsupported model type: {model_type}, using default schema")
            return cls._get_default_schema(model_type, model_path)


class SchemaGenerator:
    """Generates various schema formats from model analysis."""
    
    @staticmethod
    def generate_openapi_schema(schema_info: str) -> Dict:
        """Convert model schema info to OpenAPI format."""
        if "float32" in schema_info:
            return {
                "type": "array",
                "items": {"type": "number", "format": "float"},
                "description": f"Input tensor with shape {schema_info}"
            }
        elif "int" in schema_info:
            return {
                "type": "array", 
                "items": {"type": "integer"},
                "description": f"Input tensor with shape {schema_info}"
            }
        else:
            return {
                "type": "array",
                "description": f"Input tensor with shape {schema_info}"
            }
    
    @staticmethod
    def generate_triton_config(model_analysis: Dict, max_batch_size: int = 32) -> str:
        """Generate Triton Inference Server model configuration."""
        model_name = model_analysis["model_name"]
        format_type = model_analysis["format"]
        
        # Platform mapping
        platform_map = {
            "onnx": "onnxruntime_onnx",
            "tensorrt": "tensorrt_plan",
            "tensorflow": "tensorflow_graphdef",
            "pytorch": "pytorch_libtorch"
        }
        platform = platform_map.get(format_type, "onnxruntime_onnx")
        
        config = f"""name: "{model_name}"
platform: "{platform}"
max_batch_size: {max_batch_size}

input [
"""
        
        # Add inputs
        for input_name, input_shape in model_analysis["inputs"].items():
            # Parse shape string like "float32[?,3,224,224]"
            dtype = input_shape.split('[')[0]
            shape_str = input_shape.split('[')[1].rstrip(']')
            dims = [d.strip() for d in shape_str.split(',') if d.strip()]
            
            # Convert dtype
            triton_dtype_map = {
                "float32": "TYPE_FP32",
                "float16": "TYPE_FP16", 
                "int32": "TYPE_INT32",
                "int64": "TYPE_INT64",
                "bool": "TYPE_BOOL"
            }
            triton_dtype = triton_dtype_map.get(dtype, "TYPE_FP32")
            
            config += f"""  {{
    name: "{input_name}"
    data_type: {triton_dtype}
    dims: [ {', '.join(dims[1:])} ]  # Skip batch dimension
  }}
"""
        
        config += "]\n\noutput [\n"
        
        # Add outputs
        for output_name, output_shape in model_analysis["outputs"].items():
            dtype = output_shape.split('[')[0]
            shape_str = output_shape.split('[')[1].rstrip(']')
            dims = [d.strip() for d in shape_str.split(',') if d.strip()]
            
            triton_dtype = triton_dtype_map.get(dtype, "TYPE_FP32")
            
            config += f"""  {{
    name: "{output_name}"
    data_type: {triton_dtype}
    dims: [ {', '.join(dims[1:])} ]  # Skip batch dimension
  }}
"""
        
        config += "]\n"
        
        # Add instance group for GPU
        config += """
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

# Dynamic batching configuration
dynamic_batching {
  preferred_batch_size: [ 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 100
}
"""
        
        return config