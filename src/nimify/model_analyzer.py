"""Model analysis utilities for automatic schema detection."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelAnalyzer:
    """Analyzes model files to extract metadata and generate schemas."""
    
    @staticmethod
    def analyze_onnx_model(model_path: str) -> dict:
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
                        shape.append("?")  # Dynamic dimension
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
    def analyze_tensorrt_model(model_path: str) -> dict:
        """Analyze TensorRT model using basic file inspection."""
        try:
            import os
            file_size = os.path.getsize(model_path)
            logger.info(f"TensorRT model size: {file_size / (1024*1024):.2f} MB")
            
            # For TensorRT, we'll use intelligent defaults based on common patterns
            model_name = Path(model_path).stem
            
            # Analyze filename for hints about model type
            if any(keyword in model_name.lower() for keyword in ['yolo', 'detect', 'object']):
                return {
                    "format": "tensorrt",
                    "inputs": {"images": "float32[?,3,640,640]"},
                    "outputs": {"detections": "float32[?,25200,85]"},
                    "model_name": model_name,
                    "estimated_type": "object_detection"
                }
            elif any(keyword in model_name.lower() for keyword in ['resnet', 'efficientnet', 'classifier']):
                return {
                    "format": "tensorrt",
                    "inputs": {"input": "float32[?,3,224,224]"},
                    "outputs": {"predictions": "float32[?,1000]"},
                    "model_name": model_name,
                    "estimated_type": "classification"
                }
            else:
                return ModelAnalyzer._get_default_schema("tensorrt", model_path)
                
        except Exception as e:
            logger.error(f"Failed to analyze TensorRT model: {e}")
            return ModelAnalyzer._get_default_schema("tensorrt", model_path)
    
    @staticmethod
    def _get_default_schema(format_type: str, model_path: str) -> dict:
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
        elif format_type == "pytorch":
            return {
                "format": "pytorch",
                "inputs": {"input": "float32[?,3,224,224]"},
                "outputs": {"predictions": "float32[?,1000]"},
                "model_name": model_name
            }
        elif format_type == "tensorflow":
            return {
                "format": "tensorflow",
                "inputs": {"input": "float32[?,224,224,3]"},
                "outputs": {"predictions": "float32[?,1000]"},
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
    def analyze_model(cls, model_path: str) -> dict:
        """Main analysis function - routes to appropriate analyzer."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_type = cls.detect_model_type(model_path)
        logger.info(f"Detected model type: {model_type} for {model_path}")
        
        if model_type == "onnx":
            return cls.analyze_onnx_model(model_path)
        elif model_type == "tensorrt":
            return cls.analyze_tensorrt_model(model_path)
        elif model_type == "pytorch":
            return cls.analyze_pytorch_model(model_path)
        elif model_type == "tensorflow":
            return cls.analyze_tensorflow_model(model_path)
        else:
            logger.warning(f"Unsupported model type: {model_type}, using default schema")
            return cls._get_default_schema(model_type, model_path)


class SchemaGenerator:
    """Generates various schema formats from model analysis."""
    
    @staticmethod
    def generate_openapi_schema(schema_info: str) -> dict:
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
    def generate_triton_config(model_analysis: dict, max_batch_size: int = 32) -> str:
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
    
    @staticmethod  
    def analyze_pytorch_model(model_path: str) -> dict:
        """Analyze PyTorch model using basic inspection."""
        try:
            import torch
            # Load model metadata only, not the full model
            model_data = torch.load(model_path, map_location='cpu')
            model_name = Path(model_path).stem
            
            # Try to extract input/output info from state dict keys
            if isinstance(model_data, dict) and 'state_dict' in model_data:
                state_dict = model_data['state_dict']
            elif isinstance(model_data, dict):
                state_dict = model_data
            else:
                logger.warning("Could not extract state dict from PyTorch model")
                return ModelAnalyzer._get_default_schema("pytorch", model_path)
            
            # Analyze layer names for hints
            layer_names = list(state_dict.keys())
            first_layer = layer_names[0] if layer_names else ""
            last_layer = layer_names[-1] if layer_names else ""
            
            logger.info(f"PyTorch model layers: {len(layer_names)}, first: {first_layer}, last: {last_layer}")
            
            return {
                "format": "pytorch",
                "inputs": {"input": "float32[?,3,224,224]"},
                "outputs": {"output": "float32[?,1000]"},
                "model_name": model_name,
                "layers_count": len(layer_names)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze PyTorch model: {e}")
            return ModelAnalyzer._get_default_schema("pytorch", model_path)
    
    @staticmethod
    def analyze_tensorflow_model(model_path: str) -> dict:
        """Analyze TensorFlow model using basic inspection."""
        try:
            import tensorflow as tf
            
            if model_path.endswith('.pb'):
                # GraphDef format
                with tf.io.gfile.GFile(model_path, 'rb') as f:
                    graph_def = tf.compat.v1.GraphDef()
                    graph_def.ParseFromString(f.read())
                
                input_nodes = []
                output_nodes = []
                
                for node in graph_def.node:
                    if node.op == 'Placeholder':
                        input_nodes.append(node.name)
                    elif 'output' in node.name.lower() or node.name.endswith('predictions'):
                        output_nodes.append(node.name)
                
                model_name = Path(model_path).stem
                
                return {
                    "format": "tensorflow",
                    "inputs": {name: "float32[?,224,224,3]" for name in input_nodes or ["input"]},
                    "outputs": {name: "float32[?,1000]" for name in output_nodes or ["predictions"]},
                    "model_name": model_name
                }
            
            else:
                # SavedModel format
                model = tf.saved_model.load(model_path)
                signatures = list(model.signatures.keys())
                
                if signatures:
                    sig = model.signatures[signatures[0]]
                    inputs = {name: f"float32[{list(spec.shape)}]" for name, spec in sig.inputs.items()}
                    outputs = {name: f"float32[{list(spec.shape)}]" for name, spec in sig.outputs.items()}
                else:
                    inputs = {"input": "float32[?,224,224,3]"}
                    outputs = {"predictions": "float32[?,1000]"}
                
                model_name = Path(model_path).stem
                
                return {
                    "format": "tensorflow",
                    "inputs": inputs,
                    "outputs": outputs,
                    "model_name": model_name,
                    "signatures": signatures
                }
                
        except Exception as e:
            logger.error(f"Failed to analyze TensorFlow model: {e}")
            return ModelAnalyzer._get_default_schema("tensorflow", model_path)
    
