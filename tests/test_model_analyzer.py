"""Tests for model analyzer functionality."""

import pytest
from unittest.mock import patch, Mock
from pathlib import Path

from nimify.model_analyzer import ModelAnalyzer, SchemaGenerator


class TestModelAnalyzer:
    """Test cases for ModelAnalyzer class."""
    
    def test_detect_model_type_onnx(self):
        """Test ONNX model type detection."""
        assert ModelAnalyzer.detect_model_type("model.onnx") == "onnx"
        assert ModelAnalyzer.detect_model_type("MODEL.ONNX") == "onnx"
    
    def test_detect_model_type_tensorrt(self):
        """Test TensorRT model type detection."""
        assert ModelAnalyzer.detect_model_type("model.trt") == "tensorrt"
        assert ModelAnalyzer.detect_model_type("model.engine") == "tensorrt"
        assert ModelAnalyzer.detect_model_type("model.plan") == "tensorrt"
    
    def test_detect_model_type_unknown(self):
        """Test unknown model type detection."""
        assert ModelAnalyzer.detect_model_type("model.xyz") == "unknown"
        assert ModelAnalyzer.detect_model_type("model") == "unknown"
    
    def test_get_default_schema_onnx(self):
        """Test default ONNX schema generation."""
        schema = ModelAnalyzer._get_default_schema("onnx", "test_model.onnx")
        
        assert schema["format"] == "onnx"
        assert schema["model_name"] == "test_model"
        assert "input" in schema["inputs"]
        assert "output" in schema["outputs"]
        assert "float32" in schema["inputs"]["input"]
    
    def test_get_default_schema_tensorrt(self):
        """Test default TensorRT schema generation."""
        schema = ModelAnalyzer._get_default_schema("tensorrt", "detector.trt")
        
        assert schema["format"] == "tensorrt"
        assert schema["model_name"] == "detector"
        assert "images" in schema["inputs"]
        assert "detections" in schema["outputs"]
    
    @patch('pathlib.Path.exists')
    def test_analyze_model_file_not_found(self, mock_exists):
        """Test model analysis with non-existent file."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            ModelAnalyzer.analyze_model("nonexistent.onnx")
    
    @patch('pathlib.Path.exists')
    @patch('nimify.model_analyzer.ModelAnalyzer.analyze_onnx_model')
    def test_analyze_model_onnx_success(self, mock_analyze, mock_exists):
        """Test successful ONNX model analysis."""
        mock_exists.return_value = True
        mock_analyze.return_value = {
            "format": "onnx",
            "inputs": {"input": "float32[1,3,224,224]"},
            "outputs": {"output": "float32[1,1000]"},
            "model_name": "test_model"
        }
        
        result = ModelAnalyzer.analyze_model("test_model.onnx")
        
        assert result["format"] == "onnx"
        assert result["model_name"] == "test_model"
        mock_analyze.assert_called_once_with("test_model.onnx")
    
    @patch('pathlib.Path.exists')
    def test_analyze_model_unsupported_format(self, mock_exists):
        """Test analysis of unsupported model format."""
        mock_exists.return_value = True
        
        result = ModelAnalyzer.analyze_model("model.xyz")
        
        assert result["format"] == "unknown"
        assert "input" in result["inputs"]
        assert "output" in result["outputs"]
    
    @patch('onnx.load')
    def test_analyze_onnx_model_success(self, mock_load):
        """Test successful ONNX model analysis."""
        # Mock ONNX model structure
        mock_model = Mock()
        
        # Mock input tensor
        mock_input = Mock()
        mock_input.name = "input"
        mock_input.type.tensor_type.elem_type = 1  # FLOAT
        
        # Mock dimensions
        dim1 = Mock()
        dim1.dim_value = 0
        dim1.dim_param = "batch"
        
        dim2 = Mock()
        dim2.dim_value = 3
        dim2.dim_param = ""
        
        dim3 = Mock()
        dim3.dim_value = 224
        dim3.dim_param = ""
        
        dim4 = Mock()
        dim4.dim_value = 224
        dim4.dim_param = ""
        
        mock_input.type.tensor_type.shape.dim = [dim1, dim2, dim3, dim4]
        
        # Mock output tensor
        mock_output = Mock()
        mock_output.name = "output"
        mock_output.type.tensor_type.elem_type = 1  # FLOAT
        
        out_dim1 = Mock()
        out_dim1.dim_value = 0
        out_dim1.dim_param = "batch"
        
        out_dim2 = Mock()
        out_dim2.dim_value = 1000
        out_dim2.dim_param = ""
        
        mock_output.type.tensor_type.shape.dim = [out_dim1, out_dim2]
        
        mock_model.graph.input = [mock_input]
        mock_model.graph.output = [mock_output]
        mock_load.return_value = mock_model
        
        result = ModelAnalyzer.analyze_onnx_model("test.onnx")
        
        assert result["format"] == "onnx"
        assert result["model_name"] == "test"
        assert "input" in result["inputs"]
        assert "output" in result["outputs"]
        assert "float32" in result["inputs"]["input"]
    
    @patch('onnx.load')
    def test_analyze_onnx_model_failure(self, mock_load):
        """Test ONNX model analysis failure."""
        mock_load.side_effect = Exception("ONNX load failed")
        
        result = ModelAnalyzer.analyze_onnx_model("test.onnx")
        
        # Should return default schema on failure
        assert result["format"] == "onnx"
        assert result["model_name"] == "test"
    
    def test_analyze_tensorrt_model(self):
        """Test TensorRT model analysis (returns default)."""
        result = ModelAnalyzer.analyze_tensorrt_model("model.trt")
        
        assert result["format"] == "tensorrt"
        assert result["model_name"] == "model"
        assert "images" in result["inputs"]


class TestSchemaGenerator:
    """Test cases for SchemaGenerator class."""
    
    def test_generate_openapi_schema_float32(self):
        """Test OpenAPI schema generation for float32."""
        schema = SchemaGenerator.generate_openapi_schema("float32[1,3,224,224]")
        
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "number"
        assert schema["items"]["format"] == "float"
    
    def test_generate_openapi_schema_int(self):
        """Test OpenAPI schema generation for int."""
        schema = SchemaGenerator.generate_openapi_schema("int32[1,10]")
        
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "integer"
    
    def test_generate_openapi_schema_unknown(self):
        """Test OpenAPI schema generation for unknown type."""
        schema = SchemaGenerator.generate_openapi_schema("unknown[1,10]")
        
        assert schema["type"] == "array"
        assert "items" not in schema or schema.get("items", {}).get("type") != "number"
    
    def test_generate_triton_config_onnx(self):
        """Test Triton config generation for ONNX model."""
        model_analysis = {
            "model_name": "test_model",
            "format": "onnx",
            "inputs": {"input": "float32[?,3,224,224]"},
            "outputs": {"output": "float32[?,1000]"}
        }
        
        config = SchemaGenerator.generate_triton_config(model_analysis)
        
        assert 'name: "test_model"' in config
        assert 'platform: "onnxruntime_onnx"' in config
        assert 'max_batch_size: 32' in config
        assert '"input"' in config
        assert '"output"' in config
        assert 'TYPE_FP32' in config
        assert 'dynamic_batching' in config
    
    def test_generate_triton_config_tensorrt(self):
        """Test Triton config generation for TensorRT model."""
        model_analysis = {
            "model_name": "detector",
            "format": "tensorrt",
            "inputs": {"images": "float32[?,3,640,640]"},
            "outputs": {"detections": "float32[?,4]"}
        }
        
        config = SchemaGenerator.generate_triton_config(model_analysis, max_batch_size=16)
        
        assert 'name: "detector"' in config
        assert 'platform: "tensorrt_plan"' in config
        assert 'max_batch_size: 16' in config
        assert '"images"' in config
        assert '"detections"' in config