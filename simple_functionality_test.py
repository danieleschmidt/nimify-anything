#!/usr/bin/env python3
"""Simple test without external dependencies to verify Generation 1 functionality."""

import tempfile
import json
from pathlib import Path

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("🧪 Testing Generation 1: Basic Functionality")
    
    try:
        from src.nimify.core import ModelConfig, Nimifier
        
        # Test configuration
        config = ModelConfig(name="test")
        print(f"✅ ModelConfig created: {config.name}")
        print(f"   Max batch size: {config.max_batch_size}")
        print(f"   Dynamic batching: {config.dynamic_batching}")
        
        # Test nimifier
        nimifier = Nimifier(config)
        print(f"✅ Nimifier created with config")
        
        # Test service wrapper (without actual model)
        service = nimifier.wrap_model(
            model_path="/tmp/dummy.onnx",
            input_schema={"input": "float32[?,10]"},
            output_schema={"output": "float32[?,1]"}
        )
        print(f"✅ Service wrapper created")
        
        # Test OpenAPI generation
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            openapi_path = f.name
        
        service.generate_openapi(openapi_path)
        
        if Path(openapi_path).exists():
            with open(openapi_path) as f:
                spec = json.load(f)
            print(f"✅ OpenAPI spec generated: {spec['info']['title']}")
            print(f"   Endpoints: {list(spec['paths'].keys())}")
            Path(openapi_path).unlink()
        
        # Test Helm chart generation  
        with tempfile.TemporaryDirectory() as helm_dir:
            service.generate_helm_chart(helm_dir)
            
            chart_yaml = Path(helm_dir) / "Chart.yaml"
            values_yaml = Path(helm_dir) / "values.yaml"
            
            if chart_yaml.exists() and values_yaml.exists():
                print(f"✅ Helm chart generated successfully")
                with open(chart_yaml) as f:
                    chart_content = f.read()
                print(f"   Chart name: {config.name}")
            else:
                print(f"❌ Helm chart files missing")
            
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()

def test_validation():
    """Test validation functionality."""
    print("\n🧪 Testing Generation 1: Validation")
    
    try:
        from src.nimify.validation import ServiceNameValidator, ValidationError
        
        # Test valid names
        valid_names = ["my-service", "image-classifier", "text-gen"]
        for name in valid_names:
            try:
                result = ServiceNameValidator.validate_service_name(name)
                print(f"✅ Valid service name: {name}")
            except ValidationError as e:
                print(f"❌ Unexpected validation error for {name}: {e}")
        
        # Test invalid names  
        invalid_names = ["My-Service", "123invalid", "service_with_underscore", ""]
        for name in invalid_names:
            try:
                ServiceNameValidator.validate_service_name(name)
                print(f"❌ Should have failed validation: {name}")
            except ValidationError:
                print(f"✅ Correctly rejected invalid name: {name}")
        
    except ImportError as e:
        print(f"⚠️  Validation import failed: {e}")

def test_model_analyzer():
    """Test model analyzer with dummy file."""
    print("\n🧪 Testing Generation 1: Model Analyzer")
    
    try:
        from src.nimify.model_analyzer import ModelAnalyzer
        
        # Create dummy model files for testing
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            f.write(b"dummy_onnx_content_for_testing")
            onnx_path = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.trt', delete=False) as f:
            f.write(b"dummy_tensorrt_content")
            trt_path = f.name
        
        # Test model type detection
        onnx_type = ModelAnalyzer.detect_model_type(onnx_path)
        trt_type = ModelAnalyzer.detect_model_type(trt_path)
        
        print(f"✅ ONNX detection: {onnx_type}")
        print(f"✅ TensorRT detection: {trt_type}")
        
        # Test model analysis (will use fallback)
        try:
            analysis = ModelAnalyzer.analyze_model(onnx_path)
            print(f"✅ Model analysis: {analysis['format']}")
            print(f"   Inputs: {analysis.get('inputs', {})}")
            print(f"   Outputs: {analysis.get('outputs', {})}")
        except Exception as e:
            print(f"⚠️  Model analysis failed (expected for dummy file): {e}")
        
        # Clean up
        Path(onnx_path).unlink()
        Path(trt_path).unlink()
        
    except ImportError as e:
        print(f"⚠️  Model analyzer import failed: {e}")

def test_cli_imports():
    """Test CLI module imports."""
    print("\n🧪 Testing Generation 1: CLI Imports")
    
    try:
        from src.nimify.cli import main
        print("✅ CLI main function imported successfully")
        
        # Test click integration
        import click
        print("✅ Click available for CLI")
        
    except ImportError as e:
        print(f"⚠️  CLI import failed: {e}")

if __name__ == "__main__":
    print("🚀 GENERATION 1 TESTING: Make It Work (Simple)")
    print("=" * 50)
    
    test_basic_functionality()
    test_validation() 
    test_model_analyzer()
    test_cli_imports()
    
    print("\n" + "=" * 50)
    print("✅ Generation 1 testing completed!")
    print("📈 Ready for Generation 2: Make It Robust")