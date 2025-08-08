#!/usr/bin/env python3
"""Simple test to demonstrate Nimify functionality without external dependencies."""

import sys
import os
import tempfile
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, 'src')

def test_basic_functionality():
    """Test basic Nimify functionality."""
    print("🧪 Testing Nimify Basic Functionality")
    
    try:
        # Test core classes
        from nimify.core import ModelConfig, Nimifier, NIMService
        print("✅ Core classes imported successfully")
        
        # Test model analyzer
        from nimify.model_analyzer import ModelAnalyzer
        print("✅ Model analyzer imported successfully")
        
        # Test validation
        from nimify.validation import ServiceNameValidator, ModelFileValidator
        print("✅ Validation utilities imported successfully")
        
        # Test performance monitoring
        from nimify.performance import MetricsCollector, ModelCache
        print("✅ Performance monitoring imported successfully")
        
        # Test security
        from nimify.security import RateLimiter, InputSanitizer
        print("✅ Security utilities imported successfully")
        
        # Test deployment
        from nimify.deployment import DeploymentConfig, KubernetesManifestGenerator
        print("✅ Deployment utilities imported successfully")
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test service name validation
    try:
        ServiceNameValidator.validate_service_name("test-service")
        print("✅ Service name validation works")
    except Exception as e:
        print(f"❌ Service name validation error: {e}")
        return False
    
    # Test model config
    try:
        config = ModelConfig(
            name="test-model",
            max_batch_size=32,
            dynamic_batching=True
        )
        print("✅ Model config creation works")
    except Exception as e:
        print(f"❌ Model config error: {e}")
        return False
    
    # Test Nimifier
    try:
        nimifier = Nimifier(config)
        print("✅ Nimifier creation works")
    except Exception as e:
        print(f"❌ Nimifier error: {e}")
        return False
    
    # Test OpenAPI generation
    try:
        service = NIMService(
            config=config,
            model_path="test.onnx",
            input_schema={"input": "float32[?,3,224,224]"},
            output_schema={"output": "float32[?,1000]"}
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            service.generate_openapi(f.name)
            
        with open(f.name, 'r') as f:
            openapi_spec = json.load(f)
        
        # Verify OpenAPI structure
        assert "openapi" in openapi_spec
        assert "info" in openapi_spec
        assert "paths" in openapi_spec
        assert "/v1/predict" in openapi_spec["paths"]
        
        os.unlink(f.name)
        print("✅ OpenAPI generation works")
        
    except Exception as e:
        print(f"❌ OpenAPI generation error: {e}")
        return False
    
    # Test Helm chart generation
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            service.generate_helm_chart(temp_dir)
            
            chart_path = Path(temp_dir)
            assert (chart_path / "Chart.yaml").exists()
            assert (chart_path / "values.yaml").exists()
            assert (chart_path / "templates" / "deployment.yaml").exists()
        
        print("✅ Helm chart generation works")
        
    except Exception as e:
        print(f"❌ Helm chart generation error: {e}")
        return False
    
    # Test Kubernetes manifest generation
    try:
        deployment_config = DeploymentConfig(
            service_name="test-service",
            image_name="test-image"
        )
        
        manifest_gen = KubernetesManifestGenerator(deployment_config)
        manifests = manifest_gen.generate_all_manifests()
        
        # Verify required manifests
        required_manifests = ["deployment", "service", "hpa"]
        for manifest_name in required_manifests:
            assert manifest_name in manifests
            assert manifests[manifest_name] is not None
        
        print("✅ Kubernetes manifest generation works")
        
    except Exception as e:
        print(f"❌ Kubernetes manifest generation error: {e}")
        return False
    
    # Test security components
    try:
        rate_limiter = RateLimiter(requests_per_minute=60)
        allowed, retry_after = rate_limiter.is_allowed("127.0.0.1")
        assert allowed == True
        
        # Test input sanitization
        clean_input = InputSanitizer.sanitize_string("Hello World")
        assert clean_input == "Hello World"
        
        print("✅ Security components work")
        
    except Exception as e:
        print(f"❌ Security components error: {e}")
        return False
    
    # Test performance monitoring
    try:
        metrics_collector = MetricsCollector()
        metrics_collector.record_request(100.0)  # 100ms latency
        metrics = metrics_collector.get_metrics()
        
        assert hasattr(metrics, 'latency_p50')
        assert hasattr(metrics, 'throughput_rps')
        
        print("✅ Performance monitoring works")
        
    except Exception as e:
        print(f"❌ Performance monitoring error: {e}")
        return False
    
    print("\n🎉 All basic functionality tests passed!")
    return True

def test_model_analysis():
    """Test model analysis functionality."""
    print("\n🔍 Testing Model Analysis")
    
    try:
        from nimify.model_analyzer import ModelAnalyzer
        
        # Test model type detection
        assert ModelAnalyzer.detect_model_type("model.onnx") == "onnx"
        assert ModelAnalyzer.detect_model_type("model.trt") == "tensorrt"
        assert ModelAnalyzer.detect_model_type("model.pt") == "pytorch"
        
        print("✅ Model type detection works")
        
        # Test default schema generation
        default_schema = ModelAnalyzer._get_default_schema("onnx", "test.onnx")
        assert "inputs" in default_schema
        assert "outputs" in default_schema
        assert default_schema["format"] == "onnx"
        
        print("✅ Default schema generation works")
        
    except Exception as e:
        print(f"❌ Model analysis error: {e}")
        return False
    
    print("✅ Model analysis tests passed!")
    return True

def test_complete_workflow():
    """Test a complete workflow from model to deployment."""
    print("\n🔄 Testing Complete Workflow")
    
    try:
        from nimify.core import ModelConfig, Nimifier
        from nimify.deployment import DeploymentConfig, DeploymentOrchestrator
        
        # Step 1: Create model config
        config = ModelConfig(
            name="demo-service",
            max_batch_size=64,
            dynamic_batching=True
        )
        
        # Step 2: Create Nimifier
        nimifier = Nimifier(config)
        
        # Step 3: Wrap model (simulate)
        service = nimifier.wrap_model(
            model_path="demo.onnx",
            input_schema={"images": "float32[?,3,224,224]"},
            output_schema={"predictions": "float32[?,1000]"}
        )
        
        # Step 4: Generate deployment artifacts
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Generate OpenAPI
            openapi_path = temp_path / "openapi.json"
            service.generate_openapi(str(openapi_path))
            
            # Generate Helm chart
            helm_path = temp_path / "helm"
            service.generate_helm_chart(str(helm_path))
            
            # Generate deployment package
            deployment_config = DeploymentConfig(
                service_name="demo-service",
                image_name="demo-service",
                replicas=3
            )
            
            orchestrator = DeploymentOrchestrator(deployment_config)
            package_dir = orchestrator.generate_deployment_package(temp_path)
            
            # Verify all artifacts exist
            assert openapi_path.exists()
            assert (helm_path / "Chart.yaml").exists()
            assert (package_dir / "helm").exists()
            assert (package_dir / "kubernetes").exists()
            assert (package_dir / "scripts").exists()
        
        print("✅ Complete workflow test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Complete workflow error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Running Nimify Functionality Tests\n")
    
    tests = [
        test_basic_functionality,
        test_model_analysis,
        test_complete_workflow
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Nimify basic functionality is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)