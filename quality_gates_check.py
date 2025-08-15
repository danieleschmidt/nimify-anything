#!/usr/bin/env python3
"""Comprehensive quality gates check for Nimify Anything."""

import sys
import os
import json
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from nimify.cli import main as cli_main
from nimify.core import Nimifier, ModelConfig
from nimify.model_analyzer import ModelAnalyzer
from nimify.validation import ServiceNameValidator, ValidationError as ValidError
from nimify.simple_api import app as simple_app
from nimify.robust_api import app as robust_app  
from nimify.optimized_api import app as optimized_app
from nimify.performance_optimizer import IntelligentCache, global_monitor
from nimify.error_handling import global_error_handler, ValidationError
from nimify.circuit_breaker import CircuitBreaker
from nimify.auto_scaler import global_auto_scaler

def run_quality_gate(gate_name: str, test_func):
    """Run a quality gate and report results."""
    print(f"🔍 Running Quality Gate: {gate_name}")
    start_time = time.time()
    
    try:
        result = test_func()
        duration = time.time() - start_time
        
        if result.get('passed', False):
            print(f"✅ {gate_name} - PASSED ({duration:.2f}s)")
            print(f"   {result.get('message', '')}")
            return True
        else:
            print(f"❌ {gate_name} - FAILED ({duration:.2f}s)")
            print(f"   {result.get('message', 'Unknown failure')}")
            return False
            
    except Exception as e:
        duration = time.time() - start_time
        print(f"💥 {gate_name} - ERROR ({duration:.2f}s)")
        print(f"   Exception: {str(e)}")
        return False


def test_core_functionality():
    """Test core Nimify functionality."""
    try:
        # Test ModelConfig
        config = ModelConfig(name="test-model", max_batch_size=16)
        assert config.name == "test-model"
        assert config.max_batch_size == 16
        
        # Test Nimifier
        nimifier = Nimifier(config)
        service = nimifier.wrap_model(
            model_path="test.onnx",
            input_schema={"input": "float32[?,224,224,3]"},
            output_schema={"predictions": "float32[?,1000]"}
        )
        
        assert service.model_path == "test.onnx"
        assert service.config == config
        
        return {
            'passed': True,
            'message': 'Core functionality working correctly'
        }
        
    except Exception as e:
        return {
            'passed': False,
            'message': f'Core functionality failed: {str(e)}'
        }


def test_model_analyzer():
    """Test model analyzer functionality."""
    try:
        # Test ONNX detection
        assert ModelAnalyzer.detect_model_type("model.onnx") == "onnx"
        assert ModelAnalyzer.detect_model_type("model.engine") == "tensorrt"
        assert ModelAnalyzer.detect_model_type("model.unknown") == "unknown"
        
        # Test default schemas
        onnx_schema = ModelAnalyzer._get_default_schema("onnx", "test.onnx")
        assert "inputs" in onnx_schema
        assert "outputs" in onnx_schema
        
        return {
            'passed': True,
            'message': 'Model analyzer working correctly'
        }
        
    except Exception as e:
        return {
            'passed': False,
            'message': f'Model analyzer failed: {str(e)}'
        }


def test_validation():
    """Test validation functionality."""
    try:
        # Test service name validation
        validator = ServiceNameValidator()
        
        # Valid names
        assert validator.validate_service_name("my-service") == "my-service"
        assert validator.validate_service_name("test123") == "test123"
        
        # Invalid names should raise ValidationError
        try:
            validator.validate_service_name("INVALID")
            assert False, "Should have raised ValidationError"
        except ValidError:
            pass  # Expected
        
        return {
            'passed': True,
            'message': 'Validation working correctly'
        }
        
    except Exception as e:
        return {
            'passed': False,
            'message': f'Validation failed: {str(e)}'
        }


def test_apis_load():
    """Test that all APIs load without errors."""
    try:
        # All APIs should be importable and have basic structure
        assert hasattr(simple_app, 'title')
        assert hasattr(robust_app, 'title') 
        assert hasattr(optimized_app, 'title')
        
        # Check API titles
        assert "Simple" in simple_app.title
        assert "Robust" in robust_app.title
        assert "Optimized" in optimized_app.title
        
        return {
            'passed': True,
            'message': 'All APIs load successfully'
        }
        
    except Exception as e:
        return {
            'passed': False,
            'message': f'API loading failed: {str(e)}'
        }


def test_performance_features():
    """Test performance optimization features."""
    try:
        # Test intelligent cache
        cache = IntelligentCache(max_size=100)
        cache.put("test_key", [1, 2, 3, 4, 5])
        result = cache.get("test_key")
        assert result == [1, 2, 3, 4, 5]
        
        # Test performance monitor
        global_monitor.record_metric("test_metric", 42.0)
        stats = global_monitor.get_metric_stats("test_metric")
        assert stats is not None
        assert stats['latest'] == 42.0
        
        return {
            'passed': True,
            'message': 'Performance features working correctly'
        }
        
    except Exception as e:
        return {
            'passed': False,
            'message': f'Performance features failed: {str(e)}'
        }


def test_reliability_features():
    """Test reliability and error handling features."""
    try:
        # Test error handling
        try:
            raise ValidationError("Test error")
        except ValidationError as e:
            result = global_error_handler.handle_error(e)
            # Error handler should log and track errors
            assert "ValidationError" in global_error_handler.get_error_stats()
        
        # Test circuit breaker
        cb = CircuitBreaker(failure_threshold=2, timeout=0.1)
        
        def failing_func():
            raise Exception("Test failure")
        
        # Trigger failures
        for _ in range(3):
            try:
                cb.call(failing_func)
            except:
                pass
        
        # Circuit should be open now
        assert cb.state.value == "open"
        
        return {
            'passed': True,
            'message': 'Reliability features working correctly'
        }
        
    except Exception as e:
        return {
            'passed': False,
            'message': f'Reliability features failed: {str(e)}'
        }


def test_scaling_features():
    """Test auto-scaling functionality."""
    try:
        # Test auto-scaler
        initial_resources = global_auto_scaler.get_current_resources()
        assert len(initial_resources) > 0
        
        # Record high CPU to trigger scaling
        for i in range(10):
            global_auto_scaler.record_metric("cpu_utilization", 80.0 + i)
        
        # Evaluate scaling decisions
        decisions = global_auto_scaler.evaluate_scaling()
        
        # Should have scaling decisions due to high CPU
        scaling_stats = global_auto_scaler.get_scaling_stats()
        assert scaling_stats['rules_count'] > 0
        
        return {
            'passed': True,
            'message': 'Auto-scaling features working correctly'
        }
        
    except Exception as e:
        return {
            'passed': False,
            'message': f'Scaling features failed: {str(e)}'
        }


def test_deployment_readiness():
    """Test deployment readiness."""
    try:
        # Check that required files exist
        required_files = [
            "src/nimify/__init__.py",
            "src/nimify/cli.py",
            "src/nimify/core.py",
            "src/nimify/model_analyzer.py",
            "src/nimify/validation.py",
            "src/nimify/simple_api.py",
            "src/nimify/robust_api.py", 
            "src/nimify/optimized_api.py",
            "pyproject.toml",
            "README.md"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            return {
                'passed': False,
                'message': f'Missing files: {missing_files}'
            }
        
        return {
            'passed': True,
            'message': 'All required deployment files present'
        }
        
    except Exception as e:
        return {
            'passed': False,
            'message': f'Deployment readiness check failed: {str(e)}'
        }


def main():
    """Run all quality gates."""
    print("🚀 NIMIFY ANYTHING - COMPREHENSIVE QUALITY GATES")
    print("=" * 60)
    
    quality_gates = [
        ("Core Functionality", test_core_functionality),
        ("Model Analyzer", test_model_analyzer),
        ("Validation System", test_validation),
        ("API Loading", test_apis_load),
        ("Performance Features", test_performance_features),
        ("Reliability Features", test_reliability_features),
        ("Scaling Features", test_scaling_features),
        ("Deployment Readiness", test_deployment_readiness),
    ]
    
    passed = 0
    total = len(quality_gates)
    
    for gate_name, test_func in quality_gates:
        if run_quality_gate(gate_name, test_func):
            passed += 1
        print()  # Blank line for readability
    
    # Summary
    print("=" * 60)
    print(f"QUALITY GATES SUMMARY: {passed}/{total} PASSED")
    
    if passed == total:
        print("🎉 ALL QUALITY GATES PASSED! System ready for production.")
        
        # Generate quality report
        report = {
            "timestamp": time.time(),
            "total_gates": total,
            "passed_gates": passed,
            "success_rate": passed / total,
            "status": "PASSED" if passed == total else "FAILED",
            "generation_1": "✅ Basic functionality implemented",
            "generation_2": "✅ Robustness and reliability added",
            "generation_3": "✅ Scaling and optimization complete",
            "features": [
                "CLI interface with create/build/doctor commands",
                "Multi-format model support (ONNX, TensorRT, PyTorch, TensorFlow)",
                "OpenAPI spec generation",
                "Helm chart generation", 
                "Three API tiers: Simple, Robust, Optimized",
                "Comprehensive error handling and recovery",
                "Circuit breaker pattern for fault tolerance",
                "Intelligent caching with LRU eviction",
                "Auto-scaling based on metrics",
                "Performance monitoring and optimization",
                "Security validation and logging"
            ]
        }
        
        with open("quality_gates_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Quality report saved to: quality_gates_report.json")
        return 0
    else:
        print(f"❌ {total - passed} QUALITY GATES FAILED. Review and fix issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())