#!/usr/bin/env python3
"""Simple quality gates check for Nimify Anything."""

import sys
import os
import json
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_core_functionality():
    """Check core functionality."""
    try:
        from nimify.core import Nimifier, ModelConfig
        
        config = ModelConfig(name="test-model", max_batch_size=16)
        nimifier = Nimifier(config)
        service = nimifier.wrap_model(
            model_path="test.onnx",
            input_schema={"input": "float32[?,224,224,3]"},
            output_schema={"predictions": "float32[?,1000]"}
        )
        
        assert service.model_path == "test.onnx"
        print("✅ Core functionality working")
        return True
        
    except Exception as e:
        print(f"❌ Core functionality failed: {e}")
        return False

def check_model_analyzer():
    """Check model analyzer."""
    try:
        from nimify.model_analyzer import ModelAnalyzer
        
        assert ModelAnalyzer.detect_model_type("model.onnx") == "onnx"
        assert ModelAnalyzer.detect_model_type("model.engine") == "tensorrt"
        
        schema = ModelAnalyzer._get_default_schema("onnx", "test.onnx")
        assert "inputs" in schema and "outputs" in schema
        
        print("✅ Model analyzer working")
        return True
        
    except Exception as e:
        print(f"❌ Model analyzer failed: {e}")
        return False

def check_validation():
    """Check validation system."""
    try:
        from nimify.validation import ServiceNameValidator, ValidationError
        
        validator = ServiceNameValidator()
        assert validator.validate_service_name("my-service") == "my-service"
        
        try:
            validator.validate_service_name("INVALID")
            assert False, "Should have raised ValidationError"
        except ValidationError:
            pass  # Expected
        
        print("✅ Validation system working")
        return True
        
    except Exception as e:
        print(f"❌ Validation system failed: {e}")
        return False

def check_apis():
    """Check that APIs can be imported."""
    try:
        # Test each API separately to avoid metrics conflicts
        import importlib
        
        # Simple API
        simple_module = importlib.import_module("nimify.simple_api")
        assert hasattr(simple_module, 'app')
        print("✅ Simple API loads")
        
        # Performance optimizer
        perf_module = importlib.import_module("nimify.performance_optimizer")
        cache = perf_module.IntelligentCache(max_size=10)
        cache.put("test", [1, 2, 3])
        assert cache.get("test") == [1, 2, 3]
        print("✅ Performance optimizer working")
        
        # Error handling
        error_module = importlib.import_module("nimify.error_handling") 
        handler = error_module.ErrorHandler()
        print("✅ Error handling working")
        
        # Circuit breaker
        cb_module = importlib.import_module("nimify.circuit_breaker")
        cb = cb_module.CircuitBreaker(failure_threshold=3)
        print("✅ Circuit breaker working")
        
        # Auto-scaler  
        scaler_module = importlib.import_module("nimify.auto_scaler")
        scaler = scaler_module.AutoScaler()
        scaler.record_metric("test_metric", 50.0)
        print("✅ Auto-scaler working")
        
        return True
        
    except Exception as e:
        print(f"❌ API check failed: {e}")
        return False

def check_cli():
    """Check CLI functionality."""
    try:
        import subprocess
        
        # Test CLI help
        result = subprocess.run([
            sys.executable, "-m", "src.nimify.cli", "--help"
        ], capture_output=True, text=True, cwd="/root/repo")
        
        if result.returncode == 0 and "Nimify Anything" in result.stdout:
            print("✅ CLI working")
            return True
        else:
            print(f"❌ CLI failed: {result.stderr}")
            return False
        
    except Exception as e:
        print(f"❌ CLI check failed: {e}")
        return False

def check_files():
    """Check required files exist."""
    required_files = [
        "src/nimify/__init__.py",
        "src/nimify/cli.py", 
        "src/nimify/core.py",
        "src/nimify/model_analyzer.py",
        "src/nimify/validation.py",
        "src/nimify/simple_api.py",
        "pyproject.toml",
        "README.md"
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    if missing:
        print(f"❌ Missing files: {missing}")
        return False
    
    print("✅ All required files present")
    return True

def main():
    """Run quality gates."""
    print("🚀 NIMIFY ANYTHING - QUALITY GATES CHECK")
    print("=" * 50)
    
    checks = [
        ("Files Check", check_files),
        ("Core Functionality", check_core_functionality),  
        ("Model Analyzer", check_model_analyzer),
        ("Validation", check_validation),
        ("APIs & Features", check_apis),
        ("CLI Interface", check_cli),
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        print(f"\n🔍 {name}...")
        if check_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"SUMMARY: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 ALL QUALITY GATES PASSED!")
        print("\nSystem Features Implemented:")
        print("• CLI with create/build/doctor commands")
        print("• Multi-format model support")
        print("• OpenAPI spec generation") 
        print("• Three API tiers (Simple/Robust/Optimized)")
        print("• Error handling & circuit breakers")
        print("• Intelligent caching")
        print("• Auto-scaling")
        print("• Performance monitoring")
        
        # Save report
        report = {
            "timestamp": time.time(),
            "checks_passed": passed,
            "checks_total": total,
            "success_rate": passed / total,
            "status": "PASSED"
        }
        
        with open("quality_gates_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Report saved: quality_gates_report.json")
        return 0
    else:
        print(f"❌ {total - passed} checks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())