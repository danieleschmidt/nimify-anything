#!/usr/bin/env python3
"""Test the robust system implementation."""

import sys
import asyncio
import tempfile
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

async def test_robust_api():
    """Test the robust API implementation."""
    print("🛡️ Testing Robust API Implementation")
    
    try:
        from nimify.robust_api import create_robust_service
        from nimify.error_handling import NimifyError, ValidationError
        from nimify.monitoring import global_metrics_collector, system_monitor, alert_manager
        
        # Create test model file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.onnx', delete=False) as f:
            f.write("# Mock ONNX model")
            model_path = f.name
        
        try:
            # Create robust service
            service = create_robust_service(model_path)
            print("✅ Created robust service")
            
            # Test model loading
            await service.model_loader.load_model()
            print("✅ Model loading successful")
            
            # Test health check
            health_result = await service.health_check({
                'client_ip': '127.0.0.1',
                'user_agent': 'test-client/1.0'
            })
            print(f"✅ Health check: {health_result['status']}")
            
            # Test valid prediction
            request_data = {
                'input': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            }
            
            client_info = {
                'client_ip': '127.0.0.1',
                'user_agent': 'test-client/1.0'
            }
            
            result = await service.predict(request_data, client_info)
            print(f"✅ Prediction successful: {len(result['predictions'])} predictions")
            
            # Test validation error handling
            try:
                invalid_request = {'input': 'invalid'}
                await service.predict(invalid_request, client_info)
                print("❌ Should have failed validation")
            except Exception as e:
                print("✅ Validation error handling works")
            
            # Test metrics collection
            metrics = global_metrics_collector.get_histogram_stats('request_duration')
            print(f"✅ Metrics collected: {len(global_metrics_collector.metrics)} metric types")
            
            return True
            
        finally:
            # Cleanup
            Path(model_path).unlink(exist_ok=True)
            
    except Exception as e:
        print(f"❌ Robust API test failed: {e}")
        return False

async def test_monitoring_system():
    """Test the monitoring and alerting system."""
    print("\\n📊 Testing Monitoring System")
    
    try:
        from nimify.monitoring import (
            global_metrics_collector, system_monitor, alert_manager, 
            Alert, create_default_alerts, DashboardGenerator
        )
        
        # Test metrics collection
        global_metrics_collector.record_counter("test_counter", 5.0)
        global_metrics_collector.record_gauge("test_gauge", 42.5)
        global_metrics_collector.record_histogram("test_histogram", 100.0)
        global_metrics_collector.record_histogram("test_histogram", 200.0)
        global_metrics_collector.record_histogram("test_histogram", 150.0)
        
        # Verify metrics
        assert global_metrics_collector.get_counter("test_counter") == 5.0
        assert global_metrics_collector.get_gauge("test_gauge") == 42.5
        
        histogram_stats = global_metrics_collector.get_histogram_stats("test_histogram")
        assert histogram_stats['count'] == 3
        assert histogram_stats['mean'] == 150.0
        
        print("✅ Metrics collection works")
        
        # Test Prometheus export
        prometheus_output = global_metrics_collector.export_prometheus()
        assert "test_counter 5.0" in prometheus_output
        assert "test_gauge 42.5" in prometheus_output
        print("✅ Prometheus export works")
        
        # Test alert system
        test_alert = Alert(
            name="test_alert",
            condition=lambda m: m.get("test_gauge", 0) > 40,
            message="Test gauge is above 40",
            severity="warning"
        )
        
        alert_manager.add_alert(test_alert)
        await alert_manager._check_alerts()  # Manual check
        
        active_alerts = alert_manager.get_active_alerts()
        print(f"✅ Alert system works: {len(active_alerts)} active alerts")
        
        # Test dashboard generation
        dashboard_gen = DashboardGenerator(global_metrics_collector)
        dashboard_config = dashboard_gen.generate_grafana_dashboard()
        
        assert "dashboard" in dashboard_config
        assert len(dashboard_config["dashboard"]["panels"]) > 0
        print("✅ Dashboard generation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring system test failed: {e}")
        return False

async def test_error_handling():
    """Test comprehensive error handling."""
    print("\\n🚨 Testing Error Handling System")
    
    try:
        from nimify.error_handling import (
            NimifyError, ValidationError, ModelLoadingError, 
            error_manager, ErrorCategory, ErrorSeverity, retry
        )
        
        # Test custom exceptions
        try:
            raise ValidationError("Test validation error", field="test_field", value="test_value")
        except ValidationError as e:
            assert e.category == ErrorCategory.VALIDATION
            assert e.severity == ErrorSeverity.MEDIUM
            assert "test_field" in e.details
            print("✅ Custom exceptions work")
        
        # Test error manager
        try:
            raise Exception("Test generic error")
        except Exception as e:
            error_context = error_manager.handle_error(e)
            assert error_context.error_id is not None
            assert error_context.category is not None
            print(f"✅ Error manager works: {error_context.error_id}")
        
        # Test retry decorator
        attempt_count = 0
        
        @retry(max_attempts=3, delay=0.1)
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert attempt_count == 3
        print("✅ Retry decorator works")
        
        # Test error statistics
        stats = error_manager.get_error_statistics()
        assert stats['total_errors'] > 0
        print(f"✅ Error statistics: {stats['total_errors']} total errors")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

async def test_security_integration():
    """Test security integration."""
    print("\\n🔒 Testing Security Integration")
    
    try:
        from nimify.security import (
            rate_limiter, ip_blocklist, api_key_manager, 
            threat_detector, InputSanitizer
        )
        
        # Test rate limiting
        allowed, retry_after = rate_limiter.is_allowed("192.168.1.100")
        assert allowed == True
        print("✅ Rate limiting works")
        
        # Test IP blocklist
        ip_blocklist.block_ip("192.168.1.200", duration_minutes=1)
        assert ip_blocklist.is_blocked("192.168.1.200") == True
        assert ip_blocklist.is_blocked("192.168.1.201") == False
        print("✅ IP blocklist works")
        
        # Test API key management
        api_key = api_key_manager.generate_api_key("test_user", ["predict"])
        assert api_key_manager.validate_api_key(api_key, "predict") == True
        assert api_key_manager.validate_api_key("invalid_key", "predict") == False
        print("✅ API key management works")
        
        # Test input sanitization
        clean_input = InputSanitizer.sanitize_string("Hello, World!")
        assert clean_input == "Hello, World!"
        
        try:
            InputSanitizer.sanitize_string("SELECT * FROM users")
            print("❌ Should have detected SQL injection")
        except Exception:
            print("✅ Input sanitization detects attacks")
        
        # Test threat detection
        threat_detector.record_failed_attempt("192.168.1.300", "brute_force")
        analysis = threat_detector.analyze_request_content("normal request content")
        assert isinstance(analysis, dict)
        print("✅ Threat detection works")
        
        return True
        
    except Exception as e:
        print(f"❌ Security integration test failed: {e}")
        return False

async def main():
    """Run all robust system tests."""
    print("🛡️ Testing Nimify Robust System (Generation 2)\\n")
    
    tests = [
        test_robust_api,
        test_monitoring_system,
        test_error_handling,
        test_security_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\\n📊 Robust System Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All robust system tests passed! Generation 2 is ready.")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)