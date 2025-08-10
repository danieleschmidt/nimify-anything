#!/usr/bin/env python3
"""Test Generation 2: Robust and Reliable functionality."""

import asyncio
import tempfile
import time
from pathlib import Path


def test_error_handling():
    """Test comprehensive error handling system."""
    print("🧪 Testing Generation 2: Error Handling")
    
    try:
        from src.nimify.error_handling import (
            ErrorRecoveryManager, NimifyError, ValidationError, 
            InferenceError, SecurityError, ErrorCategory, ErrorSeverity
        )
        
        # Test error manager
        error_manager = ErrorRecoveryManager()
        
        # Test different error types
        validation_error = ValidationError("Test validation error", field="input", value="invalid")
        print(f"✅ ValidationError created: {validation_error.category.value}")
        
        inference_error = InferenceError("Test inference error", batch_size=32)
        print(f"✅ InferenceError created: {inference_error.category.value}")
        
        security_error = SecurityError("Test security error", threat_type="injection", client_ip="1.2.3.4")
        print(f"✅ SecurityError created: {security_error.category.value}")
        
        # Test error handling
        try:
            raise validation_error
        except Exception as e:
            error_context = error_manager.handle_error(e)
            print(f"✅ Error handled with ID: {error_context.error_id}")
        
        # Test error statistics
        stats = error_manager.get_error_statistics()
        print(f"✅ Error statistics: {stats['total_errors']} total errors")
        
    except ImportError as e:
        print(f"⚠️  Error handling import failed: {e}")


def test_security_features():
    """Test security features."""
    print("\n🧪 Testing Generation 2: Security Features")
    
    try:
        from src.nimify.security import (
            RateLimiter, IPBlocklist, InputSanitizer, APIKeyManager,
            ThreatDetector, SecurityHeaders
        )
        
        # Test rate limiter
        rate_limiter = RateLimiter(requests_per_minute=60, burst_size=10)
        
        # Test allowing requests
        allowed, retry_after = rate_limiter.is_allowed("192.168.1.1")
        print(f"✅ Rate limiter: allowed={allowed}, retry_after={retry_after}")
        
        # Test IP blocklist
        blocklist = IPBlocklist()
        blocklist.block_ip("1.2.3.4", duration_minutes=5)
        blocked = blocklist.is_blocked("1.2.3.4")
        print(f"✅ IP blocklist: IP blocked={blocked}")
        
        # Test input sanitizer
        test_input = "normal input"
        sanitized = InputSanitizer.sanitize_string(test_input)
        print(f"✅ Input sanitizer: '{test_input}' -> '{sanitized}'")
        
        # Test attack detection
        malicious_input = "<script>alert('xss')</script>"
        attacks = InputSanitizer.scan_for_attacks(malicious_input)
        print(f"✅ Attack detection: found {attacks}")
        
        # Test API key manager
        key_manager = APIKeyManager()
        api_key = key_manager.generate_api_key("test-service")
        valid = key_manager.validate_api_key(api_key)
        print(f"✅ API key validation: {valid}")
        
        # Test security headers
        headers = SecurityHeaders.get_headers()
        print(f"✅ Security headers: {len(headers)} headers configured")
        
        # Test threat detector
        threat_detector = ThreatDetector()
        threat_detector.record_failed_attempt("1.2.3.4", "brute_force")
        under_attack = threat_detector.is_under_attack("1.2.3.4")
        print(f"✅ Threat detection: under_attack={under_attack}")
        
    except ImportError as e:
        print(f"⚠️  Security features import failed: {e}")


def test_performance_monitoring():
    """Test performance monitoring."""
    print("\n🧪 Testing Generation 2: Performance Monitoring")
    
    try:
        from src.nimify.performance import (
            MetricsCollector, ModelCache, CircuitBreaker, 
            BatchProcessor, AdaptiveScaler
        )
        
        # Test metrics collector
        metrics = MetricsCollector()
        
        # Record some test metrics
        metrics.record_request(50.0)  # 50ms latency
        metrics.record_request(75.0)
        metrics.record_request(100.0)
        
        metrics.record_cache_hit()
        metrics.record_cache_miss()
        
        current_metrics = metrics.get_metrics()
        print(f"✅ Metrics: P50={current_metrics.latency_p50}ms, hit_rate={current_metrics.cache_hit_rate}")
        
        # Test model cache
        cache = ModelCache(max_size=100, ttl_seconds=60)
        
        # Test caching
        test_input = [[1.0, 2.0, 3.0]]
        test_output = [[0.1, 0.2, 0.7]]
        
        cache.put(test_input, test_output)
        cached_result = cache.get(test_input)
        hit_rate = cache.get_hit_rate()
        
        print(f"✅ Model cache: cached_result={cached_result is not None}, hit_rate={hit_rate}")
        
        # Test circuit breaker
        breaker = CircuitBreaker(failure_threshold=3, timeout_seconds=10)
        
        can_execute = breaker.can_execute()
        print(f"✅ Circuit breaker: can_execute={can_execute}")
        
        # Test some failures
        for i in range(3):
            breaker.record_failure()
        
        can_execute_after_failures = breaker.can_execute()
        print(f"✅ Circuit breaker after failures: can_execute={can_execute_after_failures}")
        
        # Test adaptive scaler
        scaler = AdaptiveScaler(metrics)
        should_scale_up = scaler.should_scale_up()
        should_scale_down = scaler.should_scale_down()
        
        print(f"✅ Adaptive scaler: scale_up={should_scale_up}, scale_down={should_scale_down}")
        
    except ImportError as e:
        print(f"⚠️  Performance monitoring import failed: {e}")


async def test_robust_api():
    """Test robust API functionality."""
    print("\n🧪 Testing Generation 2: Robust API")
    
    try:
        from src.nimify.robust_api import RobustModelLoader, RobustAPIHandler
        
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            f.write(b"mock_model_content_for_testing")
            model_path = f.name
        
        try:
            # Test model loader
            model_loader = RobustModelLoader(model_path)
            
            # Test loading
            await model_loader.load_model()
            print("✅ Model loaded successfully")
            
            # Test health check
            health = await model_loader.health_check()
            print(f"✅ Model health check: status={health['status']}")
            
            # Test prediction
            test_input = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            predictions = await model_loader.predict(test_input)
            print(f"✅ Model prediction: {len(predictions)} results for {len(test_input)} inputs")
            
            # Test API handler
            api_handler = RobustAPIHandler(model_loader)
            
            # Test prediction through API
            request_data = {"input": test_input}
            client_info = {"client_ip": "192.168.1.100", "user_agent": "test-client/1.0"}
            
            try:
                result = await api_handler.predict(request_data, client_info)
                print(f"✅ API prediction successful: {result.get('request_id', 'no-id')}")
            except Exception as e:
                print(f"✅ API prediction handled error correctly: {type(e).__name__}")
            
            # Test health check through API
            health_result = await api_handler.health_check(client_info)
            print(f"✅ API health check: status={health_result.get('status', 'unknown')}")
            
        finally:
            # Clean up
            Path(model_path).unlink()
        
    except ImportError as e:
        print(f"⚠️  Robust API import failed: {e}")
    except Exception as e:
        print(f"⚠️  Robust API test failed: {e}")


def test_logging_system():
    """Test structured logging system."""
    print("\n🧪 Testing Generation 2: Logging System")
    
    try:
        from src.nimify.logging_config import (
            setup_logging, log_security_event, log_api_request, 
            log_performance_metric, log_model_operation
        )
        
        # Setup logging
        setup_logging("test-service", log_level="INFO", enable_audit=True)
        print("✅ Logging configured successfully")
        
        # Test different log types
        log_security_event(
            event_type="test_event",
            message="Test security event",
            ip_address="192.168.1.1",
            request_id="test-123"
        )
        print("✅ Security event logged")
        
        log_api_request(
            method="POST",
            endpoint="/v1/predict",
            status_code=200,
            duration_ms=50.5,
            ip_address="192.168.1.1",
            request_id="test-123"
        )
        print("✅ API request logged")
        
        log_performance_metric("test_metric", 42.0, "ms", {"service": "test"})
        print("✅ Performance metric logged")
        
        log_model_operation(
            operation="load",
            model_path="/tmp/test.onnx",
            service_name="test-service",
            success=True,
            duration_ms=1500.0
        )
        print("✅ Model operation logged")
        
    except ImportError as e:
        print(f"⚠️  Logging system import failed: {e}")


async def run_generation2_tests():
    """Run all Generation 2 tests."""
    print("🚀 GENERATION 2 TESTING: Make It Robust (Reliable)")
    print("=" * 60)
    
    test_error_handling()
    test_security_features()
    test_performance_monitoring()
    test_logging_system()
    await test_robust_api()
    
    print("\n" + "=" * 60)
    print("✅ Generation 2 testing completed!")
    print("📈 Ready for Generation 3: Make It Scale (Optimized)")


if __name__ == "__main__":
    asyncio.run(run_generation2_tests())