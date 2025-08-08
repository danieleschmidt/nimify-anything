"""Robust API implementation with comprehensive error handling and security."""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
import json
from pathlib import Path

# Simulated FastAPI components for demonstration
from .error_handling import (
    NimifyError, ValidationError, InferenceError, SecurityError, 
    error_manager, ErrorCategory, ErrorSeverity, retry, async_retry
)
from .validation import RequestValidator
from .security import (
    rate_limiter, ip_blocklist, api_key_manager, threat_detector,
    SecurityHeaders, InputSanitizer
)
from .performance import metrics_collector, model_cache, circuit_breaker
from .logging_config import log_security_event, log_api_request, log_performance_metric

logger = logging.getLogger(__name__)


class RobustModelLoader:
    """Model loader with comprehensive error handling and recovery."""
    
    def __init__(self, model_path: str, max_retries: int = 3):
        self.model_path = model_path
        self.max_retries = max_retries
        self.model_session = None
        self.load_time = None
        self.is_healthy = False
        self._lock = asyncio.Lock()
    
    @async_retry(max_attempts=3, delay=2.0, exceptions=(Exception,))
    async def load_model(self):
        """Load model with retry logic and health checking."""
        async with self._lock:
            if self.model_session is not None:
                return self.model_session
            
            start_time = time.time()
            
            try:
                logger.info(f"Loading model from {self.model_path}")
                
                # Validate model file exists
                model_file = Path(self.model_path)
                if not model_file.exists():
                    raise ValidationError(
                        f"Model file not found: {self.model_path}",
                        field="model_path",
                        value=self.model_path
                    )
                
                # Check file size (basic validation)
                file_size = model_file.stat().st_size
                if file_size == 0:
                    raise ValidationError(
                        "Model file is empty",
                        field="model_size",
                        value=file_size
                    )
                
                if file_size > 10 * 1024 * 1024 * 1024:  # 10GB limit
                    raise ValidationError(
                        f"Model file too large: {file_size} bytes",
                        field="model_size", 
                        value=file_size
                    )
                
                # Simulate model loading (in real implementation, use ONNX Runtime)
                await asyncio.sleep(0.1)  # Simulate loading time
                
                # Create mock session
                self.model_session = {
                    'model_path': self.model_path,
                    'input_names': ['input'],
                    'output_names': ['output'],
                    'loaded_at': time.time()
                }
                
                self.load_time = time.time() - start_time
                self.is_healthy = True
                
                logger.info(f"Model loaded successfully in {self.load_time:.2f}s")
                
                # Log performance metric
                log_performance_metric("model_load_time", self.load_time * 1000, "ms")
                
                return self.model_session
                
            except Exception as e:
                self.is_healthy = False
                logger.error(f"Model loading failed: {e}")
                
                # Handle with error manager
                error_context = error_manager.handle_error(e, {
                    'model_path': self.model_path,
                    'operation': 'model_loading'
                })
                
                raise NimifyError(
                    f"Failed to load model: {str(e)}",
                    category=ErrorCategory.MODEL_LOADING,
                    severity=ErrorSeverity.CRITICAL,
                    details={'model_path': self.model_path, 'error_id': error_context.error_id}
                )
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive model health check."""
        if not self.model_session:
            return {
                'status': 'unhealthy',
                'reason': 'Model not loaded',
                'loaded': False
            }
        
        try:
            # Test inference with dummy data
            test_input = [[1.0, 2.0, 3.0]]
            result = await self.predict(test_input, health_check=True)
            
            return {
                'status': 'healthy',
                'loaded': True,
                'load_time': self.load_time,
                'last_inference': time.time()
            }
            
        except Exception as e:
            self.is_healthy = False
            logger.error(f"Model health check failed: {e}")
            
            return {
                'status': 'unhealthy',
                'reason': str(e),
                'loaded': True,
                'load_time': self.load_time
            }
    
    @async_retry(max_attempts=2, delay=0.5, exceptions=(InferenceError,))
    async def predict(self, input_data: List[List[float]], health_check: bool = False) -> List[List[float]]:
        """Run inference with error handling and caching."""
        if not self.model_session:
            raise InferenceError("Model not loaded")
        
        if not circuit_breaker.can_execute():
            raise InferenceError("Circuit breaker is open - service temporarily unavailable")
        
        start_time = time.time()
        
        try:
            # Check cache first (unless health check)
            if not health_check:
                cached_result = model_cache.get(input_data)
                if cached_result:
                    metrics_collector.record_cache_hit()
                    logger.debug("Cache hit for inference request")
                    return cached_result
                else:
                    metrics_collector.record_cache_miss()
            
            # Validate input dimensions
            if not input_data or not isinstance(input_data, list):
                raise ValidationError("Input must be a non-empty list")
            
            batch_size = len(input_data)
            if batch_size > 64:
                raise ValidationError(f"Batch size too large: {batch_size} (max: 64)")
            
            # Simulate inference
            await asyncio.sleep(0.01 * batch_size)  # Simulate processing time
            
            # Generate mock predictions
            predictions = [[0.1, 0.2, 0.7] for _ in input_data]
            
            inference_time = time.time() - start_time
            
            # Record metrics
            metrics_collector.record_request(inference_time * 1000)  # Convert to ms
            log_performance_metric("inference_time", inference_time * 1000, "ms")
            log_performance_metric("batch_size", batch_size, "count")
            
            # Cache result (unless health check)
            if not health_check:
                model_cache.put(input_data, predictions)
            
            # Record success in circuit breaker
            circuit_breaker.record_success()
            
            logger.debug(f"Inference completed in {inference_time:.3f}s for batch size {batch_size}")
            
            return predictions
            
        except ValidationError:
            # Don't retry validation errors
            circuit_breaker.record_failure()
            raise
            
        except Exception as e:
            circuit_breaker.record_failure()
            
            # Handle with error manager
            error_context = error_manager.handle_error(e, {
                'batch_size': len(input_data) if input_data else 0,
                'operation': 'inference'
            })
            
            raise InferenceError(
                f"Inference failed: {str(e)}",
                batch_size=len(input_data) if input_data else 0,
                input_shape=f"{len(input_data)}x{len(input_data[0]) if input_data and input_data[0] else 0}"
            )


class RobustAPIHandler:
    """API handler with comprehensive security and error handling."""
    
    def __init__(self, model_loader: RobustModelLoader):
        self.model_loader = model_loader
        self.active_requests = 0
        self.max_concurrent_requests = 100
        self._request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
    
    async def predict(self, request_data: Dict[str, Any], client_info: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction request with comprehensive validation and security."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        client_ip = client_info.get('client_ip', 'unknown')
        user_agent = client_info.get('user_agent', 'unknown')
        
        # Increment active requests counter
        metrics_collector.increment_concurrent()
        
        try:
            async with self._request_semaphore:
                return await self._process_request(request_data, request_id, client_ip, user_agent, start_time)
        finally:
            metrics_collector.decrement_concurrent()
    
    async def _process_request(self, request_data: Dict[str, Any], request_id: str, 
                             client_ip: str, user_agent: str, start_time: float) -> Dict[str, Any]:
        """Process individual request with security checks."""
        
        try:
            # Security checks
            await self._security_checks(request_data, client_ip, user_agent, request_id)
            
            # Validate request structure
            input_data = self._validate_request(request_data)
            
            # Load model if not already loaded
            if not self.model_loader.model_session:
                await self.model_loader.load_model()
            
            # Run inference
            predictions = await self.model_loader.predict(input_data)
            
            inference_time_ms = (time.time() - start_time) * 1000
            
            # Log successful request
            log_api_request(
                method="POST",
                endpoint="/v1/predict",
                status_code=200,
                duration_ms=inference_time_ms,
                ip_address=client_ip,
                user_agent=user_agent,
                request_id=request_id
            )
            
            return {
                'predictions': predictions,
                'inference_time_ms': inference_time_ms,
                'request_id': request_id,
                'model_version': '1.0.0',
                'cache_hit': False  # Would be determined by model_cache
            }
            
        except ValidationError as e:
            # Log validation error
            log_security_event(
                event_type="validation_error",
                message=f"Input validation failed: {str(e)}",
                ip_address=client_ip,
                request_id=request_id
            )
            
            raise {
                'error': 'Validation Error',
                'message': str(e),
                'request_id': request_id,
                'timestamp': time.time()
            }
            
        except SecurityError as e:
            # Log security event
            log_security_event(
                event_type="security_violation",
                message=f"Security violation: {str(e)}",
                ip_address=client_ip,
                request_id=request_id,
                level=logging.CRITICAL
            )
            
            # Block IP for repeated security violations
            if threat_detector.is_under_attack(client_ip):
                ip_blocklist.block_ip(client_ip, duration_minutes=60)
            
            raise {
                'error': 'Security Error',
                'message': 'Access denied',
                'request_id': request_id,
                'timestamp': time.time()
            }
            
        except InferenceError as e:
            # Log inference error
            logger.error(f"Inference error for request {request_id}: {e}")
            
            raise {
                'error': 'Inference Error',
                'message': 'Model inference failed',
                'request_id': request_id,
                'timestamp': time.time()
            }
            
        except Exception as e:
            # Handle unexpected errors
            error_context = error_manager.handle_error(e, {
                'request_id': request_id,
                'client_ip': client_ip,
                'operation': 'api_request'
            })
            
            logger.error(f"Unexpected error in request {request_id}: {e}")
            
            raise {
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred',
                'request_id': request_id,
                'error_id': error_context.error_id,
                'timestamp': time.time()
            }
    
    async def _security_checks(self, request_data: Dict[str, Any], client_ip: str, 
                             user_agent: str, request_id: str):
        """Comprehensive security validation."""
        
        # Check IP blocklist
        if ip_blocklist.is_blocked(client_ip):
            log_security_event(
                event_type="blocked_ip_attempt",
                message=f"Request from blocked IP: {client_ip}",
                ip_address=client_ip,
                request_id=request_id
            )
            raise SecurityError("IP address is blocked", client_ip=client_ip)
        
        # Rate limiting
        allowed, retry_after = rate_limiter.is_allowed(client_ip)
        if not allowed:
            log_security_event(
                event_type="rate_limit_exceeded",
                message=f"Rate limit exceeded for IP: {client_ip}",
                ip_address=client_ip,
                request_id=request_id
            )
            raise SecurityError(f"Rate limit exceeded. Retry after {retry_after} seconds", client_ip=client_ip)
        
        # Check for suspicious activity
        if rate_limiter.is_suspicious_activity(client_ip):
            log_security_event(
                event_type="suspicious_activity",
                message=f"Suspicious activity detected from IP: {client_ip}",
                ip_address=client_ip,
                request_id=request_id
            )
            threat_detector.record_failed_attempt(client_ip, "suspicious_pattern")
        
        # Validate user agent
        if not user_agent or len(user_agent) > 500:
            log_security_event(
                event_type="invalid_user_agent",
                message=f"Invalid user agent: {user_agent[:100]}",
                ip_address=client_ip,
                request_id=request_id
            )
            raise SecurityError("Invalid user agent", client_ip=client_ip)
        
        # Content analysis
        content_str = json.dumps(request_data)
        analysis = threat_detector.analyze_request_content(content_str)
        
        # Check for high-risk content
        risk_score = (
            analysis['suspicious_keywords'] * 2 +
            analysis['encoded_content'] * 3 +
            analysis['sql_patterns'] * 5 +
            analysis['script_patterns'] * 5
        )
        
        if risk_score > 10:
            log_security_event(
                event_type="high_risk_content",
                message=f"High-risk content detected (score: {risk_score}): {analysis}",
                ip_address=client_ip,
                request_id=request_id,
                level=logging.WARNING
            )
            # Don't block immediately, but record the attempt
            threat_detector.record_failed_attempt(client_ip, "high_risk_content")
    
    def _validate_request(self, request_data: Dict[str, Any]) -> List[List[float]]:
        """Validate and sanitize request data."""
        
        # Check required fields
        if 'input' not in request_data:
            raise ValidationError("Missing required field 'input'", field="input")
        
        input_data = request_data['input']
        
        # Type validation
        if not isinstance(input_data, list):
            raise ValidationError("Input must be a list", field="input", value=type(input_data).__name__)
        
        if not input_data:
            raise ValidationError("Input cannot be empty", field="input")
        
        # Batch size validation
        if len(input_data) > 64:
            raise ValidationError(f"Batch size too large: {len(input_data)} (max: 64)", 
                                field="batch_size", value=len(input_data))
        
        # Validate each item in batch
        processed_input = []
        for i, item in enumerate(input_data):
            if not isinstance(item, list):
                raise ValidationError(f"Input item {i} must be a list", 
                                    field=f"input[{i}]", value=type(item).__name__)
            
            # Validate numeric values
            processed_item = []
            for j, value in enumerate(item):
                if not isinstance(value, (int, float)):
                    raise ValidationError(f"Input item [{i}][{j}] must be numeric", 
                                        field=f"input[{i}][{j}]", value=type(value).__name__)
                
                # Check for NaN or infinity
                if isinstance(value, float):
                    import math
                    if math.isnan(value) or math.isinf(value):
                        raise ValidationError(f"Invalid numeric value at [{i}][{j}]", 
                                            field=f"input[{i}][{j}]", value=str(value))
                
                processed_item.append(float(value))
            
            # Check input dimensions consistency
            if i == 0:
                expected_length = len(processed_item)
            elif len(processed_item) != expected_length:
                raise ValidationError(f"Input dimension mismatch at item {i}: expected {expected_length}, got {len(processed_item)}", 
                                    field=f"input[{i}]")
            
            processed_input.append(processed_item)
        
        return processed_input
    
    async def health_check(self, client_info: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive health check endpoint."""
        client_ip = client_info.get('client_ip', 'unknown')
        
        # Basic security check (no rate limiting for health checks)
        if ip_blocklist.is_blocked(client_ip):
            return {
                'status': 'blocked',
                'message': 'IP address is blocked',
                'timestamp': time.time()
            }
        
        try:
            # Check model health
            model_health = await self.model_loader.health_check()
            
            # Get system metrics
            metrics = metrics_collector.get_metrics()
            
            # Get error statistics
            from .error_handling import get_system_health
            system_health = get_system_health()
            
            # Get cache statistics
            cache_stats = model_cache.get_hit_rate()
            
            overall_status = "healthy"
            if not model_health['status'] == 'healthy':
                overall_status = "unhealthy"
            elif system_health['status'] in ['critical', 'degraded']:
                overall_status = system_health['status']
            
            return {
                'status': overall_status,
                'timestamp': time.time(),
                'model': model_health,
                'performance': {
                    'latency_p50': metrics.latency_p50,
                    'latency_p95': metrics.latency_p95,
                    'throughput_rps': metrics.throughput_rps,
                    'cache_hit_rate': cache_stats,
                    'concurrent_requests': metrics.concurrent_requests
                },
                'system': system_health,
                'version': '1.0.0'
            }
            
        except Exception as e:
            error_context = error_manager.handle_error(e, {
                'client_ip': client_ip,
                'operation': 'health_check'
            })
            
            return {
                'status': 'unhealthy',
                'error': str(e),
                'error_id': error_context.error_id,
                'timestamp': time.time()
            }


# Initialize robust components
def create_robust_service(model_path: str) -> RobustAPIHandler:
    """Create a robust NIM service with comprehensive error handling."""
    
    # Initialize model loader
    model_loader = RobustModelLoader(model_path)
    
    # Create API handler
    api_handler = RobustAPIHandler(model_loader)
    
    logger.info(f"Created robust NIM service for model: {model_path}")
    
    return api_handler