"""Robust core functionality with comprehensive error handling and reliability features."""

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import threading
import hashlib

from .core import ModelConfig, Nimifier, NIMService
from .circuit_breaker import CircuitBreaker
from .rate_limiter import RateLimiter
from .caching_system import CacheManager
from .monitoring import MetricsCollector, HealthChecker
from .security import SecurityManager, TokenValidator
from .error_handling import ErrorHandler, CustomException
from .validation import ServiceValidator, ModelValidator


class RobustModelConfig(ModelConfig):
    """Enhanced model configuration with robustness features."""
    
    # Reliability settings
    retry_attempts: int = 3
    retry_delay: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    health_check_interval: int = 30
    
    # Security settings
    api_key_required: bool = False
    rate_limit_per_minute: int = 1000
    max_request_size_mb: int = 100
    allowed_origins: List[str] = None
    
    # Monitoring settings
    enable_detailed_metrics: bool = True
    enable_request_tracing: bool = True
    log_level: str = "INFO"
    
    # Performance settings
    request_timeout: int = 30
    max_concurrent_requests: int = 100
    memory_limit_mb: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.allowed_origins is None:
            self.allowed_origins = ["*"]


class RobustNimifier(Nimifier):
    """Enhanced Nimifier with robust error handling and monitoring."""
    
    def __init__(self, config: RobustModelConfig, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.config: RobustModelConfig = config
        
        # Initialize robust components
        self.error_handler = ErrorHandler()
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker()
        self.security_manager = SecurityManager()
        self.model_validator = ModelValidator()
        
        # Setup monitoring
        self._setup_monitoring()
        self._setup_error_tracking()
    
    def _setup_monitoring(self):
        """Setup comprehensive monitoring."""
        self.health_checker.add_check("model_loading", self._check_model_health)
        self.health_checker.add_check("memory_usage", self._check_memory_usage)
        self.health_checker.add_check("disk_space", self._check_disk_space)
        
        # Start health monitoring thread
        self.health_thread = threading.Thread(
            target=self._run_health_monitoring, 
            daemon=True
        )
        self.health_thread.start()
    
    def _setup_error_tracking(self):
        """Setup error tracking and recovery."""
        self.error_handler.add_handler(FileNotFoundError, self._handle_file_not_found)
        self.error_handler.add_handler(MemoryError, self._handle_memory_error)
        self.error_handler.add_handler(Exception, self._handle_generic_error)
    
    def _check_model_health(self) -> Dict[str, Any]:
        """Check model health status."""
        return {
            "status": "healthy",
            "last_check": datetime.utcnow().isoformat(),
            "model_loaded": True  # This would check actual model status
        }
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            return {
                "status": "healthy" if memory_percent < 90 else "warning",
                "memory_mb": memory_info.rss / 1024 / 1024,
                "memory_percent": memory_percent,
                "available_mb": psutil.virtual_memory().available / 1024 / 1024
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space availability."""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_percent = (free / total) * 100
            
            return {
                "status": "healthy" if free_percent > 10 else "warning",
                "free_gb": free / (1024**3),
                "total_gb": total / (1024**3),
                "free_percent": free_percent
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _run_health_monitoring(self):
        """Run continuous health monitoring."""
        while True:
            try:
                health_results = self.health_checker.run_all_checks()
                self.metrics_collector.record_health_status(health_results)
                
                # Log critical health issues
                for check_name, result in health_results.items():
                    if result.get("status") == "error":
                        self.logger.error(f"Health check failed: {check_name} - {result}")
                    elif result.get("status") == "warning":
                        self.logger.warning(f"Health check warning: {check_name} - {result}")
                
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(self.config.health_check_interval)
    
    def _handle_file_not_found(self, error: FileNotFoundError, context: Dict[str, Any]) -> Any:
        """Handle file not found errors with recovery."""
        self.logger.error(f"File not found: {error}, Context: {context}")
        
        # Attempt recovery based on context
        if "model_path" in context:
            # Try to find model in alternative locations
            alternative_paths = [
                "/models/",
                "./models/",
                os.path.expanduser("~/models/")
            ]
            
            model_filename = Path(context["model_path"]).name
            for alt_path in alternative_paths:
                alt_model_path = Path(alt_path) / model_filename
                if alt_model_path.exists():
                    self.logger.info(f"Found model at alternative location: {alt_model_path}")
                    return str(alt_model_path)
        
        raise error
    
    def _handle_memory_error(self, error: MemoryError, context: Dict[str, Any]) -> Any:
        """Handle memory errors with cleanup."""
        self.logger.error(f"Memory error occurred: {error}")
        
        # Attempt memory cleanup
        import gc
        gc.collect()
        
        # Reduce batch size if possible
        if "batch_size" in context:
            new_batch_size = max(1, context["batch_size"] // 2)
            self.logger.info(f"Reducing batch size from {context['batch_size']} to {new_batch_size}")
            context["batch_size"] = new_batch_size
            return context
        
        raise error
    
    def _handle_generic_error(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Handle generic errors with logging."""
        error_id = str(uuid.uuid4())
        self.logger.error(
            f"Generic error [{error_id}]: {type(error).__name__}: {error}",
            extra={"context": context, "error_id": error_id}
        )
        
        # Record error metrics
        self.metrics_collector.record_error(
            error_type=type(error).__name__,
            error_message=str(error),
            context=context
        )
        
        raise error
    
    async def wrap_model_robust(
        self,
        model_path: str,
        input_schema: Dict[str, str],
        output_schema: Dict[str, str],
        preprocessing_config: Optional[Dict[str, Any]] = None,
        postprocessing_config: Optional[Dict[str, Any]] = None
    ) -> 'RobustNIMService':
        """Wrap a model with comprehensive robustness features."""
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting robust model wrapping [{operation_id}]: {model_path}")
            
            # Validate model path with retries
            model_path = await self._validate_model_path_with_retry(model_path)
            
            # Validate model format and structure
            self.model_validator.validate_model_file(model_path)
            
            # Validate schemas
            ServiceValidator.validate_schemas(input_schema, output_schema)
            
            # Create robust service
            service = RobustNIMService(
                config=self.config,
                model_path=model_path,
                input_schema=input_schema,
                output_schema=output_schema,
                preprocessing_config=preprocessing_config or {},
                postprocessing_config=postprocessing_config or {},
                logger=self.logger,
                error_handler=self.error_handler,
                metrics_collector=self.metrics_collector
            )
            
            # Initialize service components
            await service.initialize()
            
            duration = time.time() - start_time
            self.metrics_collector.record_operation_duration("model_wrapping", duration)
            self.logger.info(f"Model wrapping completed [{operation_id}] in {duration:.2f}s")
            
            return service
            
        except Exception as e:
            duration = time.time() - start_time
            context = {
                "operation_id": operation_id,
                "model_path": model_path,
                "duration": duration
            }
            
            # Use error handler
            try:
                return await self.error_handler.handle_error(e, context)
            except Exception:
                self.logger.error(
                    f"Model wrapping failed [{operation_id}]: {e}",
                    extra={"context": context}
                )
                raise
    
    async def _validate_model_path_with_retry(self, model_path: str) -> str:
        """Validate model path with retry logic."""
        for attempt in range(self.config.retry_attempts):
            try:
                validated_path = self._validate_model_path(model_path)
                return validated_path
            except FileNotFoundError as e:
                if attempt < self.config.retry_attempts - 1:
                    self.logger.warning(
                        f"Model path validation failed (attempt {attempt + 1}), retrying..."
                    )
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise e
    
    def _validate_model_path(self, model_path: str) -> str:
        """Validate and resolve model path."""
        path_obj = Path(model_path)
        
        # Check if file exists
        if not path_obj.exists():
            # Try error handler for recovery
            context = {"model_path": model_path}
            try:
                recovered_path = self.error_handler.handle_error(
                    FileNotFoundError(f"Model file not found: {model_path}"),
                    context
                )
                return recovered_path
            except FileNotFoundError:
                raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Check file permissions
        if not os.access(path_obj, os.R_OK):
            raise PermissionError(f"Cannot read model file: {model_path}")
        
        # Check file size
        file_size_mb = path_obj.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_request_size_mb:
            raise ValueError(f"Model file too large: {file_size_mb:.2f}MB > {self.config.max_request_size_mb}MB")
        
        return str(path_obj.absolute())


class RobustNIMService(NIMService):
    """Enhanced NIM service with comprehensive robustness features."""
    
    def __init__(
        self,
        config: RobustModelConfig,
        model_path: str,
        input_schema: Dict[str, str],
        output_schema: Dict[str, str],
        preprocessing_config: Optional[Dict[str, Any]] = None,
        postprocessing_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        error_handler: Optional[ErrorHandler] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        super().__init__(
            config, model_path, input_schema, output_schema,
            preprocessing_config, postprocessing_config, logger
        )
        
        self.error_handler = error_handler or ErrorHandler()
        self.metrics_collector = metrics_collector or MetricsCollector()
        
        # Initialize robust components
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            timeout=config.circuit_breaker_timeout
        )
        self.rate_limiter = RateLimiter(
            max_requests=config.rate_limit_per_minute,
            window_seconds=60
        )
        self.cache_manager = CacheManager(
            max_size=1000,
            ttl_seconds=300
        )
        self.security_manager = SecurityManager(
            api_key_required=config.api_key_required,
            allowed_origins=config.allowed_origins
        )
        
        # Service state tracking
        self.is_initialized = False
        self.initialization_time = None
        self.last_health_check = None
        self.error_count = 0
        self.success_count = 0
    
    async def initialize(self):
        """Initialize service with comprehensive setup."""
        if self.is_initialized:
            return
        
        start_time = time.time()
        self.logger.info(f"Initializing robust NIM service: {self.config.name}")
        
        try:
            # Initialize components
            await self._initialize_model_loader()
            await self._initialize_security()
            await self._initialize_monitoring()
            await self._setup_graceful_shutdown()
            
            self.is_initialized = True
            self.initialization_time = time.time() - start_time
            
            self.logger.info(
                f"Service initialized successfully in {self.initialization_time:.2f}s"
            )
            
            # Record initialization metrics
            self.metrics_collector.record_initialization_success(self.initialization_time)
            
        except Exception as e:
            self.logger.error(f"Service initialization failed: {e}")
            self.metrics_collector.record_initialization_failure(str(e))
            raise
    
    async def _initialize_model_loader(self):
        """Initialize model loading with validation."""
        self.logger.debug("Initializing model loader...")
        
        # Validate model file
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Check model format
        model_format = self._detect_model_format()
        self.logger.info(f"Detected model format: {model_format}")
        
        # Initialize model-specific loader
        # This would be implemented based on the actual model format
        self.logger.info("Model loader initialized")
    
    def _detect_model_format(self) -> str:
        """Detect model format with validation."""
        ext = Path(self.model_path).suffix.lower()
        format_map = {
            '.onnx': 'ONNX',
            '.trt': 'TensorRT',
            '.engine': 'TensorRT',
            '.plan': 'TensorRT',
            '.pb': 'TensorFlow',
            '.pt': 'PyTorch',
            '.pth': 'PyTorch'
        }
        
        format_name = format_map.get(ext, 'Unknown')
        if format_name == 'Unknown':
            self.logger.warning(f"Unknown model format: {ext}")
        
        return format_name
    
    async def _initialize_security(self):
        """Initialize security components."""
        self.logger.debug("Initializing security...")
        
        # Setup API key validation if required
        if self.config.api_key_required:
            self.security_manager.setup_api_key_validation()
        
        # Setup CORS validation
        self.security_manager.setup_cors_validation(self.config.allowed_origins)
        
        self.logger.info("Security initialized")
    
    async def _initialize_monitoring(self):
        """Initialize monitoring components."""
        self.logger.debug("Initializing monitoring...")
        
        # Setup metrics collection
        self.metrics_collector.initialize_service_metrics(self.service_id)
        
        # Setup health checking
        self._schedule_health_checks()
        
        self.logger.info("Monitoring initialized")
    
    def _schedule_health_checks(self):
        """Schedule periodic health checks."""
        def run_health_check():
            while self.is_initialized:
                try:
                    self._perform_health_check()
                    time.sleep(self.config.health_check_interval)
                except Exception as e:
                    self.logger.error(f"Health check error: {e}")
                    time.sleep(self.config.health_check_interval)
        
        health_thread = threading.Thread(target=run_health_check, daemon=True)
        health_thread.start()
    
    def _perform_health_check(self):
        """Perform comprehensive health check."""
        health_status = {
            "service_id": self.service_id,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": time.time() - self.created_at.timestamp(),
            "error_rate": self.error_count / max(1, self.success_count + self.error_count),
            "circuit_breaker_state": "open" if self.circuit_breaker.is_open() else "closed",
            "memory_usage": self._get_memory_usage(),
            "cache_status": self._get_cache_status()
        }
        
        self.last_health_check = health_status
        self.metrics_collector.record_health_check(health_status)
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            return {
                "rss_mb": process.memory_info().rss / 1024 / 1024,
                "percent": process.memory_percent()
            }
        except Exception:
            return {"error": "Unable to get memory usage"}
    
    def _get_cache_status(self) -> Dict[str, Any]:
        """Get cache status."""
        return {
            "size": self.cache_manager.size(),
            "hit_rate": self.cache_manager.get_hit_rate(),
            "capacity": self.cache_manager.max_size
        }
    
    async def _setup_graceful_shutdown(self):
        """Setup graceful shutdown handling."""
        import signal
        
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, starting graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def shutdown(self):
        """Gracefully shutdown the service."""
        self.logger.info("Starting graceful shutdown...")
        
        try:
            # Mark service as shutting down
            self.is_initialized = False
            
            # Finish processing existing requests (with timeout)
            timeout = 30  # seconds
            start_time = time.time()
            
            while (time.time() - start_time) < timeout:
                if self.metrics_collector.get_active_requests() == 0:
                    break
                await asyncio.sleep(0.1)
            
            # Clean up resources
            await self._cleanup_resources()
            
            self.logger.info("Graceful shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def _cleanup_resources(self):
        """Clean up service resources."""
        try:
            # Clear cache
            self.cache_manager.clear()
            
            # Close any open connections
            # This would be implemented based on specific resources
            
            self.logger.debug("Resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during resource cleanup: {e}")
    
    async def predict_robust(
        self,
        input_data: List[List[float]],
        request_id: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Make predictions with comprehensive error handling and monitoring."""
        request_id = request_id or str(uuid.uuid4())
        timeout = timeout or self.config.request_timeout
        start_time = time.time()
        
        # Check if service is ready
        if not self.is_initialized:
            raise CustomException(
                "Service not initialized",
                error_code="SERVICE_NOT_READY",
                request_id=request_id
            )
        
        # Check circuit breaker
        if self.circuit_breaker.is_open():
            raise CustomException(
                "Service temporarily unavailable (circuit breaker open)",
                error_code="CIRCUIT_BREAKER_OPEN",
                request_id=request_id
            )
        
        try:
            # Rate limiting check
            client_id = request_id  # In practice, this would be client IP or user ID
            if not self.rate_limiter.allow_request(client_id):
                raise CustomException(
                    "Rate limit exceeded",
                    error_code="RATE_LIMIT_EXCEEDED",
                    request_id=request_id
                )
            
            # Check cache first
            cache_key = self._generate_cache_key(input_data)
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                self.metrics_collector.record_cache_hit()
                return cached_result
            
            self.metrics_collector.record_cache_miss()
            
            # Input validation
            self._validate_input_data(input_data)
            
            # Execute prediction with timeout
            result = await asyncio.wait_for(
                self._execute_prediction(input_data, request_id),
                timeout=timeout
            )
            
            # Cache result
            self.cache_manager.set(cache_key, result)
            
            # Record success metrics
            processing_time = time.time() - start_time
            self.success_count += 1
            self.circuit_breaker.record_success()
            
            self.metrics_collector.record_prediction_success(
                processing_time=processing_time,
                batch_size=len(input_data),
                request_id=request_id
            )
            
            return result
            
        except asyncio.TimeoutError:
            self.error_count += 1
            self.circuit_breaker.record_failure()
            
            error = CustomException(
                f"Request timeout after {timeout}s",
                error_code="REQUEST_TIMEOUT",
                request_id=request_id
            )
            
            self.metrics_collector.record_prediction_error("timeout", request_id)
            raise error
            
        except Exception as e:
            self.error_count += 1
            self.circuit_breaker.record_failure()
            
            # Use error handler for recovery attempts
            try:
                context = {
                    "request_id": request_id,
                    "input_data_shape": [len(input_data), len(input_data[0]) if input_data else 0],
                    "processing_time": time.time() - start_time
                }
                
                recovered_result = await self.error_handler.handle_error(e, context)
                if recovered_result is not None:
                    return recovered_result
                    
            except Exception:
                pass  # Recovery failed, continue with original error handling
            
            self.metrics_collector.record_prediction_error(
                error_type=type(e).__name__,
                request_id=request_id
            )
            
            # Wrap exception with additional context
            if isinstance(e, CustomException):
                raise e
            else:
                raise CustomException(
                    f"Prediction failed: {str(e)}",
                    error_code="PREDICTION_ERROR",
                    request_id=request_id,
                    original_error=e
                )
    
    def _generate_cache_key(self, input_data: List[List[float]]) -> str:
        """Generate cache key for input data."""
        # Convert input to string and hash it
        input_str = json.dumps(input_data, sort_keys=True)
        return hashlib.md5(input_str.encode()).hexdigest()
    
    def _validate_input_data(self, input_data: List[List[float]]):
        """Validate input data with comprehensive checks."""
        if not input_data:
            raise CustomException(
                "Input data cannot be empty",
                error_code="VALIDATION_ERROR"
            )
        
        if len(input_data) > self.config.max_batch_size:
            raise CustomException(
                f"Batch size {len(input_data)} exceeds maximum {self.config.max_batch_size}",
                error_code="BATCH_SIZE_EXCEEDED"
            )
        
        # Check data consistency
        if len(set(len(row) for row in input_data)) > 1:
            raise CustomException(
                "All input rows must have the same length",
                error_code="INCONSISTENT_INPUT_SHAPE"
            )
        
        # Check for invalid values
        for i, row in enumerate(input_data):
            for j, value in enumerate(row):
                if not isinstance(value, (int, float)):
                    raise CustomException(
                        f"Invalid value type at position [{i}][{j}]: {type(value)}",
                        error_code="INVALID_VALUE_TYPE"
                    )
                
                if not (-1e10 < value < 1e10):  # Check for reasonable range
                    raise CustomException(
                        f"Value out of range at position [{i}][{j}]: {value}",
                        error_code="VALUE_OUT_OF_RANGE"
                    )
    
    async def _execute_prediction(
        self, 
        input_data: List[List[float]], 
        request_id: str
    ) -> Dict[str, Any]:
        """Execute the actual prediction logic."""
        # This is a mock implementation
        # In practice, this would use the actual model inference
        
        await asyncio.sleep(0.01 * len(input_data))  # Simulate processing time
        
        predictions = [
            [float(i * 0.1 + j * 0.01) for j in range(len(input_data[0]))]
            for i, row in enumerate(input_data)
        ]
        
        return {
            "predictions": predictions,
            "model_version": self.config.version,
            "processing_time_ms": 10 * len(input_data),  # Mock processing time
            "batch_size": len(input_data),
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return {
            "service_id": self.service_id,
            "name": self.config.name,
            "status": "healthy" if self.is_initialized else "not_ready",
            "uptime_seconds": time.time() - self.created_at.timestamp(),
            "initialization_time": self.initialization_time,
            "last_health_check": self.last_health_check,
            "error_rate": self.error_count / max(1, self.success_count + self.error_count),
            "total_requests": self.success_count + self.error_count,
            "circuit_breaker_state": "open" if self.circuit_breaker.is_open() else "closed",
            "cache_hit_rate": self.cache_manager.get_hit_rate(),
            "version": self.config.version
        }