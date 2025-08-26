"""Enhanced error handling system with comprehensive recovery mechanisms."""

import asyncio
import logging
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Type
from enum import Enum
import json


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    CIRCUIT_BREAKER = "circuit_breaker"
    MODEL_INFERENCE = "model_inference"
    SYSTEM_RESOURCE = "system_resource"
    NETWORK = "network"
    DATABASE = "database"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class CustomException(Exception):
    """Enhanced custom exception with detailed context."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
        recoverable: bool = True
    ):
        self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.request_id = request_id or str(uuid.uuid4())
        self.context = context or {}
        self.original_error = original_error
        self.recoverable = recoverable
        self.timestamp = datetime.utcnow()
        self.error_id = str(uuid.uuid4())
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "error_id": self.error_id,
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "recoverable": self.recoverable,
            "context": self.context,
            "original_error": str(self.original_error) if self.original_error else None
        }
    
    def to_json(self) -> str:
        """Convert exception to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def __init__(self, name: str, max_attempts: int = 3, delay: float = 1.0):
        self.name = name
        self.max_attempts = max_attempts
        self.delay = delay
    
    async def attempt_recovery(
        self, 
        error: Exception, 
        context: Dict[str, Any], 
        attempt: int
    ) -> Optional[Any]:
        """Attempt to recover from the error."""
        raise NotImplementedError("Subclasses must implement attempt_recovery")
    
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if this strategy can recover from the given error."""
        raise NotImplementedError("Subclasses must implement can_recover")


class RetryStrategy(ErrorRecoveryStrategy):
    """Simple retry strategy with exponential backoff."""
    
    def __init__(self, name: str = "retry", max_attempts: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
        super().__init__(name, max_attempts, initial_delay)
        self.backoff_factor = backoff_factor
    
    async def attempt_recovery(
        self, 
        error: Exception, 
        context: Dict[str, Any], 
        attempt: int
    ) -> Optional[Any]:
        """Retry the original operation with exponential backoff."""
        if attempt >= self.max_attempts:
            return None
        
        # Calculate delay with exponential backoff
        delay = self.delay * (self.backoff_factor ** attempt)
        await asyncio.sleep(delay)
        
        # If there's a retry_function in context, call it
        if "retry_function" in context:
            try:
                return await context["retry_function"]()
            except Exception as retry_error:
                context["last_retry_error"] = retry_error
                raise retry_error
        
        return None
    
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if retry is applicable."""
        # Don't retry for validation errors or authentication errors
        if isinstance(error, CustomException):
            return error.category not in [
                ErrorCategory.VALIDATION,
                ErrorCategory.AUTHENTICATION,
                ErrorCategory.AUTHORIZATION
            ]
        
        # Retry for network errors, timeouts, etc.
        return isinstance(error, (ConnectionError, TimeoutError, asyncio.TimeoutError))


class FallbackStrategy(ErrorRecoveryStrategy):
    """Fallback strategy using alternative implementations."""
    
    def __init__(self, name: str = "fallback", fallback_function: Optional[Callable] = None):
        super().__init__(name, max_attempts=1)
        self.fallback_function = fallback_function
    
    async def attempt_recovery(
        self, 
        error: Exception, 
        context: Dict[str, Any], 
        attempt: int
    ) -> Optional[Any]:
        """Use fallback implementation."""
        if self.fallback_function:
            try:
                return await self.fallback_function(context)
            except Exception as fallback_error:
                context["fallback_error"] = fallback_error
                raise fallback_error
        
        # Check for fallback in context
        if "fallback_function" in context:
            try:
                return await context["fallback_function"](context)
            except Exception as fallback_error:
                context["fallback_error"] = fallback_error
                raise fallback_error
        
        return None
    
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if fallback is available."""
        return (
            self.fallback_function is not None or 
            "fallback_function" in context
        )


class CircuitBreakerStrategy(ErrorRecoveryStrategy):
    """Circuit breaker error recovery strategy."""
    
    def __init__(self, name: str = "circuit_breaker"):
        super().__init__(name, max_attempts=1)
    
    async def attempt_recovery(
        self, 
        error: Exception, 
        context: Dict[str, Any], 
        attempt: int
    ) -> Optional[Any]:
        """Handle circuit breaker state."""
        # Return cached response if available
        if "cached_response" in context:
            return context["cached_response"]
        
        # Return degraded response if available
        if "degraded_response_function" in context:
            try:
                return await context["degraded_response_function"](context)
            except Exception as degraded_error:
                context["degraded_error"] = degraded_error
                raise degraded_error
        
        return None
    
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if circuit breaker recovery is possible."""
        return (
            "cached_response" in context or 
            "degraded_response_function" in context
        )


class ErrorHandler:
    """Comprehensive error handling system with recovery strategies."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_handlers: Dict[Type[Exception], List[Callable]] = {}
        self.recovery_strategies: List[ErrorRecoveryStrategy] = []
        self.error_stats: Dict[str, int] = {}
        self.recent_errors: List[Dict[str, Any]] = []
        self.max_recent_errors = 1000
        
        # Add default recovery strategies
        self.add_recovery_strategy(RetryStrategy())
        self.add_recovery_strategy(FallbackStrategy())
        self.add_recovery_strategy(CircuitBreakerStrategy())
    
    def add_handler(self, exception_type: Type[Exception], handler: Callable):
        """Add a specific error handler for an exception type."""
        if exception_type not in self.error_handlers:
            self.error_handlers[exception_type] = []
        self.error_handlers[exception_type].append(handler)
    
    def add_recovery_strategy(self, strategy: ErrorRecoveryStrategy):
        """Add a recovery strategy."""
        self.recovery_strategies.append(strategy)
    
    async def handle_error(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Handle an error with comprehensive recovery mechanisms."""
        context = context or {}
        error_id = str(uuid.uuid4())
        
        # Create or enhance custom exception
        if not isinstance(error, CustomException):
            custom_error = self._create_custom_exception(error, context)
        else:
            custom_error = error
        
        # Log the error
        self._log_error(custom_error, context, error_id)
        
        # Record error statistics
        self._record_error_stats(custom_error)
        
        # Store recent error
        self._store_recent_error(custom_error, context, error_id)
        
        # Try specific handlers first
        recovery_result = await self._try_specific_handlers(error, context)
        if recovery_result is not None:
            return recovery_result
        
        # Try recovery strategies
        if custom_error.recoverable:
            recovery_result = await self._try_recovery_strategies(error, context)
            if recovery_result is not None:
                return recovery_result
        
        # If no recovery was possible, raise the original or custom error
        raise custom_error
    
    def _create_custom_exception(
        self, 
        error: Exception, 
        context: Dict[str, Any]
    ) -> CustomException:
        """Create a CustomException from a regular exception."""
        error_category = self._classify_error(error)
        error_severity = self._assess_severity(error, error_category)
        error_code = self._generate_error_code(error, error_category)
        
        return CustomException(
            message=str(error),
            error_code=error_code,
            category=error_category,
            severity=error_severity,
            request_id=context.get("request_id"),
            context=context,
            original_error=error,
            recoverable=self._is_recoverable(error, error_category)
        )
    
    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify error into appropriate category."""
        error_type = type(error)
        error_msg = str(error).lower()
        
        # Classification rules
        if error_type in [ValueError, TypeError]:
            return ErrorCategory.VALIDATION
        elif "authentication" in error_msg or "unauthorized" in error_msg:
            return ErrorCategory.AUTHENTICATION
        elif "forbidden" in error_msg or "permission" in error_msg:
            return ErrorCategory.AUTHORIZATION
        elif "rate limit" in error_msg or "too many requests" in error_msg:
            return ErrorCategory.RATE_LIMIT
        elif "circuit breaker" in error_msg:
            return ErrorCategory.CIRCUIT_BREAKER
        elif "inference" in error_msg or "model" in error_msg:
            return ErrorCategory.MODEL_INFERENCE
        elif error_type in [MemoryError, OSError]:
            return ErrorCategory.SYSTEM_RESOURCE
        elif error_type in [ConnectionError, TimeoutError]:
            return ErrorCategory.NETWORK
        elif "config" in error_msg or "setting" in error_msg:
            return ErrorCategory.CONFIGURATION
        else:
            return ErrorCategory.UNKNOWN
    
    def _assess_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Assess error severity."""
        # Critical errors
        if isinstance(error, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        
        # High severity by category
        if category in [ErrorCategory.SYSTEM_RESOURCE, ErrorCategory.MODEL_INFERENCE]:
            return ErrorSeverity.HIGH
        
        # Medium severity
        if category in [ErrorCategory.NETWORK, ErrorCategory.CIRCUIT_BREAKER]:
            return ErrorSeverity.MEDIUM
        
        # Low severity
        return ErrorSeverity.LOW
    
    def _generate_error_code(self, error: Exception, category: ErrorCategory) -> str:
        """Generate error code."""
        error_type_name = type(error).__name__.upper()
        category_prefix = category.value.upper()
        return f"{category_prefix}_{error_type_name}"
    
    def _is_recoverable(self, error: Exception, category: ErrorCategory) -> bool:
        """Determine if error is recoverable."""
        # Non-recoverable categories
        non_recoverable = [
            ErrorCategory.VALIDATION,
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.AUTHORIZATION
        ]
        
        if category in non_recoverable:
            return False
        
        # Non-recoverable error types
        if isinstance(error, (SyntaxError, ImportError)):
            return False
        
        return True
    
    def _log_error(
        self, 
        error: CustomException, 
        context: Dict[str, Any], 
        error_id: str
    ):
        """Log error with appropriate level."""
        log_data = {
            "error_id": error_id,
            "error_code": error.error_code,
            "category": error.category.value,
            "severity": error.severity.value,
            "request_id": error.request_id,
            "context": context,
            "traceback": traceback.format_exc() if error.original_error else None
        }
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"Critical error: {error.message}", extra=log_data)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(f"High severity error: {error.message}", extra=log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"Medium severity error: {error.message}", extra=log_data)
        else:
            self.logger.info(f"Low severity error: {error.message}", extra=log_data)
    
    def _record_error_stats(self, error: CustomException):
        """Record error statistics."""
        error_key = f"{error.category.value}_{error.error_code}"
        self.error_stats[error_key] = self.error_stats.get(error_key, 0) + 1
    
    def _store_recent_error(
        self, 
        error: CustomException, 
        context: Dict[str, Any], 
        error_id: str
    ):
        """Store recent error for analysis."""
        error_record = {
            "error_id": error_id,
            "timestamp": error.timestamp.isoformat(),
            "error_code": error.error_code,
            "category": error.category.value,
            "severity": error.severity.value,
            "message": error.message,
            "request_id": error.request_id,
            "context_keys": list(context.keys())  # Store only keys to avoid memory issues
        }
        
        self.recent_errors.append(error_record)
        
        # Keep only recent errors
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors = self.recent_errors[-self.max_recent_errors:]
    
    async def _try_specific_handlers(
        self, 
        error: Exception, 
        context: Dict[str, Any]
    ) -> Optional[Any]:
        """Try specific error handlers."""
        error_type = type(error)
        
        if error_type in self.error_handlers:
            for handler in self.error_handlers[error_type]:
                try:
                    result = handler(error, context)
                    if asyncio.iscoroutine(result):
                        result = await result
                    
                    if result is not None:
                        self.logger.info(f"Error recovered using specific handler: {handler.__name__}")
                        return result
                        
                except Exception as handler_error:
                    self.logger.warning(
                        f"Specific handler {handler.__name__} failed: {handler_error}"
                    )
                    continue
        
        return None
    
    async def _try_recovery_strategies(
        self, 
        error: Exception, 
        context: Dict[str, Any]
    ) -> Optional[Any]:
        """Try recovery strategies in order."""
        for strategy in self.recovery_strategies:
            if not strategy.can_recover(error, context):
                continue
            
            self.logger.info(f"Attempting recovery with strategy: {strategy.name}")
            
            for attempt in range(strategy.max_attempts):
                try:
                    result = await strategy.attempt_recovery(error, context, attempt)
                    if result is not None:
                        self.logger.info(
                            f"Error recovered using strategy {strategy.name} "
                            f"(attempt {attempt + 1}/{strategy.max_attempts})"
                        )
                        return result
                        
                except Exception as recovery_error:
                    self.logger.warning(
                        f"Recovery strategy {strategy.name} attempt {attempt + 1} failed: "
                        f"{recovery_error}"
                    )
                    
                    if attempt == strategy.max_attempts - 1:
                        # Last attempt failed
                        context[f"{strategy.name}_last_error"] = recovery_error
                        break
                    
                    continue
        
        return None
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "total_errors": sum(self.error_stats.values()),
            "error_breakdown": dict(self.error_stats),
            "recent_errors_count": len(self.recent_errors),
            "recovery_strategies": [s.name for s in self.recovery_strategies]
        }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors."""
        return self.recent_errors[-limit:] if limit > 0 else self.recent_errors
    
    def clear_error_stats(self):
        """Clear error statistics (useful for testing)."""
        self.error_stats.clear()
        self.recent_errors.clear()


class ErrorReporter:
    """Error reporting system for external monitoring."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self.logger = logging.getLogger(__name__)
    
    async def report_error(self, error: CustomException, context: Dict[str, Any]):
        """Report error to external systems."""
        try:
            # Report to webhook if configured
            if self.webhook_url:
                await self._send_webhook_notification(error, context)
            
            # Report to monitoring system
            await self._send_to_monitoring(error, context)
            
        except Exception as report_error:
            self.logger.error(f"Failed to report error: {report_error}")
    
    async def _send_webhook_notification(
        self, 
        error: CustomException, 
        context: Dict[str, Any]
    ):
        """Send error notification to webhook."""
        import aiohttp
        
        payload = {
            "error_id": error.error_id,
            "timestamp": error.timestamp.isoformat(),
            "severity": error.severity.value,
            "category": error.category.value,
            "error_code": error.error_code,
            "message": error.message,
            "request_id": error.request_id,
            "service": context.get("service_name", "nimify-service")
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url, 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        self.logger.debug("Error webhook notification sent successfully")
                    else:
                        self.logger.warning(f"Webhook notification failed: {response.status}")
                        
        except Exception as webhook_error:
            self.logger.error(f"Webhook notification error: {webhook_error}")
    
    async def _send_to_monitoring(self, error: CustomException, context: Dict[str, Any]):
        """Send error to monitoring system (e.g., Prometheus)."""
        # This would integrate with your monitoring system
        # For now, we'll just log it
        self.logger.info(f"Monitoring alert: {error.error_code} - {error.message}")


# Create global error handler instance
global_error_handler = ErrorHandler()