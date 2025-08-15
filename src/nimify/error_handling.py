"""Comprehensive error handling and recovery mechanisms."""

import logging
import traceback
import time
from typing import Any, Dict, Optional, Callable
from functools import wraps
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error category types."""
    VALIDATION = "validation"
    INFRASTRUCTURE = "infrastructure"
    MODEL = "model"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    EXTERNAL = "external"


class NimifyError(Exception):
    """Base exception for Nimify errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.recovery_suggestions = recovery_suggestions
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_msg": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "details": self.details,
            "recovery_suggestions": self.recovery_suggestions,
            "timestamp": self.timestamp
        }


class ValidationError(NimifyError):
    """Validation-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class ModelError(NimifyError):
    """Model loading or inference errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.MODEL,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class SecurityError(NimifyError):
    """Security-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )


class ConfigurationError(NimifyError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class InfrastructureError(NimifyError):
    """Infrastructure-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.INFRASTRUCTURE,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class ErrorHandler:
    """Centralized error handling with recovery mechanisms."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts = {}
        self.recovery_strategies = {}
        
    def register_recovery_strategy(
        self,
        error_type: type,
        strategy: Callable[[Exception], Any]
    ):
        """Register a recovery strategy for specific error type."""
        self.recovery_strategies[error_type] = strategy
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        attempt_recovery: bool = True
    ) -> Optional[Any]:
        """Handle error with optional recovery attempt."""
        context = context or {}
        
        # Log error with context
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "error_traceback": traceback.format_exc()
        }
        
        if isinstance(error, NimifyError):
            error_data.update(error.to_dict())
            
        self.logger.error(f"Error handled: {error}", extra=error_data)
        
        # Track error counts
        error_type_name = type(error).__name__
        self.error_counts[error_type_name] = self.error_counts.get(error_type_name, 0) + 1
        
        # Attempt recovery if enabled and strategy exists
        if attempt_recovery and type(error) in self.recovery_strategies:
            try:
                recovery_result = self.recovery_strategies[type(error)](error)
                self.logger.info(f"Recovery successful for {error_type_name}")
                return recovery_result
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed for {error_type_name}: {recovery_error}")
        
        return None
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics."""
        return self.error_counts.copy()


def with_error_handling(
    error_handler: ErrorHandler,
    reraise_on_failure: bool = True,
    context_provider: Optional[Callable] = None
):
    """Decorator for automatic error handling."""
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {}
                if context_provider:
                    try:
                        context = context_provider(*args, **kwargs)
                    except:
                        pass
                
                recovery_result = error_handler.handle_error(e, context=context)
                
                if recovery_result is not None:
                    return recovery_result
                elif reraise_on_failure:
                    raise
                else:
                    return None
                    
        return wrapper
    return decorator


class RetryableError(NimifyError):
    """Error that can be retried."""
    
    def __init__(self, message: str, max_retries: int = 3, **kwargs):
        super().__init__(message=message, **kwargs)
        self.max_retries = max_retries


def retry_on_error(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    exponential_backoff: bool = True,
    retryable_exceptions: tuple = (RetryableError,)
):
    """Decorator for automatic retry on specific exceptions."""
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = retry_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    logging.getLogger(__name__).warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {e}"
                    )
                    
                    time.sleep(current_delay)
                    
                    if exponential_backoff:
                        current_delay *= 2
            
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


# Global error handler instance
global_error_handler = ErrorHandler()


# Recovery strategies
def model_loading_recovery(error: Exception) -> Optional[Any]:
    """Recovery strategy for model loading errors."""
    if "CUDA" in str(error):
        logging.getLogger(__name__).info("CUDA error detected, falling back to CPU")
        # Would attempt CPU fallback here
        return None
    return None


def infrastructure_recovery(error: Exception) -> Optional[Any]:
    """Recovery strategy for infrastructure errors."""
    if "connection" in str(error).lower():
        logging.getLogger(__name__).info("Connection error detected, implementing backoff")
        time.sleep(5)  # Simple backoff
        return None
    return None


# Register default recovery strategies
global_error_handler.register_recovery_strategy(ModelError, model_loading_recovery)
global_error_handler.register_recovery_strategy(InfrastructureError, infrastructure_recovery)