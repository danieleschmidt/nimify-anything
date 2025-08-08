"""Comprehensive error handling and recovery mechanisms."""

import sys
import traceback
import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
import asyncio

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    MODEL_LOADING = "model_loading"
    INFERENCE = "inference"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"
    DEPLOYMENT = "deployment"
    SECURITY = "security"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    traceback: Optional[str] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3


class NimifyError(Exception):
    """Base exception for Nimify-specific errors."""
    
    def __init__(
        self, 
        message: str, 
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Dict[str, Any] = None,
        cause: Exception = None
    ):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.cause = cause
        self.timestamp = time.time()


class ValidationError(NimifyError):
    """Input validation errors."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        details = {}
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = str(value)[:100]  # Truncate long values
        
        super().__init__(
            message, 
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            details=details
        )


class ModelLoadingError(NimifyError):
    """Model loading and initialization errors."""
    
    def __init__(self, message: str, model_path: str, model_type: str = None):
        details = {'model_path': model_path}
        if model_type:
            details['model_type'] = model_type
        
        super().__init__(
            message,
            category=ErrorCategory.MODEL_LOADING,
            severity=ErrorSeverity.HIGH,
            details=details
        )


class InferenceError(NimifyError):
    """Model inference errors."""
    
    def __init__(self, message: str, input_shape: str = None, batch_size: int = None):
        details = {}
        if input_shape:
            details['input_shape'] = input_shape
        if batch_size:
            details['batch_size'] = batch_size
        
        super().__init__(
            message,
            category=ErrorCategory.INFERENCE,
            severity=ErrorSeverity.HIGH,
            details=details
        )


class ResourceError(NimifyError):
    """Resource allocation and management errors."""
    
    def __init__(self, message: str, resource_type: str, current_usage: str = None):
        details = {'resource_type': resource_type}
        if current_usage:
            details['current_usage'] = current_usage
        
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.HIGH,
            details=details
        )


class SecurityError(NimifyError):
    """Security-related errors."""
    
    def __init__(self, message: str, threat_type: str = None, client_ip: str = None):
        details = {}
        if threat_type:
            details['threat_type'] = threat_type
        if client_ip:
            details['client_ip'] = client_ip
        
        super().__init__(
            message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            details=details
        )


class ErrorRecoveryManager:
    """Manages error recovery strategies and retry logic."""
    
    def __init__(self):
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        self.error_history: List[ErrorContext] = []
        self.max_history = 1000
        self._lock = threading.Lock()
    
    def register_recovery_strategy(self, category: ErrorCategory, strategy: Callable):
        """Register a recovery strategy for a specific error category."""
        if category not in self.recovery_strategies:
            self.recovery_strategies[category] = []
        self.recovery_strategies[category].append(strategy)
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorContext:
        """Handle an error with appropriate recovery strategies."""
        # Create error context
        if isinstance(error, NimifyError):
            error_context = ErrorContext(
                error_id=f"err_{int(time.time())}_{hash(str(error)) % 10000:04d}",
                category=error.category,
                severity=error.severity,
                message=str(error),
                details=error.details.copy(),
                traceback=traceback.format_exc(),
                **{k: v for k, v in (context or {}).items() if k in ['request_id', 'user_id']}
            )
        else:
            # Handle non-Nimify exceptions
            error_context = ErrorContext(
                error_id=f"err_{int(time.time())}_{hash(str(error)) % 10000:04d}",
                category=self._classify_error(error),
                severity=ErrorSeverity.MEDIUM,
                message=str(error),
                traceback=traceback.format_exc(),
                **{k: v for k, v in (context or {}).items() if k in ['request_id', 'user_id']}
            )
        
        # Record error
        with self._lock:
            self.error_history.append(error_context)
            if len(self.error_history) > self.max_history:
                self.error_history.pop(0)
        
        # Log error
        self._log_error(error_context)
        
        # Attempt recovery if strategies are available
        if error_context.category in self.recovery_strategies:
            self._attempt_recovery(error_context)
        
        return error_context
    
    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify error by type and message."""
        error_type = type(error).__name__.lower()
        error_msg = str(error).lower()
        
        if 'validation' in error_msg or 'invalid' in error_msg:
            return ErrorCategory.VALIDATION
        elif 'model' in error_msg or 'load' in error_msg:
            return ErrorCategory.MODEL_LOADING
        elif 'inference' in error_msg or 'predict' in error_msg:
            return ErrorCategory.INFERENCE
        elif 'network' in error_msg or 'connection' in error_msg:
            return ErrorCategory.NETWORK
        elif 'auth' in error_msg or 'permission' in error_msg:
            return ErrorCategory.AUTHENTICATION
        elif 'memory' in error_msg or 'resource' in error_msg:
            return ErrorCategory.RESOURCE
        elif 'config' in error_msg or 'setting' in error_msg:
            return ErrorCategory.CONFIGURATION
        elif 'deploy' in error_msg or 'kubernetes' in error_msg:
            return ErrorCategory.DEPLOYMENT
        elif 'security' in error_msg or 'threat' in error_msg:
            return ErrorCategory.SECURITY
        else:
            return ErrorCategory.UNKNOWN
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level."""
        log_data = {
            'error_id': error_context.error_id,
            'category': error_context.category.value,
            'severity': error_context.severity.value,
            'details': error_context.details,
            'request_id': error_context.request_id,
            'user_id': error_context.user_id
        }
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR: {error_context.message}", extra=log_data)
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY: {error_context.message}", extra=log_data)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY: {error_context.message}", extra=log_data)
        else:
            logger.info(f"LOW SEVERITY: {error_context.message}", extra=log_data)
    
    def _attempt_recovery(self, error_context: ErrorContext):
        """Attempt recovery using registered strategies."""
        if error_context.recovery_attempts >= error_context.max_recovery_attempts:
            logger.error(f"Max recovery attempts reached for error {error_context.error_id}")
            return False
        
        error_context.recovery_attempts += 1
        strategies = self.recovery_strategies.get(error_context.category, [])
        
        for strategy in strategies:
            try:
                if strategy(error_context):
                    logger.info(f"Recovery successful for error {error_context.error_id}")
                    return True
            except Exception as e:
                logger.error(f"Recovery strategy failed: {e}")
        
        return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        with self._lock:
            if not self.error_history:
                return {
                    'total_errors': 0,
                    'by_category': {},
                    'by_severity': {},
                    'recent_errors': []
                }
            
            # Count by category
            by_category = {}
            for ctx in self.error_history:
                category = ctx.category.value
                by_category[category] = by_category.get(category, 0) + 1
            
            # Count by severity
            by_severity = {}
            for ctx in self.error_history:
                severity = ctx.severity.value
                by_severity[severity] = by_severity.get(severity, 0) + 1
            
            # Recent errors (last 10)
            recent_errors = [
                {
                    'error_id': ctx.error_id,
                    'category': ctx.category.value,
                    'severity': ctx.severity.value,
                    'message': ctx.message,
                    'timestamp': ctx.timestamp
                }
                for ctx in self.error_history[-10:]
            ]
            
            return {
                'total_errors': len(self.error_history),
                'by_category': by_category,
                'by_severity': by_severity,
                'recent_errors': recent_errors
            }


# Retry decorator with exponential backoff
def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Callable = None
):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    if attempts == max_attempts:
                        logger.error(f"Max retry attempts ({max_attempts}) reached for {func.__name__}: {e}")
                        raise
                    
                    if on_retry:
                        on_retry(e, attempts)
                    
                    logger.warning(f"Retry {attempts}/{max_attempts} for {func.__name__} after {current_delay}s: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        return wrapper
    return decorator


# Async retry decorator
def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Callable = None
):
    """Async retry decorator with exponential backoff."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    if attempts == max_attempts:
                        logger.error(f"Max retry attempts ({max_attempts}) reached for {func.__name__}: {e}")
                        raise
                    
                    if on_retry:
                        await on_retry(e, attempts) if asyncio.iscoroutinefunction(on_retry) else on_retry(e, attempts)
                    
                    logger.warning(f"Retry {attempts}/{max_attempts} for {func.__name__} after {current_delay}s: {e}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        return wrapper
    return decorator


@contextmanager
def error_context(context_info: Dict[str, Any]):
    """Context manager for error handling."""
    try:
        yield
    except Exception as e:
        error_manager.handle_error(e, context_info)
        raise


# Default recovery strategies
def model_loading_recovery(error_context: ErrorContext) -> bool:
    """Recovery strategy for model loading errors."""
    logger.info(f"Attempting model loading recovery for error {error_context.error_id}")
    
    # Try clearing model cache
    try:
        import gc
        gc.collect()
        logger.info("Cleared Python garbage collector")
        return True
    except Exception as e:
        logger.error(f"Model loading recovery failed: {e}")
        return False


def resource_recovery(error_context: ErrorContext) -> bool:
    """Recovery strategy for resource errors."""
    logger.info(f"Attempting resource recovery for error {error_context.error_id}")
    
    # Try freeing up memory
    try:
        import gc
        gc.collect()
        
        # Clear caches if available
        if hasattr(sys.modules.get('nimify.performance'), 'model_cache'):
            sys.modules['nimify.performance'].model_cache.clear()
            logger.info("Cleared model cache")
        
        return True
    except Exception as e:
        logger.error(f"Resource recovery failed: {e}")
        return False


def network_recovery(error_context: ErrorContext) -> bool:
    """Recovery strategy for network errors."""
    logger.info(f"Attempting network recovery for error {error_context.error_id}")
    
    # Wait and retry (simple strategy)
    try:
        time.sleep(1.0)
        logger.info("Completed network recovery delay")
        return True
    except Exception as e:
        logger.error(f"Network recovery failed: {e}")
        return False


# Global error manager instance
error_manager = ErrorRecoveryManager()

# Register default recovery strategies
error_manager.register_recovery_strategy(ErrorCategory.MODEL_LOADING, model_loading_recovery)
error_manager.register_recovery_strategy(ErrorCategory.RESOURCE, resource_recovery)
error_manager.register_recovery_strategy(ErrorCategory.NETWORK, network_recovery)


# Health check function
def get_system_health() -> Dict[str, Any]:
    """Get comprehensive system health status."""
    error_stats = error_manager.get_error_statistics()
    
    # Calculate health score (0-100)
    total_errors = error_stats['total_errors']
    critical_errors = error_stats['by_severity'].get('critical', 0)
    high_errors = error_stats['by_severity'].get('high', 0)
    
    if total_errors == 0:
        health_score = 100
        status = "healthy"
    elif critical_errors > 0:
        health_score = max(0, 50 - (critical_errors * 10))
        status = "critical"
    elif high_errors > 5:
        health_score = max(30, 80 - (high_errors * 5))
        status = "degraded"
    elif total_errors > 20:
        health_score = max(60, 90 - total_errors)
        status = "warning"
    else:
        health_score = max(70, 100 - total_errors)
        status = "healthy"
    
    return {
        'status': status,
        'health_score': health_score,
        'error_statistics': error_stats,
        'timestamp': time.time()
    }