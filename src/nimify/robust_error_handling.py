"""Comprehensive error handling and recovery for NIM services."""

import asyncio
import logging
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for proper escalation."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for proper handling strategies."""
    VALIDATION = "validation"
    INFERENCE = "inference"
    RESOURCE = "resource"
    NETWORK = "network"
    SECURITY = "security"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Rich error context for better debugging and recovery."""
    error_id: str
    timestamp: float
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: dict[str, Any]
    stack_trace: str
    request_id: Optional[str] = None
    model_name: Optional[str] = None
    recovery_attempted: bool = False


class CircuitBreakerState(Enum):
    """Circuit breaker states for fault tolerance."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    monitor_window: float = 300.0


class RobustCircuitBreaker:
    """Advanced circuit breaker with health monitoring and adaptive recovery."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.state_change_time = time.time()
        self._failure_history = []
        
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.config.recovery_timeout:
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                self.state = CircuitBreakerState.HALF_OPEN
                self.state_change_time = time.time()
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await self._on_success()
            return result
            
        except Exception as e:
            await self._on_failure(e)
            raise
    
    async def _on_success(self):
        """Handle successful execution."""
        current_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                logger.info("Circuit breaker transitioning to CLOSED after successful recovery")
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.state_change_time = current_time
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success in normal operation
            self.failure_count = max(0, self.failure_count - 1)
    
    async def _on_failure(self, exception: Exception):
        """Handle failed execution."""
        current_time = time.time()
        self.failure_count += 1
        self.last_failure_time = current_time
        self._failure_history.append({
            "timestamp": current_time,
            "error": str(exception),
            "type": type(exception).__name__
        })
        
        # Clean old failures from history
        self._cleanup_failure_history()
        
        if self.state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN]:
            if self.failure_count >= self.config.failure_threshold:
                logger.warning(f"Circuit breaker OPENING due to {self.failure_count} failures")
                self.state = CircuitBreakerState.OPEN
                self.success_count = 0
                self.state_change_time = current_time
    
    def _cleanup_failure_history(self):
        """Remove old failures outside monitoring window."""
        current_time = time.time()
        cutoff_time = current_time - self.config.monitor_window
        self._failure_history = [
            failure for failure in self._failure_history
            if failure["timestamp"] > cutoff_time
        ]
    
    def get_health_status(self) -> dict[str, Any]:
        """Get current health status and metrics."""
        current_time = time.time()
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_rate": len(self._failure_history) / max(1, self.config.monitor_window / 60),
            "time_in_current_state": current_time - self.state_change_time,
            "recent_failures": self._failure_history[-5:],  # Last 5 failures
            "next_retry_available": max(0, self.config.recovery_timeout - (current_time - self.last_failure_time))
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class AdaptiveRetryPolicy:
    """Intelligent retry policy with exponential backoff and jitter."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with adaptive retry policy."""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"Function succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_attempts - 1:
                    logger.error(f"Function failed after {self.max_attempts} attempts: {e}")
                    break
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.base_delay * (self.exponential_base ** attempt),
                    self.max_delay
                )
                
                # Add jitter to prevent thundering herd
                if self.jitter:
                    import random
                    delay += random.uniform(0, delay * 0.1)
                
                logger.warning(f"Function failed on attempt {attempt + 1}, retrying in {delay:.2f}s: {e}")
                await asyncio.sleep(delay)
        
        raise last_exception


class ErrorRecoveryManager:
    """Centralized error recovery and escalation management."""
    
    def __init__(self):
        self.circuit_breakers = {}
        self.error_handlers = {}
        self.recovery_strategies = {}
        self._error_history = []
        
    def register_circuit_breaker(self, service_name: str, config: CircuitBreakerConfig = None) -> RobustCircuitBreaker:
        """Register circuit breaker for a service."""
        config = config or CircuitBreakerConfig()
        circuit_breaker = RobustCircuitBreaker(config)
        self.circuit_breakers[service_name] = circuit_breaker
        return circuit_breaker
    
    def register_error_handler(self, category: ErrorCategory, handler: Callable):
        """Register custom error handler for specific category."""
        self.error_handlers[category] = handler
    
    def register_recovery_strategy(self, error_type: type, strategy: Callable):
        """Register recovery strategy for specific error type."""
        self.recovery_strategies[error_type] = strategy
    
    async def handle_error(
        self,
        exception: Exception,
        context: dict[str, Any] = None
    ) -> ErrorContext:
        """Comprehensive error handling with recovery attempts."""
        error_context = self._create_error_context(exception, context or {})
        
        # Log error with full context
        logger.error(
            f"Error {error_context.error_id}: {error_context.message}",
            extra={
                "error_context": error_context.__dict__,
                "stack_trace": error_context.stack_trace
            }
        )
        
        # Store in error history
        self._error_history.append(error_context)
        self._cleanup_error_history()
        
        # Attempt recovery if strategy exists
        recovery_strategy = self.recovery_strategies.get(type(exception))
        if recovery_strategy and not error_context.recovery_attempted:
            try:
                logger.info(f"Attempting recovery for error {error_context.error_id}")
                await recovery_strategy(exception, error_context)
                error_context.recovery_attempted = True
                logger.info(f"Recovery successful for error {error_context.error_id}")
            except Exception as recovery_error:
                logger.error(f"Recovery failed for error {error_context.error_id}: {recovery_error}")
        
        # Execute category-specific handler
        handler = self.error_handlers.get(error_context.category)
        if handler:
            try:
                await handler(error_context)
            except Exception as handler_error:
                logger.error(f"Error handler failed: {handler_error}")
        
        # Escalate if critical
        if error_context.severity == ErrorSeverity.CRITICAL:
            await self._escalate_error(error_context)
        
        return error_context
    
    def _create_error_context(self, exception: Exception, context: dict[str, Any]) -> ErrorContext:
        """Create rich error context from exception and metadata."""
        import uuid
        
        error_id = str(uuid.uuid4())
        category = self._classify_error(exception)
        severity = self._assess_severity(exception, category)
        
        return ErrorContext(
            error_id=error_id,
            timestamp=time.time(),
            category=category,
            severity=severity,
            message=str(exception),
            details=context,
            stack_trace=traceback.format_exc(),
            request_id=context.get("request_id"),
            model_name=context.get("model_name")
        )
    
    def _classify_error(self, exception: Exception) -> ErrorCategory:
        """Classify error into appropriate category."""
        error_type = type(exception).__name__.lower()
        
        if "validation" in error_type or "pydantic" in error_type:
            return ErrorCategory.VALIDATION
        elif "inference" in error_type or "model" in error_type:
            return ErrorCategory.INFERENCE
        elif "memory" in error_type or "resource" in error_type or "timeout" in error_type:
            return ErrorCategory.RESOURCE
        elif "connection" in error_type or "network" in error_type or "http" in error_type:
            return ErrorCategory.NETWORK
        elif "permission" in error_type or "auth" in error_type or "security" in error_type:
            return ErrorCategory.SECURITY
        else:
            return ErrorCategory.SYSTEM
    
    def _assess_severity(self, exception: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Assess error severity based on type and category."""
        error_type = type(exception).__name__.lower()
        
        # Critical errors that require immediate attention
        if any(keyword in error_type for keyword in ["security", "permission", "auth"]):
            return ErrorSeverity.CRITICAL
        
        # High severity for resource and system issues
        if category in [ErrorCategory.RESOURCE, ErrorCategory.SYSTEM]:
            return ErrorSeverity.HIGH
        
        # Medium for network and inference issues
        if category in [ErrorCategory.NETWORK, ErrorCategory.INFERENCE]:
            return ErrorSeverity.MEDIUM
        
        # Low for validation errors
        return ErrorSeverity.LOW
    
    async def _escalate_error(self, error_context: ErrorContext):
        """Escalate critical errors to monitoring systems."""
        logger.critical(
            f"CRITICAL ERROR ESCALATED: {error_context.error_id}",
            extra={"error_context": error_context.__dict__}
        )
        
        # Here you would integrate with alerting systems
        # Example: Send to PagerDuty, Slack, etc.
        await self._send_alert(error_context)
    
    async def _send_alert(self, error_context: ErrorContext):
        """Send alert to external monitoring systems."""
        # Placeholder for alert integration
        # In production, this would integrate with:
        # - PagerDuty
        # - Slack webhooks
        # - Email notifications
        # - Custom monitoring systems
        logger.info(f"Alert sent for critical error: {error_context.error_id}")
    
    def _cleanup_error_history(self):
        """Clean up old errors from history (keep last 1000 or 24 hours)."""
        current_time = time.time()
        cutoff_time = current_time - 86400  # 24 hours
        
        self._error_history = [
            error for error in self._error_history[-1000:]  # Keep last 1000
            if error.timestamp > cutoff_time  # Within 24 hours
        ]
    
    def get_error_statistics(self) -> dict[str, Any]:
        """Get comprehensive error statistics and health metrics."""
        current_time = time.time()
        recent_errors = [e for e in self._error_history if current_time - e.timestamp < 3600]  # Last hour
        
        category_counts = {}
        severity_counts = {}
        
        for error in recent_errors:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        circuit_breaker_status = {
            name: cb.get_health_status() 
            for name, cb in self.circuit_breakers.items()
        }
        
        return {
            "total_errors": len(self._error_history),
            "recent_errors_1h": len(recent_errors),
            "error_rate_per_hour": len(recent_errors),
            "category_breakdown": category_counts,
            "severity_breakdown": severity_counts,
            "circuit_breakers": circuit_breaker_status,
            "recent_critical_errors": [
                {
                    "error_id": e.error_id,
                    "timestamp": e.timestamp,
                    "message": e.message,
                    "category": e.category.value
                }
                for e in recent_errors
                if e.severity == ErrorSeverity.CRITICAL
            ]
        }


# Global error recovery manager instance
global_error_manager = ErrorRecoveryManager()


def robust_error_handler(
    circuit_breaker_name: str = None,
    retry_policy: AdaptiveRetryPolicy = None,
    recovery_strategy: Callable = None
):
    """Decorator for robust error handling with circuit breaker and retry."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get or create circuit breaker
            circuit_breaker = None
            if circuit_breaker_name:
                circuit_breaker = global_error_manager.circuit_breakers.get(circuit_breaker_name)
                if not circuit_breaker:
                    circuit_breaker = global_error_manager.register_circuit_breaker(circuit_breaker_name)
            
            # Create retry policy if not provided
            if retry_policy is None:
                retry_policy_instance = AdaptiveRetryPolicy()
            else:
                retry_policy_instance = retry_policy
            
            try:
                if circuit_breaker:
                    return await circuit_breaker.execute(retry_policy_instance.execute, func, *args, **kwargs)
                else:
                    return await retry_policy_instance.execute(func, *args, **kwargs)
                    
            except Exception as e:
                # Handle error through global error manager
                context = {
                    "function_name": func.__name__,
                    "args": str(args)[:200],  # Truncate for logging
                    "kwargs": str(kwargs)[:200],
                    "circuit_breaker": circuit_breaker_name
                }
                
                await global_error_manager.handle_error(e, context)
                raise
        
        return wrapper
    return decorator