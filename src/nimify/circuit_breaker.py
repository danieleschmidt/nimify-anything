"""Circuit breaker pattern for fault tolerance."""

import time
import logging
from typing import Callable, Any, Optional, Dict
from functools import wraps
from enum import Enum
from threading import Lock


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type = Exception,
        success_threshold: int = 3
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before transitioning to half-open
            expected_exception: Exception type that counts as failure
            success_threshold: Successes needed in half-open to close circuit
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED
        self.lock = Lock()
        
        self.logger = logging.getLogger(__name__)
    
    def _can_attempt_call(self) -> bool:
        """Check if call can be attempted based on current state."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.timeout:
                self._transition_to_half_open()
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        return False
    
    def _transition_to_half_open(self):
        """Transition circuit breaker to half-open state."""
        with self.lock:
            self.state = CircuitState.HALF_OPEN
            self.success_count = 0
            self.logger.info("Circuit breaker transitioned to HALF_OPEN")
    
    def _transition_to_closed(self):
        """Transition circuit breaker to closed state."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.logger.info("Circuit breaker transitioned to CLOSED")
    
    def _transition_to_open(self):
        """Transition circuit breaker to open state."""
        with self.lock:
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
            self.logger.warning(
                f"Circuit breaker OPENED after {self.failure_count} failures"
            )
    
    def _on_success(self):
        """Handle successful call."""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        with self.lock:
            self.failure_count += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self._transition_to_open()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if not self._can_attempt_call():
            raise CircuitBreakerException(
                f"Circuit breaker is {self.state.value}, failing fast"
            )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_threshold": self.failure_threshold,
            "timeout": self.timeout,
            "last_failure_time": self.last_failure_time
        }


def with_circuit_breaker(
    failure_threshold: int = 5,
    timeout: float = 60.0,
    expected_exception: type = Exception,
    success_threshold: int = 3
):
    """Decorator to add circuit breaker to a function."""
    
    def decorator(func: Callable):
        circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            timeout=timeout,
            expected_exception=expected_exception,
            success_threshold=success_threshold
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return circuit_breaker.call(func, *args, **kwargs)
        
        # Attach circuit breaker to function for external access
        wrapper.circuit_breaker = circuit_breaker
        return wrapper
    
    return decorator


class GlobalCircuitBreakers:
    """Manage multiple circuit breakers globally."""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.lock = Lock()
    
    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type = Exception,
        success_threshold: int = 3
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        with self.lock:
            if name not in self.breakers:
                self.breakers[name] = CircuitBreaker(
                    failure_threshold=failure_threshold,
                    timeout=timeout,
                    expected_exception=expected_exception,
                    success_threshold=success_threshold
                )
            return self.breakers[name]
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {
            name: breaker.get_status() 
            for name, breaker in self.breakers.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers to closed state."""
        with self.lock:
            for breaker in self.breakers.values():
                breaker._transition_to_closed()


# Global instance
global_circuit_breakers = GlobalCircuitBreakers()


def protected_call(
    name: str,
    func: Callable,
    *args,
    failure_threshold: int = 5,
    timeout: float = 60.0,
    **kwargs
) -> Any:
    """Make a protected call using named circuit breaker."""
    breaker = global_circuit_breakers.get_or_create(
        name=name,
        failure_threshold=failure_threshold,
        timeout=timeout
    )
    
    return breaker.call(func, *args, **kwargs)


# Example usage for model inference
class ModelInferenceCircuitBreaker:
    """Specialized circuit breaker for model inference."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.breaker = global_circuit_breakers.get_or_create(
            name=f"model_inference_{model_name}",
            failure_threshold=3,  # Lower threshold for model failures
            timeout=30.0,         # Shorter recovery time
            expected_exception=Exception,
            success_threshold=2   # Faster recovery
        )
    
    def predict(self, predict_func: Callable, *args, **kwargs):
        """Make prediction with circuit breaker protection."""
        return self.breaker.call(predict_func, *args, **kwargs)
    
    def get_status(self):
        """Get circuit breaker status for this model."""
        return self.breaker.get_status()