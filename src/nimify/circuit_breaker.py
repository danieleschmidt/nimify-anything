"""Circuit breaker pattern for fault tolerance with quantum-inspired adaptive thresholds."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from threading import Lock
from typing import Any

import numpy as np


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance with quantum-inspired adaptation."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type = Exception,
        success_threshold: int = 3,
        enable_quantum_adaptation: bool = True
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before transitioning to half-open
            expected_exception: Exception type that counts as failure
            success_threshold: Successes needed in half-open to close circuit
            enable_quantum_adaptation: Enable quantum-inspired adaptive thresholds
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold
        self.enable_quantum_adaptation = enable_quantum_adaptation
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED
        self.lock = Lock()
        
        # Quantum-inspired adaptive parameters
        self.failure_history = []  # Track failure patterns
        self.response_time_history = []  # Track response times
        self.adaptive_threshold = failure_threshold
        self.quantum_coherence = 1.0  # Coherence measure for adaptation
        self.entanglement_factor = 0.1  # Cross-correlation with system state
        
        self.logger = logging.getLogger(__name__)
    
    def _update_quantum_parameters(self, success: bool, response_time: float = None):
        """Update quantum-inspired parameters based on system behavior."""
        
        if not self.enable_quantum_adaptation:
            return
        
        current_time = time.time()
        
        # Update failure history (keep last 50 events)
        self.failure_history.append({
            'timestamp': current_time,
            'success': success,
            'response_time': response_time
        })
        if len(self.failure_history) > 50:
            self.failure_history.pop(0)
        
        # Track response times
        if response_time is not None:
            self.response_time_history.append(response_time)
            if len(self.response_time_history) > 100:
                self.response_time_history.pop(0)
        
        # Compute adaptive threshold using quantum-inspired algorithm
        if len(self.failure_history) >= 10:
            self._compute_adaptive_threshold()
    
    def _compute_adaptive_threshold(self):
        """Compute adaptive threshold using quantum-inspired metrics."""
        
        # Analyze failure patterns
        recent_failures = [
            event for event in self.failure_history[-20:] 
            if not event['success']
        ]
        failure_rate = len(recent_failures) / 20
        
        # Compute quantum coherence (system stability)
        if self.response_time_history:
            response_variance = np.var(self.response_time_history)
            mean_response = np.mean(self.response_time_history)
            
            # High coherence = low variance = stable system
            if mean_response > 0:
                self.quantum_coherence = 1.0 / (1.0 + response_variance / mean_response)
            else:
                self.quantum_coherence = 0.5
        
        # Compute entanglement factor (correlation with failure patterns)
        if len(recent_failures) > 1:
            failure_times = [f['timestamp'] for f in recent_failures]
            time_diffs = np.diff(failure_times)
            if len(time_diffs) > 1:
                # High entanglement = correlated failures
                self.entanglement_factor = min(1.0, np.std(time_diffs) / np.mean(time_diffs))
        
        # Adaptive threshold using quantum-inspired formula
        base_threshold = self.failure_threshold
        
        # Quantum tunneling effect: allow more failures when system is coherent
        coherence_adjustment = self.quantum_coherence * 2.0
        
        # Entanglement effect: reduce threshold when failures are correlated
        entanglement_adjustment = (1.0 - self.entanglement_factor) * 1.5
        
        # Failure rate momentum: adjust based on recent trend
        momentum_adjustment = (1.0 - failure_rate) * 1.2
        
        self.adaptive_threshold = max(
            2,  # Minimum threshold
            int(base_threshold * coherence_adjustment * entanglement_adjustment * momentum_adjustment)
        )
        
        self.logger.debug(
            f"Adaptive threshold updated: {self.adaptive_threshold} "
            f"(coherence={self.quantum_coherence:.3f}, "
            f"entanglement={self.entanglement_factor:.3f}, "
            f"failure_rate={failure_rate:.3f})"
        )
    
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
        except self.expected_exception:
            self._on_failure()
            raise
    
    def get_status(self) -> dict[str, Any]:
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
        self.breakers: dict[str, CircuitBreaker] = {}
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
    
    def get_all_status(self) -> dict[str, dict[str, Any]]:
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


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breakers."""
    failure_threshold: int = 5
    timeout: float = 60.0
    expected_exception: type = Exception
    success_threshold: int = 3
    name: str = "default"


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