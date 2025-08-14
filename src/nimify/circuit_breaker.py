"""Circuit breaker pattern implementation for robust service resilience."""

import asyncio
import time
import logging
from enum import Enum
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass
import threading
from collections import deque

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failures detected, requests rejected
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes before closing from half-open
    timeout: float = 60.0              # Seconds before trying half-open
    recovery_timeout: float = 30.0     # Seconds to test in half-open
    slow_call_threshold: float = 5.0   # Seconds for slow call detection
    max_calls_in_half_open: int = 10   # Max calls allowed in half-open


class CircuitBreakerException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation with advanced features:
    - Automatic failure detection and recovery
    - Slow call detection 
    - Configurable thresholds and timeouts
    - Thread-safe operation
    - Metrics collection
    """
    
    def __init__(self, config: CircuitBreakerConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.half_open_start_time = 0
        self.half_open_calls = 0
        self.lock = threading.RLock()
        
        # Metrics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.recent_calls: deque = deque(maxlen=100)  # Store recent call results
        
        logger.info(f"Circuit breaker '{name}' initialized with config: {config}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    async def acall(self, func: Callable, *args, **kwargs) -> Any:
        """Async version of call method."""
        with self.lock:
            if not self._can_proceed():
                self._record_rejection()
                raise CircuitBreakerException(
                    f"Circuit breaker '{self.name}' is {self.state.value}"
                )
            
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1
        
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            call_duration = time.time() - start_time
            self._record_success(call_duration)
            return result
            
        except Exception as e:
            call_duration = time.time() - start_time
            self._record_failure(call_duration, e)
            raise
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            if not self._can_proceed():
                self._record_rejection()
                raise CircuitBreakerException(
                    f"Circuit breaker '{self.name}' is {self.state.value}"
                )
            
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            call_duration = time.time() - start_time
            self._record_success(call_duration)
            return result
            
        except Exception as e:
            call_duration = time.time() - start_time
            self._record_failure(call_duration, e)
            raise
    
    def _can_proceed(self) -> bool:
        """Check if call can proceed based on circuit state."""
        current_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            return True
        
        elif self.state == CircuitState.OPEN:
            # Check if timeout has passed to move to half-open
            if current_time - self.last_failure_time >= self.config.timeout:
                self._transition_to_half_open()
                return True
            return False
        
        elif self.state == CircuitState.HALF_OPEN:
            # Check if we've exceeded max calls in half-open
            if self.half_open_calls >= self.config.max_calls_in_half_open:
                return False
            
            # Check if recovery timeout has passed
            if current_time - self.half_open_start_time >= self.config.recovery_timeout:
                self._transition_to_open()
                return False
            
            return True
        
        return False
    
    def _record_success(self, duration: float):
        """Record successful call."""
        with self.lock:
            self.total_calls += 1
            self.total_successes += 1
            self.recent_calls.append(('success', duration, time.time()))
            
            # Check for slow calls
            if duration > self.config.slow_call_threshold:
                logger.warning(f"Slow call detected in circuit '{self.name}': {duration:.2f}s")
                # Treat slow calls as partial failures
                self.failure_count += 0.5
            else:
                self.failure_count = max(0, self.failure_count - 0.1)  # Gradual recovery
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
    
    def _record_failure(self, duration: float, exception: Exception):
        """Record failed call."""
        with self.lock:
            self.total_calls += 1
            self.total_failures += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.recent_calls.append(('failure', duration, time.time(), str(exception)))
            
            logger.warning(
                f"Circuit breaker '{self.name}' recorded failure: {exception} "
                f"(duration: {duration:.2f}s, failures: {self.failure_count})"
            )
            
            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self._transition_to_open()
            
            elif self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state opens the circuit
                self._transition_to_open()
    
    def _record_rejection(self):
        """Record rejected call due to open circuit."""
        with self.lock:
            self.total_calls += 1
            self.recent_calls.append(('rejected', 0, time.time()))
            
            logger.debug(f"Circuit breaker '{self.name}' rejected call (state: {self.state.value})")
    
    def _transition_to_open(self):
        """Transition to OPEN state."""
        previous_state = self.state
        self.state = CircuitState.OPEN
        self.success_count = 0
        self.half_open_calls = 0
        
        logger.warning(
            f"Circuit breaker '{self.name}' transitioned from {previous_state.value} to OPEN "
            f"(failures: {self.failure_count}/{self.config.failure_threshold})"
        )
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        previous_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.half_open_calls = 0
        self.half_open_start_time = time.time()
        
        logger.info(
            f"Circuit breaker '{self.name}' transitioned from {previous_state.value} to HALF_OPEN "
            f"(testing recovery)"
        )
    
    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        previous_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        
        logger.info(
            f"Circuit breaker '{self.name}' transitioned from {previous_state.value} to CLOSED "
            f"(service recovered)"
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self.lock:
            current_time = time.time()
            
            # Calculate success rate from recent calls
            recent_successes = sum(1 for call in self.recent_calls if call[0] == 'success')
            recent_total = len(self.recent_calls)
            success_rate = (recent_successes / recent_total) if recent_total > 0 else 0
            
            # Calculate average response time
            response_times = [call[1] for call in self.recent_calls if call[0] in ['success', 'failure']]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            return {
                'name': self.name,
                'state': self.state.value,
                'total_calls': self.total_calls,
                'total_successes': self.total_successes,
                'total_failures': self.total_failures,
                'current_failure_count': self.failure_count,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'time_since_last_failure': current_time - self.last_failure_time if self.last_failure_time > 0 else 0,
                'half_open_calls': self.half_open_calls if self.state == CircuitState.HALF_OPEN else None,
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'success_threshold': self.config.success_threshold,
                    'timeout': self.config.timeout,
                    'slow_call_threshold': self.config.slow_call_threshold
                }
            }
    
    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        with self.lock:
            previous_state = self.state
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.half_open_calls = 0
            self.last_failure_time = 0
            
            logger.info(f"Circuit breaker '{self.name}' manually reset from {previous_state.value} to CLOSED")
    
    def force_open(self):
        """Manually force circuit breaker to OPEN state."""
        with self.lock:
            previous_state = self.state
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
            
            logger.warning(f"Circuit breaker '{self.name}' manually forced from {previous_state.value} to OPEN")


class CircuitBreakerRegistry:
    """Registry to manage multiple circuit breakers."""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.lock = threading.RLock()
    
    def get_or_create(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        with self.lock:
            if name not in self.breakers:
                if config is None:
                    config = CircuitBreakerConfig()
                self.breakers[name] = CircuitBreaker(config, name)
            return self.breakers[name]
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all registered circuit breakers."""
        with self.lock:
            return {name: breaker.get_metrics() for name, breaker in self.breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self.lock:
            for breaker in self.breakers.values():
                breaker.reset()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all circuit breakers."""
        with self.lock:
            total_calls = sum(breaker.total_calls for breaker in self.breakers.values())
            total_failures = sum(breaker.total_failures for breaker in self.breakers.values())
            
            states = {}
            for breaker in self.breakers.values():
                state = breaker.state.value
                states[state] = states.get(state, 0) + 1
            
            return {
                'total_breakers': len(self.breakers),
                'total_calls': total_calls,
                'total_failures': total_failures,
                'overall_success_rate': (total_calls - total_failures) / total_calls if total_calls > 0 else 0,
                'states_distribution': states
            }


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to add circuit breaker protection to functions."""
    def decorator(func: Callable) -> Callable:
        breaker = circuit_breaker_registry.get_or_create(name, config)
        return breaker(func)
    return decorator