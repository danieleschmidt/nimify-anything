"""Advanced rate limiting with multiple algorithms and adaptive behavior."""

import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    ADAPTIVE = "adaptive"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    max_requests: int = 100          # Max requests per window
    window_size: int = 60            # Window size in seconds
    burst_size: int = 150            # Max burst size (for token bucket)
    refill_rate: float = 1.67        # Tokens per second (100/60)
    
    # Adaptive settings
    min_requests: int = 10           # Minimum requests when load is high
    max_requests_adaptive: int = 500 # Maximum requests when load is low
    adaptation_factor: float = 0.1   # How quickly to adapt (0-1)
    
    # Per-client limits
    per_client_enabled: bool = True
    per_client_max: int = 10         # Max requests per client per window
    
    # Penalties
    penalty_multiplier: float = 2.0  # Multiplier for penalty duration
    max_penalty_duration: int = 300  # Max penalty in seconds


class RateLimitException(Exception):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: float | None = None, 
                 limit_type: str = "general"):
        super().__init__(message)
        self.retry_after = retry_after
        self.limit_type = limit_type


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""
    
    @abstractmethod
    async def is_allowed(self, key: str, tokens: int = 1) -> tuple[bool, float | None]:
        """Check if request is allowed. Returns (allowed, retry_after)."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> dict:
        """Get rate limiter metrics."""
        pass
    
    @abstractmethod
    def reset(self, key: str | None = None):
        """Reset rate limiter state."""
        pass


class TokenBucketRateLimiter(RateLimiter):
    """Token bucket rate limiter with burst capacity."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.buckets: dict[str, dict] = {}
        self.lock = threading.RLock()
        self.total_requests = 0
        self.total_rejected = 0
        
        logger.info(f"Token bucket rate limiter initialized: {config.max_requests} req/{config.window_size}s")
    
    async def is_allowed(self, key: str, tokens: int = 1) -> tuple[bool, float | None]:
        """Check if request is allowed using token bucket algorithm."""
        current_time = time.time()
        
        with self.lock:
            self.total_requests += tokens
            
            if key not in self.buckets:
                self.buckets[key] = {
                    'tokens': self.config.burst_size,
                    'last_refill': current_time,
                    'total_requests': 0,
                    'total_rejected': 0
                }
            
            bucket = self.buckets[key]
            
            # Refill tokens based on time elapsed
            time_elapsed = current_time - bucket['last_refill']
            tokens_to_add = time_elapsed * self.config.refill_rate
            bucket['tokens'] = min(self.config.burst_size, bucket['tokens'] + tokens_to_add)
            bucket['last_refill'] = current_time
            bucket['total_requests'] += tokens
            
            # Check if enough tokens available
            if bucket['tokens'] >= tokens:
                bucket['tokens'] -= tokens
                return True, None
            else:
                bucket['total_rejected'] += tokens
                self.total_rejected += tokens
                
                # Calculate retry after time
                tokens_needed = tokens - bucket['tokens']
                retry_after = tokens_needed / self.config.refill_rate
                
                return False, retry_after
    
    def get_metrics(self) -> dict:
        """Get token bucket metrics."""
        with self.lock:
            client_metrics = {}
            for key, bucket in self.buckets.items():
                client_metrics[key] = {
                    'tokens_available': bucket['tokens'],
                    'total_requests': bucket['total_requests'],
                    'total_rejected': bucket['total_rejected'],
                    'rejection_rate': bucket['total_rejected'] / bucket['total_requests'] if bucket['total_requests'] > 0 else 0
                }
            
            return {
                'algorithm': self.config.algorithm.value,
                'total_requests': self.total_requests,
                'total_rejected': self.total_rejected,
                'overall_rejection_rate': self.total_rejected / self.total_requests if self.total_requests > 0 else 0,
                'active_clients': len(self.buckets),
                'client_metrics': client_metrics,
                'config': {
                    'max_requests': self.config.max_requests,
                    'burst_size': self.config.burst_size,
                    'refill_rate': self.config.refill_rate
                }
            }
    
    def reset(self, key: str | None = None):
        """Reset bucket state."""
        with self.lock:
            if key:
                if key in self.buckets:
                    self.buckets[key] = {
                        'tokens': self.config.burst_size,
                        'last_refill': time.time(),
                        'total_requests': 0,
                        'total_rejected': 0
                    }
            else:
                self.buckets.clear()
                self.total_requests = 0
                self.total_rejected = 0


class SlidingWindowRateLimiter(RateLimiter):
    """Sliding window rate limiter with precise timing."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.windows: dict[str, deque] = defaultdict(lambda: deque())
        self.lock = threading.RLock()
        self.total_requests = 0
        self.total_rejected = 0
        
        logger.info(f"Sliding window rate limiter initialized: {config.max_requests} req/{config.window_size}s")
    
    async def is_allowed(self, key: str, tokens: int = 1) -> tuple[bool, float | None]:
        """Check if request is allowed using sliding window algorithm."""
        current_time = time.time()
        window_start = current_time - self.config.window_size
        
        with self.lock:
            self.total_requests += tokens
            
            # Clean old entries
            window = self.windows[key]
            while window and window[0] < window_start:
                window.popleft()
            
            # Check if adding request would exceed limit
            current_count = len(window)
            if current_count + tokens <= self.config.max_requests:
                # Add request timestamps
                for _ in range(tokens):
                    window.append(current_time)
                return True, None
            else:
                self.total_rejected += tokens
                
                # Calculate retry after time
                if window:
                    oldest_request = window[0]
                    retry_after = oldest_request + self.config.window_size - current_time
                    retry_after = max(0, retry_after)
                else:
                    retry_after = self.config.window_size
                
                return False, retry_after
    
    def get_metrics(self) -> dict:
        """Get sliding window metrics."""
        with self.lock:
            current_time = time.time()
            window_start = current_time - self.config.window_size
            
            client_metrics = {}
            for key, window in self.windows.items():
                # Clean old entries for accurate count
                while window and window[0] < window_start:
                    window.popleft()
                
                client_metrics[key] = {
                    'current_requests': len(window),
                    'utilization': len(window) / self.config.max_requests,
                }
            
            return {
                'algorithm': self.config.algorithm.value,
                'total_requests': self.total_requests,
                'total_rejected': self.total_rejected,
                'overall_rejection_rate': self.total_rejected / self.total_requests if self.total_requests > 0 else 0,
                'active_clients': len(self.windows),
                'client_metrics': client_metrics,
                'config': {
                    'max_requests': self.config.max_requests,
                    'window_size': self.config.window_size
                }
            }
    
    def reset(self, key: str | None = None):
        """Reset window state."""
        with self.lock:
            if key:
                if key in self.windows:
                    self.windows[key].clear()
            else:
                self.windows.clear()
                self.total_requests = 0
                self.total_rejected = 0


class AdaptiveRateLimiter(RateLimiter):
    """Adaptive rate limiter that adjusts limits based on system load."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.base_limiter = TokenBucketRateLimiter(config)
        self.current_limit = config.max_requests
        self.load_metrics: deque = deque(maxlen=60)  # Store last 60 measurements
        self.last_adaptation = time.time()
        self.lock = threading.RLock()
        
        logger.info(f"Adaptive rate limiter initialized: {config.min_requests}-{config.max_requests_adaptive} req/{config.window_size}s")
    
    async def is_allowed(self, key: str, tokens: int = 1) -> tuple[bool, float | None]:
        """Check if request is allowed with adaptive limits."""
        # Update current load metrics
        await self._update_load_metrics()
        
        # Adapt limits if needed
        await self._adapt_limits()
        
        # Use base limiter with current limit
        return await self.base_limiter.is_allowed(key, tokens)
    
    async def _update_load_metrics(self):
        """Update system load metrics."""
        current_time = time.time()
        
        # Simulate load metrics (in real implementation, gather from system)
        try:
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Calculate composite load score
            load_score = (cpu_percent + memory_percent) / 2
            
        except ImportError:
            # Fallback: use request rate as load indicator
            metrics = self.base_limiter.get_metrics()
            rejection_rate = metrics['overall_rejection_rate']
            load_score = rejection_rate * 100
        
        with self.lock:
            self.load_metrics.append((current_time, load_score))
    
    async def _adapt_limits(self):
        """Adapt rate limits based on load."""
        current_time = time.time()
        
        # Only adapt every 10 seconds
        if current_time - self.last_adaptation < 10:
            return
        
        with self.lock:
            if len(self.load_metrics) < 5:
                return
            
            # Calculate average load over last period
            recent_loads = [load for _, load in self.load_metrics[-10:]]
            avg_load = sum(recent_loads) / len(recent_loads)
            
            # Determine target limit based on load
            if avg_load > 80:  # High load
                target_limit = self.config.min_requests
            elif avg_load < 30:  # Low load
                target_limit = self.config.max_requests_adaptive
            else:  # Medium load
                # Linear interpolation
                ratio = (80 - avg_load) / 50  # (80-30)
                target_limit = int(self.config.min_requests + 
                                 ratio * (self.config.max_requests_adaptive - self.config.min_requests))
            
            # Gradually adapt to target
            diff = target_limit - self.current_limit
            adaptation = diff * self.config.adaptation_factor
            new_limit = int(self.current_limit + adaptation)
            
            # Update if significant change
            if abs(new_limit - self.current_limit) >= 5:
                old_limit = self.current_limit
                self.current_limit = max(self.config.min_requests, 
                                       min(self.config.max_requests_adaptive, new_limit))
                
                # Update base limiter config
                self.base_limiter.config.max_requests = self.current_limit
                self.base_limiter.config.refill_rate = self.current_limit / self.config.window_size
                
                self.last_adaptation = current_time
                
                logger.info(f"Adaptive rate limiter adjusted: {old_limit} -> {self.current_limit} "
                           f"(load: {avg_load:.1f}%)")
    
    def get_metrics(self) -> dict:
        """Get adaptive rate limiter metrics."""
        base_metrics = self.base_limiter.get_metrics()
        
        with self.lock:
            recent_loads = [load for _, load in self.load_metrics[-10:]] if self.load_metrics else [0]
            avg_load = sum(recent_loads) / len(recent_loads)
            
            base_metrics.update({
                'adaptive_current_limit': self.current_limit,
                'adaptive_avg_load': avg_load,
                'adaptive_min_limit': self.config.min_requests,
                'adaptive_max_limit': self.config.max_requests_adaptive,
                'load_samples': len(self.load_metrics)
            })
        
        return base_metrics
    
    def reset(self, key: str | None = None):
        """Reset adaptive limiter state."""
        self.base_limiter.reset(key)
        if not key:  # Full reset
            with self.lock:
                self.current_limit = self.config.max_requests
                self.load_metrics.clear()
                self.last_adaptation = time.time()


class MultiTierRateLimiter:
    """Multi-tier rate limiter with global and per-client limits."""
    
    def __init__(self, global_config: RateLimitConfig, client_config: RateLimitConfig | None = None):
        self.global_limiter = self._create_limiter(global_config)
        
        if client_config:
            self.client_config = client_config
        else:
            # Create default client config
            self.client_config = RateLimitConfig(
                algorithm=global_config.algorithm,
                max_requests=global_config.per_client_max,
                window_size=global_config.window_size,
                burst_size=global_config.per_client_max * 2,
                refill_rate=global_config.per_client_max / global_config.window_size
            )
        
        self.client_limiters: dict[str, RateLimiter] = {}
        self.penalties: dict[str, float] = {}  # Client -> penalty end time
        self.lock = threading.RLock()
        
        logger.info("Multi-tier rate limiter initialized with global and per-client limits")
    
    def _create_limiter(self, config: RateLimitConfig) -> RateLimiter:
        """Create appropriate rate limiter based on algorithm."""
        if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return TokenBucketRateLimiter(config)
        elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return SlidingWindowRateLimiter(config)
        elif config.algorithm == RateLimitAlgorithm.ADAPTIVE:
            return AdaptiveRateLimiter(config)
        else:
            return TokenBucketRateLimiter(config)  # Default
    
    async def is_allowed(self, client_id: str, tokens: int = 1) -> tuple[bool, float | None, str]:
        """Check if request is allowed. Returns (allowed, retry_after, limit_type)."""
        current_time = time.time()
        
        # Check if client is under penalty
        with self.lock:
            if client_id in self.penalties:
                if current_time < self.penalties[client_id]:
                    remaining_penalty = self.penalties[client_id] - current_time
                    return False, remaining_penalty, "penalty"
                else:
                    del self.penalties[client_id]
        
        # Check global limit first
        global_allowed, global_retry = await self.global_limiter.is_allowed("global", tokens)
        if not global_allowed:
            return False, global_retry, "global"
        
        # Check per-client limit
        if self.client_config.per_client_enabled:
            with self.lock:
                if client_id not in self.client_limiters:
                    self.client_limiters[client_id] = self._create_limiter(self.client_config)
            
            client_allowed, client_retry = await self.client_limiters[client_id].is_allowed(client_id, tokens)
            if not client_allowed:
                # Apply penalty for repeated violations
                await self._apply_penalty(client_id)
                return False, client_retry, "client"
        
        return True, None, "allowed"
    
    async def _apply_penalty(self, client_id: str):
        """Apply penalty to client for rate limit violations."""
        current_time = time.time()
        
        with self.lock:
            # Calculate penalty duration
            base_penalty = self.client_config.window_size
            penalty_duration = base_penalty * self.client_config.penalty_multiplier
            penalty_duration = min(penalty_duration, self.client_config.max_penalty_duration)
            
            self.penalties[client_id] = current_time + penalty_duration
            
            logger.warning(f"Applied penalty to client {client_id}: {penalty_duration}s")
    
    def get_metrics(self) -> dict:
        """Get comprehensive rate limiting metrics."""
        global_metrics = self.global_limiter.get_metrics()
        
        with self.lock:
            client_metrics = {}
            for client_id, limiter in self.client_limiters.items():
                client_metrics[client_id] = limiter.get_metrics()
            
            current_time = time.time()
            active_penalties = {
                client_id: penalty_end - current_time 
                for client_id, penalty_end in self.penalties.items() 
                if penalty_end > current_time
            }
            
            return {
                'global': global_metrics,
                'clients': client_metrics,
                'penalties': {
                    'active': active_penalties,
                    'total_clients_penalized': len(self.penalties)
                },
                'summary': {
                    'total_clients': len(self.client_limiters),
                    'penalized_clients': len(active_penalties)
                }
            }
    
    def reset(self, client_id: str | None = None):
        """Reset rate limiter state."""
        if client_id:
            with self.lock:
                if client_id in self.client_limiters:
                    self.client_limiters[client_id].reset()
                if client_id in self.penalties:
                    del self.penalties[client_id]
        else:
            self.global_limiter.reset()
            with self.lock:
                for limiter in self.client_limiters.values():
                    limiter.reset()
                self.client_limiters.clear()
                self.penalties.clear()


def get_client_id(request_info: dict) -> str:
    """Extract client identifier from request information."""
    # Try different identification methods
    if 'api_key' in request_info:
        return f"api_key:{request_info['api_key']}"
    elif 'user_id' in request_info:
        return f"user:{request_info['user_id']}"
    elif 'ip_address' in request_info:
        # Hash IP for privacy
        ip_hash = hashlib.sha256(request_info['ip_address'].encode()).hexdigest()[:16]
        return f"ip:{ip_hash}"
    else:
        return "anonymous"