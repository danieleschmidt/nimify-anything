"""Advanced security features for NIM services including authentication, authorization, and threat protection."""

import hashlib
import hmac
import jwt
import logging
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations."""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"  
    AUTHORIZED = "authorized"
    ADMIN = "admin"


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityConfig:
    """Security configuration for NIM services."""
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    enable_api_key_auth: bool = True
    enable_request_signing: bool = False
    enable_threat_detection: bool = True
    max_request_size_mb: int = 10
    allowed_origins: list[str] = None


class APIKey(BaseModel):
    """API Key model with metadata."""
    key_id: str
    key_hash: str
    name: str
    created_at: float
    expires_at: Optional[float] = None
    permissions: list[str] = []
    is_active: bool = True
    usage_count: int = 0
    last_used: Optional[float] = None


class User(BaseModel):
    """User model for authentication."""
    user_id: str
    username: str
    email: str
    roles: list[str] = []
    permissions: list[str] = []
    is_active: bool = True
    created_at: float
    last_login: Optional[float] = None


class ThreatDetectionRule:
    """Rule for detecting security threats."""
    
    def __init__(self, name: str, description: str, threat_level: ThreatLevel, check_func: callable):
        self.name = name
        self.description = description
        self.threat_level = threat_level
        self.check_func = check_func
    
    async def evaluate(self, request: Request, context: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Evaluate rule against request and context."""
        try:
            result = await self.check_func(request, context)
            if result:
                return {
                    "rule_name": self.name,
                    "description": self.description,
                    "threat_level": self.threat_level.value,
                    "details": result
                }
        except Exception as e:
            logger.error(f"Threat detection rule {self.name} failed: {e}")
        
        return None


class RateLimiter:
    """Token bucket rate limiter with different limits per endpoint."""
    
    def __init__(self):
        self.buckets = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    async def check_rate_limit(
        self,
        identifier: str,
        requests_per_window: int = 100,
        window_seconds: int = 60
    ) -> tuple[bool, dict[str, Any]]:
        """Check if request is within rate limits."""
        current_time = time.time()
        
        # Cleanup old buckets periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            await self._cleanup_buckets(current_time)
        
        # Get or create bucket
        if identifier not in self.buckets:
            self.buckets[identifier] = {
                "tokens": requests_per_window,
                "last_refill": current_time,
                "window_seconds": window_seconds,
                "requests_per_window": requests_per_window,
                "total_requests": 0
            }
        
        bucket = self.buckets[identifier]
        
        # Refill tokens based on time passed
        time_passed = current_time - bucket["last_refill"]
        if time_passed > 0:
            tokens_to_add = (time_passed / window_seconds) * requests_per_window
            bucket["tokens"] = min(requests_per_window, bucket["tokens"] + tokens_to_add)
            bucket["last_refill"] = current_time
        
        # Check if request is allowed
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            bucket["total_requests"] += 1
            
            return True, {
                "allowed": True,
                "tokens_remaining": int(bucket["tokens"]),
                "reset_time": current_time + window_seconds,
                "total_requests": bucket["total_requests"]
            }
        else:
            return False, {
                "allowed": False,
                "tokens_remaining": 0,
                "reset_time": current_time + window_seconds,
                "total_requests": bucket["total_requests"],
                "retry_after": window_seconds - (current_time % window_seconds)
            }
    
    async def _cleanup_buckets(self, current_time: float):
        """Remove old, unused buckets to prevent memory leaks."""
        cutoff_time = current_time - 3600  # 1 hour
        
        buckets_to_remove = []
        for identifier, bucket in self.buckets.items():
            if bucket["last_refill"] < cutoff_time:
                buckets_to_remove.append(identifier)
        
        for identifier in buckets_to_remove:
            del self.buckets[identifier]
        
        self.last_cleanup = current_time
        
        if buckets_to_remove:
            logger.info(f"Cleaned up {len(buckets_to_remove)} old rate limit buckets")


class SecurityManager:
    """Comprehensive security manager for NIM services."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.rate_limiter = RateLimiter()
        self.api_keys = {}  # In production, use secure storage
        self.users = {}     # In production, use database
        self.threat_rules = []
        self.security_events = []
        
        # Initialize default threat detection rules
        self._initialize_threat_detection()
    
    def _initialize_threat_detection(self):
        """Initialize default threat detection rules."""
        
        async def check_sql_injection(request: Request, context: dict) -> Optional[dict]:
            """Detect potential SQL injection attempts."""
            body = context.get("body", "")
            query_params = str(request.query_params)
            
            sql_patterns = [
                "' OR '1'='1",
                "'; DROP TABLE",
                "UNION SELECT",
                "SELECT * FROM",
                "--",
                "/*",
                "xp_cmdshell"
            ]
            
            for pattern in sql_patterns:
                if pattern.lower() in body.lower() or pattern.lower() in query_params.lower():
                    return {
                        "pattern_matched": pattern,
                        "location": "body" if pattern.lower() in body.lower() else "query_params"
                    }
            
            return None
        
        async def check_request_size(request: Request, context: dict) -> Optional[dict]:
            """Check for unusually large requests."""
            content_length = request.headers.get("content-length")
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > self.config.max_request_size_mb:
                    return {
                        "request_size_mb": size_mb,
                        "limit_mb": self.config.max_request_size_mb
                    }
            
            return None
        
        async def check_suspicious_user_agent(request: Request, context: dict) -> Optional[dict]:
            """Detect suspicious user agents."""
            user_agent = request.headers.get("user-agent", "").lower()
            
            suspicious_patterns = [
                "sqlmap",
                "nikto", 
                "nmap",
                "masscan",
                "curl/7.0",  # Very old curl versions
                "python-requests/0.0",  # Default requests without user agent
            ]
            
            for pattern in suspicious_patterns:
                if pattern in user_agent:
                    return {
                        "user_agent": user_agent,
                        "suspicious_pattern": pattern
                    }
            
            return None
        
        async def check_rapid_requests(request: Request, context: dict) -> Optional[dict]:
            """Detect rapid successive requests from same IP."""
            client_ip = context.get("client_ip", "unknown")
            current_time = time.time()
            
            # Check if more than 50 requests in last 10 seconds
            recent_events = [
                event for event in self.security_events
                if event.get("client_ip") == client_ip and 
                   current_time - event.get("timestamp", 0) < 10
            ]
            
            if len(recent_events) > 50:
                return {
                    "client_ip": client_ip,
                    "request_count": len(recent_events),
                    "time_window_seconds": 10
                }
            
            return None
        
        # Register threat detection rules
        self.threat_rules = [
            ThreatDetectionRule("sql_injection", "SQL injection attempt detection", ThreatLevel.HIGH, check_sql_injection),
            ThreatDetectionRule("request_size", "Oversized request detection", ThreatLevel.MEDIUM, check_request_size),
            ThreatDetectionRule("suspicious_user_agent", "Suspicious user agent detection", ThreatLevel.MEDIUM, check_suspicious_user_agent),
            ThreatDetectionRule("rapid_requests", "Rapid request detection", ThreatLevel.HIGH, check_rapid_requests),
        ]
    
    def create_api_key(self, name: str, permissions: list[str] = None, expires_hours: int = None) -> tuple[str, APIKey]:
        """Create new API key with specified permissions."""
        key_id = secrets.token_urlsafe(16)
        raw_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        expires_at = None
        if expires_hours:
            expires_at = time.time() + (expires_hours * 3600)
        
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            created_at=time.time(),
            expires_at=expires_at,
            permissions=permissions or [],
            is_active=True,
            usage_count=0
        )
        
        self.api_keys[key_id] = api_key
        
        # Return the full key (only time it's available in plain text)
        full_key = f"nim_{key_id}_{raw_key}"
        return full_key, api_key
    
    def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """Validate API key and return associated metadata."""
        try:
            if not api_key.startswith("nim_"):
                return None
            
            parts = api_key.split("_", 2)
            if len(parts) != 3:
                return None
            
            key_id = parts[1]
            raw_key = parts[2]
            
            stored_key = self.api_keys.get(key_id)
            if not stored_key or not stored_key.is_active:
                return None
            
            # Check expiry
            if stored_key.expires_at and time.time() > stored_key.expires_at:
                stored_key.is_active = False
                return None
            
            # Verify key hash
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
            if not hmac.compare_digest(stored_key.key_hash, key_hash):
                return None
            
            # Update usage statistics
            stored_key.usage_count += 1
            stored_key.last_used = time.time()
            
            return stored_key
            
        except Exception as e:
            logger.error(f"API key validation error: {e}")
            return None
    
    def create_jwt_token(self, user: User, additional_claims: dict = None) -> str:
        """Create JWT token for authenticated user."""
        now = time.time()
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "roles": user.roles,
            "permissions": user.permissions,
            "iat": now,
            "exp": now + (self.config.jwt_expiry_hours * 3600)
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        token = jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
        return token
    
    def validate_jwt_token(self, token: str) -> Optional[dict]:
        """Validate JWT token and return payload."""
        try:
            payload = jwt.decode(token, self.config.jwt_secret_key, algorithms=[self.config.jwt_algorithm])
            
            # Check if user is still active
            user = self.users.get(payload.get("sub"))
            if user and not user.is_active:
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    
    async def check_permissions(self, required_permissions: list[str], user_permissions: list[str]) -> bool:
        """Check if user has required permissions."""
        return all(perm in user_permissions for perm in required_permissions)
    
    async def enforce_security(
        self,
        request: Request,
        security_level: SecurityLevel = SecurityLevel.PUBLIC,
        required_permissions: list[str] = None,
        rate_limit_key: str = None
    ) -> dict[str, Any]:
        """Comprehensive security enforcement."""
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")
        
        security_context = {
            "client_ip": client_ip,
            "user_agent": user_agent,
            "timestamp": time.time(),
            "endpoint": str(request.url.path),
            "method": request.method,
            "user": None,
            "api_key": None
        }
        
        # Rate limiting
        if rate_limit_key:
            allowed, rate_info = await self.rate_limiter.check_rate_limit(
                rate_limit_key,
                self.config.rate_limit_requests,
                self.config.rate_limit_window_seconds
            )
            
            if not allowed:
                await self._log_security_event("rate_limit_exceeded", security_context, rate_info)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(int(rate_info.get("retry_after", 60)))}
                )
        
        # Authentication for non-public endpoints
        if security_level != SecurityLevel.PUBLIC:
            auth_result = await self._authenticate_request(request)
            
            if not auth_result:
                await self._log_security_event("authentication_failed", security_context)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            security_context.update(auth_result)
        
        # Authorization check
        if required_permissions and security_context.get("permissions"):
            has_permission = await self.check_permissions(
                required_permissions,
                security_context["permissions"]
            )
            
            if not has_permission:
                await self._log_security_event("authorization_failed", security_context, {
                    "required_permissions": required_permissions,
                    "user_permissions": security_context["permissions"]
                })
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
        
        # Threat detection
        if self.config.enable_threat_detection:
            await self._run_threat_detection(request, security_context)
        
        return security_context
    
    async def _authenticate_request(self, request: Request) -> Optional[dict[str, Any]]:
        """Authenticate request using API key or JWT token."""
        # Check for API key in header
        api_key = request.headers.get("x-api-key")
        if api_key:
            validated_key = self.validate_api_key(api_key)
            if validated_key:
                return {
                    "auth_type": "api_key",
                    "api_key": validated_key,
                    "permissions": validated_key.permissions
                }
        
        # Check for JWT token in Authorization header
        authorization = request.headers.get("authorization")
        if authorization and authorization.startswith("Bearer "):
            token = authorization.split(" ", 1)[1]
            payload = self.validate_jwt_token(token)
            if payload:
                return {
                    "auth_type": "jwt",
                    "user_id": payload.get("sub"),
                    "username": payload.get("username"),
                    "roles": payload.get("roles", []),
                    "permissions": payload.get("permissions", [])
                }
        
        return None
    
    async def _run_threat_detection(self, request: Request, context: dict[str, Any]):
        """Run threat detection rules against request."""
        # Get request body for analysis
        body = ""
        try:
            if request.method in ["POST", "PUT", "PATCH"]:
                body = await request.body()
                body = body.decode("utf-8", errors="ignore")
        except Exception:
            pass
        
        threat_context = {**context, "body": body}
        
        for rule in self.threat_rules:
            threat_result = await rule.evaluate(request, threat_context)
            if threat_result:
                await self._log_security_event("threat_detected", context, threat_result)
                
                # Block high/critical threats
                if rule.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Security threat detected: {rule.description}"
                    )
    
    async def _log_security_event(self, event_type: str, context: dict[str, Any], details: dict[str, Any] = None):
        """Log security event for monitoring and analysis."""
        event = {
            "event_type": event_type,
            "timestamp": time.time(),
            "client_ip": context.get("client_ip"),
            "endpoint": context.get("endpoint"),
            "user_agent": context.get("user_agent"),
            "details": details or {}
        }
        
        self.security_events.append(event)
        
        # Keep only last 10000 events to prevent memory issues
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-5000:]
        
        logger.warning(f"Security event: {event_type}", extra={"security_event": event})
    
    def get_security_statistics(self) -> dict[str, Any]:
        """Get comprehensive security statistics."""
        current_time = time.time()
        
        # Events in last hour
        recent_events = [
            event for event in self.security_events
            if current_time - event["timestamp"] < 3600
        ]
        
        event_counts = {}
        for event in recent_events:
            event_type = event["event_type"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Active API keys
        active_keys = len([key for key in self.api_keys.values() if key.is_active])
        
        # Rate limiter statistics
        rate_limit_buckets = len(self.rate_limiter.buckets)
        
        return {
            "total_security_events": len(self.security_events),
            "recent_events_1h": len(recent_events),
            "event_type_breakdown": event_counts,
            "active_api_keys": active_keys,
            "total_api_keys": len(self.api_keys),
            "rate_limit_buckets": rate_limit_buckets,
            "threat_detection_rules": len(self.threat_rules),
            "recent_threats": [
                event for event in recent_events
                if event["event_type"] == "threat_detected"
            ][-10:]  # Last 10 threats
        }


# Security middleware for FastAPI
class SecurityMiddleware:
    """FastAPI middleware for automatic security enforcement."""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
    
    async def __call__(self, request: Request, call_next):
        """Security middleware handler."""
        # Skip security for health check endpoints
        if request.url.path in ["/health", "/metrics", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        try:
            # Enforce basic security
            rate_limit_key = f"{request.client.host}:{request.url.path}" if request.client else request.url.path
            
            security_context = await self.security_manager.enforce_security(
                request,
                SecurityLevel.PUBLIC,  # Default level, override in endpoint
                rate_limit_key=rate_limit_key
            )
            
            # Add security context to request state
            request.state.security_context = security_context
            
            response = await call_next(request)
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            # Don't block requests on security middleware errors
            return await call_next(request)