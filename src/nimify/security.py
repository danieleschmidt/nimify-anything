"""Security utilities and hardening measures."""

import hashlib
import hmac
import secrets
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple
import ipaddress
import re
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter with IP-based tracking."""
    
    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens_per_second = requests_per_minute / 60.0
        
        # Track tokens per IP
        self.buckets: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"tokens": burst_size, "last_refill": time.time()}
        )
        
        # Track request history for analysis
        self.request_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
    
    def is_allowed(self, client_ip: str) -> Tuple[bool, int]:
        """Check if request is allowed, return (allowed, retry_after_seconds)."""
        now = time.time()
        bucket = self.buckets[client_ip]
        
        # Refill tokens based on time elapsed
        time_elapsed = now - bucket["last_refill"]
        bucket["tokens"] = min(
            self.burst_size,
            bucket["tokens"] + (time_elapsed * self.tokens_per_second)
        )
        bucket["last_refill"] = now
        
        # Record request attempt
        self.request_history[client_ip].append(now)
        
        # Check if we have tokens available
        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True, 0
        else:
            # Calculate retry after time
            retry_after = int((1.0 - bucket["tokens"]) / self.tokens_per_second)
            return False, retry_after
    
    def is_suspicious_activity(self, client_ip: str) -> bool:
        """Detect suspicious activity patterns."""
        history = self.request_history[client_ip]
        if len(history) < 10:
            return False
        
        now = time.time()
        recent_requests = [req for req in history if now - req < 60]  # Last minute
        
        # Suspicious if more than 3x the normal rate
        if len(recent_requests) > self.requests_per_minute * 3:
            return True
        
        # Check for burst patterns (many requests in short time)
        if len(recent_requests) > 20:
            time_span = max(recent_requests) - min(recent_requests)
            if time_span < 10:  # 20+ requests in 10 seconds
                return True
        
        return False


class IPBlocklist:
    """Manage IP address blocklist with automatic expiration."""
    
    def __init__(self):
        self.blocked_ips: Dict[str, datetime] = {}
        self.blocked_networks: Dict[str, datetime] = {}
        self.whitelist_networks: Set[str] = {
            "127.0.0.0/8",    # localhost
            "10.0.0.0/8",     # private
            "172.16.0.0/12",  # private
            "192.168.0.0/16"  # private
        }
    
    def block_ip(self, ip: str, duration_minutes: int = 60):
        """Block an IP address for specified duration."""
        expiry = datetime.now() + timedelta(minutes=duration_minutes)
        self.blocked_ips[ip] = expiry
        logger.warning(f"Blocked IP {ip} until {expiry}")
    
    def block_network(self, network: str, duration_minutes: int = 60):
        """Block a network range for specified duration."""
        try:
            ipaddress.ip_network(network)  # Validate network format
            expiry = datetime.now() + timedelta(minutes=duration_minutes)
            self.blocked_networks[network] = expiry
            logger.warning(f"Blocked network {network} until {expiry}")
        except ValueError as e:
            logger.error(f"Invalid network format {network}: {e}")
    
    def is_blocked(self, ip: str) -> bool:
        """Check if an IP is blocked."""
        now = datetime.now()
        
        # Clean expired entries
        self.blocked_ips = {k: v for k, v in self.blocked_ips.items() if v > now}
        self.blocked_networks = {k: v for k, v in self.blocked_networks.items() if v > now}
        
        # Check direct IP block
        if ip in self.blocked_ips:
            return True
        
        # Check network blocks
        try:
            ip_obj = ipaddress.ip_address(ip)
            for network, expiry in self.blocked_networks.items():
                if expiry > now and ip_obj in ipaddress.ip_network(network):
                    return True
        except ValueError:
            logger.error(f"Invalid IP address: {ip}")
            return True  # Block invalid IPs
        
        return False
    
    def is_whitelisted(self, ip: str) -> bool:
        """Check if IP is in whitelist."""
        try:
            ip_obj = ipaddress.ip_address(ip)
            for network in self.whitelist_networks:
                if ip_obj in ipaddress.ip_network(network):
                    return True
        except ValueError:
            return False
        
        return False


class InputSanitizer:
    """Sanitize and validate inputs for security."""
    
    # Patterns that indicate potential attacks
    ATTACK_PATTERNS = [
        # SQL injection
        (re.compile(r'(\bunion\b|\bselect\b|\binsert\b|\bdelete\b|\bdrop\b)', re.IGNORECASE), "sql_injection"),
        
        # Script injection
        (re.compile(r'<script|javascript:|data:text/html', re.IGNORECASE), "script_injection"),
        
        # Command injection
        (re.compile(r'[;&|`$]|\$\(|\|\||\&\&', re.IGNORECASE), "command_injection"),
        
        # Path traversal
        (re.compile(r'\.\.[\\/]|[\\/]\.\.', re.IGNORECASE), "path_traversal"),
        
        # LDAP injection
        (re.compile(r'[()&|!]', re.IGNORECASE), "ldap_injection"),
        
        # NoSQL injection
        (re.compile(r'\$where|\$ne|\$gt|\$lt', re.IGNORECASE), "nosql_injection"),
    ]
    
    @classmethod
    def scan_for_attacks(cls, content: str) -> List[str]:
        """Scan content for potential attack patterns."""
        detected_attacks = []
        
        for pattern, attack_type in cls.ATTACK_PATTERNS:
            if pattern.search(content):
                detected_attacks.append(attack_type)
        
        return detected_attacks
    
    @classmethod
    def sanitize_string(cls, input_str: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not isinstance(input_str, str):
            raise ValueError("Input must be a string")
        
        # Truncate if too long
        if len(input_str) > max_length:
            input_str = input_str[:max_length]
        
        # Remove null bytes and control characters
        sanitized = ''.join(char for char in input_str if ord(char) >= 32 or char in '\t\n\r')
        
        # Check for attack patterns
        attacks = cls.scan_for_attacks(sanitized)
        if attacks:
            logger.warning(f"Potential attacks detected: {attacks}")
            raise ValueError(f"Input contains potentially malicious content: {attacks}")
        
        return sanitized


class APIKeyManager:
    """Manage API keys for service authentication."""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict] = {}
        self.key_usage: Dict[str, List[datetime]] = defaultdict(list)
    
    def generate_api_key(self, name: str, permissions: List[str] = None) -> str:
        """Generate a new API key."""
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        self.api_keys[key_hash] = {
            "name": name,
            "permissions": permissions or ["predict"],
            "created": datetime.now(),
            "last_used": None,
            "usage_count": 0
        }
        
        logger.info(f"Generated API key for {name}")
        return api_key
    
    def validate_api_key(self, api_key: str, required_permission: str = "predict") -> bool:
        """Validate API key and check permissions."""
        if not api_key:
            return False
        
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash not in self.api_keys:
            logger.warning(f"Invalid API key attempt: {key_hash[:8]}...")
            return False
        
        key_info = self.api_keys[key_hash]
        
        # Check permissions
        if required_permission not in key_info["permissions"]:
            logger.warning(f"Insufficient permissions for key {key_info['name']}")
            return False
        
        # Update usage
        key_info["last_used"] = datetime.now()
        key_info["usage_count"] += 1
        self.key_usage[key_hash].append(datetime.now())
        
        return True
    
    def is_key_rate_limited(self, api_key: str, requests_per_hour: int = 1000) -> bool:
        """Check if API key is rate limited."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        
        recent_usage = [
            usage for usage in self.key_usage[key_hash]
            if usage > one_hour_ago
        ]
        
        return len(recent_usage) >= requests_per_hour


class SecurityHeaders:
    """Security headers for HTTP responses."""
    
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
    }
    
    @classmethod
    def get_headers(cls) -> Dict[str, str]:
        """Get security headers dictionary."""
        return cls.SECURITY_HEADERS.copy()


class ThreatDetector:
    """Detect various types of threats and attacks."""
    
    def __init__(self):
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.suspicious_patterns: Dict[str, int] = defaultdict(int)
    
    def record_failed_attempt(self, ip: str, attempt_type: str):
        """Record a failed authentication/validation attempt."""
        self.failed_attempts[ip].append(datetime.now())
        
        # Clean old attempts (older than 1 hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        self.failed_attempts[ip] = [
            attempt for attempt in self.failed_attempts[ip]
            if attempt > one_hour_ago
        ]
    
    def is_under_attack(self, ip: str, threshold: int = 5) -> bool:
        """Check if IP is showing signs of attack."""
        return len(self.failed_attempts[ip]) >= threshold
    
    def detect_pattern_anomaly(self, pattern: str) -> bool:
        """Detect unusual patterns in requests."""
        self.suspicious_patterns[pattern] += 1
        
        # If pattern appears frequently, it might be an attack
        return self.suspicious_patterns[pattern] > 10
    
    def analyze_request_content(self, content: str) -> Dict[str, int]:
        """Analyze request content for suspicious elements."""
        analysis = {
            "suspicious_keywords": 0,
            "encoded_content": 0,
            "sql_patterns": 0,
            "script_patterns": 0
        }
        
        # Suspicious keywords
        suspicious_words = ["admin", "root", "password", "token", "secret", "config"]
        for word in suspicious_words:
            if word.lower() in content.lower():
                analysis["suspicious_keywords"] += 1
        
        # Base64 encoding (potential payload)
        if re.search(r'[A-Za-z0-9+/]{20,}={0,2}', content):
            analysis["encoded_content"] += 1
        
        # SQL injection patterns
        sql_patterns = ["union", "select", "insert", "delete", "drop", "exec"]
        for pattern in sql_patterns:
            if re.search(rf'\b{pattern}\b', content, re.IGNORECASE):
                analysis["sql_patterns"] += 1
        
        # Script injection patterns
        script_patterns = ["<script", "javascript:", "onload=", "onerror="]
        for pattern in script_patterns:
            if pattern.lower() in content.lower():
                analysis["script_patterns"] += 1
        
        return analysis


# Global instances (singleton pattern)
rate_limiter = RateLimiter()
ip_blocklist = IPBlocklist()
api_key_manager = APIKeyManager()
threat_detector = ThreatDetector()