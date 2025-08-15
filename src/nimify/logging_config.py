"""Logging configuration for NIM services."""

import logging
import json
import time
from typing import Optional, Dict, Any


def setup_logging(service_name: str, log_level: str = "INFO", enable_audit: bool = True):
    """Set up structured logging for the service."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, log_level))
    
    return logger


def log_api_request(
    method: str, 
    endpoint: str, 
    status_code: int, 
    duration_ms: float,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    request_id: Optional[str] = None
):
    """Log API request with structured format."""
    logger = logging.getLogger("nim-service")
    
    log_data = {
        "method": method,
        "endpoint": endpoint,
        "status_code": status_code,
        "duration_ms": duration_ms,
        "ip_address": ip_address,
        "user_agent": user_agent,
        "request_id": request_id,
        "timestamp": time.time()
    }
    
    if status_code >= 400:
        logger.warning(f"API Request - {method} {endpoint} - {status_code} - {duration_ms:.2f}ms", extra=log_data)
    else:
        logger.info(f"API Request - {method} {endpoint} - {status_code} - {duration_ms:.2f}ms", extra=log_data)


def log_security_event(
    event_type: str,
    message: str,
    ip_address: Optional[str] = None,
    request_id: Optional[str] = None,
    **kwargs
):
    """Log security-related events."""
    logger = logging.getLogger("nim-service.security")
    
    log_data = {
        "event_type": event_type,
        "message": message,
        "ip_address": ip_address,
        "request_id": request_id,
        "timestamp": time.time(),
        **kwargs
    }
    
    logger.warning(f"Security Event - {event_type}: {message}", extra=log_data)


def log_performance_metric(metric_name: str, value: float, unit: str):
    """Log performance metrics."""
    logger = logging.getLogger("nim-service.performance")
    
    log_data = {
        "metric_name": metric_name,
        "value": value,
        "unit": unit,
        "timestamp": time.time()
    }
    
    logger.info(f"Performance - {metric_name}: {value} {unit}", extra=log_data)