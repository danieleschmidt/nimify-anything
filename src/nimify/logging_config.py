"""Logging configuration for Nimify services."""

import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import os


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info'):
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)


class SecurityAuditHandler(logging.Handler):
    """Special handler for security-related log events."""
    
    def __init__(self, audit_file: str):
        super().__init__()
        self.audit_file = Path(audit_file)
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
    
    def emit(self, record: logging.LogRecord):
        """Emit security audit log."""
        if hasattr(record, 'security_event'):
            audit_entry = {
                'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
                'event_type': getattr(record, 'security_event', 'unknown'),
                'level': record.levelname,
                'message': record.getMessage(),
                'source': record.name,
                'user_id': getattr(record, 'user_id', None),
                'ip_address': getattr(record, 'ip_address', None),
                'request_id': getattr(record, 'request_id', None)
            }
            
            with open(self.audit_file, 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')


def setup_logging(
    service_name: str,
    log_level: str = "INFO",
    log_format: str = "structured",
    log_file: Optional[str] = None,
    enable_audit: bool = True
) -> None:
    """Set up logging configuration for a Nimify service."""
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if log_format.lower() == "structured":
        console_formatter = StructuredFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(numeric_level)
        
        if log_format.lower() == "structured":
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Security audit handler
    if enable_audit:
        audit_file = f"/tmp/{service_name}_security_audit.log"
        audit_handler = SecurityAuditHandler(audit_file)
        audit_handler.setLevel(logging.WARNING)
        audit_handler.setFormatter(StructuredFormatter())
        
        # Add to security logger specifically
        security_logger = logging.getLogger('nimify.security')
        security_logger.addHandler(audit_handler)
    
    # Set up specific logger levels
    logging.getLogger('nimify').setLevel(numeric_level)
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('fastapi').setLevel(logging.WARNING)
    
    # Add service context to all logs
    class ServiceContextFilter(logging.Filter):
        def filter(self, record):
            record.service_name = service_name
            return True
    
    service_filter = ServiceContextFilter()
    for handler in root_logger.handlers:
        handler.addFilter(service_filter)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


def log_security_event(
    event_type: str,
    message: str,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    request_id: Optional[str] = None,
    level: int = logging.WARNING
) -> None:
    """Log a security-related event."""
    security_logger = logging.getLogger('nimify.security')
    
    # Create extra context
    extra = {
        'security_event': event_type,
        'user_id': user_id,
        'ip_address': ip_address,
        'request_id': request_id
    }
    
    security_logger.log(level, message, extra=extra)


def log_model_operation(
    operation: str,
    model_path: str,
    service_name: str,
    success: bool,
    duration_ms: float,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log model-related operations."""
    logger = logging.getLogger('nimify.model')
    
    extra = {
        'operation': operation,
        'model_path': model_path,
        'service_name': service_name,
        'success': success,
        'duration_ms': duration_ms,
        'metadata': metadata or {}
    }
    
    level = logging.INFO if success else logging.ERROR
    message = f"Model operation '{operation}' {'succeeded' if success else 'failed'}"
    
    logger.log(level, message, extra=extra)


def log_api_request(
    method: str,
    endpoint: str,
    status_code: int,
    duration_ms: float,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    request_id: Optional[str] = None
) -> None:
    """Log API requests."""
    logger = logging.getLogger('nimify.api')
    
    extra = {
        'http_method': method,
        'endpoint': endpoint,
        'status_code': status_code,
        'duration_ms': duration_ms,
        'ip_address': ip_address,
        'user_agent': user_agent,
        'request_id': request_id
    }
    
    if status_code >= 400:
        level = logging.WARNING if status_code < 500 else logging.ERROR
    else:
        level = logging.INFO
    
    message = f"{method} {endpoint} - {status_code} ({duration_ms:.2f}ms)"
    
    logger.log(level, message, extra=extra)


# Performance monitoring logger
def log_performance_metric(
    metric_name: str,
    value: float,
    unit: str,
    tags: Optional[Dict[str, str]] = None
) -> None:
    """Log performance metrics."""
    logger = logging.getLogger('nimify.metrics')
    
    extra = {
        'metric_name': metric_name,
        'value': value,
        'unit': unit,
        'tags': tags or {}
    }
    
    logger.info(f"Metric: {metric_name}={value}{unit}", extra=extra)