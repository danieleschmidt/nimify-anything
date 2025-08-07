"""Input validation and security utilities."""

import re
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error."""
    pass


class ModelFileValidator:
    """Validates model files for security and integrity."""
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'.onnx', '.trt', '.engine', '.plan', '.pb', '.pt', '.pth'}
    
    # Maximum file size (1GB)
    MAX_FILE_SIZE = 1024 * 1024 * 1024
    
    # Minimum file size (1KB) - prevent empty files
    MIN_FILE_SIZE = 1024
    
    @classmethod
    def validate_file_path(cls, file_path: str) -> Path:
        """Validate model file path for security."""
        path = Path(file_path).resolve()
        
        # Check if file exists
        if not path.exists():
            raise ValidationError(f"Model file not found: {file_path}")
        
        # Check if it's actually a file
        if not path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")
        
        # Check file extension
        if path.suffix.lower() not in cls.ALLOWED_EXTENSIONS:
            raise ValidationError(
                f"Unsupported file extension: {path.suffix}. "
                f"Allowed: {', '.join(cls.ALLOWED_EXTENSIONS)}"
            )
        
        # Check file size
        file_size = path.stat().st_size
        if file_size > cls.MAX_FILE_SIZE:
            raise ValidationError(f"File too large: {file_size} bytes (max: {cls.MAX_FILE_SIZE})")
        
        if file_size < cls.MIN_FILE_SIZE:
            raise ValidationError(f"File too small: {file_size} bytes (min: {cls.MIN_FILE_SIZE})")
        
        # Check for path traversal attempts
        if '..' in str(path):
            raise ValidationError(f"Invalid path detected (path traversal): {file_path}")
        
        return path
    
    @classmethod
    def compute_file_hash(cls, file_path: Path) -> str:
        """Compute SHA-256 hash of the model file."""
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except IOError as e:
            raise ValidationError(f"Cannot read file for hashing: {e}")


class ServiceNameValidator:
    """Validates service names for compliance and security."""
    
    # Pattern for valid service names (DNS-safe)
    VALID_NAME_PATTERN = re.compile(r'^[a-z]([a-z0-9-]*[a-z0-9])?$')
    
    # Reserved names that should not be used
    RESERVED_NAMES = {
        'nimify', 'api', 'health', 'metrics', 'docs', 'openapi', 'swagger',
        'admin', 'root', 'system', 'default', 'kubernetes', 'docker',
        'localhost', 'nvidia', 'triton', 'nim'
    }
    
    @classmethod
    def validate_service_name(cls, name: str) -> str:
        """Validate service name for compliance."""
        if not isinstance(name, str):
            raise ValidationError("Service name must be a string")
        
        # Check length
        if len(name) < 2:
            raise ValidationError("Service name must be at least 2 characters long")
        
        if len(name) > 63:
            raise ValidationError("Service name must be 63 characters or less")
        
        # Check pattern (lowercase, alphanumeric, hyphens)
        if not cls.VALID_NAME_PATTERN.match(name):
            raise ValidationError(
                "Service name must start with a letter, contain only lowercase "
                "letters, numbers, and hyphens, and end with a letter or number"
            )
        
        # Check reserved names
        if name.lower() in cls.RESERVED_NAMES:
            raise ValidationError(f"'{name}' is a reserved name and cannot be used")
        
        return name


class ConfigurationValidator:
    """Validates configuration parameters."""
    
    @classmethod
    def validate_port(cls, port: int) -> int:
        """Validate port number."""
        if not isinstance(port, int):
            raise ValidationError("Port must be an integer")
        
        if port < 1024 or port > 65535:
            raise ValidationError("Port must be between 1024 and 65535")
        
        return port
    
    @classmethod
    def validate_batch_size(cls, batch_size: int) -> int:
        """Validate batch size parameter."""
        if not isinstance(batch_size, int):
            raise ValidationError("Batch size must be an integer")
        
        if batch_size < 1 or batch_size > 1024:
            raise ValidationError("Batch size must be between 1 and 1024")
        
        return batch_size
    
    @classmethod
    def validate_memory_limit(cls, memory: str) -> str:
        """Validate memory limit string (e.g., '4Gi', '512Mi')."""
        pattern = re.compile(r'^(\d+(\.\d+)?)(Mi|Gi|Ti)$')
        if not pattern.match(memory):
            raise ValidationError(
                f"Invalid memory format: {memory}. Use format like '4Gi', '512Mi'"
            )
        
        return memory


class SecurityValidator:
    """Security-focused validators."""
    
    DANGEROUS_PATTERNS = [
        # Command injection patterns
        r'[;&|`$]',
        r'\$\(',
        r'`.*`',
        # Path traversal
        r'\.\.',
        # Script injection
        r'<script',
        r'javascript:',
        # SQL injection patterns
        r'(union|select|insert|delete|drop)\s',
        # Shell metacharacters
        r'[<>|&;]'
    ]
    
    @classmethod
    def scan_for_dangerous_content(cls, content: str) -> List[str]:
        """Scan content for potentially dangerous patterns."""
        dangerous_matches = []
        
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                dangerous_matches.append(pattern)
        
        return dangerous_matches
    
    @classmethod
    def validate_environment_variables(cls, env_vars: Dict[str, str]) -> Dict[str, str]:
        """Validate environment variables for security."""
        validated = {}
        dangerous_keys = {'PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH'}
        
        for key, value in env_vars.items():
            # Check for dangerous environment variable names
            if key.upper() in dangerous_keys:
                logger.warning(f"Potentially dangerous environment variable: {key}")
            
            # Scan value for dangerous content
            dangerous_patterns = cls.scan_for_dangerous_content(value)
            if dangerous_patterns:
                raise ValidationError(
                    f"Environment variable '{key}' contains dangerous patterns: {dangerous_patterns}"
                )
            
            validated[key] = value
        
        return validated


class RequestValidator(BaseModel):
    """Pydantic model for validating API requests."""
    
    input: List[List[float]] = Field(
        ..., 
        description="Input data for inference",
        min_items=1,
        max_items=64  # Maximum batch size
    )
    
    @validator('input')
    def validate_input_dimensions(cls, v):
        """Validate input tensor dimensions."""
        if not v:
            raise ValueError("Input cannot be empty")
        
        # Check that all rows have the same length
        first_row_len = len(v[0]) if v else 0
        for i, row in enumerate(v):
            if len(row) != first_row_len:
                raise ValueError(f"All input rows must have the same length. Row {i} has length {len(row)}, expected {first_row_len}")
        
        # Check for NaN or infinite values
        for i, row in enumerate(v):
            for j, val in enumerate(row):
                if not isinstance(val, (int, float)):
                    raise ValueError(f"All values must be numeric. Found {type(val)} at position [{i}][{j}]")
                
                import math
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(f"Invalid value (NaN or Inf) at position [{i}][{j}]")
        
        return v


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and other issues."""
    # Remove path separators and dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "unnamed"
    
    return sanitized


def validate_docker_image_name(image_name: str) -> str:
    """Validate Docker image name format."""
    # Docker image name pattern
    pattern = re.compile(
        r'^(?:[a-z0-9]([a-z0-9-]*[a-z0-9])?\.)*[a-z0-9]([a-z0-9-]*[a-z0-9])?'
        r'(:[0-9]+)?(/[a-z0-9]([a-z0-9._-]*[a-z0-9])?)*'
        r'(:[a-zA-Z0-9][a-zA-Z0-9._-]{0,127})?$'
    )
    
    if not pattern.match(image_name):
        raise ValidationError(f"Invalid Docker image name format: {image_name}")
    
    return image_name