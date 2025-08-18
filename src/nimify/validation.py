"""Input validation and security utilities for bioneuro-olfactory fusion."""

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error."""
    pass


class BioneuroDataValidator:
    """Validates bioneuro-olfactory data files for security and integrity."""
    
    # Allowed file extensions for neural data
    NEURAL_DATA_EXTENSIONS = {'.edf', '.bdf', '.fif', '.mat', '.npy', '.h5', '.hdf5'}
    
    # Allowed file extensions for olfactory data
    OLFACTORY_DATA_EXTENSIONS = {'.json', '.csv', '.xml', '.sdf', '.mol'}
    
    # Combined allowed extensions
    ALLOWED_EXTENSIONS = NEURAL_DATA_EXTENSIONS | OLFACTORY_DATA_EXTENSIONS
    
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
        except OSError as e:
            raise ValidationError(f"Cannot read file for hashing: {e}")


class ServiceNameValidator:
    """Validates bioneuro service names for compliance and security."""
    
    # Pattern for valid service names (DNS-safe)
    VALID_NAME_PATTERN = re.compile(r'^[a-z]([a-z0-9-]*[a-z0-9])?$')
    
    # Reserved names that should not be used
    RESERVED_NAMES = {
        'bioneuro', 'olfactory', 'neural', 'fusion', 'api', 'health', 'metrics', 
        'docs', 'openapi', 'swagger', 'admin', 'root', 'system', 'default', 
        'kubernetes', 'docker', 'localhost', 'nvidia', 'triton', 'nim',
        'eeg', 'fmri', 'meg', 'ephys', 'calcium'
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


class BioneuroConfigValidator:
    """Validates bioneuro-olfactory fusion configuration parameters."""
    
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
    def scan_for_dangerous_content(cls, content: str) -> list[str]:
        """Scan content for potentially dangerous patterns."""
        dangerous_matches = []
        
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                dangerous_matches.append(pattern)
        
        return dangerous_matches
    
    @classmethod
    def validate_environment_variables(cls, env_vars: dict[str, str]) -> dict[str, str]:
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
    
    input: list[list[float]] = Field(
        ..., 
        description="Input data for inference",
        min_items=1,
        max_items=64  # Maximum batch size
    )
    
    @validator('input')
    def validate_input_dimensions(self, v):
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
                if not isinstance(val, int | float):
                    raise ValueError(f"All values must be numeric. Found {type(val)} at position [{i}][{j}]")
                
                import math
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(f"Invalid value (NaN or Inf) at position [{i}][{j}]")
        
        return v


class ModelFileValidator:
    """Validates model files for security and compatibility."""
    
    ALLOWED_EXTENSIONS = {'.onnx', '.trt', '.engine', '.plan', '.pb', '.pth', '.pt'}
    MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5GB
    
    @classmethod
    def validate_model_file(cls, file_path: str) -> Path:
        """Validate model file."""
        path = Path(file_path).resolve()
        
        if not path.exists():
            raise ValidationError(f"Model file not found: {file_path}")
        
        if path.suffix.lower() not in cls.ALLOWED_EXTENSIONS:
            raise ValidationError(f"Unsupported file extension: {path.suffix}")
        
        file_size = path.stat().st_size
        if file_size > cls.MAX_FILE_SIZE:
            raise ValidationError(f"File too large: {file_size} bytes")
        
        return path


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


class DataQualityValidator:
    """Validates data quality for bioneuro-olfactory fusion."""
    
    @staticmethod
    def assess_neural_data_quality(neural_data: list[list[float]], sampling_rate: int) -> dict[str, Any]:
        """Assess quality of neural data."""
        import numpy as np
        
        data_array = np.array(neural_data)
        
        quality_metrics = {
            'n_channels': data_array.shape[0],
            'n_timepoints': data_array.shape[1],
            'duration_seconds': data_array.shape[1] / sampling_rate,
            'sampling_rate': sampling_rate
        }
        
        # Signal quality assessment
        signal_power = np.mean(data_array ** 2, axis=1)
        noise_estimate = np.median(np.abs(np.diff(data_array, axis=1)), axis=1) / 0.6745
        snr = signal_power / (noise_estimate ** 2 + 1e-10)
        
        quality_metrics.update({
            'mean_snr': float(np.mean(snr)),
            'min_snr': float(np.min(snr)),
            'signal_power_db': float(10 * np.log10(np.mean(signal_power) + 1e-10)),
            'noise_level_db': float(10 * np.log10(np.mean(noise_estimate ** 2) + 1e-10)),
        })
        
        # Artifact detection
        amplitude_threshold = 5 * np.std(data_array)
        artifact_fraction = np.mean(np.abs(data_array) > amplitude_threshold)
        
        quality_metrics.update({
            'artifact_fraction': float(artifact_fraction),
            'amplitude_range': float(np.ptp(data_array)),
            'zero_channels': int(np.sum(np.all(data_array == 0, axis=1))),
            'saturated_channels': int(np.sum(np.any(np.abs(data_array) > 1e6, axis=1)))
        })
        
        # Overall quality score (0-1)
        quality_factors = [
            min(1.0, quality_metrics['mean_snr'] / 10.0),  # SNR factor
            max(0.0, 1.0 - artifact_fraction * 5),         # Artifact penalty
            1.0 if quality_metrics['zero_channels'] == 0 else 0.5,  # Zero channel penalty
            1.0 if quality_metrics['saturated_channels'] == 0 else 0.3  # Saturation penalty
        ]
        
        quality_metrics['overall_quality'] = float(np.mean(quality_factors))
        quality_metrics['quality_grade'] = (
            'excellent' if quality_metrics['overall_quality'] > 0.8 else
            'good' if quality_metrics['overall_quality'] > 0.6 else
            'fair' if quality_metrics['overall_quality'] > 0.4 else
            'poor'
        )
        
        return quality_metrics
    
    @staticmethod
    def assess_olfactory_data_quality(molecule_data: dict[str, Any], concentration: float) -> dict[str, Any]:
        """Assess quality of olfactory stimulus data."""
        
        quality_metrics = {
            'has_name': 'name' in molecule_data,
            'has_molecular_weight': 'molecular_weight' in molecule_data,
            'has_functional_groups': 'functional_groups' in molecule_data,
            'has_odor_character': 'odor_character' in molecule_data,
            'concentration_ppm': concentration
        }
        
        # Completeness score
        completeness_factors = [
            quality_metrics['has_name'],
            quality_metrics['has_molecular_weight'],
            quality_metrics['has_functional_groups'],
            quality_metrics['has_odor_character']
        ]
        
        quality_metrics['completeness_score'] = sum(completeness_factors) / len(completeness_factors)
        
        # Concentration validity
        quality_metrics['concentration_valid'] = 0.001 <= concentration <= 100.0
        quality_metrics['concentration_realistic'] = 0.01 <= concentration <= 10.0
        
        # Molecular weight validation
        if 'molecular_weight' in molecule_data:
            mw = molecule_data['molecular_weight']
            quality_metrics['mw_realistic'] = 50 <= mw <= 500  # Realistic range for odorants
        else:
            quality_metrics['mw_realistic'] = False
        
        # Overall quality
        quality_factors = [
            quality_metrics['completeness_score'],
            1.0 if quality_metrics['concentration_valid'] else 0.0,
            1.0 if quality_metrics['mw_realistic'] else 0.5
        ]
        
        quality_metrics['overall_quality'] = sum(quality_factors) / len(quality_factors)
        quality_metrics['quality_grade'] = (
            'excellent' if quality_metrics['overall_quality'] > 0.8 else
            'good' if quality_metrics['overall_quality'] > 0.6 else
            'fair' if quality_metrics['overall_quality'] > 0.4 else
            'poor'
        )
        
        return quality_metrics