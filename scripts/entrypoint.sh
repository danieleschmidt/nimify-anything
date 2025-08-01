#!/bin/bash
# Entrypoint script for Nimify Anything containers
# Handles initialization, configuration, and graceful shutdown

set -euo pipefail

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${GREEN}[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $1${NC}" >&2
}

warn() {
    echo -e "${YELLOW}[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] WARNING: $1${NC}" >&2
}

error() {
    echo -e "${RED}[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] ERROR: $1${NC}" >&2
    exit 1
}

info() {
    echo -e "${BLUE}[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] INFO: $1${NC}" >&2
}

# Signal handlers for graceful shutdown
cleanup() {
    log "Received shutdown signal, cleaning up..."
    
    # Kill background processes
    jobs -p | xargs -r kill -TERM
    
    # Wait for processes to terminate
    wait
    
    log "Cleanup completed"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Environment validation
validate_environment() {
    log "Validating environment..."
    
    # Check required directories
    for dir in /models /cache /tmp/nimify; do
        if [[ ! -d "$dir" ]]; then
            warn "Directory $dir does not exist, creating..."
            mkdir -p "$dir"
        fi
        
        if [[ ! -w "$dir" ]]; then
            error "Directory $dir is not writable"
        fi
    done
    
    # Check Python installation
    if ! python -c "import nimify" 2>/dev/null; then
        error "Nimify package not properly installed"
    fi
    
    # Validate configuration
    if [[ "${NIMIFY_ENV:-}" == "production" ]]; then
        if [[ -z "${NGC_API_KEY:-}" ]]; then
            warn "NGC_API_KEY not set - NIM features may not work"
        fi
    fi
    
    log "Environment validation completed"
}

# System initialization
initialize_system() {
    log "Initializing system..."
    
    # Set timezone if specified
    if [[ -n "${TZ:-}" ]]; then
        if [[ -f "/usr/share/zoneinfo/$TZ" ]]; then
            ln -sf "/usr/share/zoneinfo/$TZ" /etc/localtime
            echo "$TZ" > /etc/timezone
            log "Timezone set to $TZ"
        else
            warn "Invalid timezone: $TZ"
        fi
    fi
    
    # Configure logging
    if [[ -f "/app/config/logging.yaml" ]]; then
        export NIMIFY_LOG_CONFIG="/app/config/logging.yaml"
    fi
    
    # Initialize model cache
    if [[ ! -d "/cache/models" ]]; then
        mkdir -p /cache/models
        log "Initialized model cache"
    fi
    
    # Generate machine ID for metrics
    if [[ ! -f "/tmp/nimify/machine-id" ]]; then
        python -c "import uuid; print(uuid.uuid4())" > /tmp/nimify/machine-id
        log "Generated machine ID"
    fi
    
    log "System initialization completed"
}

# Health check function
health_check() {
    log "Running health check..."
    
    # Check if nimify command works
    if ! nimify --version >/dev/null 2>&1; then
        error "Nimify command not working"
    fi
    
    # Check if we can connect to required services
    if [[ "${NIMIFY_ENV:-}" == "production" ]]; then
        # Check Kubernetes connectivity if in cluster
        if [[ -f "/var/run/secrets/kubernetes.io/serviceaccount/token" ]]; then
            if ! kubectl version --client >/dev/null 2>&1; then
                warn "kubectl not available - Kubernetes features disabled"
            fi
        fi
    fi
    
    log "Health check passed"
}

# Pre-flight checks
preflight_checks() {
    log "Running pre-flight checks..."
    
    # Check disk space
    DISK_USAGE=$(df /cache | tail -1 | awk '{print $5}' | sed 's/%//')
    if [[ $DISK_USAGE -gt 80 ]]; then
        warn "Cache disk usage is ${DISK_USAGE}% - consider cleanup"
    fi
    
    # Check memory
    if [[ -f "/proc/meminfo" ]]; then
        AVAILABLE_MEM=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
        if [[ $AVAILABLE_MEM -lt 1000000 ]]; then  # Less than 1GB
            warn "Low available memory: ${AVAILABLE_MEM}KB"
        fi
    fi
    
    # Check GPU if available
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi >/dev/null 2>&1; then
            log "GPU detected and available"
            export CUDA_AVAILABLE=true
        else
            warn "nvidia-smi failed - GPU may not be available"
            export CUDA_AVAILABLE=false
        fi
    else
        info "No GPU detected, running in CPU mode"
        export CUDA_AVAILABLE=false
    fi
    
    log "Pre-flight checks completed"
}

# Configuration setup
setup_configuration() {
    log "Setting up configuration..."
    
    # Set default values
    export NIMIFY_MODEL_CACHE="${NIMIFY_MODEL_CACHE:-/cache/models}"
    export NIMIFY_LOG_LEVEL="${NIMIFY_LOG_LEVEL:-INFO}"
    export NIMIFY_METRICS_PORT="${NIMIFY_METRICS_PORT:-9090}"
    
    # Production optimizations
    if [[ "${NIMIFY_ENV:-}" == "production" ]]; then
        export PYTHONOPTIMIZE=1
        export MALLOC_TRIM_THRESHOLD_=100000
    fi
    
    # Development mode settings
    if [[ "${DEBUG:-false}" == "true" ]]; then
        export NIMIFY_LOG_LEVEL="DEBUG"
        export PYTHONDONTWRITEBYTECODE=1
    fi
    
    log "Configuration setup completed"
}

# Start background services
start_services() {
    log "Starting background services..."
    
    # Start metrics endpoint if enabled
    if [[ "${ENABLE_METRICS:-true}" == "true" ]]; then
        python -m nimify.metrics --port "${NIMIFY_METRICS_PORT}" &
        log "Metrics service started on port ${NIMIFY_METRICS_PORT}"
    fi
    
    # Start model cache cleanup if enabled
    if [[ "${ENABLE_CACHE_CLEANUP:-true}" == "true" ]]; then
        python -m nimify.cache.cleanup --interval 3600 &
        log "Cache cleanup service started"
    fi
    
    log "Background services started"
}

# Main execution
main() {
    log "Starting Nimify Anything container..."
    log "Version: $(nimify --version 2>/dev/null || echo 'unknown')"
    log "Environment: ${NIMIFY_ENV:-development}"
    log "User: $(whoami)"
    log "Working directory: $(pwd)"
    
    # Run initialization sequence
    validate_environment
    initialize_system
    setup_configuration
    preflight_checks
    health_check
    start_services
    
    log "Container initialization completed"
    
    # If no arguments provided, show help
    if [[ $# -eq 0 ]]; then
        log "No command provided, showing help..."
        exec nimify --help
    fi
    
    # Execute the provided command
    log "Executing command: $*"
    exec "$@"
}

# Only run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi