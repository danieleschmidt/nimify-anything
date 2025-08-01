#!/bin/bash
# Build script for Nimify Anything containers
# Supports multi-platform builds and caching

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="${IMAGE_NAME:-nimify-anything}"
REGISTRY="${REGISTRY:-localhost:5000}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
BUILD_ARGS="${BUILD_ARGS:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build Nimify Anything container images with multi-platform support.

OPTIONS:
    -t, --tag TAG           Image tag (default: latest)
    -f, --file DOCKERFILE   Dockerfile to use (default: Dockerfile)
    -p, --platforms LIST    Target platforms (default: linux/amd64,linux/arm64)
    -r, --registry URL      Container registry (default: localhost:5000)
    --push                  Push image after build
    --cache                 Use build cache
    --no-cache              Disable build cache
    --dev                   Build development image
    --prod                  Build production image (default)
    --debug                 Enable debug output
    -h, --help              Show this help message

Examples:
    $0 --tag v1.0.0 --push
    $0 --dev --tag dev-latest
    $0 --file Dockerfile.production --tag prod-v1.0.0 --push
EOF
}

# Parse command line arguments
TAG="latest"
DOCKERFILE="Dockerfile"
PUSH=false
USE_CACHE=true
TARGET="production"
DEBUG=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -f|--file)
            DOCKERFILE="$2"
            shift 2
            ;;
        -p|--platforms)
            PLATFORMS="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --cache)
            USE_CACHE=true
            shift
            ;;
        --no-cache)
            USE_CACHE=false
            shift
            ;;
        --dev)
            TARGET="development"
            DOCKERFILE="Dockerfile.production"
            shift
            ;;
        --prod)
            TARGET="production"
            DOCKERFILE="Dockerfile.production"
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Validate requirements
if ! command -v docker &> /dev/null; then
    error "Docker is required but not installed"
fi

# Enable BuildKit
export DOCKER_BUILDKIT=1

# Set debug mode
if [[ "$DEBUG" == "true" ]]; then
    set -x
    export BUILDKIT_PROGRESS=plain
fi

# Build metadata
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "dev")

# Full image name
FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${TAG}"

log "Building container image..."
log "Image: ${FULL_IMAGE_NAME}"
log "Target: ${TARGET}"
log "Platforms: ${PLATFORMS}"
log "Dockerfile: ${DOCKERFILE}"

# Prepare build arguments
BUILD_ARGS_LIST=(
    "--build-arg" "BUILD_DATE=${BUILD_DATE}"
    "--build-arg" "VCS_REF=${VCS_REF}"
    "--build-arg" "VERSION=${VERSION}"
    "--target" "${TARGET}"
)

# Add cache arguments
if [[ "$USE_CACHE" == "true" ]]; then
    BUILD_ARGS_LIST+=(
        "--cache-from" "type=registry,ref=${FULL_IMAGE_NAME}-cache"
        "--cache-to" "type=registry,ref=${FULL_IMAGE_NAME}-cache,mode=max"
    )
else
    BUILD_ARGS_LIST+=("--no-cache")
fi

# Add custom build args
if [[ -n "$BUILD_ARGS" ]]; then
    IFS=' ' read -ra CUSTOM_ARGS <<< "$BUILD_ARGS"
    for arg in "${CUSTOM_ARGS[@]}"; do
        BUILD_ARGS_LIST+=("--build-arg" "$arg")
    done
fi

# Create builder if needed for multi-platform
if [[ "$PLATFORMS" == *","* ]]; then
    log "Setting up multi-platform builder..."
    
    BUILDER_NAME="nimify-builder"
    if ! docker buildx inspect "$BUILDER_NAME" &>/dev/null; then
        docker buildx create --name "$BUILDER_NAME" --driver docker-container --use
        docker buildx inspect --bootstrap
    else
        docker buildx use "$BUILDER_NAME"
    fi
    
    BUILD_CMD="docker buildx build"
    BUILD_ARGS_LIST+=("--platform" "$PLATFORMS")
    
    if [[ "$PUSH" == "true" ]]; then
        BUILD_ARGS_LIST+=("--push")
    else
        warn "Multi-platform builds require --push to store images"
        BUILD_ARGS_LIST+=("--push")
        PUSH=true
    fi
else
    BUILD_CMD="docker build"
fi

# Execute build
log "Executing build command..."
cd "$PROJECT_ROOT"

if ! $BUILD_CMD \
    -f "$DOCKERFILE" \
    -t "$FULL_IMAGE_NAME" \
    "${BUILD_ARGS_LIST[@]}" \
    .; then
    error "Build failed"
fi

# Push if requested and not already pushed
if [[ "$PUSH" == "true" && "$PLATFORMS" != *","* ]]; then
    log "Pushing image to registry..."
    if ! docker push "$FULL_IMAGE_NAME"; then
        error "Push failed"
    fi
fi

# Security scan
if command -v trivy &> /dev/null; then
    log "Running security scan..."
    if ! trivy image --severity HIGH,CRITICAL "$FULL_IMAGE_NAME"; then
        warn "Security vulnerabilities found"
    fi
else
    warn "Trivy not found, skipping security scan"
fi

# Generate SBOM
if command -v syft &> /dev/null; then
    log "Generating Software Bill of Materials..."
    syft "$FULL_IMAGE_NAME" -o spdx-json > "SBOM-${TAG}.json"
    log "SBOM saved as SBOM-${TAG}.json"
else
    warn "Syft not found, skipping SBOM generation"
fi

# Test the built image
log "Testing built image..."
if ! docker run --rm "$FULL_IMAGE_NAME" nimify --version; then
    error "Image test failed"
fi

log "Build completed successfully!"
log "Image: ${FULL_IMAGE_NAME}"
log "Size: $(docker images --format "table {{.Size}}" "$FULL_IMAGE_NAME" | tail -n 1)"

# Cleanup
if [[ "$PLATFORMS" == *","* ]]; then
    log "Cleaning up builder..."
    docker buildx rm "$BUILDER_NAME" || true
fi