# Multi-stage Dockerfile optimized for NVIDIA NIM AI/ML services
# Build stage for dependencies and setup
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04 AS builder

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python3.10 -m pip install --upgrade pip
RUN python3.10 -m pip install virtualenv
RUN python3.10 -m virtualenv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY pyproject.toml /tmp/
RUN pip install --no-cache-dir build wheel
RUN cd /tmp && pip install --no-cache-dir -e .

# Production stage
FROM nvidia/cuda:12.9.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN groupadd -r nimify && useradd -r -g nimify -u 1001 nimify
RUN mkdir -p /app /models /cache && chown -R nimify:nimify /app /models /cache

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=nimify:nimify src/ ./src/
COPY --chown=nimify:nimify pyproject.toml ./

# Install the package in the container
RUN pip install --no-cache-dir -e .

# Switch to non-root user
USER nimify

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD nimify --version || exit 1

# Default command
CMD ["nimify", "--help"]

# Labels for metadata
LABEL org.opencontainers.image.title="Nimify Anything"
LABEL org.opencontainers.image.description="CLI for wrapping ONNX/TensorRT models into NVIDIA NIM services"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/yourusername/nimify-anything"
LABEL org.opencontainers.image.licenses="MIT"