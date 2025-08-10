# Multi-stage production Dockerfile for nimify-anything
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Production stage
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 as runtime

# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -r nimuser && useradd -r -g nimuser nimuser

# Copy application code
COPY src/ /app/src/
COPY requirements.txt /app/

# Set working directory and ownership
WORKDIR /app
RUN chown -R nimuser:nimuser /app

# Switch to non-root user
USER nimuser

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "-m", "uvicorn", "src.nimify.api:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--access-log"]
