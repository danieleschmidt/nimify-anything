#!/bin/bash
set -e

echo "🏗️ Building nimify-anything..."

# Build Docker image
docker build -f docker/Dockerfile -t nimify-anything:latest .

# Run tests
echo "🧪 Running tests..."
python -m pytest tests/ -v

# Security scan
echo "🔒 Running security scan..."
# docker run --rm -v "$(pwd):/code" securecodewarrior/docker-action

# Performance test
echo "⚡ Running performance test..."
# locust -f tests/performance/locustfile.py --headless -u 10 -r 2 -t 30s

echo "✅ Build completed successfully!"
