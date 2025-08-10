#!/bin/bash
set -e

NAMESPACE=${1:-default}
TIMEOUT=${2:-60}

echo "🏥 Checking nimify-anything health..."

# Get service endpoint
SERVICE_IP=$(kubectl get service nimify-anything -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

if [ -z "$SERVICE_IP" ]; then
    echo "Using port-forward for health check..."
    kubectl port-forward svc/nimify-anything 8080:80 -n $NAMESPACE &
    PID=$!
    sleep 5
    SERVICE_URL="http://localhost:8080"
else
    SERVICE_URL="http://$SERVICE_IP"
fi

# Health check
echo "Checking health endpoint..."
curl -f $SERVICE_URL/health

echo "Checking metrics endpoint..."
curl -f $SERVICE_URL/metrics

# Kill port-forward if used
if [ ! -z "$PID" ]; then
    kill $PID
fi

echo "✅ Health check passed!"
