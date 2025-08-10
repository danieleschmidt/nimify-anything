#!/bin/bash
set -e

ENVIRONMENT=${1:-development}
NAMESPACE=${2:-default}

echo "🚀 Deploying nimify-anything to $ENVIRONMENT environment..."

# Validate kubectl access
kubectl cluster-info > /dev/null

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Deploy using Kustomize
kubectl apply -k kubernetes/overlays/$ENVIRONMENT -n $NAMESPACE

# Wait for rollout
kubectl rollout status deployment/nimify-anything -n $NAMESPACE

# Verify deployment
kubectl get pods -n $NAMESPACE -l app=nimify-anything

echo "✅ Deployment to $ENVIRONMENT completed successfully!"
echo "🔗 Access your service:"
echo "   kubectl port-forward svc/nimify-anything 8080:80 -n $NAMESPACE"
