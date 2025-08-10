#!/bin/bash
set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <region>"
    echo "Available regions: us-east-1, us-west-2, eu-west-1, eu-central-1, ap-southeast-1, ap-northeast-1"
    exit 1
fi

REGION=$1
echo "📍 Deploying nimify-anything to $REGION..."

# Validate region
if [ ! -d "regions/$REGION" ]; then
    echo "❌ Region $REGION not found"
    exit 1
fi

# Set regional context
kubectl config use-context "$REGION"

# Apply regional manifests  
kubectl apply -f "regions/$REGION/"

# Wait for deployment
kubectl rollout status deployment/nimify-anything-$REGION -n nim-global

echo "✅ Deployment to $REGION complete"
