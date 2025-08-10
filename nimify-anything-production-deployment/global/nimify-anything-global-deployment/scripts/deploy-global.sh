#!/bin/bash
set -e

echo "🌍 Deploying nimify-anything globally..."

# Deploy to all regions in parallel
regions=("us-east-1 us-west-2 eu-west-1 eu-central-1 ap-southeast-1 ap-northeast-1")

for region in "${regions[@]}"; do
    echo "📍 Deploying to $region..."
    
    # Set regional context
    kubectl config use-context "$region"
    
    # Apply regional manifests
    kubectl apply -f "regions/$region/"
    
    # Wait for deployment to be ready
    kubectl rollout status deployment/nimify-anything-$region -n nim-global
    
    echo "✅ $region deployment complete"
done

echo "🎉 Global deployment complete!"
echo "🔗 Service available at: https://nimify-anything.api.nimify.ai"
