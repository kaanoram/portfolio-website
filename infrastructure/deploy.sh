#!/bin/bash

# Deployment script for portfolio website demo infrastructure
# This script automates the terraform deployment process

set -e

echo "======================================"
echo "Portfolio Website Deployment Script"
echo "======================================"
echo ""

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ AWS CLI not configured. Please run 'aws configure' first."
    exit 1
fi

echo "✅ AWS CLI configured"

echo "📦 Packaging Lambda function..."
cd "$(dirname "$0")"
./package_lambda.sh

cd terraform

echo "🔧 Initializing Terraform..."
terraform init

echo "📋 Planning deployment..."
terraform plan -var-file="demo.tfvars"

echo ""
read -p "Do you want to proceed with deployment? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

echo "🚀 Deploying infrastructure..."
terraform apply -var-file="demo.tfvars" -auto-approve

echo ""
echo "📊 Deployment completed! Here are the important outputs:"
echo ""
terraform output

echo ""
echo "🎉 Deployment successful!"
echo ""
echo "Next steps:"
echo "1. Copy the API Gateway URL from the outputs above"
echo "2. Create a .env file in the project root with:"
echo "   VITE_API_ENDPOINT=<your-api-gateway-url>"
echo "   VITE_DEMO_MODE=true"
echo "3. Build and deploy the frontend: npm run build"
echo "4. Upload the build folder to the S3 bucket shown in outputs"
echo ""
