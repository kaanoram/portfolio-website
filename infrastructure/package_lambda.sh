#!/bin/bash
# Script to package Lambda function and dependencies

echo "Packaging Lambda function for deployment..."

# Create clean build directory
rm -rf lambda_build
mkdir -p lambda_build

# Copy Lambda handler
cp ../src/backend/lambda_handler.py lambda_build/

# Create Lambda layer for ML dependencies (one-time)
if [ ! -f ml_layer.zip ]; then
    echo "Creating ML dependencies layer..."
    
    # Create layer directory
    mkdir -p ml_layer/python
    
    # Install heavy dependencies in layer
    pip install --target ml_layer/python \
        tensorflow==2.14.0 \
        scikit-learn==1.3.0 \
        joblib==1.3.2 \
        numpy==1.24.3
    
    # Create layer zip
    cd ml_layer
    zip -r ../ml_layer.zip python/
    cd ..
    
    echo "ML layer created: ml_layer.zip"
fi

# Install only lightweight dependencies in main package
pip install --target lambda_build \
    boto3 \
    requests

# Create deployment package
cd lambda_build
zip -r ../lambda_deployment.zip .
cd ..

echo "Lambda deployment package created: lambda_deployment.zip"

# Show package sizes
echo -e "\nPackage sizes:"
ls -lh lambda_deployment.zip ml_layer.zip 2>/dev/null

echo -e "\nDeployment packages ready!"
echo "1. Upload ml_layer.zip as Lambda Layer"
echo "2. Upload lambda_deployment.zip as Lambda function"