#!/bin/bash


set -e

echo "======================================"
echo "Lambda Function Packaging Script"
echo "======================================"
echo ""

# Create temporary directory
TEMP_DIR=$(mktemp -d)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Working directory: $TEMP_DIR"
echo "Project root: $PROJECT_ROOT"
echo ""

# Copy Lambda handler
echo "📦 Copying Lambda handler..."
cp "$PROJECT_ROOT/src/backend/lambda_handler.py" "$TEMP_DIR/"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
if [ -f "$PROJECT_ROOT/src/backend/requirements.txt" ]; then
    pip install -r "$PROJECT_ROOT/src/backend/requirements.txt" -t "$TEMP_DIR/" --no-deps --quiet
else
    echo "⚠️  No requirements.txt found, installing minimal dependencies..."
fi

# Install specific ML dependencies that are needed
echo "📦 Installing ML dependencies..."
pip install numpy==1.24.3 -t "$TEMP_DIR/" --no-deps --quiet
pip install scikit-learn==1.3.0 -t "$TEMP_DIR/" --no-deps --quiet
pip install joblib==1.3.2 -t "$TEMP_DIR/" --no-deps --quiet
pip install boto3==1.28.57 -t "$TEMP_DIR/" --no-deps --quiet

# Create deployment package
echo "📦 Creating deployment package..."
cd "$TEMP_DIR"
zip -r "$SCRIPT_DIR/terraform/lambda_deployment.zip" . > /dev/null

echo ""
echo "✅ Lambda deployment package created!"
echo "📍 Location: $SCRIPT_DIR/terraform/lambda_deployment.zip"
echo "📊 Package size: $(du -h "$SCRIPT_DIR/terraform/lambda_deployment.zip" | cut -f1)"

rm -rf "$TEMP_DIR"

echo ""
echo "🎉 Lambda packaging completed successfully!"
echo ""
