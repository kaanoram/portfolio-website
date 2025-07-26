# E-commerce Analytics Platform - Deployment Guide

## Architecture Overview

This project demonstrates a production-ready ML system with two deployment modes:

### 1. **Demo Mode (Portfolio)** - $10-20/month
- **Frontend**: S3 + CloudFront CDN
- **Backend**: AWS Lambda + API Gateway
- **ML Models**: Stored in S3, cached in Lambda
- **Database**: DynamoDB (pay-per-request)
- **Real ML predictions** with <100ms latency

### 2. **Production Mode** - $400-700/month
- **Frontend**: ECS Fargate with auto-scaling
- **Backend**: ECS + Apache Beam on Kinesis
- **ML Pipeline**: Real-time processing of 1M+ transactions/day
- **Database**: RDS PostgreSQL + DynamoDB
- **Full monitoring** with CloudWatch

## Quick Start Deployment (Demo Mode)

### Prerequisites
- AWS Account
- AWS CLI configured
- Terraform installed
- Node.js 18+ and Python 3.9+
- Your trained ML models in `src/backend/models/`

### Step 1: Prepare ML Models

```bash
# Package models for S3
cd src/backend
python -c "
import joblib
import shutil
import os

# Create deployment directory
os.makedirs('deploy_models', exist_ok=True)

# Copy model files
files = ['scaler.joblib', 'model_fold_0.keras', 'model_metrics.json']
for f in files:
    if os.path.exists(f'models/{f}'):
        shutil.copy(f'models/{f}', f'deploy_models/{f}')

print('Models ready for deployment in deploy_models/')
"
```

### Step 2: Build Frontend

```bash
# Install dependencies
npm install

# Set environment variables
echo "VITE_API_ENDPOINT=https://your-api-gateway-url.execute-api.us-east-1.amazonaws.com" > .env
echo "VITE_DEMO_MODE=true" >> .env

# Build for production
npm run build
```

### Step 3: Package Lambda Function

```bash
cd infrastructure
chmod +x package_lambda.sh
./package_lambda.sh
```

### Step 4: Deploy Infrastructure

```bash
# Initialize Terraform
terraform init

# Plan deployment
terraform plan -var-file="demo.tfvars"

# Deploy (will prompt for confirmation)
terraform apply -var-file="demo.tfvars"

# Note the outputs - you'll need these!
```

### Step 5: Upload Assets

```bash
# Upload frontend to S3
aws s3 sync ../dist/ s3://your-frontend-bucket/ --delete

# Upload models to S3
aws s3 cp ../src/backend/deploy_models/ s3://your-models-bucket/models/v1.0.0/ --recursive

# Invalidate CloudFront cache
aws cloudfront create-invalidation --distribution-id YOUR_DIST_ID --paths "/*"
```

### Step 6: Update Frontend Config

```bash
# Update .env with actual API endpoint from Terraform output
echo "VITE_API_ENDPOINT=$(terraform output -raw api_endpoint)" > .env

# Rebuild and redeploy frontend
npm run build
aws s3 sync dist/ s3://your-frontend-bucket/ --delete
```

## Testing Your Deployment

### 1. Check API Health
```bash
curl https://your-api-gateway-url.execute-api.us-east-1.amazonaws.com/api/capabilities
```

### 2. Generate Test Transaction
```bash
curl -X POST https://your-api-gateway-url.execute-api.us-east-1.amazonaws.com/api/demo/transaction
```

### 3. View Metrics
```bash
curl https://your-api-gateway-url.execute-api.us-east-1.amazonaws.com/api/demo/metrics
```

## Monitoring & Costs

### CloudWatch Dashboard
```bash
# Create custom dashboard
aws cloudwatch put-dashboard \
  --dashboard-name EcommerceAnalyticsDemo \
  --dashboard-body file://cloudwatch-dashboard.json
```

### Cost Tracking
Monitor your AWS costs:
1. AWS Cost Explorer
2. Set up billing alerts
3. Use AWS Budgets

Expected costs (Demo Mode):
- **First month**: ~$5-10 (mostly free tier)
- **Ongoing**: ~$10-20/month

## Scaling to Production

When ready to demonstrate full capabilities:

1. **Load Testing**
   ```bash
   # Run included load tests
   cd src/backend
   python test_pipeline_local.py
   
   # WebSocket load test
   python load_test_websocket.py
   ```

2. **Deploy Production Infrastructure**
   ```bash
   # Use production Terraform config
   terraform apply -var-file="production.tfvars"
   ```

3. **Performance Benchmarks**
   - Document your load test results
   - Include screenshots in portfolio
   - Show architecture diagrams

## Architecture Diagrams

### Demo Mode Architecture
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  CloudFront │────▶│  S3 Static   │     │   Lambda    │
│     CDN     │     │   Website    │     │  ML Models  │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
┌─────────────┐     ┌──────────────┐            │
│   Browser   │────▶│ API Gateway  │────────────┘
│   (React)   │     │   (HTTP)     │
└─────────────┘     └──────────────┘
                           │
                    ┌──────▼───────┐
                    │  DynamoDB    │
                    │ (On-Demand)  │
                    └──────────────┘
```

### Production Architecture
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│    ALB      │────▶│  ECS Tasks   │────▶│   Kinesis   │
│             │     │  (Frontend)  │     │   Streams   │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
┌─────────────┐     ┌──────────────┐     ┌──────▼───────┐
│  Users      │────▶│  ECS Tasks   │────▶│ Apache Beam │
│  (2000+)    │     │  (Backend)   │     │  Pipeline   │
└─────────────┘     └──────────────┘     └──────────────┘
                           │
                    ┌──────▼───────┐     ┌─────────────┐
                    │     RDS      │     │     S3      │
                    │ PostgreSQL   │     │   Models    │
                    └──────────────┘     └─────────────┘
```

## Troubleshooting

### Lambda Cold Starts
- First request may take 3-5 seconds
- Subsequent requests: <100ms
- Consider provisioned concurrency for demos

### CORS Issues
- Check API Gateway CORS configuration
- Verify CloudFront behaviors
- Update Lambda response headers

### Model Loading
- Ensure S3 bucket permissions
- Check Lambda memory (512MB minimum)
- Verify model file paths

## Security Considerations

1. **API Keys**: Add API key requirement for production
2. **WAF**: Enable AWS WAF for DDoS protection
3. **Secrets**: Use AWS Secrets Manager for sensitive data
4. **IAM**: Follow least-privilege principle

## Next Steps

1. **Custom Domain**
   ```bash
   # Route 53 + ACM for HTTPS
   terraform apply -var="domain_name=kaanoram.io"
   ```

2. **CI/CD Pipeline**
   - GitHub Actions workflow included
   - Automated testing and deployment

3. **Enhanced Monitoring**
   - Set up alerts
   - Create custom metrics
   - Add distributed tracing

## Support

For issues or questions:
- Check CloudWatch Logs
- Review Terraform state
- See troubleshooting section

Remember: This demo showcases enterprise-grade architecture at minimal cost!