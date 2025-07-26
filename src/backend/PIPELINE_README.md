# Apache Beam Production Pipeline

This directory contains the production-ready Apache Beam pipeline for processing 1M+ daily e-commerce transactions with sub-second latency.

## Architecture Overview

The pipeline uses AWS Kinesis Data Streams for real-time data ingestion and processing:

```
Kinesis Input Stream → Apache Beam Pipeline → Kinesis Output Stream
                              ↓
                         ML Predictions
                              ↓
                      Enriched Transactions
```

## Key Features

- **Scalability**: Handles 1M+ transactions daily (12+ TPS average, 50 TPS peak)
- **Low Latency**: Sub-second processing with optimized model inference
- **Fault Tolerance**: Automatic retry and dead-letter queue for failed records
- **Auto-scaling**: Dynamic worker scaling based on load
- **Monitoring**: CloudWatch metrics and alarms

## Files

- `analytics_pipeline_production.py` - Main production pipeline code
- `pipeline_config.yaml` - Configuration for AWS resources and pipeline settings
- `deploy_pipeline.py` - Deployment script for AWS infrastructure
- `test_pipeline_local.py` - Local testing and validation script
- `Dockerfile.pipeline` - Docker image for pipeline deployment

## Prerequisites

1. AWS Account with appropriate permissions
2. Python 3.9+
3. Docker
4. AWS CLI configured

## Local Testing

Before deploying to production, test the pipeline locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run local tests
python test_pipeline_local.py
```

Expected output:
- Component tests should pass
- Throughput simulation should show ~12 TPS capability
- Prediction latency should be < 100ms average

## Deployment

### 1. Configure AWS Credentials

```bash
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_REGION=us-east-1
```

### 2. Deploy Infrastructure

```bash
# Deploy all AWS resources
python deploy_pipeline.py --config pipeline_config.yaml

# Or skip infrastructure if already created
python deploy_pipeline.py --skip-infrastructure
```

This will:
- Create Kinesis streams with 10 shards each
- Set up DynamoDB table for customer profiles
- Create S3 buckets for models and checkpoints
- Upload trained models to S3
- Configure CloudWatch alarms

### 3. Build and Deploy Pipeline

```bash
# Build Docker image
docker build -t ecommerce-pipeline:production -f Dockerfile.pipeline .

# Tag and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin [your-ecr-uri]
docker tag ecommerce-pipeline:production [your-ecr-uri]/ecommerce-pipeline:production
docker push [your-ecr-uri]/ecommerce-pipeline:production
```

### 4. Run Pipeline on AWS

For production deployment, you have several options:

**Option A: AWS Batch (Recommended for batch processing)**
```bash
# Update the ECS task definition to use the pipeline image
# Then submit batch job
```

**Option B: ECS Fargate (Recommended for streaming)**
```bash
# Create ECS service with the pipeline container
# Configure auto-scaling based on Kinesis lag
```

**Option C: EMR (For large-scale processing)**
```bash
# Deploy to EMR cluster with Flink runner
```

## Monitoring

### CloudWatch Metrics

The pipeline publishes the following metrics:
- `transactions_per_second` - Current processing rate
- `prediction_latency_ms` - Model inference latency
- `error_rate` - Percentage of failed transactions
- `unique_customers_per_minute` - Customer activity metric

### Alarms

Configured alarms:
- **HighErrorRate**: Triggers when error rate > 5%
- **LowThroughput**: Triggers when TPS < 10 for 5 minutes

### Viewing Metrics

```bash
# View recent metrics
aws cloudwatch get-metric-statistics \
  --namespace EcommerceAnalytics \
  --metric-name transactions_per_second \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-01T01:00:00Z \
  --period 300 \
  --statistics Average
```

## Performance Tuning

### Kinesis Shards

Current configuration uses 10 shards per stream:
- Each shard: 1MB/sec input, 2MB/sec output
- Total capacity: 10MB/sec input (enough for 100k+ transactions/sec)

To increase capacity:
```bash
aws kinesis update-shard-count \
  --stream-name ecommerce-transactions \
  --target-shard-count 20
```

### Pipeline Workers

Adjust in `pipeline_config.yaml`:
```yaml
pipeline:
  autoscaling:
    min_workers: 5
    max_workers: 100  # Increase for higher load
```

### Model Optimization

- Models are cached in memory for 5 minutes
- Batch predictions when possible
- Consider using TensorFlow Lite for faster inference

## Troubleshooting

### Common Issues

1. **High latency**
   - Check model loading time
   - Verify Kinesis shard distribution
   - Review CloudWatch logs for bottlenecks

2. **Memory issues**
   - Increase worker memory in ECS task definition
   - Enable memory profiling with `--profile_memory`

3. **Failed predictions**
   - Check model file permissions in S3
   - Verify feature preprocessing matches training

### Debug Mode

Run pipeline with debug logging:
```bash
python analytics_pipeline_production.py --log-level DEBUG
```

## Cost Optimization

Estimated monthly costs (1M transactions/day):
- Kinesis: ~$150 (3 streams, 10 shards each)
- DynamoDB: ~$50 (provisioned capacity)
- S3: ~$10 (model storage + checkpoints)
- Compute: ~$200-500 (depending on auto-scaling)

Total: ~$400-700/month

## Next Steps

1. Set up CI/CD pipeline for automated deployments
2. Implement A/B testing for model versions
3. Add data quality monitoring
4. Configure cross-region replication for DR