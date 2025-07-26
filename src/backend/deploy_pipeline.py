#!/usr/bin/env python3
"""
Deployment script for the production Apache Beam pipeline on AWS.
Handles infrastructure setup and pipeline deployment.
"""

import boto3
import yaml
import json
import time
import logging
import argparse
import subprocess
import os
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineDeployer:
    """Handles deployment of Apache Beam pipeline to AWS"""
    
    def __init__(self, config_path: str, environment: str = 'production'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.environment = environment
        self.aws_region = self.config['aws']['region']
        
        # Initialize AWS clients
        self.kinesis = boto3.client('kinesis', region_name=self.aws_region)
        self.dynamodb = boto3.client('dynamodb', region_name=self.aws_region)
        self.s3 = boto3.client('s3', region_name=self.aws_region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=self.aws_region)
        
    def create_kinesis_streams(self):
        """Create Kinesis Data Streams for input and output"""
        streams = [
            (self.config['aws']['kinesis']['input_stream'], 'Input stream for raw transactions'),
            (self.config['aws']['kinesis']['output_stream'], 'Output stream for enriched transactions'),
            (self.config['aws']['kinesis']['metrics_stream'], 'Metrics stream for monitoring'),
            (self.config['pipeline']['dead_letter_stream'], 'Dead letter queue for failed records')
        ]
        
        for stream_name, description in streams:
            try:
                # Check if stream exists
                self.kinesis.describe_stream(StreamName=stream_name)
                logger.info(f"Stream {stream_name} already exists")
            except self.kinesis.exceptions.ResourceNotFoundException:
                # Create stream
                logger.info(f"Creating Kinesis stream: {stream_name}")
                self.kinesis.create_stream(
                    StreamName=stream_name,
                    ShardCount=self.config['aws']['kinesis']['shard_count'],
                    StreamModeDetails={'StreamMode': 'PROVISIONED'}
                )
                
                # Wait for stream to become active
                waiter = self.kinesis.get_waiter('stream_exists')
                waiter.wait(StreamName=stream_name)
                logger.info(f"Stream {stream_name} created successfully")
    
    def create_dynamodb_table(self):
        """Create DynamoDB table for customer profiles"""
        table_name = self.config['aws']['dynamodb']['customer_table']
        
        try:
            self.dynamodb.describe_table(TableName=table_name)
            logger.info(f"Table {table_name} already exists")
        except self.dynamodb.exceptions.ResourceNotFoundException:
            logger.info(f"Creating DynamoDB table: {table_name}")
            
            self.dynamodb.create_table(
                TableName=table_name,
                KeySchema=[
                    {'AttributeName': 'customer_id', 'KeyType': 'HASH'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'customer_id', 'AttributeType': 'S'}
                ],
                BillingMode='PROVISIONED',
                ProvisionedThroughput={
                    'ReadCapacityUnits': self.config['aws']['dynamodb']['read_capacity'],
                    'WriteCapacityUnits': self.config['aws']['dynamodb']['write_capacity']
                },
                StreamSpecification={
                    'StreamEnabled': True,
                    'StreamViewType': 'NEW_AND_OLD_IMAGES'
                }
            )
            
            # Wait for table to become active
            waiter = self.dynamodb.get_waiter('table_exists')
            waiter.wait(TableName=table_name)
            logger.info(f"Table {table_name} created successfully")
    
    def create_s3_buckets(self):
        """Create S3 buckets for models and checkpoints"""
        buckets = [
            self.config['aws']['s3']['model_bucket'],
            self.config['aws']['s3']['checkpoint_bucket'],
            self.config['aws']['s3']['data_lake_bucket']
        ]
        
        for bucket_name in buckets:
            try:
                self.s3.head_bucket(Bucket=bucket_name)
                logger.info(f"Bucket {bucket_name} already exists")
            except:
                logger.info(f"Creating S3 bucket: {bucket_name}")
                
                if self.aws_region == 'us-east-1':
                    self.s3.create_bucket(Bucket=bucket_name)
                else:
                    self.s3.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': self.aws_region}
                    )
                
                # Enable versioning for model bucket
                if bucket_name == self.config['aws']['s3']['model_bucket']:
                    self.s3.put_bucket_versioning(
                        Bucket=bucket_name,
                        VersioningConfiguration={'Status': 'Enabled'}
                    )
                
                logger.info(f"Bucket {bucket_name} created successfully")
    
    def upload_models(self):
        """Upload trained models to S3"""
        model_bucket = self.config['aws']['s3']['model_bucket']
        model_version = self.config['models']['version']
        local_model_dir = 'models'
        
        logger.info(f"Uploading models to s3://{model_bucket}/models/{model_version}/")
        
        # Upload model files
        model_files = [
            'scaler.joblib',
            'model_fold_0.keras',
            'model_fold_1.keras',
            'model_fold_2.keras',
            'model_fold_3.keras',
            'model_fold_4.keras',
            'model_metrics.json'
        ]
        
        for file_name in model_files:
            local_path = os.path.join(local_model_dir, file_name)
            if os.path.exists(local_path):
                s3_key = f"models/{model_version}/{file_name}"
                logger.info(f"Uploading {file_name} to {s3_key}")
                self.s3.upload_file(local_path, model_bucket, s3_key)
            else:
                logger.warning(f"Model file not found: {local_path}")
    
    def create_cloudwatch_alarms(self):
        """Create CloudWatch alarms for monitoring"""
        namespace = self.config['monitoring']['cloudwatch']['namespace']
        
        for alarm_config in self.config['monitoring']['cloudwatch']['alarms']:
            alarm_name = f"{self.environment}-{alarm_config['name']}"
            
            logger.info(f"Creating CloudWatch alarm: {alarm_name}")
            
            self.cloudwatch.put_metric_alarm(
                AlarmName=alarm_name,
                ComparisonOperator=alarm_config.get('comparison', 'GreaterThanThreshold'),
                EvaluationPeriods=alarm_config['evaluation_periods'],
                MetricName=alarm_config['metric'],
                Namespace=namespace,
                Period=60,  # 1 minute
                Statistic='Average',
                Threshold=alarm_config['threshold'],
                ActionsEnabled=True,
                AlarmDescription=f"Alarm for {alarm_config['metric']} in {self.environment}",
                TreatMissingData='breaching'
            )
    
    def deploy_pipeline(self):
        """Deploy the Apache Beam pipeline"""
        logger.info("Deploying Apache Beam pipeline...")
        
        # Set environment variables
        env = os.environ.copy()
        env.update({
            'AWS_REGION': self.aws_region,
            'KINESIS_STREAM_INPUT': self.config['aws']['kinesis']['input_stream'],
            'KINESIS_STREAM_OUTPUT': self.config['aws']['kinesis']['output_stream'],
            'S3_MODEL_BUCKET': self.config['aws']['s3']['model_bucket'],
            'S3_CHECKPOINT_BUCKET': self.config['aws']['s3']['checkpoint_bucket'],
            'MODEL_VERSION': self.config['models']['version']
        })
        
        # Build Docker image for the pipeline
        logger.info("Building Docker image for pipeline...")
        subprocess.run([
            'docker', 'build',
            '-t', f'ecommerce-pipeline:{self.environment}',
            '-f', 'Dockerfile.pipeline',
            '.'
        ], check=True)
        
        # Deploy to AWS Batch or ECS (simplified for this example)
        logger.info("Pipeline Docker image built successfully")
        logger.info("To deploy to production, push the image to ECR and update ECS task definition")
        
        return True
    
    def verify_deployment(self):
        """Verify that all components are properly deployed"""
        logger.info("Verifying deployment...")
        
        # Check Kinesis streams
        for stream_name in [self.config['aws']['kinesis']['input_stream'],
                           self.config['aws']['kinesis']['output_stream']]:
            response = self.kinesis.describe_stream(StreamName=stream_name)
            status = response['StreamDescription']['StreamStatus']
            if status != 'ACTIVE':
                raise Exception(f"Stream {stream_name} is not active: {status}")
        
        # Check DynamoDB table
        table_name = self.config['aws']['dynamodb']['customer_table']
        response = self.dynamodb.describe_table(TableName=table_name)
        status = response['Table']['TableStatus']
        if status != 'ACTIVE':
            raise Exception(f"Table {table_name} is not active: {status}")
        
        logger.info("All components verified successfully")
    
    def deploy_all(self):
        """Execute full deployment"""
        logger.info(f"Starting deployment for environment: {self.environment}")
        
        # Create infrastructure
        self.create_kinesis_streams()
        self.create_dynamodb_table()
        self.create_s3_buckets()
        
        # Upload models
        self.upload_models()
        
        # Set up monitoring
        self.create_cloudwatch_alarms()
        
        # Deploy pipeline
        self.deploy_pipeline()
        
        # Verify deployment
        self.verify_deployment()
        
        logger.info("Deployment completed successfully!")


def main():
    parser = argparse.ArgumentParser(description='Deploy Apache Beam pipeline to AWS')
    parser.add_argument('--config', default='pipeline_config.yaml', help='Path to configuration file')
    parser.add_argument('--environment', default='production', help='Deployment environment')
    parser.add_argument('--skip-infrastructure', action='store_true', help='Skip infrastructure creation')
    
    args = parser.parse_args()
    
    deployer = PipelineDeployer(args.config, args.environment)
    
    if args.skip_infrastructure:
        logger.info("Skipping infrastructure creation")
        deployer.upload_models()
        deployer.deploy_pipeline()
    else:
        deployer.deploy_all()


if __name__ == '__main__':
    main()