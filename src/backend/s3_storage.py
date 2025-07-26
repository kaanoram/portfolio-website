"""
S3 Storage integration for model artifacts and data lake
Handles model versioning and data archival
"""

import os
import boto3
import json
import joblib
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from botocore.exceptions import ClientError
import tensorflow as tf
import pandas as pd
import io

logger = logging.getLogger(__name__)

class S3StorageManager:
    """Manages S3 operations for models and data storage."""
    
    def __init__(self, model_bucket: Optional[str] = None, data_bucket: Optional[str] = None):
        self.model_bucket = model_bucket or os.getenv('S3_MODEL_BUCKET', 'ecommerce-analytics-ml-models')
        self.data_bucket = data_bucket or os.getenv('S3_DATA_BUCKET', 'ecommerce-analytics-data-lake')
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3')
        self.s3_resource = boto3.resource('s3')
        
    def _ensure_bucket_exists(self, bucket_name: str):
        """Ensure bucket exists, create if not."""
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.info(f"Creating bucket: {bucket_name}")
                self.s3_client.create_bucket(Bucket=bucket_name)
            else:
                raise
    
    async def upload_model(self, model, model_name: str, version: str, metadata: Dict[str, Any]):
        """Upload a trained model to S3 with versioning."""
        self._ensure_bucket_exists(self.model_bucket)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create model key with version
        model_key = f"models/{model_name}/v{version}/{model_name}_{timestamp}.keras"
        metadata_key = f"models/{model_name}/v{version}/metadata_{timestamp}.json"
        
        try:
            # Save model to temporary file
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
                if isinstance(model, tf.keras.Model):
                    model.save(tmp_file.name)
                else:
                    joblib.dump(model, tmp_file.name)
                
                # Upload to S3
                self.s3_client.upload_file(
                    tmp_file.name,
                    self.model_bucket,
                    model_key,
                    ExtraArgs={
                        'Metadata': {
                            'model_name': model_name,
                            'version': version,
                            'timestamp': timestamp,
                            'accuracy': str(metadata.get('accuracy', 0))
                        }
                    }
                )
                
                # Clean up temp file
                os.unlink(tmp_file.name)
            
            # Upload metadata
            self.s3_client.put_object(
                Bucket=self.model_bucket,
                Key=metadata_key,
                Body=json.dumps(metadata, indent=2),
                ContentType='application/json'
            )
            
            # Update latest model pointer
            latest_key = f"models/{model_name}/latest.json"
            latest_info = {
                'version': version,
                'model_key': model_key,
                'metadata_key': metadata_key,
                'timestamp': timestamp,
                'accuracy': metadata.get('accuracy', 0)
            }
            
            self.s3_client.put_object(
                Bucket=self.model_bucket,
                Key=latest_key,
                Body=json.dumps(latest_info, indent=2),
                ContentType='application/json'
            )
            
            logger.info(f"Model uploaded successfully: {model_key}")
            return model_key
            
        except Exception as e:
            logger.error(f"Error uploading model: {str(e)}")
            raise
    
    async def download_model(self, model_name: str, version: Optional[str] = None):
        """Download a model from S3."""
        try:
            if version is None:
                # Get latest version
                latest_key = f"models/{model_name}/latest.json"
                response = self.s3_client.get_object(Bucket=self.model_bucket, Key=latest_key)
                latest_info = json.loads(response['Body'].read())
                model_key = latest_info['model_key']
            else:
                # List objects with the specific version
                prefix = f"models/{model_name}/v{version}/"
                response = self.s3_client.list_objects_v2(
                    Bucket=self.model_bucket,
                    Prefix=prefix
                )
                
                # Find the model file
                model_files = [obj['Key'] for obj in response.get('Contents', []) 
                              if obj['Key'].endswith('.keras') or obj['Key'].endswith('.joblib')]
                
                if not model_files:
                    raise ValueError(f"No model found for {model_name} version {version}")
                
                model_key = model_files[0]  # Get the most recent
            
            # Download model to temporary file
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
                self.s3_client.download_file(self.model_bucket, model_key, tmp_file.name)
                
                # Load model based on file type
                if model_key.endswith('.keras'):
                    model = tf.keras.models.load_model(tmp_file.name)
                else:
                    model = joblib.load(tmp_file.name)
                
                # Clean up temp file
                os.unlink(tmp_file.name)
            
            logger.info(f"Model downloaded successfully: {model_key}")
            return model
            
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            raise
    
    async def list_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """List all versions of a model."""
        prefix = f"models/{model_name}/"
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.model_bucket,
                Prefix=prefix,
                Delimiter='/'
            )
            
            versions = []
            for prefix_info in response.get('CommonPrefixes', []):
                version_prefix = prefix_info['Prefix']
                if '/v' in version_prefix:
                    version = version_prefix.split('/v')[-1].rstrip('/')
                    
                    # Get metadata for this version
                    metadata_prefix = f"{version_prefix}metadata_"
                    metadata_response = self.s3_client.list_objects_v2(
                        Bucket=self.model_bucket,
                        Prefix=metadata_prefix
                    )
                    
                    if metadata_response.get('Contents'):
                        metadata_key = metadata_response['Contents'][0]['Key']
                        metadata_obj = self.s3_client.get_object(
                            Bucket=self.model_bucket, 
                            Key=metadata_key
                        )
                        metadata = json.loads(metadata_obj['Body'].read())
                        
                        versions.append({
                            'version': version,
                            'metadata': metadata,
                            'upload_time': metadata_response['Contents'][0]['LastModified']
                        })
            
            return sorted(versions, key=lambda x: x['version'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing model versions: {str(e)}")
            raise
    
    async def archive_transaction_data(self, transactions: pd.DataFrame, date: datetime):
        """Archive transaction data to S3 data lake."""
        self._ensure_bucket_exists(self.data_bucket)
        
        # Create partitioned path
        year = date.year
        month = date.month
        day = date.day
        
        key = f"transactions/year={year}/month={month:02d}/day={day:02d}/transactions_{date.strftime('%Y%m%d')}.parquet"
        
        try:
            # Convert DataFrame to Parquet in memory
            buffer = io.BytesIO()
            transactions.to_parquet(buffer, engine='pyarrow', compression='snappy')
            buffer.seek(0)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.data_bucket,
                Key=key,
                Body=buffer.getvalue(),
                ContentType='application/octet-stream',
                Metadata={
                    'record_count': str(len(transactions)),
                    'date': date.strftime('%Y-%m-%d')
                }
            )
            
            logger.info(f"Archived {len(transactions)} transactions to {key}")
            return key
            
        except Exception as e:
            logger.error(f"Error archiving transactions: {str(e)}")
            raise
    
    async def get_archived_transactions(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Retrieve archived transactions from S3 data lake."""
        all_transactions = []
        
        # Generate date range
        current_date = start_date
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            day = current_date.day
            
            key = f"transactions/year={year}/month={month:02d}/day={day:02d}/"
            
            try:
                # List objects for this date
                response = self.s3_client.list_objects_v2(
                    Bucket=self.data_bucket,
                    Prefix=key
                )
                
                for obj in response.get('Contents', []):
                    if obj['Key'].endswith('.parquet'):
                        # Download and read parquet file
                        obj_response = self.s3_client.get_object(
                            Bucket=self.data_bucket,
                            Key=obj['Key']
                        )
                        
                        df = pd.read_parquet(io.BytesIO(obj_response['Body'].read()))
                        all_transactions.append(df)
                
            except Exception as e:
                logger.warning(f"Error reading data for {current_date}: {str(e)}")
            
            current_date = current_date + pd.Timedelta(days=1)
        
        if all_transactions:
            return pd.concat(all_transactions, ignore_index=True)
        else:
            return pd.DataFrame()
    
    async def save_analytics_report(self, report_data: Dict[str, Any], report_type: str):
        """Save analytics reports to S3."""
        self._ensure_bucket_exists(self.data_bucket)
        
        timestamp = datetime.now()
        key = f"reports/{report_type}/{timestamp.strftime('%Y/%m/%d')}/{report_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            self.s3_client.put_object(
                Bucket=self.data_bucket,
                Key=key,
                Body=json.dumps(report_data, indent=2),
                ContentType='application/json',
                Metadata={
                    'report_type': report_type,
                    'timestamp': timestamp.isoformat()
                }
            )
            
            logger.info(f"Saved {report_type} report to {key}")
            return key
            
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            raise

# Singleton instance
s3_manager = S3StorageManager()