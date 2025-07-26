"""
Production-ready Apache Beam pipeline for processing 1M+ daily e-commerce transactions.
Designed for AWS infrastructure with Kinesis Data Streams integration.
"""

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
from apache_beam.transforms import window
from apache_beam.io.kinesis import ReadFromKinesis, WriteToKinesis
from apache_beam.metrics import Metrics
import tensorflow as tf
import numpy as np
import json
import datetime
import joblib
import boto3
import logging
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class PipelineConfig:
    """Configuration for the production pipeline"""
    # AWS Configuration
    aws_region: str = os.getenv('AWS_REGION', 'us-east-1')
    kinesis_stream_input: str = os.getenv('KINESIS_STREAM_INPUT', 'ecommerce-transactions')
    kinesis_stream_output: str = os.getenv('KINESIS_STREAM_OUTPUT', 'ecommerce-analytics')
    s3_model_bucket: str = os.getenv('S3_MODEL_BUCKET', 'ecommerce-models')
    s3_checkpoint_bucket: str = os.getenv('S3_CHECKPOINT_BUCKET', 'ecommerce-checkpoints')
    
    # Pipeline Configuration
    window_duration: int = 1  # seconds
    batch_size: int = 100
    max_retry_attempts: int = 3
    checkpoint_interval: int = 60  # seconds
    
    # Model Configuration
    model_version: str = os.getenv('MODEL_VERSION', 'latest')
    feature_columns: List[str] = None
    
    def __post_init__(self):
        """Initialize feature columns after dataclass creation"""
        self.feature_columns = [
            'daily_amount', 'num_transactions', 'total_items', 'unique_items',
            'basket_diversity', 'avg_purchase_value', 'hour_of_day', 'is_weekend',
            'morning_shopper', 'evening_shopper', 'month', 'quarter',
            'days_since_first', 'days_since_prev', 'active_days',
            'engagement_ratio', 'recency', 'days_between_purchases',
            'purchase_acceleration', 'spending_growth_rate', 'prev_amount',
            'rolling_avg_amount', 'rolling_avg_daily_amount', 'frequency',
            'tenure_months', 'customer_segment'
        ]


class ModelCache:
    """Singleton cache for ML models to avoid reloading"""
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance
    
    def get_model(self, model_path: str, force_reload: bool = False):
        """Get cached model or load from S3"""
        if model_path in self._models and not force_reload:
            return self._models[model_path]
        
        logger.info(f"Loading model from {model_path}")
        model = self._load_model_from_s3(model_path)
        self._models[model_path] = model
        return model
    
    def _load_model_from_s3(self, s3_path: str):
        """Load model from S3 bucket"""
        s3 = boto3.client('s3')
        bucket, key = s3_path.replace('s3://', '').split('/', 1)
        
        local_path = f'/tmp/{os.path.basename(key)}'
        s3.download_file(bucket, key, local_path)
        
        if key.endswith('.keras'):
            return tf.keras.models.load_model(local_path)
        elif key.endswith('.joblib'):
            return joblib.load(local_path)
        else:
            raise ValueError(f"Unsupported model format: {key}")


class TransactionEnricher(beam.DoFn):
    """Enriches transactions with customer history and temporal features"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.customer_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    def setup(self):
        """Initialize connections and load reference data"""
        # Initialize DynamoDB client for customer data
        self.dynamodb = boto3.resource('dynamodb', region_name=self.config.aws_region)
        self.customer_table = self.dynamodb.Table('customer-profiles')
        
        # Metrics
        self.processed_counter = Metrics.counter('transactions', 'processed')
        self.error_counter = Metrics.counter('transactions', 'errors')
        
    def process(self, element: bytes) -> List[Dict]:
        """Process and enrich a single transaction"""
        try:
            # Parse transaction
            transaction = json.loads(element.decode('utf-8'))
            
            # Add timestamp if not present
            if 'timestamp' not in transaction:
                transaction['timestamp'] = datetime.datetime.now().timestamp()
            
            # Convert timestamp to datetime
            dt = datetime.datetime.fromtimestamp(transaction['timestamp'])
            
            # Add temporal features
            transaction.update({
                'hour_of_day': dt.hour,
                'day_of_week': dt.weekday(),
                'is_weekend': 1 if dt.weekday() >= 5 else 0,
                'morning_shopper': 1 if dt.hour < 12 else 0,
                'evening_shopper': 1 if dt.hour > 17 else 0,
                'month': dt.month,
                'quarter': (dt.month - 1) // 3 + 1
            })
            
            # Enrich with customer data
            customer_id = transaction.get('customer_id')
            if customer_id:
                customer_data = self._get_customer_data(customer_id)
                transaction.update(customer_data)
            
            # Calculate derived features
            transaction['basket_diversity'] = (
                transaction.get('unique_items', 0) / 
                (transaction.get('total_items', 1) or 1)
            )
            transaction['avg_purchase_value'] = (
                transaction.get('daily_amount', 0) / 
                (transaction.get('num_transactions', 1) or 1)
            )
            
            self.processed_counter.inc()
            yield transaction
            
        except Exception as e:
            logger.error(f"Error processing transaction: {str(e)}")
            self.error_counter.inc()
            yield {
                'error': str(e),
                'timestamp': datetime.datetime.now().timestamp(),
                'raw_data': element.decode('utf-8', errors='ignore')
            }
    
    def _get_customer_data(self, customer_id: str) -> Dict:
        """Fetch customer historical data with caching"""
        # Check cache first
        cache_key = f"{customer_id}:{int(datetime.datetime.now().timestamp() / self.cache_ttl)}"
        if cache_key in self.customer_cache:
            return self.customer_cache[cache_key]
        
        try:
            # Query DynamoDB
            response = self.customer_table.get_item(Key={'customer_id': customer_id})
            
            if 'Item' in response:
                customer_data = response['Item']
            else:
                # New customer - return defaults
                customer_data = {
                    'days_since_first': 0,
                    'days_since_prev': 0,
                    'active_days': 1,
                    'engagement_ratio': 0.0,
                    'recency': 0,
                    'days_between_purchases': 0,
                    'purchase_acceleration': 0,
                    'spending_growth_rate': 0,
                    'prev_amount': 0,
                    'rolling_avg_amount': 0,
                    'rolling_avg_daily_amount': 0,
                    'frequency': 0,
                    'tenure_months': 0,
                    'customer_segment': 0
                }
            
            # Cache the result
            self.customer_cache[cache_key] = customer_data
            
            # Limit cache size
            if len(self.customer_cache) > 10000:
                # Remove oldest entries
                oldest_keys = sorted(self.customer_cache.keys())[:5000]
                for key in oldest_keys:
                    del self.customer_cache[key]
            
            return customer_data
            
        except Exception as e:
            logger.error(f"Error fetching customer data: {str(e)}")
            return {}


class PurchasePredictor(beam.DoFn):
    """Predicts purchase probability using ensemble models"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model_cache = ModelCache()
        
    def setup(self):
        """Load models and scalers"""
        # Load ensemble models
        self.models = []
        for i in range(5):  # 5-fold ensemble
            model_path = f"s3://{self.config.s3_model_bucket}/models/{self.config.model_version}/model_fold_{i}.keras"
            try:
                model = self.model_cache.get_model(model_path)
                self.models.append(model)
            except Exception as e:
                logger.error(f"Failed to load model {i}: {str(e)}")
        
        # Load scaler
        scaler_path = f"s3://{self.config.s3_model_bucket}/models/{self.config.model_version}/scaler.joblib"
        self.scaler = self.model_cache.get_model(scaler_path)
        
        # Metrics
        self.prediction_counter = Metrics.counter('predictions', 'successful')
        self.prediction_error_counter = Metrics.counter('predictions', 'errors')
        
    def process(self, element: Dict) -> List[Dict]:
        """Generate purchase predictions"""
        try:
            # Extract features in the correct order
            features = []
            for col in self.config.feature_columns:
                features.append(element.get(col, 0))
            
            # Scale features
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)
            
            # Ensemble prediction
            predictions = []
            for model in self.models:
                pred = model.predict(features_scaled, verbose=0)[0][0]
                predictions.append(pred)
            
            # Average predictions
            element['purchase_probability'] = float(np.mean(predictions))
            element['prediction_confidence'] = float(1 - np.std(predictions))
            
            # Add risk category
            prob = element['purchase_probability']
            if prob < 0.3:
                element['risk_category'] = 'high_churn_risk'
            elif prob < 0.6:
                element['risk_category'] = 'medium_risk'
            else:
                element['risk_category'] = 'low_risk'
            
            self.prediction_counter.inc()
            yield element
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
            self.prediction_error_counter.inc()
            element['prediction_error'] = str(e)
            yield element


class MetricsAggregator(beam.CombineFn):
    """Aggregates metrics across windows"""
    
    def create_accumulator(self):
        return {
            'total_amount': 0.0,
            'transaction_count': 0,
            'unique_customers': set(),
            'risk_distribution': {'high_churn_risk': 0, 'medium_risk': 0, 'low_risk': 0},
            'avg_purchase_probability': [],
            'processing_times': []
        }
    
    def add_input(self, accumulator, element):
        accumulator['total_amount'] += element.get('daily_amount', 0)
        accumulator['transaction_count'] += 1
        
        if 'customer_id' in element:
            accumulator['unique_customers'].add(element['customer_id'])
        
        if 'risk_category' in element:
            accumulator['risk_distribution'][element['risk_category']] += 1
        
        if 'purchase_probability' in element:
            accumulator['avg_purchase_probability'].append(element['purchase_probability'])
        
        return accumulator
    
    def merge_accumulators(self, accumulators):
        merged = self.create_accumulator()
        
        for acc in accumulators:
            merged['total_amount'] += acc['total_amount']
            merged['transaction_count'] += acc['transaction_count']
            merged['unique_customers'].update(acc['unique_customers'])
            
            for risk, count in acc['risk_distribution'].items():
                merged['risk_distribution'][risk] += count
            
            merged['avg_purchase_probability'].extend(acc['avg_purchase_probability'])
            merged['processing_times'].extend(acc['processing_times'])
        
        return merged
    
    def extract_output(self, accumulator):
        output = {
            'timestamp': datetime.datetime.now().isoformat(),
            'total_amount': accumulator['total_amount'],
            'transaction_count': accumulator['transaction_count'],
            'unique_customer_count': len(accumulator['unique_customers']),
            'risk_distribution': accumulator['risk_distribution'],
            'transactions_per_second': accumulator['transaction_count'],  # Since window is 1 second
        }
        
        if accumulator['avg_purchase_probability']:
            output['avg_purchase_probability'] = np.mean(accumulator['avg_purchase_probability'])
        
        return output


def format_for_kinesis(element):
    """Format element for Kinesis output"""
    return json.dumps(element).encode('utf-8')


def run_production_pipeline():
    """Run the production pipeline"""
    config = PipelineConfig()
    
    # Pipeline options
    pipeline_options = PipelineOptions()
    pipeline_options.view_as(StandardOptions).streaming = True
    pipeline_options.view_as(StandardOptions).runner = 'DataflowRunner'  # For production
    
    # Additional options for Dataflow
    pipeline_options.view_as(StandardOptions).project = os.getenv('GCP_PROJECT', 'your-project')
    pipeline_options.view_as(StandardOptions).region = 'us-central1'
    pipeline_options.view_as(StandardOptions).temp_location = f'gs://{config.s3_checkpoint_bucket}/temp'
    pipeline_options.view_as(StandardOptions).staging_location = f'gs://{config.s3_checkpoint_bucket}/staging'
    
    with beam.Pipeline(options=pipeline_options) as pipeline:
        # Read from Kinesis
        raw_transactions = (
            pipeline
            | 'Read from Kinesis' >> ReadFromKinesis(
                stream_name=config.kinesis_stream_input,
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region=config.aws_region,
                max_num_records=1000,
                max_read_time=1
            )
            | 'Window Transactions' >> beam.WindowInto(
                window.FixedWindows(config.window_duration),
                allowed_lateness=30,
                accumulation_mode=beam.transforms.window.AccumulationMode.DISCARDING
            )
        )
        
        # Process transactions
        enriched_transactions = (
            raw_transactions
            | 'Enrich Transactions' >> beam.ParDo(TransactionEnricher(config))
            | 'Predict Purchases' >> beam.ParDo(PurchasePredictor(config))
        )
        
        # Real-time analytics
        window_metrics = (
            enriched_transactions
            | 'Aggregate Metrics' >> beam.CombineGlobally(
                MetricsAggregator()
            ).without_defaults()
        )
        
        # Write enriched transactions to output stream
        enriched_output = (
            enriched_transactions
            | 'Format for Output' >> beam.Map(format_for_kinesis)
            | 'Write Transactions' >> WriteToKinesis(
                stream_name=config.kinesis_stream_output,
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region=config.aws_region,
                partition_key_fn=lambda x: json.loads(x.decode('utf-8')).get('customer_id', 'unknown')
            )
        )
        
        # Write metrics to monitoring stream
        metrics_output = (
            window_metrics
            | 'Format Metrics' >> beam.Map(format_for_kinesis)
            | 'Write Metrics' >> WriteToKinesis(
                stream_name=f"{config.kinesis_stream_output}-metrics",
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region=config.aws_region,
                partition_key_fn=lambda x: 'metrics'
            )
        )


if __name__ == '__main__':
    logger.info("Starting production analytics pipeline...")
    run_production_pipeline()