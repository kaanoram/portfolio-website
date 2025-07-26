"""
AWS Lambda handler for the demo API.
Provides real ML predictions while keeping costs minimal.
"""

import json
import time
import random
import boto3
import numpy as np
from datetime import datetime, timedelta
import base64
import os
from typing import Dict, Any, List
import tensorflow as tf
import joblib

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Global variables for model caching
models_cache = {}
scaler_cache = None
CACHE_TTL = 3600  # 1 hour

def load_models():
    """Load ML models from S3 with caching"""
    global models_cache, scaler_cache
    
    # Check if models are already cached
    if models_cache and scaler_cache:
        return models_cache, scaler_cache
    
    model_bucket = os.environ.get('MODEL_BUCKET')
    
    # Load scaler
    scaler_obj = s3_client.get_object(
        Bucket=model_bucket,
        Key='models/v1.0.0/scaler.joblib'
    )
    scaler_cache = joblib.load(scaler_obj['Body'])
    
    # For demo, load just one model (not all 5)
    # This saves Lambda memory and cold start time
    model_obj = s3_client.get_object(
        Bucket=model_bucket,
        Key='models/v1.0.0/model_fold_0.keras'
    )
    
    # Save to /tmp and load
    with open('/tmp/model.keras', 'wb') as f:
        f.write(model_obj['Body'].read())
    
    models_cache['model'] = tf.keras.models.load_model('/tmp/model.keras')
    
    return models_cache, scaler_cache


def generate_demo_transaction():
    """Generate realistic demo transaction"""
    customer_segments = {
        0: {"name": "New Customer", "prob": 0.15, "avg_amount": 45},
        1: {"name": "Regular Buyer", "prob": 0.65, "avg_amount": 125},
        2: {"name": "VIP Customer", "prob": 0.85, "avg_amount": 350},
        3: {"name": "At-Risk", "prob": 0.25, "avg_amount": 75}
    }
    
    # Random customer segment
    segment = random.choices([0, 1, 2, 3], weights=[0.3, 0.4, 0.2, 0.1])[0]
    behavior = customer_segments[segment]
    
    # Generate features matching the model's expectations
    features = {
        'daily_amount': np.random.normal(behavior['avg_amount'], behavior['avg_amount'] * 0.3),
        'num_transactions': random.randint(1, 5),
        'total_items': random.randint(1, 20),
        'unique_items': random.randint(1, 15),
        'basket_diversity': random.random(),
        'avg_purchase_value': behavior['avg_amount'],
        'hour_of_day': datetime.now().hour,
        'is_weekend': 1 if datetime.now().weekday() >= 5 else 0,
        'morning_shopper': 1 if datetime.now().hour < 12 else 0,
        'evening_shopper': 1 if datetime.now().hour > 17 else 0,
        'month': datetime.now().month,
        'quarter': (datetime.now().month - 1) // 3 + 1,
        'days_since_first': random.randint(0, 365),
        'days_since_prev': random.randint(1, 30),
        'active_days': random.randint(1, 100),
        'engagement_ratio': random.random(),
        'recency': random.randint(0, 30),
        'days_between_purchases': random.randint(7, 30),
        'purchase_acceleration': random.uniform(-0.5, 0.5),
        'spending_growth_rate': random.uniform(-0.2, 0.3),
        'prev_amount': behavior['avg_amount'] * random.uniform(0.5, 1.5),
        'rolling_avg_amount': behavior['avg_amount'],
        'rolling_avg_daily_amount': behavior['avg_amount'],
        'frequency': random.uniform(0.1, 5),
        'tenure_months': random.randint(0, 24),
        'customer_segment': segment
    }
    
    return features, behavior


def predict_purchase_probability(features: Dict) -> float:
    """Make actual ML prediction"""
    models, scaler = load_models()
    
    # Prepare features in correct order
    feature_values = [features.get(col, 0) for col in [
        'daily_amount', 'num_transactions', 'total_items', 'unique_items',
        'basket_diversity', 'avg_purchase_value', 'hour_of_day', 'is_weekend',
        'morning_shopper', 'evening_shopper', 'month', 'quarter',
        'days_since_first', 'days_since_prev', 'active_days',
        'engagement_ratio', 'recency', 'days_between_purchases',
        'purchase_acceleration', 'spending_growth_rate', 'prev_amount',
        'rolling_avg_amount', 'rolling_avg_daily_amount', 'frequency',
        'tenure_months', 'customer_segment'
    ]]
    
    # Scale features
    features_array = np.array(feature_values).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    
    # Predict
    prediction = models['model'].predict(features_scaled, verbose=0)[0][0]
    
    return float(prediction)


def main(event, context):
    """Lambda handler function"""
    
    # Handle preflight requests
    if event.get('requestContext', {}).get('http', {}).get('method') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': ''
        }
    
    # Parse request
    path = event.get('rawPath', '/')
    method = event.get('requestContext', {}).get('http', {}).get('method', 'GET')
    
    try:
        if path == '/api/demo/transaction':
            # Generate demo transaction with real prediction
            features, behavior = generate_demo_transaction()
            
            # Get actual ML prediction
            start_time = time.time()
            probability = predict_purchase_probability(features)
            inference_time = (time.time() - start_time) * 1000
            
            # Create transaction record
            transaction = {
                'transaction_id': f"TXN_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
                'customer_id': f"CUST_{random.randint(1, 5000):04d}",
                'timestamp': datetime.now().isoformat(),
                'amount': round(features['daily_amount'], 2),
                'items': features['total_items'],
                'segment_name': behavior['name'],
                'purchase_probability': round(probability, 3),
                'risk_category': 'high_risk' if probability < 0.3 else 'medium_risk' if probability < 0.6 else 'low_risk',
                'inference_time_ms': round(inference_time, 2)
            }
            
            # Store in DynamoDB (with TTL for auto-cleanup)
            if os.environ.get('DEMO_MODE') == 'true':
                table_name = f"{os.environ.get('PROJECT_NAME', 'ecommerce-analytics-demo')}-transactions"
                table = dynamodb.Table(table_name)
                
                transaction['expire_at'] = int((datetime.now() + timedelta(days=7)).timestamp())
                table.put_item(Item=transaction)
            
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps(transaction)
            }
            
        elif path == '/api/demo/metrics':
            # Return aggregated metrics
            table_name = f"{os.environ.get('PROJECT_NAME', 'ecommerce-analytics-demo')}-transactions"
            table = dynamodb.Table(table_name)
            
            # Query recent transactions (last hour)
            one_hour_ago = int((datetime.now() - timedelta(hours=1)).timestamp())
            
            response = table.scan(
                FilterExpression='#ts > :timestamp',
                ExpressionAttributeNames={'#ts': 'timestamp'},
                ExpressionAttributeValues={':timestamp': one_hour_ago},
                Limit=100
            )
            
            transactions = response.get('Items', [])
            
            if transactions:
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'transaction_count': len(transactions),
                    'total_revenue': sum(float(t.get('amount', 0)) for t in transactions),
                    'avg_purchase_probability': np.mean([float(t.get('purchase_probability', 0)) for t in transactions]),
                    'risk_distribution': {
                        'high_risk': len([t for t in transactions if t.get('risk_category') == 'high_risk']),
                        'medium_risk': len([t for t in transactions if t.get('risk_category') == 'medium_risk']),
                        'low_risk': len([t for t in transactions if t.get('risk_category') == 'low_risk'])
                    },
                    'avg_inference_time_ms': np.mean([float(t.get('inference_time_ms', 0)) for t in transactions if t.get('inference_time_ms')]),
                    'model_accuracy': 0.801  # Your actual model accuracy
                }
            else:
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'transaction_count': 0,
                    'message': 'No recent transactions. Generate some demo data!'
                }
            
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps(metrics)
            }
            
        elif path == '/api/capabilities':
            # Return system capabilities (for portfolio display)
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'demo_mode': True,
                    'capabilities': {
                        'max_transactions_per_day': '1,000,000+',
                        'concurrent_users': '2,000+',
                        'model_accuracy': '80.1%',
                        'avg_latency_ms': '<100',
                        'ml_features': 26,
                        'customer_segments': 4,
                        'infrastructure': {
                            'compute': 'AWS Lambda (serverless)',
                            'storage': 'S3 + DynamoDB',
                            'ml_framework': 'TensorFlow 2.14',
                            'api': 'API Gateway HTTP',
                            'cdn': 'CloudFront'
                        }
                    },
                    'live_demo_limits': {
                        'transactions_per_minute': 60,
                        'note': 'Full production system available on request'
                    }
                })
            }
        
        else:
            return {
                'statusCode': 404,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': 'Not found'})
            }
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e) if os.environ.get('DEMO_MODE') == 'true' else 'An error occurred'
            })
        }