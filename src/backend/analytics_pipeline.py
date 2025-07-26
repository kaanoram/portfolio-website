import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms import window
from apache_beam.io import ReadFromPubSub
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
import json
import datetime
import joblib  # Added missing import
from sklearn.preprocessing import StandardScaler  # Added for type hints
import logging  # Added for better error handling

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionProcessor:
    """
    Processes raw e-commerce transactions in real-time, enriching them with 
    customer data and preparing them for analysis.
    """
    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialize the processor with paths to the trained model and scaler.
        
        Args:
            model_path (str): Path to the saved TensorFlow model
            scaler_path (str): Path to the saved StandardScaler object
        """
        try:
            # Load the trained prediction model and scaler
            self.model = tf.keras.models.load_model(model_path)
            self.scaler: StandardScaler = joblib.load(scaler_path)
            logger.info("Successfully loaded model and scaler")
        except (OSError, ValueError) as e:
            logger.error(f"Error loading model or scaler: {str(e)}")
            raise
        
    def preprocess_transaction(self, element: Dict) -> Dict:
        """
        Preprocesses a single transaction by normalizing values and adding features.
        
        Args:
            element (Dict): Raw transaction data
            
        Returns:
            Dict: Processed transaction with additional features
        """
        try:
            # Extract basic transaction info
            transaction = json.loads(element.decode('utf-8'))
            
            # Add temporal features
            timestamp = datetime.datetime.fromtimestamp(transaction['timestamp'])
            transaction['hour_of_day'] = timestamp.hour
            transaction['day_of_week'] = timestamp.weekday()
            transaction['is_weekend'] = 1 if timestamp.weekday() >= 5 else 0
            
            # Normalize monetary values using the scaler
            transaction['amount_normalized'] = float(self.scaler.transform(
                [[transaction['amount']]])[0][0])
            
            # Add customer context
            customer_id = transaction['customer_id']
            transaction['customer_history'] = self.get_customer_history(customer_id)
            
            return transaction
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error processing transaction: {str(e)}")
            # Return a simplified version of the transaction for error handling
            return {
                'error': str(e),
                'timestamp': datetime.datetime.now().timestamp(),
                'status': 'failed'
            }

    def get_customer_history(self, customer_id: str) -> Dict:
        """
        Fetches customer's historical behavior data.
        
        Args:
            customer_id (str): Unique identifier for the customer
            
        Returns:
            Dict: Customer's historical behavior metrics
        """
        # In production, this would query a database
        # For now, we'll return dummy data
        try:
            # Simulate database lookup
            return {
                'total_purchases': 10,
                'average_order_value': 150.0,
                'days_since_last_purchase': 7
            }
        except Exception as e:
            logger.error(f"Error fetching customer history: {str(e)}")
            # Return safe default values
            return {
                'total_purchases': 0,
                'average_order_value': 0.0,
                'days_since_last_purchase': 999
            }
        
class PurchasePredictorDoFn(beam.DoFn):
    """
    Predicts the likelihood of future purchases using the loaded model.
    """
    def process(self, element: Dict) -> List[Dict]:
        # Extract features for prediction
        features = [
            element['amount_normalized'],
            element['hour_of_day'] / 24.0,  # Normalize hour
            element['is_weekend'],
            element['customer_history']['total_purchases'] / 100.0,  # Normalize purchases
            element['customer_history']['days_since_last_purchase'] / 30.0  # Normalize days
        ]
        
        # Make prediction
        features_array = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features_array)[0][0]
        
        # Add prediction to transaction data
        element['purchase_probability'] = float(prediction)
        return [element]

def run_pipeline(input_subscription: str, output_topic: str):
    """
    Runs the main data processing pipeline.
    """
    pipeline_options = PipelineOptions(
        streaming=True,
        project='your-project-id',
        region='us-central1',
        job_name='ecommerce-analytics-pipeline',
        temp_location='gs://your-bucket/temp'
    )
    
    with beam.Pipeline(options=pipeline_options) as pipeline:
        # Read transactions from Pub/Sub
        transactions = (
            pipeline 
            | 'Read from Pub/Sub' >> ReadFromPubSub(subscription=input_subscription)
            # Group transactions into 1-second windows
            | 'Window into 1s' >> beam.WindowInto(window.FixedWindows(1))
        )
        
        # Process transactions
        processed_transactions = (
            transactions
            | 'Preprocess Transactions' >> beam.Map(TransactionProcessor().preprocess_transaction)
            | 'Predict Future Purchases' >> beam.ParDo(PurchasePredictorDoFn())
        )
        
        # Calculate real-time metrics
        metrics = (
            processed_transactions
            | 'Calculate Metrics' >> beam.CombineGlobally(
                lambda transactions: {
                    'total_amount': sum(t['amount'] for t in transactions),
                    'transaction_count': len(transactions),
                    'average_probability': np.mean([t['purchase_probability'] for t in transactions])
                }
            ).without_defaults()
        )
        
        # Write results back to Pub/Sub
        metrics | 'Write to Pub/Sub' >> beam.io.WriteToPubSub(output_topic)

if __name__ == '__main__':
    input_subscription = 'projects/your-project/subscriptions/transactions-sub'
    output_topic = 'projects/your-project/topics/analytics-output'
    run_pipeline(input_subscription, output_topic)