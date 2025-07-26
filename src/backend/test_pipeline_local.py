#!/usr/bin/env python3
"""
Local testing script for the Apache Beam pipeline.
Simulates Kinesis streams using local files for development.
"""

import json
import time
import random
import datetime
import numpy as np
from typing import List, Dict
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.testing.test_pipeline import TestPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionGenerator:
    """Generates realistic e-commerce transactions for testing"""
    
    def __init__(self, num_customers: int = 1000):
        self.num_customers = num_customers
        self.products = self._generate_products()
        self.customer_profiles = self._generate_customer_profiles()
    
    def _generate_products(self) -> List[Dict]:
        """Generate sample products"""
        categories = ['Electronics', 'Clothing', 'Home', 'Books', 'Sports']
        products = []
        
        for i in range(100):
            products.append({
                'product_id': f'PROD_{i:04d}',
                'category': random.choice(categories),
                'price': round(random.uniform(10, 500), 2)
            })
        
        return products
    
    def _generate_customer_profiles(self) -> Dict:
        """Generate customer profiles with behavior patterns"""
        profiles = {}
        
        for i in range(self.num_customers):
            customer_id = f'CUST_{i:04d}'
            
            # Assign customer segments
            segment = random.choices(
                ['frequent_buyer', 'occasional_buyer', 'new_customer', 'churning'],
                weights=[0.2, 0.5, 0.2, 0.1]
            )[0]
            
            profiles[customer_id] = {
                'customer_id': customer_id,
                'segment': segment,
                'avg_order_value': random.uniform(50, 300),
                'purchase_frequency': {
                    'frequent_buyer': random.uniform(5, 10),
                    'occasional_buyer': random.uniform(1, 3),
                    'new_customer': random.uniform(0.5, 1),
                    'churning': random.uniform(0.1, 0.5)
                }[segment],
                'preferred_categories': random.sample(
                    ['Electronics', 'Clothing', 'Home', 'Books', 'Sports'],
                    k=random.randint(1, 3)
                )
            }
        
        return profiles
    
    def generate_transaction(self) -> Dict:
        """Generate a single transaction"""
        customer_id = random.choice(list(self.customer_profiles.keys()))
        profile = self.customer_profiles[customer_id]
        
        # Select products based on customer preferences
        num_items = random.randint(1, 10)
        selected_products = []
        
        for _ in range(num_items):
            if random.random() < 0.7:  # 70% chance to buy from preferred category
                preferred_products = [
                    p for p in self.products 
                    if p['category'] in profile['preferred_categories']
                ]
                if preferred_products:
                    selected_products.append(random.choice(preferred_products))
                else:
                    selected_products.append(random.choice(self.products))
            else:
                selected_products.append(random.choice(self.products))
        
        # Calculate transaction details
        total_amount = sum(p['price'] * random.randint(1, 3) for p in selected_products)
        
        transaction = {
            'transaction_id': f'TXN_{int(time.time() * 1000)}_{random.randint(1000, 9999)}',
            'customer_id': customer_id,
            'timestamp': time.time(),
            'daily_amount': round(total_amount, 2),
            'num_transactions': 1,
            'total_items': len(selected_products),
            'unique_items': len(set(p['product_id'] for p in selected_products)),
            'products': [p['product_id'] for p in selected_products]
        }
        
        return transaction
    
    def generate_batch(self, size: int) -> List[Dict]:
        """Generate a batch of transactions"""
        return [self.generate_transaction() for _ in range(size)]


def test_pipeline_components():
    """Test individual pipeline components"""
    logger.info("Testing pipeline components...")
    
    # Test transaction generation
    generator = TransactionGenerator()
    
    # Generate sample transactions
    transactions = generator.generate_batch(100)
    logger.info(f"Generated {len(transactions)} test transactions")
    
    # Test with local Apache Beam pipeline
    with TestPipeline() as pipeline:
        # Create test input
        test_input = pipeline | 'Create' >> beam.Create(transactions)
        
        # Test enrichment (simplified version)
        def enrich_transaction(transaction):
            # Add temporal features
            dt = datetime.datetime.fromtimestamp(transaction['timestamp'])
            transaction['hour_of_day'] = dt.hour
            transaction['is_weekend'] = 1 if dt.weekday() >= 5 else 0
            transaction['morning_shopper'] = 1 if dt.hour < 12 else 0
            transaction['evening_shopper'] = 1 if dt.hour > 17 else 0
            return transaction
        
        enriched = test_input | 'Enrich' >> beam.Map(enrich_transaction)
        
        # Test aggregation
        def aggregate_metrics(transactions):
            return {
                'timestamp': datetime.datetime.now().isoformat(),
                'transaction_count': len(list(transactions)),
                'total_amount': sum(t['daily_amount'] for t in transactions),
                'unique_customers': len(set(t['customer_id'] for t in transactions))
            }
        
        metrics = (
            enriched 
            | 'Window' >> beam.WindowInto(beam.transforms.window.FixedWindows(1))
            | 'Aggregate' >> beam.CombineGlobally(aggregate_metrics).without_defaults()
        )
        
        # Print results
        enriched | 'Print Transactions' >> beam.Map(
            lambda x: logger.info(f"Enriched transaction: {x['transaction_id']}")
        )
        
        metrics | 'Print Metrics' >> beam.Map(
            lambda x: logger.info(f"Window metrics: {x}")
        )
    
    logger.info("Component testing completed successfully")


def simulate_streaming_load():
    """Simulate streaming load to test throughput"""
    logger.info("Starting streaming load simulation...")
    
    generator = TransactionGenerator(num_customers=10000)
    
    # Target: 1M transactions per day = ~11.6 transactions per second
    target_tps = 12
    duration_seconds = 60  # Run for 1 minute
    
    start_time = time.time()
    transaction_count = 0
    
    logger.info(f"Simulating {target_tps} transactions per second for {duration_seconds} seconds")
    
    while time.time() - start_time < duration_seconds:
        batch_start = time.time()
        
        # Generate batch of transactions
        batch = generator.generate_batch(target_tps)
        transaction_count += len(batch)
        
        # Simulate processing
        for transaction in batch:
            # Add some processing delay
            time.sleep(0.001)  # 1ms per transaction
        
        # Calculate actual TPS
        elapsed = time.time() - start_time
        actual_tps = transaction_count / elapsed
        
        logger.info(f"Processed {transaction_count} transactions. "
                   f"Actual TPS: {actual_tps:.2f}, Target TPS: {target_tps}")
        
        # Sleep to maintain target rate
        batch_duration = time.time() - batch_start
        sleep_time = max(0, 1.0 - batch_duration)
        time.sleep(sleep_time)
    
    # Final statistics
    total_duration = time.time() - start_time
    final_tps = transaction_count / total_duration
    
    logger.info(f"\nSimulation completed:")
    logger.info(f"Total transactions: {transaction_count}")
    logger.info(f"Duration: {total_duration:.2f} seconds")
    logger.info(f"Average TPS: {final_tps:.2f}")
    logger.info(f"Daily projection: {final_tps * 86400:,.0f} transactions")


def test_prediction_latency():
    """Test model prediction latency"""
    logger.info("Testing prediction latency...")
    
    # Simulate model inference
    import tensorflow as tf
    
    # Create a dummy model for testing
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(26,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Generate test data
    test_features = np.random.rand(1000, 26)
    
    # Warm up
    _ = model.predict(test_features[:10], verbose=0)
    
    # Measure latency
    latencies = []
    
    for i in range(100):
        start = time.time()
        _ = model.predict(test_features[i:i+1], verbose=0)
        latency = (time.time() - start) * 1000  # Convert to ms
        latencies.append(latency)
    
    # Calculate statistics
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    logger.info(f"\nPrediction latency statistics:")
    logger.info(f"Average: {avg_latency:.2f} ms")
    logger.info(f"P50: {p50_latency:.2f} ms")
    logger.info(f"P95: {p95_latency:.2f} ms")
    logger.info(f"P99: {p99_latency:.2f} ms")
    
    if avg_latency < 100:
        logger.info("✓ Latency target met (< 100ms average)")
    else:
        logger.warning("✗ Latency target not met (> 100ms average)")


def main():
    """Run all tests"""
    logger.info("Starting Apache Beam pipeline local tests\n")
    
    # Test 1: Component testing
    logger.info("=" * 50)
    logger.info("Test 1: Component Testing")
    logger.info("=" * 50)
    test_pipeline_components()
    
    # Test 2: Throughput simulation
    logger.info("\n" + "=" * 50)
    logger.info("Test 2: Throughput Simulation")
    logger.info("=" * 50)
    simulate_streaming_load()
    
    # Test 3: Latency testing
    logger.info("\n" + "=" * 50)
    logger.info("Test 3: Prediction Latency")
    logger.info("=" * 50)
    test_prediction_latency()
    
    logger.info("\n" + "=" * 50)
    logger.info("All tests completed!")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()