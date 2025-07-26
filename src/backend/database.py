"""
Database integration layer for AWS RDS PostgreSQL
Handles transaction storage and retrieval for analytics
"""

import os
import json
import asyncio
import asyncpg
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and operations for the analytics platform."""
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.getenv('DATABASE_URL')
        self.pool: Optional[asyncpg.Pool] = None
        
    async def initialize(self):
        """Initialize database connection pool."""
        try:
            # Parse connection string from AWS Secrets Manager format
            if self.connection_string and self.connection_string.startswith('{'):
                conn_data = json.loads(self.connection_string)
                self.connection_string = (
                    f"postgresql://{conn_data['username']}:{conn_data['password']}"
                    f"@{conn_data['host']}:{conn_data['port']}/{conn_data['dbname']}"
                )
            
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=10,
                max_size=100,  # Handle high concurrent connections
                max_queries=50000,
                max_cached_statement_lifetime=300,
                command_timeout=10
            )
            
            # Create tables if they don't exist
            await self.create_tables()
            
            logger.info("Database connection pool initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a database connection from the pool."""
        async with self.pool.acquire() as connection:
            yield connection
    
    async def create_tables(self):
        """Create necessary database tables."""
        async with self.acquire() as conn:
            # Transactions table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id BIGSERIAL PRIMARY KEY,
                    transaction_id VARCHAR(50) UNIQUE NOT NULL,
                    customer_id VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    amount DECIMAL(10, 2) NOT NULL,
                    quantity INTEGER NOT NULL,
                    product_id VARCHAR(50) NOT NULL,
                    product_category VARCHAR(100),
                    prediction_score DECIMAL(5, 4),
                    will_purchase_prediction BOOLEAN,
                    customer_segment INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    INDEX idx_customer_id (customer_id),
                    INDEX idx_timestamp (timestamp),
                    INDEX idx_created_at (created_at)
                )
            ''')
            
            # Customer profiles table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS customer_profiles (
                    customer_id VARCHAR(50) PRIMARY KEY,
                    total_spent DECIMAL(12, 2) DEFAULT 0,
                    transaction_count INTEGER DEFAULT 0,
                    first_purchase_date TIMESTAMP,
                    last_purchase_date TIMESTAMP,
                    avg_transaction_value DECIMAL(10, 2),
                    purchase_frequency_days DECIMAL(6, 2),
                    customer_segment INTEGER,
                    loyalty_score DECIMAL(5, 4),
                    churn_risk_score DECIMAL(5, 4),
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    INDEX idx_segment (customer_segment),
                    INDEX idx_last_updated (last_updated)
                )
            ''')
            
            # Analytics metrics table (for aggregated data)
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS analytics_metrics (
                    id SERIAL PRIMARY KEY,
                    metric_date DATE NOT NULL,
                    metric_hour INTEGER,
                    total_transactions INTEGER DEFAULT 0,
                    total_revenue DECIMAL(12, 2) DEFAULT 0,
                    unique_customers INTEGER DEFAULT 0,
                    avg_transaction_value DECIMAL(10, 2),
                    conversion_rate DECIMAL(5, 4),
                    new_customers INTEGER DEFAULT 0,
                    returning_customers INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(metric_date, metric_hour),
                    INDEX idx_metric_date (metric_date)
                )
            ''')
            
            # Model performance tracking
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id SERIAL PRIMARY KEY,
                    model_version VARCHAR(50) NOT NULL,
                    evaluation_date DATE NOT NULL,
                    accuracy DECIMAL(5, 4),
                    precision DECIMAL(5, 4),
                    recall DECIMAL(5, 4),
                    f1_score DECIMAL(5, 4),
                    auc_roc DECIMAL(5, 4),
                    total_predictions INTEGER,
                    true_positives INTEGER,
                    false_positives INTEGER,
                    true_negatives INTEGER,
                    false_negatives INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    INDEX idx_evaluation_date (evaluation_date),
                    INDEX idx_model_version (model_version)
                )
            ''')
            
            logger.info("Database tables created successfully")
    
    async def insert_transaction(self, transaction_data: Dict[str, Any]):
        """Insert a single transaction into the database."""
        async with self.acquire() as conn:
            await conn.execute('''
                INSERT INTO transactions (
                    transaction_id, customer_id, timestamp, amount, quantity,
                    product_id, product_category, prediction_score, 
                    will_purchase_prediction, customer_segment
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (transaction_id) DO NOTHING
            ''',
                transaction_data['transaction_id'],
                transaction_data['customer_id'],
                datetime.fromtimestamp(transaction_data['timestamp']),
                transaction_data['amount'],
                transaction_data['quantity'],
                transaction_data['product_id'],
                transaction_data.get('product_category'),
                transaction_data.get('prediction_score'),
                transaction_data.get('will_purchase_prediction'),
                transaction_data.get('customer_segment')
            )
    
    async def batch_insert_transactions(self, transactions: List[Dict[str, Any]]):
        """Batch insert multiple transactions for efficiency."""
        async with self.acquire() as conn:
            # Prepare data for batch insert
            values = [
                (
                    t['transaction_id'],
                    t['customer_id'],
                    datetime.fromtimestamp(t['timestamp']),
                    t['amount'],
                    t['quantity'],
                    t['product_id'],
                    t.get('product_category'),
                    t.get('prediction_score'),
                    t.get('will_purchase_prediction'),
                    t.get('customer_segment')
                )
                for t in transactions
            ]
            
            # Use COPY for maximum performance
            await conn.copy_records_to_table(
                'transactions',
                records=values,
                columns=[
                    'transaction_id', 'customer_id', 'timestamp', 'amount',
                    'quantity', 'product_id', 'product_category', 
                    'prediction_score', 'will_purchase_prediction', 'customer_segment'
                ]
            )
    
    async def update_customer_profile(self, customer_id: str, profile_data: Dict[str, Any]):
        """Update or create customer profile."""
        async with self.acquire() as conn:
            await conn.execute('''
                INSERT INTO customer_profiles (
                    customer_id, total_spent, transaction_count,
                    first_purchase_date, last_purchase_date,
                    avg_transaction_value, purchase_frequency_days,
                    customer_segment, loyalty_score, churn_risk_score
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (customer_id) DO UPDATE SET
                    total_spent = EXCLUDED.total_spent,
                    transaction_count = EXCLUDED.transaction_count,
                    last_purchase_date = EXCLUDED.last_purchase_date,
                    avg_transaction_value = EXCLUDED.avg_transaction_value,
                    purchase_frequency_days = EXCLUDED.purchase_frequency_days,
                    customer_segment = EXCLUDED.customer_segment,
                    loyalty_score = EXCLUDED.loyalty_score,
                    churn_risk_score = EXCLUDED.churn_risk_score,
                    last_updated = CURRENT_TIMESTAMP
            ''',
                customer_id,
                profile_data['total_spent'],
                profile_data['transaction_count'],
                profile_data['first_purchase_date'],
                profile_data['last_purchase_date'],
                profile_data['avg_transaction_value'],
                profile_data.get('purchase_frequency_days'),
                profile_data.get('customer_segment'),
                profile_data.get('loyalty_score'),
                profile_data.get('churn_risk_score')
            )
    
    async def get_customer_history(self, customer_id: str, days: int = 30) -> List[Dict]:
        """Get customer transaction history."""
        async with self.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM transactions
                WHERE customer_id = $1
                AND timestamp > CURRENT_TIMESTAMP - INTERVAL '%s days'
                ORDER BY timestamp DESC
            ''' % days, customer_id)
            
            return [dict(row) for row in rows]
    
    async def update_analytics_metrics(self, metrics_data: Dict[str, Any]):
        """Update hourly analytics metrics."""
        current_hour = datetime.now().hour
        current_date = datetime.now().date()
        
        async with self.acquire() as conn:
            await conn.execute('''
                INSERT INTO analytics_metrics (
                    metric_date, metric_hour, total_transactions,
                    total_revenue, unique_customers, avg_transaction_value,
                    conversion_rate, new_customers, returning_customers
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (metric_date, metric_hour) DO UPDATE SET
                    total_transactions = analytics_metrics.total_transactions + EXCLUDED.total_transactions,
                    total_revenue = analytics_metrics.total_revenue + EXCLUDED.total_revenue,
                    unique_customers = EXCLUDED.unique_customers,
                    avg_transaction_value = EXCLUDED.avg_transaction_value,
                    conversion_rate = EXCLUDED.conversion_rate,
                    new_customers = analytics_metrics.new_customers + EXCLUDED.new_customers,
                    returning_customers = analytics_metrics.returning_customers + EXCLUDED.returning_customers
            ''',
                current_date,
                current_hour,
                metrics_data['total_transactions'],
                metrics_data['total_revenue'],
                metrics_data['unique_customers'],
                metrics_data['avg_transaction_value'],
                metrics_data['conversion_rate'],
                metrics_data.get('new_customers', 0),
                metrics_data.get('returning_customers', 0)
            )
    
    async def get_realtime_metrics(self) -> Dict[str, Any]:
        """Get real-time analytics metrics."""
        async with self.acquire() as conn:
            # Get today's metrics
            today_metrics = await conn.fetchrow('''
                SELECT 
                    SUM(total_transactions) as total_transactions,
                    SUM(total_revenue) as total_revenue,
                    AVG(conversion_rate) as avg_conversion_rate
                FROM analytics_metrics
                WHERE metric_date = CURRENT_DATE
            ''')
            
            # Get active customers in last hour
            active_customers = await conn.fetchval('''
                SELECT COUNT(DISTINCT customer_id)
                FROM transactions
                WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour'
            ''')
            
            # Get trending products
            trending_products = await conn.fetch('''
                SELECT product_id, COUNT(*) as purchase_count
                FROM transactions
                WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '24 hours'
                GROUP BY product_id
                ORDER BY purchase_count DESC
                LIMIT 10
            ''')
            
            return {
                'total_transactions': today_metrics['total_transactions'] or 0,
                'total_revenue': float(today_metrics['total_revenue'] or 0),
                'conversion_rate': float(today_metrics['avg_conversion_rate'] or 0),
                'active_customers': active_customers,
                'trending_products': [dict(row) for row in trending_products]
            }
    
    async def record_model_performance(self, performance_data: Dict[str, Any]):
        """Record model performance metrics."""
        async with self.acquire() as conn:
            await conn.execute('''
                INSERT INTO model_performance (
                    model_version, evaluation_date, accuracy, precision,
                    recall, f1_score, auc_roc, total_predictions,
                    true_positives, false_positives, true_negatives, false_negatives
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ''',
                performance_data['model_version'],
                datetime.now().date(),
                performance_data['accuracy'],
                performance_data['precision'],
                performance_data['recall'],
                performance_data['f1_score'],
                performance_data['auc_roc'],
                performance_data['total_predictions'],
                performance_data['true_positives'],
                performance_data['false_positives'],
                performance_data['true_negatives'],
                performance_data['false_negatives']
            )
    
    async def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old transaction data to manage storage."""
        async with self.acquire() as conn:
            deleted = await conn.execute('''
                DELETE FROM transactions
                WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '%s days'
            ''' % days_to_keep)
            
            logger.info(f"Cleaned up {deleted} old transactions")

# Singleton instance
db_manager = DatabaseManager()