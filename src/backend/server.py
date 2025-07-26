from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import asyncio
import json
from datetime import datetime
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import random
import os
from contextlib import asynccontextmanager
from database import db_manager
from s3_storage import s3_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up...")
    if os.getenv('DATABASE_URL'):
        await db_manager.initialize()
        logger.info("Database initialized")
    yield
    # Shutdown
    logger.info("Shutting down...")
    if db_manager.pool:
        await db_manager.close()

app = FastAPI(lifespan=lifespan)

# Enable CORS for React frontend
origins = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    """
    Manages WebSocket connections and handles real-time data broadcasting.
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.models = []
        self.scaler = None
        self.kmeans = None
        self.clustering_data = None
        self.models_loaded = False
        
        # Load sample data for demo
        self.sample_data = self._load_sample_data()
        self.transaction_index = 0
        
        # Store aggregate metrics
        self.metrics = {
            'total_revenue': 0,
            'total_orders': 0,
            'total_customers': {},  # Changed to dict to store customer history
            'conversion_rate': 0,
            'start_time': datetime.now()
        }
    
    async def _load_models(self):
        """Load models from S3 in production or local files in development."""
        try:
            if os.getenv('ENVIRONMENT') == 'production' and os.getenv('S3_MODEL_BUCKET'):
                # Load from S3 in production
                logger.info("Loading models from S3...")
                
                # Load scaler
                self.scaler = await s3_manager.download_model("feature_scaler")
                logger.info("Loaded scaler from S3")
                
                # Load prediction models
                for i in range(5):
                    try:
                        model = await s3_manager.download_model(f"purchase_prediction_fold_{i}")
                        self.models.append(model)
                        logger.info(f"Loaded prediction model fold {i} from S3")
                    except Exception as e:
                        logger.warning(f"Could not load fold {i} from S3: {str(e)}")
                
                # Load clustering model
                clustering_data = await s3_manager.download_model("customer_segmentation")
                if isinstance(clustering_data, dict):
                    self.clustering_data = clustering_data
                    self.kmeans = clustering_data.get('kmeans')
                logger.info("Loaded clustering model from S3")
                
            else:
                # Load from local files in development
                logger.info("Loading models from local files...")
                
                if os.path.exists("models/scaler.joblib"):
                    self.scaler = joblib.load("models/scaler.joblib")
                    logger.info("Loaded scaler from local file")
                
                # Load the ensemble of models
                for i in range(5):
                    model_path = f"models/model_fold_{i}.keras"
                    if os.path.exists(model_path):
                        model = tf.keras.models.load_model(model_path, compile=False)
                        self.models.append(model)
                        logger.info(f"Loaded model fold {i} from local file")
                
                # Load clustering model
                if os.path.exists("models/enhanced_clustering_model.joblib"):
                    self.clustering_data = joblib.load("models/enhanced_clustering_model.joblib")
                    self.kmeans = self.clustering_data.get('kmeans')
                    logger.info("Loaded clustering model from local file")
            
            logger.info(f"Successfully loaded {len(self.models)} prediction models")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.models = []

    async def connect(self, websocket: WebSocket):
        """Establish WebSocket connection and send initial state."""
        try:
            # Load models on first connection if not already loaded
            if not self.models_loaded:
                await self._load_models()
                self.models_loaded = True
                
            await websocket.accept()
            self.active_connections.append(websocket)
            # Send current metrics immediately upon connection
            await self.send_metrics(websocket)
            logger.info(f"New client connected. Total connections: {len(self.active_connections)}")
        except Exception as e:
            logger.error(f"Error during connection: {str(e)}")
            raise

    def disconnect(self, websocket: WebSocket):
        """Handle client disconnection."""
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Remaining connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                disconnected.append(connection)
            except Exception as e:
                logger.error(f"Error broadcasting message: {str(e)}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    def _load_sample_data(self):
        """Load sample data from the training dataset."""
        try:
            # Load the original data file if it exists
            if os.path.exists('data/Online_Retail.csv'):
                df = pd.read_csv('data/Online_Retail.csv', nrows=10000)
                # Basic cleaning
                df = df.dropna()
                df = df[df['Quantity'] > 0]
                df = df[df['UnitPrice'] > 0]
                return df
            else:
                logger.warning("Sample data not found, will generate synthetic data")
                return None
        except Exception as e:
            logger.error(f"Error loading sample data: {str(e)}")
            return None
    
    def _generate_realistic_transaction(self):
        """Generate a realistic transaction based on patterns."""
        if self.sample_data is not None and len(self.sample_data) > 0:
            # Use real data as basis
            idx = self.transaction_index % len(self.sample_data)
            row = self.sample_data.iloc[idx]
            self.transaction_index += 1
            
            # Create transaction with some variation
            amount = float(row['UnitPrice'] * row['Quantity'] * random.uniform(0.8, 1.2))
            customer_id = str(row.get('CustomerID', f'CUST-{random.randint(1000, 9999)}'))
            
            return {
                'customer_id': customer_id,
                'amount': round(amount, 2),
                'quantity': int(row['Quantity']),
                'unit_price': float(row['UnitPrice']),
                'stock_code': str(row.get('StockCode', 'ITEM-001')),
                'country': str(row.get('Country', 'United Kingdom'))
            }
        else:
            # Generate synthetic transaction
            customer_segment = random.choice(['high_value', 'regular', 'occasional', 'new'])
            
            # Different patterns for different segments
            if customer_segment == 'high_value':
                amount = random.uniform(150, 500)
                quantity = random.randint(5, 20)
            elif customer_segment == 'regular':
                amount = random.uniform(50, 150)
                quantity = random.randint(2, 8)
            elif customer_segment == 'occasional':
                amount = random.uniform(20, 80)
                quantity = random.randint(1, 4)
            else:  # new
                amount = random.uniform(10, 50)
                quantity = random.randint(1, 3)
            
            return {
                'customer_id': f'CUST-{random.randint(1000, 9999)}',
                'amount': round(amount, 2),
                'quantity': quantity,
                'unit_price': round(amount / quantity, 2),
                'stock_code': f'ITEM-{random.randint(100, 999)}',
                'country': random.choice(['United Kingdom', 'USA', 'Germany', 'France'])
            }
    
    def _predict_purchase_probability(self, features):
        """Make ensemble predictions using the loaded models."""
        if not self.models:
            # Return random probability if models not loaded
            return random.uniform(0.3, 0.9)
        
        try:
            # Make predictions with each model and average
            predictions = []
            for model in self.models:
                pred = model.predict(features.reshape(1, -1), verbose=0)[0][0]
                predictions.append(pred)
            
            return float(np.mean(predictions))
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return 0.5

    async def process_transaction(self, transaction: dict) -> dict:
        """Process a transaction and update metrics."""
        try:
            # Create feature vector for prediction
            # These should match the features used in training
            features = np.zeros(26)  # 26 features as per model_metrics.json
            
            # Basic features
            features[0] = transaction['amount']  # daily_amount
            features[1] = 1  # num_transactions
            features[2] = transaction['quantity']  # total_items
            features[3] = 1  # unique_items
            features[4] = 1 / max(transaction['quantity'], 1)  # basket_diversity
            features[5] = transaction['amount']  # avg_purchase_value
            features[6] = datetime.now().hour  # hour_of_day
            features[7] = 1 if datetime.now().weekday() >= 5 else 0  # is_weekend
            features[8] = 1 if datetime.now().hour < 12 else 0  # morning_shopper
            features[9] = 1 if datetime.now().hour > 17 else 0  # evening_shopper
            features[10] = datetime.now().month  # month
            features[11] = (datetime.now().month - 1) // 3 + 1  # quarter
            
            # Customer history features (simulated for demo)
            customer_history = self.metrics['total_customers'].get(transaction['customer_id'], {
                'days_since_first': random.randint(0, 365),
                'days_since_prev': random.randint(1, 30),
                'active_days': random.randint(1, 50),
                'prev_amount': random.uniform(20, 200)
            })
            
            features[12] = customer_history.get('days_since_first', 30)  # days_since_first
            features[13] = customer_history.get('days_since_prev', 7)  # days_since_prev
            features[14] = customer_history.get('active_days', 5)  # active_days
            features[15] = features[14] / max(features[12], 1)  # engagement_ratio
            features[16] = random.randint(0, 30)  # recency
            features[17] = random.uniform(5, 30)  # days_between_purchases
            features[18] = random.uniform(-0.5, 0.5)  # purchase_acceleration
            features[19] = random.uniform(-0.2, 0.2)  # spending_growth_rate
            features[20] = customer_history.get('prev_amount', 50)  # prev_amount
            features[21] = transaction['amount'] * 0.9  # rolling_avg_amount
            features[22] = transaction['amount'] * 0.85  # rolling_avg_daily_amount
            features[23] = 1 / max(features[17], 1)  # frequency
            features[24] = features[12] / 30  # tenure_months
            features[25] = random.randint(0, 3)  # customer_segment
            
            # Scale features
            if hasattr(self, 'scaler'):
                features_scaled = self.scaler.transform(features.reshape(1, -1))[0]
            else:
                features_scaled = features
            
            # Get prediction
            purchase_probability = self._predict_purchase_probability(features_scaled)
            
            # Add prediction to transaction
            transaction['purchase_probability'] = purchase_probability
            transaction['predicted_action'] = 'likely_purchase' if purchase_probability > 0.5 else 'unlikely_purchase'
            transaction['confidence'] = abs(purchase_probability - 0.5) * 2  # Convert to confidence score
            transaction['prediction_score'] = purchase_probability
            transaction['will_purchase_prediction'] = purchase_probability > 0.5
            transaction['customer_segment'] = int(features[25])
            
            # Save to database if available
            if db_manager.pool:
                try:
                    await db_manager.insert_transaction(transaction)
                    
                    # Update customer profile
                    profile_data = {
                        'total_spent': self.metrics['total_customers'].get(transaction['customer_id'], {}).get('total_spent', 0) + transaction['amount'],
                        'transaction_count': self.metrics['total_customers'].get(transaction['customer_id'], {}).get('transaction_count', 0) + 1,
                        'first_purchase_date': datetime.now(),
                        'last_purchase_date': datetime.now(),
                        'avg_transaction_value': transaction['amount'],
                        'customer_segment': transaction['customer_segment'],
                        'loyalty_score': random.uniform(0.3, 0.9),
                        'churn_risk_score': 1 - purchase_probability
                    }
                    await db_manager.update_customer_profile(transaction['customer_id'], profile_data)
                except Exception as e:
                    logger.error(f"Database error: {str(e)}")
            
            # Update metrics
            self.metrics['total_revenue'] += transaction['amount']
            self.metrics['total_orders'] += 1
            self.metrics['total_customers'][transaction['customer_id']] = customer_history
            
            # Calculate aggregate metrics
            total_time = (datetime.now() - self.metrics['start_time']).total_seconds()
            orders_per_second = self.metrics['total_orders'] / total_time if total_time > 0 else 0
            
            # Update conversion rate as rolling average
            self.metrics['conversion_rate'] = (
                self.metrics['conversion_rate'] * 0.95 + purchase_probability * 100 * 0.05
            )
            
            metrics_update = {
                'revenue': self.metrics['total_revenue'],
                'orders': self.metrics['total_orders'],
                'customers': len(self.metrics['total_customers']),
                'orders_per_second': round(orders_per_second, 2),
                'conversion_rate': round(self.metrics['conversion_rate'], 1)
            }
            
            return {
                'transaction': transaction,
                'metrics': metrics_update
            }
            
        except Exception as e:
            logger.error(f"Error processing transaction: {str(e)}")
            raise

    async def send_metrics(self, websocket: WebSocket):
        """Send current metrics to a specific client."""
        try:
            await websocket.send_json({
                'type': 'metrics',
                'data': {
                    'revenue': self.metrics['total_revenue'],
                    'orders': self.metrics['total_orders'],
                    'customers': len(self.metrics['total_customers']),
                    'conversion_rate': self.metrics['conversion_rate']
                }
            })
        except Exception as e:
            logger.error(f"Error sending metrics: {str(e)}")
            raise

manager = ConnectionManager()

@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    try:
        # Models are loaded on first connection, so we just check status
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models_loaded": len(manager.models) if manager.models_loaded else 0,
            "active_connections": len(manager.active_connections),
            "models_status": "loaded" if manager.models_loaded else "pending"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "E-commerce Analytics API",
        "version": "1.0.0",
        "endpoints": {
            "websocket": "/ws",
            "health": "/health",
            "metrics": "/api/metrics"
        }
    }

@app.get("/api/metrics")
async def get_metrics():
    """Get real-time metrics from database."""
    if not db_manager.pool:
        # Return in-memory metrics if database not available
        return {
            "source": "memory",
            "metrics": {
                "total_revenue": manager.metrics['total_revenue'],
                "total_orders": manager.metrics['total_orders'],
                "active_customers": len(manager.metrics['total_customers']),
                "conversion_rate": manager.metrics['conversion_rate'],
                "active_connections": len(manager.active_connections)
            }
        }
    
    try:
        # Get database metrics
        db_metrics = await db_manager.get_realtime_metrics()
        return {
            "source": "database",
            "metrics": db_metrics,
            "active_connections": len(manager.active_connections)
        }
    except Exception as e:
        logger.error(f"Error fetching metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch metrics")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections and data streaming."""
    await manager.connect(websocket)
    try:
        while True:
            # Generate realistic transaction based on historical patterns
            transaction = manager._generate_realistic_transaction()
            transaction['timestamp'] = datetime.now().timestamp()
            
            # Process transaction and get updates with ML predictions
            updates = await manager.process_transaction(transaction)
            
            # Add additional info for visualization
            updates['transaction']['timestamp'] = transaction['timestamp']
            updates['transaction']['formatted_time'] = datetime.now().strftime('%H:%M:%S')
            
            # Broadcast to all connected clients
            await manager.broadcast(updates)
            
            # Variable delay to simulate real-world patterns
            # Higher traffic during business hours
            hour = datetime.now().hour
            if 9 <= hour <= 17:  # Business hours
                delay = random.uniform(0.5, 2)
            else:
                delay = random.uniform(2, 5)
            
            await asyncio.sleep(delay)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Error in websocket endpoint: {str(e)}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)