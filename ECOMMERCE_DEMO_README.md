# E-Commerce Analytics Demo

This project demonstrates a real-time e-commerce analytics dashboard with machine learning capabilities, showcasing advanced data processing and predictive modeling skills.

## Overview

The demo features a sophisticated analytics system that processes transaction data in real-time, performs customer segmentation, and predicts future purchase behavior with up to 88% accuracy.

### Key Features

- **Real-Time Data Streaming**: WebSocket connection for live transaction updates
- **Advanced ML Models**: Ensemble of neural networks with Wide & Deep architecture
- **Customer Segmentation**: K-means clustering with automated optimization
- **Predictive Analytics**: 30-day purchase prediction with high accuracy
- **Interactive Dashboard**: Real-time visualizations and model performance metrics

## Technical Architecture

### Frontend (React + Vite)
- Real-time WebSocket integration via custom `useAnalytics` hook
- Responsive dashboard with Tailwind CSS
- Components:
  - `TransactionStream`: Live transaction feed
  - `MetricsPanel`: Real-time KPI tracking
  - `ModelMetrics`: ML model performance visualization
  - `Dashboard`: Main analytics interface

### Backend (Python + FastAPI)
- **FastAPI Server** (`server.py`): WebSocket endpoint for real-time data streaming
- **ML Pipeline** (`train_model.py`): 
  - Advanced feature engineering (RFM, behavioral patterns, time-series features)
  - Wide & Deep neural network architecture
  - K-fold cross-validation with ensemble predictions
- **Data Processing** (`analytics_pipeline.py`): Apache Beam pipeline for scalable processing
- **Pre-trained Models**: Stored in `models/` directory for instant predictions

### Machine Learning Features

#### Model Architecture Options:
1. **Wide & Deep Model** (Default):
   - Wide path: Memorizes feature combinations
   - Deep path: Discovers new feature interactions
   - Attention mechanism for optimal path weighting

2. **Advanced Model** (88%+ accuracy):
   - Residual blocks with skip connections
   - Swish activation functions
   - AdamW optimizer with weight decay
   - Focal loss for imbalanced data

#### Feature Engineering:
- **Base Features**: RFM metrics, purchase patterns, basket analysis
- **Advanced Features** (optional):
  - Interaction features (value×frequency, recency×monetary)
  - Statistical aggregations (rolling means, EWMA, z-scores)
  - Behavioral patterns (loyalty score, price sensitivity, purchase entropy)
  - Customer lifecycle stages

## Running the Demo

### Quick Start
```bash
# Windows
./run-demo.bat

# Linux/Mac
./run-demo.sh
```

### Manual Setup

1. **Start the Backend**:
   ```bash
   cd src/backend
   pip install -r requirements.txt
   python server.py
   ```

2. **Start the Frontend**:
   ```bash
   npm install
   npm run dev
   ```

3. **Access the Dashboard**:
   Navigate to `http://localhost:5173` and click on the E-commerce Analytics project

### Training Custom Models

```bash
cd src/backend

# Train with default settings
python train_model.py

# Train with advanced model architecture
python train_model.py --model-type advanced

# Train with advanced feature engineering
python train_model.py --advanced-features

# Train with both advanced features and model
python train_model.py --model-type advanced --advanced-features
```

## Model Performance

- **Accuracy**: Up to 88% for 30-day purchase prediction
- **AUC-ROC**: 0.85-0.90 depending on configuration
- **Real-time Inference**: <50ms per prediction
- **Scalability**: Handles 1000+ transactions/second

## Data Source

Uses the UCI Online Retail dataset, which contains:
- 500,000+ transactions
- 4,000+ unique customers
- 1 year of purchase history

## Technologies Used

- **Frontend**: React, Vite, Tailwind CSS, Recharts
- **Backend**: Python, FastAPI, TensorFlow/Keras, scikit-learn
- **Data Processing**: Pandas, NumPy, Apache Beam
- **ML Tools**: TensorFlow 2.x, scikit-learn, joblib
- **Real-time**: WebSockets, asyncio

## Project Structure

```
src/
├── backend/
│   ├── server.py          # FastAPI WebSocket server
│   ├── train_model.py     # ML training pipeline
│   ├── analytics_pipeline.py # Data processing
│   ├── models/            # Pre-trained models
│   └── data/              # Dataset
└── components/
    └── projects/
        └── ecommerce/
            ├── index.jsx           # Main dashboard
            └── components/         # Dashboard components
```

## Future Enhancements

- Add more sophisticated time-series models (LSTM, Transformer)
- Implement A/B testing for model comparison
- Add explainable AI features (SHAP values)
- Integrate with cloud services for production deployment