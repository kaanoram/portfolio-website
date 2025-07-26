import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import tensorflow as tf
import joblib
import os
import datetime
import requests
import json
import logging
from transformers import CategoryWeightTransformer
from utils import apply_weights

def setup_logger(reset=False):
    """
    Set up or reset the logger.
    
    Args:
        reset (bool): If True, reset existing handlers before setting up new ones
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(__name__)
    
    # Reset the logger if requested
    if reset and logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
    
    # Set logging level
    logger.setLevel(logging.INFO)
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Generate timestamped log filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_{timestamp}.log"
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized. Log file: {log_file}")
    
    return logger

# Set seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Create module-level logger
logger = logging.getLogger(__name__)

def download_data():
    """Downloads the Online Retail dataset"""
    # The dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Download and save the file if we haven't already
        if not os.path.exists('data/Online_Retail.xlsx'):
            logger.info("Downloading dataset...")
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            
            with open('data/Online_Retail.xlsx', 'wb') as f:
                f.write(response.content)
            logger.info("Dataset downloaded successfully!")
        else:
            logger.info("Using existing dataset file")
        
        # Read the Excel file
        logger.info("Reading the Excel file...")
        # Use xlrd for older Excel files or openpyxl for newer ones
        df = pd.read_excel('data/Online_Retail.xlsx')
        logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        return df
    
    except Exception as e:
        logger.error(f"Critical error during data preparation: {str(e)}")
        logger.error("Please check the data source and try again.")
        raise

def clean_data(df):       
    # Basic data cleaning
    logger.info("Cleaning and preparing data...")
    initial_rows = len(df)
    
    # Drop rows with missing values
    df = df.dropna()
    logger.info(f"Dropped {initial_rows - len(df)} rows with missing values")
    
    # Filter for positive quantities and prices
    qty_filter = df['Quantity'] > 0
    price_filter = df['UnitPrice'] > 0
    df = df[qty_filter & price_filter]
    logger.info(f"Filtered to {len(df)} rows with positive quantities and prices")

    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Extract purchase date
    df['purchase_date'] = pd.to_datetime(df['InvoiceDate'].dt.date)
    
    logger.info(f"Data cleaning complete. Final shape: {df.shape}")
    return df

def add_features(df):
    """
    Add advanced customer behavior and transaction-based features.
    
    Args:
        df (pd.DataFrame): Raw transaction data
        
    Returns:
        pd.DataFrame: Processed daily purchase data with additional features
    """
    logger.info("Adding advanced features...")

    # Add basic features
    df['total_amount'] = df['Quantity'] * df['UnitPrice']
    
    # Aggregate to daily level first
    daily_purchases = df.groupby(['CustomerID', 'purchase_date']).agg({
        'total_amount': 'sum',          # Total spent that day
        'InvoiceNo': 'nunique',         # Number of transactions that day
        'Quantity': 'sum',              # Total items bought that day
        'StockCode': 'nunique',         # Number of unique items bought
        'InvoiceDate': 'min'            # First transaction time of the day
    }).reset_index()

    # Rename aggregated columns
    daily_purchases.rename(columns={
        'total_amount': 'daily_amount',
        'InvoiceNo': 'num_transactions',
        'Quantity': 'total_items',
        'StockCode': 'unique_items',
        'InvoiceDate': 'first_transaction'
    }, inplace=True)

    # Sort by customer and date for proper calculations
    daily_purchases.sort_values(['CustomerID', 'purchase_date'], inplace=True)

    # Calculate days between purchases safely
    daily_purchases['prev_purchase_date'] = daily_purchases.groupby('CustomerID')['purchase_date'].shift(1)
    daily_purchases['days_since_prev'] = (daily_purchases['purchase_date'] - daily_purchases['prev_purchase_date']).dt.days.fillna(0)
    daily_purchases['days_between_purchases'] = (
        daily_purchases['purchase_date'] - daily_purchases['prev_purchase_date']
    ).dt.total_seconds() / (24 * 3600)
    
    # Calculate purchase acceleration more safely
    daily_purchases['prev_days_between'] = daily_purchases.groupby('CustomerID')['days_between_purchases'].shift(1)
    daily_purchases['purchase_acceleration'] = daily_purchases.apply(
        lambda x: (x['prev_days_between'] - x['days_between_purchases']) / (x['prev_days_between'] + 1e-6) 
        if pd.notnull(x['prev_days_between']) else 0, 
        axis=1
    )

    # Add temporal features
    daily_purchases['first_purchase_date'] = daily_purchases.groupby('CustomerID')['purchase_date'].transform('min')
    daily_purchases['days_since_first'] = (
        daily_purchases['purchase_date'] - daily_purchases['first_purchase_date']
    ).dt.total_seconds() / (24 * 3600)

    # Rolling metrics
    daily_purchases['prev_amount'] = daily_purchases.groupby('CustomerID')['daily_amount'].shift(1)
    daily_purchases['rolling_avg_amount'] = daily_purchases.groupby('CustomerID')['daily_amount'].transform(
        lambda x: x.expanding().mean().shift(1)
    )

    # Create target variable
    daily_purchases['next_purchase_date'] = daily_purchases.groupby('CustomerID')['purchase_date'].shift(-1)
    daily_purchases['days_to_next'] = (
        daily_purchases['next_purchase_date'] - daily_purchases['purchase_date']
    ).dt.total_seconds() / (24 * 3600)
    daily_purchases['will_purchase_30d'] = (daily_purchases['days_to_next'] <= 30).astype(np.float32)

    # Add RFM features
    daily_purchases['recency'] = (daily_purchases['purchase_date'].max() - daily_purchases['purchase_date']).dt.days
    daily_purchases['tenure_months'] = ((daily_purchases['purchase_date'] - daily_purchases['first_purchase_date']).dt.days / 30)
    daily_purchases['frequency'] = daily_purchases['num_transactions'] / (daily_purchases['tenure_months'] + 1e-6)
    daily_purchases['avg_purchase_value'] = daily_purchases['daily_amount'] / (daily_purchases['num_transactions'] + 1e-6)

    # Add time-based features
    daily_purchases['hour_of_day'] = daily_purchases['first_transaction'].dt.hour
    daily_purchases['is_weekend'] = daily_purchases['first_transaction'].dt.dayofweek.isin([5, 6]).astype(int)
    daily_purchases['month'] = daily_purchases['purchase_date'].dt.month
    daily_purchases['quarter'] = daily_purchases['purchase_date'].dt.quarter

    # Basket diversity
    daily_purchases['basket_diversity'] = (
        daily_purchases['unique_items'] / (daily_purchases['total_items'] + 1e-6)
    ).clip(0, 1)

    # Time preferences
    daily_purchases['morning_shopper'] = (daily_purchases['hour_of_day'] < 12).astype(int)
    daily_purchases['evening_shopper'] = (daily_purchases['hour_of_day'] > 17).astype(int)

    # Rolling averages with proper window
    daily_purchases['rolling_avg_daily_amount'] = daily_purchases.groupby('CustomerID')['daily_amount'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )

    # Customer engagement metrics
    daily_purchases['active_days'] = daily_purchases.groupby('CustomerID')['purchase_date'].transform('count')
    daily_purchases['engagement_ratio'] = daily_purchases['active_days'] / (daily_purchases['tenure_months'] + 1e-6)

    # Calculate spending growth more safely
    daily_purchases['spending_growth_rate'] = daily_purchases.groupby('CustomerID')['daily_amount'].pct_change().fillna(0)

    # Fill NaN values
    daily_purchases = daily_purchases.fillna(0)

    logger.info(f"Added features successfully. Feature dataframe shape: {daily_purchases.shape}")
    return daily_purchases

def add_advanced_features(df):
    """Add more sophisticated features for deeper analysis"""
    logger.info("Adding advanced behavioral features...")
    
    # Add seasonality patterns
    df['season'] = df['month'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    })
    
    # Add purchase volatility
    df['amount_volatility'] = df.groupby('CustomerID')['daily_amount'].transform(
        lambda x: x.std() / (x.mean() + 1e-6)
    )
    
    # Add time-based momentum features
    df['recent_purchase_momentum'] = df.groupby('CustomerID')['daily_amount'].transform(
        lambda x: x.ewm(span=7, min_periods=1).mean()
    )
    
    # Add basket composition changes
    df['basket_composition_change'] = df.groupby('CustomerID')['unique_items'].pct_change().fillna(0)
    
    logger.info("Advanced features added successfully")
    return df

def analyze_purchase_windows(df):
    """
    Analyze purchase patterns across different time windows.
    
    Args:
        df: DataFrame with purchase data
    """
    logger.info("Analyzing purchase windows...")
    windows = [7, 14, 30, 60, 90]
    stats = {}
    
    for window in windows:
        # Calculate whether purchase happened within each window
        df[f'will_purchase_{window}d'] = (
            df['days_to_next'] <= window
        ).astype(np.float32)
        
        # Calculate positive class ratio
        positive_ratio = df[f'will_purchase_{window}d'].mean()
        
        # Calculate average time between purchases
        avg_time = df['days_between_purchases'].mean()
        
        stats[window] = {
            'positive_ratio': positive_ratio,
            'negative_ratio': 1 - positive_ratio,
            'class_imbalance': min(positive_ratio, 1-positive_ratio) / 
                              max(positive_ratio, 1-positive_ratio)
        }
    
    # Print analysis
    logger.info("\nPurchase Window Analysis:")
    logger.info("-" * 50)
    for window, metrics in stats.items():
        logger.info(f"\n{window}-Day Window:")
        logger.info(f"Positive class ratio: {metrics['positive_ratio']:.2%}")
        logger.info(f"Negative class ratio: {metrics['negative_ratio']:.2%}")
        logger.info(f"Class imbalance ratio: {metrics['class_imbalance']:.2f}")
    
    return stats

def get_feature_columns():
    """
    Returns the list of feature columns used for modeling.
    
    This function provides a single source of truth for features used in both 
    clustering and prediction.
    
    Returns:
        list: Feature column names organized by category
    """
    return [
        # Core transaction metrics
        'daily_amount', 'num_transactions', 'total_items', 'unique_items',
        'basket_diversity', 'avg_purchase_value',
        
        # Time-based patterns
        'hour_of_day', 'is_weekend', 'morning_shopper', 'evening_shopper',
        'month', 'quarter',
        
        # Customer engagement metrics
        'days_since_first', 'days_since_prev', 'active_days',
        'engagement_ratio', 'recency',
        
        # Purchase behavior patterns
        'days_between_purchases', 'purchase_acceleration', 'spending_growth_rate',
        
        # Historical and rolling metrics
        'prev_amount', 'rolling_avg_amount', 'rolling_avg_daily_amount',
        'frequency', 'tenure_months'
    ]

def prepare_xy(features_df, cluster_labels=None):
    """
    Prepares feature matrix X and target vector y for model training.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing all features
        cluster_labels (np.array, optional): Customer segment assignments from clustering
            
    Returns:
        tuple: (X, y) where:
            - X is the feature matrix using columns defined in get_feature_columns()
            - y is the binary target indicating 30-day purchase probability
    """
    logger.info("Preparing feature matrix X and target vector y...")
    X = features_df[get_feature_columns()].copy()
    
    if cluster_labels is not None:
        X['customer_segment'] = cluster_labels
        logger.info("Added cluster labels to feature matrix")
    
    y = features_df['will_purchase_30d']
    
    # Check for any NaN values and handle them
    if X.isnull().any().any():
        logger.warning("NaN values found in feature matrix X. Filling with 0.")
        X = X.fillna(0)
    
    if y.isnull().any():
        logger.warning("NaN values found in target vector y. Filling with 0.")
        y = y.fillna(0)
    
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    return X, y


def perform_enhanced_customer_segmentation(features_df, n_clusters_range=(3, 8), min_cluster_size=50):
    """
    Performs comprehensive customer segmentation with improved cluster selection, validation,
    and imbalance prevention.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing all engineered features
        n_clusters_range (tuple): Range of clusters to test (min, max)
        min_cluster_size (int): Minimum acceptable cluster size
        
    Returns:
        tuple: (KMeans model, cluster labels, cluster profiles, quality_metrics)
    """
    logger.info("Starting enhanced customer segmentation...")
    
    # Add advanced features
    logger.info("Adding advanced behavioral features...")
    features_df = add_advanced_features(features_df)
    logger.info("Advanced features added successfully")
    
    # Define feature groups (keeping your original structure)
    feature_groups = {
        'monetary': {
            'columns': ['daily_amount', 'rolling_avg_amount', 'rolling_avg_daily_amount',
                       'prev_amount', 'amount_volatility', 'spending_growth_rate',
                       'avg_purchase_value'],
            'weight': 1.5
        },
        'temporal': {
            'columns': ['days_between_purchases', 'purchase_acceleration',
                       'days_since_first', 'days_since_prev', 'recency',
                       'recent_purchase_momentum'],
            'weight': 1.2
        },
        'engagement': {
            'columns': ['num_transactions', 'total_items', 'unique_items',
                       'active_days', 'engagement_ratio', 'frequency',
                       'tenure_months'],
            'weight': 1.0
        },
        'behavioral': {
            'columns': ['basket_diversity', 'basket_composition_change',
                       'morning_shopper', 'evening_shopper'],
            'weight': 1.0
        },
        'cyclical': {
            'columns': ['hour_of_day', 'is_weekend', 'month', 'quarter'],
            'weight': 0.8
        }
    }

    # Verify and refine feature groups
    existing_columns = set(features_df.columns)
    refined_feature_groups = {}
    for group_name, group_info in feature_groups.items():
        existing_group_columns = [col for col in group_info['columns'] if col in existing_columns]
        if existing_group_columns:
            refined_feature_groups[group_name] = {
                'columns': existing_group_columns,
                'weight': group_info['weight']
            }
    
    # Handle missing features
    feature_columns = get_feature_columns()
    all_grouped_features = [col for group in refined_feature_groups.values() for col in group['columns']]
    missing_features = [col for col in feature_columns if col in existing_columns and col not in all_grouped_features]
    if missing_features:
        logger.warning(f"Some features are not grouped: {missing_features}")
        refined_feature_groups['additional'] = {'columns': missing_features, 'weight': 1.0}

    # Handle NaN values and outliers
    for group in refined_feature_groups.values():
        for column in group['columns']:
            if column in features_df.columns:
                if features_df[column].dtype in ['int64', 'float64']:
                    # Outlier clipping using IQR
                    Q1 = features_df[column].quantile(0.25)
                    Q3 = features_df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    features_df[column] = features_df[column].clip(lower, upper)
                    # Fill NaN with median
                    features_df[column] = features_df[column].fillna(features_df[column].median())
                elif features_df[column].dtype == 'bool':
                    features_df[column] = features_df[column].fillna(features_df[column].mode()[0])
                else:
                    features_df[column] = features_df[column].fillna(features_df[column].mode()[0])

    # Prepare features
    selected_columns = [col for group in refined_feature_groups.values() for col in group['columns']]
    feature_weights = [group['weight'] for group in refined_feature_groups.values() for _ in group['columns']]
    
    logger.info(f"Selected columns for clustering: {selected_columns}")
    logger.info(f"Feature weights length: {len(feature_weights)}")
    
    # Separate numerical and categorical columns
    numerical_columns = [col for col in selected_columns if features_df[col].dtype in ['int64', 'float64']]
    categorical_columns = [col for col in selected_columns if col not in numerical_columns]
    logger.info(f"Using {len(numerical_columns)} numerical features and {len(categorical_columns)} categorical features")

    num_weights = np.array(feature_weights[:len(numerical_columns)])

    # Preprocessing pipeline
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('weighted_transformer', FunctionTransformer(
            apply_weights, 
            kw_args={'weights': num_weights}
        ))
    ])

    preprocessor_steps = [('num', numeric_transformer, numerical_columns)]
    if categorical_columns:
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')),
            ('weighter', CategoryWeightTransformer(
                feature_groups=refined_feature_groups,
                original_columns=categorical_columns
            ))
        ])
        preprocessor_steps.append(('cat', categorical_transformer, categorical_columns))

    preprocessor = ColumnTransformer(preprocessor_steps)
    
    # Transform data
    logger.info(f"Fitting and transforming data for clustering with {len(selected_columns)} columns...")
    try:
        features_transformed = preprocessor.fit_transform(features_df[selected_columns])
        logger.info(f"Transformed feature shape: {features_transformed.shape}")
    except Exception as e:
        logger.error(f"Error during feature transformation: {str(e)}")
        logger.error(f"Selected columns: {selected_columns}")
        logger.error(f"Feature weights: {feature_weights}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Fallback to simpler preprocessing
        logger.info("Attempting simplified clustering with numerical features only...")
        simple_preprocessor = ColumnTransformer([('num', StandardScaler(), numerical_columns)])
        features_transformed = simple_preprocessor.fit_transform(features_df[numerical_columns])
        logger.info(f"Simplified transformation complete with shape: {features_transformed.shape}")

    # Enhanced cluster optimization with size constraint
    logger.info("Finding optimal number of clusters with enhanced analysis...")
    silhouette_scores = []
    calinski_scores = []
    davies_scores = []
    K = range(n_clusters_range[0], n_clusters_range[1] + 1)
    
    for k in K:
        logger.info(f"Testing {k} clusters...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_transformed)
        
        silhouette_avg = silhouette_score(features_transformed, cluster_labels)
        calinski_avg = calinski_harabasz_score(features_transformed, cluster_labels)
        davies_avg = davies_bouldin_score(features_transformed, cluster_labels)
        
        cluster_sizes = pd.Series(cluster_labels).value_counts()
        min_size = cluster_sizes.min()
        
        # Adjust score based on minimum cluster size
        size_penalty = max(0, (min_cluster_size - min_size) / min_cluster_size) if min_size < min_cluster_size else 0
        adjusted_silhouette = silhouette_avg * (1 - size_penalty)
        
        silhouette_scores.append(adjusted_silhouette)
        calinski_scores.append(calinski_avg)
        davies_scores.append(davies_avg)
        
        logger.info(f"Silhouette score (adjusted) for {k} clusters: {adjusted_silhouette:.4f}")
        logger.info(f"Raw silhouette: {silhouette_avg:.4f}, Calinski-Harabasz: {calinski_avg:.4f}, "
                   f"Davies-Bouldin: {davies_avg:.4f}, Min cluster size: {min_size}")

    # Select optimal clusters using combined metrics
    silhouette_norm = (silhouette_scores - min(silhouette_scores)) / (max(silhouette_scores) - min(silhouette_scores) + 1e-6)
    calinski_norm = (calinski_scores - min(calinski_scores)) / (max(calinski_scores) - min(calinski_scores) + 1e-6)
    davies_norm = (max(davies_scores) - davies_scores) / (max(davies_scores) - min(davies_scores) + 1e-6)
    combined_scores = 0.4 * silhouette_norm + 0.3 * calinski_norm + 0.3 * davies_norm
    
    optimal_clusters = K[np.argmax(combined_scores)]
    logger.info(f"Optimal number of clusters: {optimal_clusters}")

    # Final clustering
    logger.info(f"Performing final clustering with {optimal_clusters} clusters...")
    final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10, max_iter=300)
    cluster_labels = final_kmeans.fit_predict(features_transformed)
    features_df['customer_segment'] = cluster_labels

    # Generate profiles and quality metrics
    logger.info("Generating cluster profiles...")
    cluster_profiles = analyze_cluster_profiles(features_df, selected_columns, cluster_labels)
    
    quality_metrics = {
        'silhouette_score': silhouette_score(features_transformed, cluster_labels),
        'calinski_harabasz_score': calinski_harabasz_score(features_transformed, cluster_labels),
        'davies_bouldin_score': davies_bouldin_score(features_transformed, cluster_labels),
        'n_clusters': optimal_clusters,
        'min_cluster_size': pd.Series(cluster_labels).value_counts().min()
    }
    logger.info(f"Final clustering quality metrics: {quality_metrics}")

    # Save components
    logger.info("Saving clustering model and components...")
    os.makedirs('models', exist_ok=True)
    try:
        joblib.dump({
            'kmeans': final_kmeans,
            'optimal_clusters': optimal_clusters,
            'feature_groups': refined_feature_groups,
            'quality_metrics': quality_metrics,
            'preprocessor': preprocessor
        }, 'models/enhanced_clustering_model.joblib')
    except Exception as e:
        logger.error(f"Error saving clustering model: {str(e)}")

    return final_kmeans, cluster_labels, cluster_profiles, quality_metrics
    

def analyze_cluster_profiles(df, feature_columns, cluster_labels):
    """
    Analyzes the characteristics of each cluster to create meaningful profiles.
    """
    logger.info("Analyzing cluster profiles...")
    profiles = {}
    
    unique_clusters = np.unique(cluster_labels)
    logger.info(f"Found {len(unique_clusters)} clusters")
    
    for cluster in range(len(unique_clusters)):
        cluster_data = df[df['customer_segment'] == cluster]
        
        profile = {
            'size': len(cluster_data),
            'size_percentage': len(cluster_data) / len(df) * 100,
            'monetary_metrics': {
                'avg_daily_amount': cluster_data['daily_amount'].mean(),
                'amount_volatility': cluster_data['amount_volatility'].mean() if 'amount_volatility' in cluster_data.columns else None
            },
            'behavioral_metrics': {
                'avg_basket_diversity': cluster_data['basket_diversity'].mean(),
                'engagement_ratio': cluster_data['engagement_ratio'].mean()
            },
            'temporal_patterns': {
                'avg_purchase_frequency': 1 / (cluster_data['days_between_purchases'].mean() + 1e-6),
                'momentum': cluster_data['recent_purchase_momentum'].mean() if 'recent_purchase_momentum' in cluster_data.columns else None
            }
        }
        
        profiles[f'cluster_{cluster}'] = profile
        logger.info(f"Cluster {cluster}: {profile['size']} customers ({profile['size_percentage']:.1f}%)")
    
    return profiles

def create_wide_and_deep_model(input_shape):
    """
    Creates a Wide & Deep architecture with matching dimensions for attention mechanism.
    
    The wide path helps memorize specific feature combinations, while the deep path
    helps discover new feature interactions. Both paths are scaled to the same
    final dimension before applying attention.
    
    Args:
        input_shape: Number of input features
        
    Returns:
        tf.keras.Model: Compiled model with wide and deep paths
    """
    logger.info(f"Creating Wide & Deep model with input shape: {input_shape}")
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=(input_shape,))
    
    # Wide path with feature crossing
    wide = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.02))(inputs)
    wide = tf.keras.layers.BatchNormalization()(wide)
    wide = tf.keras.layers.LeakyReLU(alpha=0.2)(wide)
    
    # Deep path with residual connections
    deep = inputs
    for units in [256, 128, 256]:  # Ensure final dimension matches wide path
        shortcut = deep
        deep = tf.keras.layers.Dense(units, kernel_regularizer=tf.keras.regularizers.l2(0.02))(deep)
        deep = tf.keras.layers.BatchNormalization()(deep)
        deep = tf.keras.layers.LeakyReLU(alpha=0.2)(deep)
        deep = tf.keras.layers.Dropout(0.3)(deep)
        
        # Add residual connection if shapes match
        if shortcut.shape[-1] == units:
            deep = tf.keras.layers.Add()([deep, shortcut])
    
    # Reshape tensors for attention - ensure same dimensions
    # Add one more dimension for attention to work properly
    wide = tf.keras.layers.Reshape((1, 256))(wide)
    deep = tf.keras.layers.Reshape((1, 256))(deep)
    
    # Apply attention mechanism
    attention_output = tf.keras.layers.Attention()([wide, deep])
    attention_output = tf.keras.layers.Flatten()(attention_output)
    
    # Combine paths
    combined = tf.keras.layers.Concatenate()([
        tf.keras.layers.Flatten()(wide),
        tf.keras.layers.Flatten()(deep),
        attention_output
    ])
    
    # Final dense layers
    combined = tf.keras.layers.Dense(128, activation='relu')(combined)
    combined = tf.keras.layers.Dropout(0.2)(combined)
    
    # Output layer
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(combined)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def focal_loss(gamma=2., alpha=.25):
    """
    Focal loss for addressing class imbalance.
    
    Args:
        gamma: Focusing parameter that controls how much to focus on hard examples
        alpha: Weighting factor for the rare class
        
    Returns:
        Focal loss function
    """
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.reduce_mean(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1 + 1e-7)) - \
               tf.reduce_mean((1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0 + 1e-7))
    return focal_loss_fixed

def create_ensemble_predictions(models, X_test):
    """
    Create ensemble predictions by averaging predictions from all models.
    
    Args:
        models: List of trained models
        X_test: Test data to generate predictions for
        
    Returns:
        numpy array of averaged predictions
    """
    logger.info(f"Creating ensemble predictions with {len(models)} models...")
    predictions = np.zeros((len(X_test),))
    for i, model in enumerate(models):
        logger.info(f"Generating predictions from model {i+1}/{len(models)}...")
        predictions += model.predict(X_test, verbose=0).flatten()
    return predictions / len(models)

def train_and_save_model(logger = None):
    """Train an ensemble of models using k-fold cross validation and save the results.
    
    This function handles the complete training pipeline:
    1. Data preparation and preprocessing
    2. K-fold cross validation training
    3. Ensemble model creation
    4. Model evaluation and metrics logging
    """
    # Use the provided logger or get the default one
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Starting model training pipeline...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    try:
        # Get the prepared data
        logger.info("Loading and preparing data...")
        df = download_data()
        cleaned_data = clean_data(df)
        features = add_features(cleaned_data)
        stats = analyze_purchase_windows(features)

        # Perform clustering
        logger.info("Performing customer segmentation...")
        kmeans, cluster_labels, cluster_profiles, quality_metrics = perform_enhanced_customer_segmentation(
            features,
            n_clusters_range=(3, 8),
            min_cluster_size=50
        )
        logger.info(f"Clustering quality metrics: {quality_metrics}")

        X, y = prepare_xy(features, cluster_labels)
        
        # Create and fit the scaler
        logger.info("Creating and fitting scaler...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, 'models/scaler.joblib')

        y_numpy = y.to_numpy()
        
        # Create train/test split for final evaluation
        logger.info("Creating train/test split...")
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_scaled, y_numpy, test_size=0.2, random_state=42, stratify=y_numpy
        )
        
        # Initialize k-fold cross validation
        logger.info("\nInitializing 5-fold cross validation training...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Lists to store models and their performances
        models = []
        fold_histories = []
        fold_scores = []
        
        # Calculate class weights for imbalanced data
        logger.info("Calculating class weights for imbalanced data...")
        n_pos = np.sum(y_train_full)
        n_neg = len(y_train_full) - n_pos
        total = n_pos + n_neg
        class_weight = {
            0: total / (2 * n_neg),  # Weight for negative class
            1: total / (2 * n_pos)   # Weight for positive class
        }
        logger.info(f"Class weights: {class_weight}")
        
        # Train models on each fold
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
            logger.info(f"\nTraining fold {fold + 1}/5")
            
            # Split data for this fold
            X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
            y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]
            
            # Create model for this fold
            model = create_wide_and_deep_model(X_train.shape[1])
            
            # Configure learning rate schedule
            initial_learning_rate = 0.001
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=1000,
                decay_rate=0.9,
                staircase=True
            )
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                loss=focal_loss(),
                metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            
            # Setup callbacks for this fold
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    f'models/model_fold_{fold}.h5',
                    save_best_only=True,
                    monitor='val_loss'
                )
            ]
            
            # Train the model
            try:
                logger.info(f"Training model for fold {fold + 1}...")
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=32,
                    callbacks=callbacks,
                    class_weight=class_weight,
                    verbose=1
                )
                
                # Store model and its performance
                models.append(model)
                fold_histories.append(history.history)
                fold_scores.append(model.evaluate(X_val, y_val, verbose=0))
                logger.info(f"Fold {fold + 1} training complete")
                
            except Exception as e:
                logger.error(f"Error during training fold {fold + 1}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Continue with the next fold instead of stopping everything
                continue
        
        # Save models
        logger.info("\nSaving trained models...")
        for i, model in enumerate(models):
            try:
                model.save(f'models/model_fold_{i}.keras')
                logger.info(f"Model fold {i} saved successfully")
            except Exception as e:
                logger.error(f"Error saving model for fold {i}: {str(e)}")
        
        # Create ensemble predictions on test set
        if len(models) > 0:
            logger.info("\nGenerating ensemble predictions...")
            ensemble_predictions = create_ensemble_predictions(models, X_test)
            
            # Find optimal threshold using validation data
            logger.info("Finding optimal threshold...")
            thresholds = np.arange(0.1, 0.9, 0.05)
            best_f1 = 0
            best_threshold = 0.5
            
            for threshold in thresholds:
                y_pred = (ensemble_predictions > threshold).astype(int)
                f1 = f1_score(y_test, y_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            logger.info(f"Optimal threshold: {best_threshold}, F1 score: {best_f1:.4f}")
            
            # Generate final predictions using best threshold
            final_predictions = (ensemble_predictions > best_threshold).astype(int)

            # Get detailed metrics for the ensemble
            ensemble_accuracy = accuracy_score(y_test, final_predictions)
            ensemble_precision = precision_score(y_test, final_predictions)
            ensemble_recall = recall_score(y_test, final_predictions)
            ensemble_f1 = f1_score(y_test, final_predictions)
            ensemble_auc = roc_auc_score(y_test, ensemble_predictions)
            
            # Calculate final metrics
            cm = confusion_matrix(y_test, final_predictions)
            classification_rep = classification_report(y_test, final_predictions)
            
            logger.info("\nEnsemble Model Final Results:")
            logger.info("\nConfusion Matrix:")
            logger.info(f"\n{cm}")
            logger.info("\nClassification Report:")
            logger.info(f"\n{classification_rep}")
            
            # Save metrics and parameters
            logger.info("\nSaving metrics and parameters...")
            
            # Save metrics and parameters
            metrics = {
                'ensemble_metrics': {
                    'confusion_matrix': cm.tolist(),
                    'classification_report': classification_rep,
                    'best_threshold': float(best_threshold),
                    'accuracy': float(ensemble_accuracy),
                    'precision': float(ensemble_precision),
                    'recall': float(ensemble_recall),
                    'f1_score': float(ensemble_f1),
                    'auc': float(ensemble_auc)
                },
                'fold_metrics': {
                    'fold_scores': [score.tolist() if isinstance(score, np.ndarray) else score for score in fold_scores],
                    'fold_mean': np.mean(fold_scores, axis=0).tolist() if len(fold_scores) > 0 else [],
                    'fold_std': np.std(fold_scores, axis=0).tolist() if len(fold_scores) > 0 else []
                },
                'training_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_size': int(len(X)),
                'feature_names': list(X.columns) if hasattr(X, 'columns') else [f"feature_{i}" for i in range(X.shape[1])],
                'positive_class_ratio': float(np.mean(y)),
                'model_parameters': {
                    'n_folds': 5,
                    'batch_size': 32,
                    'initial_learning_rate': float(initial_learning_rate),
                    'class_weights': {str(k): float(v) for k, v in class_weight.items()}
                }
            }
            
            with open('models/model_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
            
            logger.info("\nTraining complete! Models and metrics have been saved.")

            # Print a comprehensive performance report
            logger.info("\nEnsemble Model Performance:")
            logger.info(f"Accuracy: {ensemble_accuracy:.3f}")
            logger.info(f"Precision: {ensemble_precision:.3f}")
            logger.info(f"Recall: {ensemble_recall:.3f}")
            logger.info(f"F1 Score: {ensemble_f1:.3f}")
            logger.info(f"AUC-ROC: {ensemble_auc:.3f}")

            # Compare with average individual model performance
            if len(fold_scores) > 0:
                fold_means = np.mean(fold_scores, axis=0)
                logger.info("\nAverage Individual Model Performance:")
                logger.info(f"Loss: {fold_means[0]:.3f}")
                logger.info(f"Accuracy: {fold_means[1]:.3f}")
                logger.info(f"AUC: {fold_means[2]:.3f}")
                logger.info(f"Precision: {fold_means[3]:.3f}")
                logger.info(f"Recall: {fold_means[4]:.3f}")
                
            return models, scaler, metrics
        else:
            logger.error("No models were successfully trained. Cannot create ensemble predictions.")
            return None, scaler, None
    except Exception as e:
        logger.error(f"Critical error in train_and_save_model: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None
    
# When you want to use the model:
if __name__ == "__main__":
    try:
        # Set up a fresh logger
        logger = setup_logger(reset=True)
        
        # Import traceback at the top level
        import traceback
        
        # Train the ensemble and get the models
        logger.info("Starting model training process...")
        models, scaler, metrics = train_and_save_model()
        
        if models is not None:
            logger.info("Training process completed successfully")
        else:
            logger.warning("Training process completed but no models were returned")
            
    except Exception as e:
        logger.error(f"Training process failed: {str(e)}")
        # Print traceback for debugging
        traceback.print_exc()