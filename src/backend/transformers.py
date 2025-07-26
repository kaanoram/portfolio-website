# transformers.py
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

class CategoryWeightTransformer(BaseEstimator, TransformerMixin):
    """
    Applies feature group weights to encoded categorical variables.
    
    This transformer handles the expanded dimensionality that results
    from one-hot encoding categorical variables by maintaining appropriate
    weights across the encoded features.
    
    Parameters:
        feature_groups (dict): Dictionary mapping feature groups to their properties
        original_columns (list): List of original categorical column names
    
    Example:
        >>> transformer = CategoryWeightTransformer(
        ...     feature_groups={'temporal': {'weight': 1.2}},
        ...     original_columns=['season']
        ... )
        >>> X_transformed = transformer.fit_transform(X)
    """
    def __init__(self, feature_groups, original_columns):
        self.feature_groups = feature_groups
        self.original_columns = original_columns
        
    def fit(self, X, y=None):
        """
        Fit the transformer (no-op as this transformer is stateless).
        
        Returns:
            self: Returns the transformer instance
        """
        # Store input shape for later verification
        self.input_shape_ = X.shape
        logger.info(f"CategoryWeightTransformer fit with input shape: {self.input_shape_}")
        
        # Pre-compute weights during fit to avoid serialization issues with lambda functions
        self._compute_weights()
        
        return self
    
    def _compute_weights(self):
        """Pre-compute weights to avoid serialization issues later"""
        try:
            # Log info about columns and their weights
            self.group_weights = {}
            for col in self.original_columns:
                group_name = self.get_feature_group(col)
                if group_name in self.feature_groups:
                    weight = self.feature_groups[group_name]['weight']
                    self.group_weights[col] = weight
                    logger.info(f"Column {col} belongs to group {group_name} with weight {weight}")
                    
        except Exception as e:
            logger.error(f"Error computing weights: {str(e)}")
            # Default all weights to 1.0 if there's an error
            self.group_weights = {col: 1.0 for col in self.original_columns}
        
    def transform(self, X):
        """
        Apply the weight transformation to the encoded categorical features.
        
        Parameters:
            X (array-like): The encoded categorical features
            
        Returns:
            array-like: The weighted features
        """
        try:
            # Log shape information for debugging
            logger.info(f"CategoryWeightTransformer transform input shape: {X.shape}")
            
            # Create weights array that matches X's shape
            weights = np.ones(X.shape[1])
            
            # If X is from one-hot encoding, the number of columns will usually be greater than
            # the number of original columns due to expansion
            features_per_col = X.shape[1] // max(len(self.original_columns), 1)
            logger.info(f"Estimated {features_per_col} encoded features per original column")
            
            col_index = 0
            for col in self.original_columns:
                weight = self.group_weights.get(col, 1.0)
                
                # Apply weight to all features derived from this column
                for i in range(features_per_col):
                    if col_index + i < len(weights):
                        weights[col_index + i] = weight
                
                col_index += features_per_col
            
            # Safely reshape weights to allow broadcasting
            weights_reshaped = weights.reshape(1, -1)
            
            # Check if weights shape matches X's columns
            if weights_reshaped.shape[1] != X.shape[1]:
                logger.warning(f"Weight shape {weights_reshaped.shape} doesn't match X columns {X.shape[1]}, adjusting...")
                # If there's a mismatch, create a properly sized weight array
                fixed_weights = np.ones((1, X.shape[1]))
                # Copy as many weights as we can
                fixed_weights[0, :min(weights_reshaped.shape[1], X.shape[1])] = weights_reshaped[0, :min(weights_reshaped.shape[1], X.shape[1])]
                weights_reshaped = fixed_weights
            
            logger.info(f"Applying weights with shape {weights_reshaped.shape} to X with shape {X.shape}")
            
            # Return weighted features
            return X * weights_reshaped
            
        except Exception as e:
            logger.error(f"Error in CategoryWeightTransformer.transform: {str(e)}")
            # Return unweighted features as fallback
            return X
            
    def get_feature_group(self, column):
        """
        Determines which feature group a column belongs to.
        
        Args:
            column (str): The name of the column/feature
            
        Returns:
            str: The name of the group containing the column
            
        Raises:
            ValueError: If the column is not found in any group
        """
        for group_name, group_info in self.feature_groups.items():
            if column in group_info['columns']:
                return group_name
        
        # If not found, log warning and return a default group
        logger.warning(f"Column '{column}' not found in any feature group, using default weight")
        return next(iter(self.feature_groups))  # Return the first group as default