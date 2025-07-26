def get_feature_group(column, feature_groups):
    """
    Determines which feature group a column belongs to.
    
    This function searches through our feature groups dictionary to find which
    group contains the given column name. This is useful when we need to 
    apply the correct weight to categorical features during preprocessing.
    
    Args:
        column (str): The name of the column/feature
        feature_groups (dict): Dictionary containing feature group definitions
        
    Returns:
        str: The name of the group containing the column
        
    Raises:
        ValueError: If the column is not found in any group
    """
    for group_name, group_info in feature_groups.items():
        if column in group_info['columns']:
            return group_name
    raise ValueError(f"Column '{column}' not found in any feature group")

def apply_weights(X, weights):
    """
    Apply weights to features. This is a named function that can be pickled,
    unlike a lambda function.
    
    Args:
        X: The feature array
        weights: The weights to apply
        
    Returns:
        The weighted feature array
    """
    # Reshape weights to ensure proper broadcasting
    weights_reshaped = weights.reshape(1, -1)
    return X * weights_reshaped