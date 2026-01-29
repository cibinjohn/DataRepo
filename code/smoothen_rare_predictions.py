import pandas as pd
import numpy as np
from collections import Counter

def smooth_predictions(df, token_col, pred_col, target_labels='all', 
                      neighbor_proximity=3, min_freq=3):
    """
    Apply smoothening to NER predictions based on frequency and neighborhood context.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with tokens and predictions
    token_col : str
        Name of the token column
    pred_col : str
        Name of the prediction column
    target_labels : str or list
        'all' to consider all labels, or a list of specific labels to smooth
    neighbor_proximity : int
        Number of tokens to look before and after for context
    min_freq : int
        Minimum frequency threshold for a consecutive sequence to be kept
    
    Returns:
    --------
    pandas.Series
        Smoothened prediction column
    """
    # Create a copy of the prediction column to avoid modifying the original
    smoothed_preds = df[pred_col].copy()
    
    # Determine which labels to process
    if target_labels == 'all':
        labels_to_process = df[pred_col].unique()
    elif isinstance(target_labels, str):
        labels_to_process = [target_labels]
    else:
        labels_to_process = target_labels
    
    # Process each target label
    for label in labels_to_process:
        # Find all positions where this label occurs
        label_positions = np.where(smoothed_preds == label)[0]
        
        if len(label_positions) == 0:
            continue
        
        # Group consecutive positions
        consecutive_groups = []
        current_group = [label_positions[0]]
        
        for i in range(1, len(label_positions)):
            if label_positions[i] == label_positions[i-1] + 1:
                current_group.append(label_positions[i])
            else:
                consecutive_groups.append(current_group)
                current_group = [label_positions[i]]
        consecutive_groups.append(current_group)
        
        # Process each consecutive group
        for group in consecutive_groups:
            group_size = len(group)
            
            # Only smooth if the group size is less than min_freq
            if group_size < min_freq:
                # Get neighbor predictions (before and after)
                start_idx = group[0]
                end_idx = group[-1]
                
                # Collect neighbor predictions
                neighbor_preds = []
                
                # Get predictions before
                before_start = max(0, start_idx - neighbor_proximity)
                before_preds = smoothed_preds.iloc[before_start:start_idx].tolist()
                neighbor_preds.extend([p for p in before_preds if p != label])
                
                # Get predictions after
                after_end = min(len(smoothed_preds), end_idx + 1 + neighbor_proximity)
                after_preds = smoothed_preds.iloc[end_idx + 1:after_end].tolist()
                neighbor_preds.extend([p for p in after_preds if p != label])
                
                # Find the most frequent alternative prediction
                if neighbor_preds:
                    pred_counts = Counter(neighbor_preds)
                    most_common_pred = pred_counts.most_common(1)[0][0]
                    
                    # Replace the current label with the most common neighbor
                    for idx in group:
                        smoothed_preds.iloc[idx] = most_common_pred
    
    return smoothed_preds


# Example usage:
if __name__ == "__main__":
    # Sample data
    data = {
        'token': ['Hello', 'I', 'need', 'help', 'with', 'Sales', 'issue', 
                  'regarding', 'Resolution', 'Resolution', 'Resolution', 
                  'of', 'my', 'problem'],
        'prediction': ['Other', 'Other', 'Other', 'Resolution', 'Resolution', 
                      'Sales', 'Sales', 'Other', 'Resolution', 'Resolution', 
                      'Resolution', 'Resolution', 'Resolution', 'Other']
    }
    
    df = pd.DataFrame(data)
    
    print("Original predictions:")
    print(df)
    print("\n")
    
    # Apply smoothening
    smoothed = smooth_predictions(
        df=df,
        token_col='token',
        pred_col='prediction',
        target_labels='Sales',
        neighbor_proximity=3,
        min_freq=3
    )
    
    df['smoothed_prediction'] = smoothed
    
    print("After smoothening:")
    print(df)
