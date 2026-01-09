import plotly.graph_objects as go

def create_metric_boxes(metrics_dict, width=600, height=300, box_colors=None):
    """
    Create metric display boxes showing metric names and values (numeric or string).
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with metric names as keys and values (numeric or string) as values
        Example: {'Total Calls': 1250, 'Agent Name': 'cibin'}
    width : int, optional
        Figure width in pixels. Default is 600
    height : int, optional
        Figure height in pixels. Default is 300
    box_colors : list, optional
        List of colors for each box. If None, uses default colors
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
    """
    num_metrics = len(metrics_dict)
    metric_names = list(metrics_dict.keys())
    metric_values = list(metrics_dict.values())
    
    # Default colors if not provided
    if box_colors is None:
        box_colors = ['#3498db', '#e67e22', '#27ae60', '#9b59b6', '#e74c3c'][:num_metrics]
    
    # Create empty figure
    fig = go.Figure()
    
    # Add shapes for boxes
    shapes = []
    for i in range(num_metrics):
        x_start = i / num_metrics
        x_end = (i + 1) / num_metrics
        shapes.append(
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=x_start + 0.01,
                y0=0.1,
                x1=x_end - 0.01,
                y1=0.9,
                fillcolor=box_colors[i],
                opacity=0.15,
                line=dict(color=box_colors[i], width=3)
            )
        )
    
    # Add annotations for metric names and values
    annotations = []
    for i, (name, value) in enumerate(metrics_dict.items()):
        x_center = (i + 0.5) / num_metrics
        
        # Metric name (top)
        annotations.append(
            dict(
                text=f"<b>{name}</b>",
                x=x_center,
                y=0.7,
                xref="paper",
                yref="paper",
                font=dict(size=18, color='#2c3e50'),
                showarrow=False
            )
        )
        
        # Metric value (bottom)
        # Format value based on type
        if isinstance(value, float):
            display_value = f"{value:.2f}"
        elif isinstance(value, int):
            display_value = str(value)
        else:
            display_value = str(value)
        
        annotations.append(
            dict(
                text=f"<b>{display_value}</b>",
                x=x_center,
                y=0.35,
                xref="paper",
                yref="paper",
                font=dict(size=40, color=box_colors[i]),
                showarrow=False
            )
        )
    
    # Update layout
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        width=width,
        height=height,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='white',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='white'
    )
    
    return fig


# Example usage:
metrics = {
    'Total Calls': 1250,
    'Agent Name': 'cibin',
    'Avg Duration': 45.3
}

custom_colors = ['#4ECDC4', '#FF6B6B', '#9b59b6']

fig = create_metric_boxes(metrics, box_colors=custom_colors)
# For Databricks:
fig.show()
