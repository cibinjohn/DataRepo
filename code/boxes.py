import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_metric_boxes(metrics_dict, width=600, height=300, box_colors=None):
    """
    Create metric display boxes showing metric names and values.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with metric names as keys and values as values
        Example: {'Total Calls': 1250, 'Avg Duration': 45.3}
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
    
    # Create subplots
    fig = make_subplots(
        rows=1, 
        cols=num_metrics,
        specs=[[{'type': 'indicator'}] * num_metrics],
        horizontal_spacing=0.1
    )
    
    # Add each metric as an indicator
    for i, (name, value) in enumerate(metrics_dict.items()):
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=value,
                title={'text': f"<b>{name}</b>", 'font': {'size': 18}},
                number={'font': {'size': 40}, 'valueformat': '.2f' if isinstance(value, float) else 'd'},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=1, col=i+1
        )
    
    # Update layout
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    # Add colored backgrounds to each subplot
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
                y0=0.05,
                x1=x_end - 0.01,
                y1=0.95,
                fillcolor=box_colors[i],
                opacity=0.2,
                line=dict(color=box_colors[i], width=2),
                layer="below"
            )
        )
    
    fig.update_layout(shapes=shapes)
    
    return fig


# Example usage:
# metrics = {
#     'Total Calls': 1250,
#     'Avg Duration': 45.3
# }
# 
# custom_colors = ['#4ECDC4', '#FF6B6B']
# 
# fig = create_metric_boxes(metrics, box_colors=custom_colors)
# # For Databricks:
# displayHTML(fig.to_html())
