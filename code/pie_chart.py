### plotly
import pandas as pd
import plotly.graph_objects as go

def create_prediction_pie_chart(df, color_dict, width=800, height=600):
    """
    Create an interactive pie chart showing the distribution of predictions using Plotly.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'prediction' and 'timetaken' columns
    color_dict : dict
        Dictionary mapping prediction values to color codes
    width : int, optional
        Figure width in pixels. Default is 800
    height : int, optional
        Figure height in pixels. Default is 600
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
    """
    # Count occurrences of each prediction
    prediction_counts = df['prediction'].value_counts()
    
    # Get colors in the same order as the counts
    colors = [color_dict.get(pred, '#808080') for pred in prediction_counts.index]
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=prediction_counts.index,
        values=prediction_counts.values,
        marker=dict(colors=colors),
        textposition='inside',
        textinfo='percent',
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Distribution of Predictions',
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=16, weight='bold')
        },
        showlegend=True,
        legend=dict(
            title=dict(text='Predictions'),
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02
        ),
        width=width,
        height=height
    )
    
    return fig


# Example usage:
# df = pd.DataFrame({
#     'prediction': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'C'],
#     'timetaken': [1.2, 2.3, 1.5, 3.1, 2.8, 1.1, 3.5, 3.2]
# })
# 
# color_dict = {
#     'A': '#FF6B6B',
#     'B': '#4ECDC4',
#     'C': '#45B7D1'
# }
# 
# fig = create_prediction_pie_chart(df, color_dict)
# fig.show()



#### Seaborn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_prediction_pie_chart(df, color_dict, figsize=(10, 8)):
    """
    Create a pie chart showing the distribution of predictions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'prediction' and 'timetaken' columns
    color_dict : dict
        Dictionary mapping prediction values to color codes
    figsize : tuple, optional
        Figure size (width, height). Default is (10, 8)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Count occurrences of each prediction
    prediction_counts = df['prediction'].value_counts()
    
    # Get colors in the same order as the counts
    colors = [color_dict.get(pred, '#gray') for pred in prediction_counts.index]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        prediction_counts.values,
        labels=prediction_counts.index,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 10}
    )
    
    # Enhance the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # Add legend
    ax.legend(
        wedges,
        [f'{pred} (n={count})' for pred, count in prediction_counts.items()],
        title="Predictions",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    ax.set_title('Distribution of Predictions', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig, ax


# Example usage:
# df = pd.DataFrame({
#     'prediction': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'C'],
#     'timetaken': [1.2, 2.3, 1.5, 3.1, 2.8, 1.1, 3.5, 3.2]
# })
# 
# color_dict = {
#     'A': '#FF6B6B',
#     'B': '#4ECDC4',
#     'C': '#45B7D1'
# }
# 
# fig, ax = create_prediction_pie_chart(df, color_dict)
# plt.show()
