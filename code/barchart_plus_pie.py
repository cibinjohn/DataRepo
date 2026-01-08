import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_combined_charts(df_predictions, df_speaker, color_dict_predictions, color_dict_speaker=None, width=1400, height=600):
    """
    Create side-by-side pie chart (predictions) and bar chart (speaker talktime) using Plotly.
    
    Parameters:
    -----------
    df_predictions : pandas.DataFrame
        DataFrame with 'prediction' and 'timetaken' columns
    df_speaker : pandas.DataFrame
        DataFrame with 'speaker' and 'talktime' columns
    color_dict_predictions : dict
        Dictionary mapping prediction values to color codes
    color_dict_speaker : dict, optional
        Dictionary mapping speaker values to color codes
    width : int, optional
        Figure width in pixels. Default is 1400
    height : int, optional
        Figure height in pixels. Default is 600
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
    """
    # Create subplots: 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribution of Predictions', 'Speaker Talk Time'),
        specs=[[{'type': 'pie'}, {'type': 'bar'}]],
        column_widths=[0.5, 0.5]
    )
    
    # === PIE CHART (Left) ===
    prediction_counts = df_predictions['prediction'].value_counts()
    colors_pie = [color_dict_predictions.get(pred, '#808080') for pred in prediction_counts.index.tolist()]
    
    fig.add_trace(
        go.Pie(
            labels=prediction_counts.index.tolist(),
            values=prediction_counts.values.tolist(),
            marker=dict(colors=colors_pie),
            textposition='inside',
            textinfo='percent',
            textfont=dict(size=12, color='white'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
            name='Predictions'
        ),
        row=1, col=1
    )
    
    # === BAR CHART (Right) ===
    speaker_data = df_speaker.groupby('speaker')['talktime'].sum().sort_values(ascending=False)
    
    if color_dict_speaker:
        colors_bar = [color_dict_speaker.get(speaker, '#808080') for speaker in speaker_data.index.tolist()]
    else:
        colors_bar = '#4ECDC4'  # Default color
    
    fig.add_trace(
        go.Bar(
            x=speaker_data.index.tolist(),
            y=speaker_data.values.tolist(),
            marker=dict(color=colors_bar),
            hovertemplate='<b>%{x}</b><br>Talk Time: %{y:.2f}<extra></extra>',
            name='Talk Time'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': '<b>Prediction Distribution & Speaker Talk Time Analysis</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=18)
        },
        showlegend=True,
        width=width,
        height=height
    )
    
    # Update y-axis for bar chart
    fig.update_yaxes(title_text="Talk Time", row=1, col=2)
    fig.update_xaxes(title_text="Speaker", row=1, col=2)
    
    return fig


# Example usage:
# df_predictions = pd.DataFrame({
#     'prediction': ['greeting', 'auth', 'greeting', 'O', 'auth', 'greeting', 'O', 'O'],
#     'timetaken': [1.2, 2.3, 1.5, 3.1, 2.8, 1.1, 3.5, 3.2]
# })
# 
# df_speaker = pd.DataFrame({
#     'speaker': ['Agent', 'Customer', 'Agent', 'Customer', 'Agent'],
#     'talktime': [45.5, 62.3, 38.7, 55.1, 42.9]
# })
# 
# color_dict_predictions = {
#     'greeting': '#FF6B6B',
#     'auth': '#4ECDC4',
#     'O': '#45B7D1'
# }
# 
# color_dict_speaker = {
#     'Agent': '#95E1D3',
#     'Customer': '#F38181'
# }
# 
# fig = create_combined_charts(df_predictions, df_speaker, color_dict_predictions, color_dict_speaker)
# # For Databricks:
# displayHTML(fig.to_html())
