def create_speaker_talktime_barchart(df, color_dict=None, width=800, height=600):
    """
    Create an interactive bar chart showing talktime by speaker using Plotly.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'speaker' and 'talktime' columns
    color_dict : dict, optional
        Dictionary mapping speaker values to color codes
        If None, uses default Plotly colors
    width : int, optional
        Figure width in pixels. Default is 800
    height : int, optional
        Figure height in pixels. Default is 600
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
    """
    # Group by speaker and sum talktime
    speaker_talktime = df.groupby('speaker')['talktime'].sum().reset_index()
    speaker_talktime = speaker_talktime.sort_values('talktime', ascending=False)
    
    # Get colors if provided
    if color_dict:
        colors = [color_dict.get(speaker, '#808080') for speaker in speaker_talktime['speaker']]
    else:
        colors = None
    
    # Create bar chart
    fig = go.Figure(data=[go.Bar(
        x=speaker_talktime['speaker'],
        y=speaker_talktime['talktime'],
        marker=dict(color=colors) if colors else {},
        text=speaker_talktime['talktime'].round(2),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Talktime: %{y:.2f}<extra></extra>'
    )])
    
    # Update layout
    fig.update_layout(
        title={
            'text': '<b>Talktime by Speaker</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=16)
        },
        xaxis_title='Speaker',
        yaxis_title='Talktime',
        showlegend=False,
        width=width,
        height=height,
        xaxis=dict(tickangle=-45) if len(speaker_talktime) > 5 else {}
    )
    
    return fig
