html_string = f"""
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
    <div>{fig1.to_html(include_plotlyjs='cdn', div_id='fig1')}</div>
    <div>{fig2.to_html(include_plotlyjs='cdn', div_id='fig2')}</div>
</div>
"""

displayHTML(html_string)
