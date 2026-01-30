# 3. Add smoothed tails and arrowheads
    for ticker, df in results.items():
        # A. Draw the smooth "Tail" line
        fig.add_trace(go.Scatter(
            x=df['x'], y=df['y'], 
            mode='lines', 
            name=ticker, 
            line=dict(width=3), 
            legendgroup=ticker,
            hoverinfo='skip'
        ))
        
        # B. Draw the "Arrow Head" at the current position
        # We use the last two points of the smoothed data to find the direction
        fig.add_trace(go.Scatter(
            x=[df['x'].iloc[-1]], 
            y=[df['y'].iloc[-1]],
            mode='markers+text',
            marker=dict(
                size=18,             # Made larger to be visible
                symbol="triangle-up", # Using a solid triangle for the arrowhead
                angleref="previous",  # Points the triangle in the direction of travel
                standoff=0           # Keeps the arrow right on the tip
            ),
            text=[ticker], 
            textposition="top right",
            name=ticker, 
            legendgroup=ticker,
            showlegend=False  # Hide the "dot" from the legend to keep it clean
        ))

    # Force the chart to be a perfect square for better RRG interpretation
    fig.update_layout(
        template="plotly_white", 
        xaxis_title="RS-Ratio", 
        yaxis_title="RS-Momentum",
        xaxis=dict(range=[96, 104], gridcolor='lightgray', zeroline=False), 
        yaxis=dict(range=[96, 104], gridcolor='lightgray', zeroline=False),
        height=800,
        width=800,
        showlegend=True
    )
