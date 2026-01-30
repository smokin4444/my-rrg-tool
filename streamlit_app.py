# --- Updated RRG CHART with Time Markers ---
fig = go.Figure()

# Quadrant Dividers
fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="rgba(0,0,0,0.3)", width=3))
fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="rgba(0,0,0,0.3)", width=3))

for i, (t, df) in enumerate(results.items()):
    color = px.colors.qualitative.Plotly[i % 10]
    
    # 1. THE TRAIL (Now with Lines + Dots)
    fig.add_trace(go.Scatter(
        x=df['x'], 
        y=df['y'], 
        mode='lines+markers',  # This adds the time dots
        name=t, 
        line=dict(width=2, color=color),
        marker=dict(
            size=5,            # Small dots for each time period
            color=color,
            opacity=0.6
        ),
        opacity=0.5, 
        legendgroup=t
    ))
    
    # 2. THE CURRENT HEAD (Large Diamond stays for clarity)
    fig.add_trace(go.Scatter(
        x=[df['x'].iloc[-1]], 
        y=[df['y'].iloc[-1]],
        mode='markers+text',
        marker=dict(
            symbol='diamond', 
            size=14, 
            color=color, 
            line=dict(width=2, color='white')
        ),
        text=[t], 
        textposition="top center",
        legendgroup=t, 
        showlegend=False
    ))
