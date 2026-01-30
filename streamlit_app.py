# ... (Imports and Math Engine as before)

# --- THE FIX IN THE RRG CHART SECTION ---
for i, (t, data) in enumerate(history[timeframe].items()):
    df_full = data["df"]
    df_sliced = df_full[df_full.index <= target_date].tail(tail_len)
    
    if len(df_sliced) < 3: continue
    
    color = px.colors.qualitative.Plotly[i % 10]
    
    # 1. THE TRAIL (Now with Lines + Period Dots)
    fig.add_trace(go.Scatter(
        x=df_sliced['x'], 
        y=df_sliced['y'], 
        mode='lines+markers',  # Adds dots for each day/week
        name=t, 
        line=dict(width=2, color=color),
        marker=dict(
            size=6,            # Size of the daily dots
            color=color,
            opacity=0.6,
            line=dict(width=1, color='white') # Makes dots crisp
        ),
        opacity=0.4, 
        legendgroup=t
    ))
    
    # 2. THE CURRENT HEAD (Large Diamond for the current date)
    fig.add_trace(go.Scatter(
        x=[df_sliced['x'].iloc[-1]], 
        y=[df_sliced['y'].iloc[-1]],
        mode='markers+text',
        marker=dict(
            symbol='diamond', 
            size=15, 
            color=color, 
            line=dict(width=2, color='white')
        ),
        text=[t], 
        textposition="top center",
        legendgroup=t, 
        showlegend=False
    ))
