import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d

st.set_page_config(page_title="Pro-RRG Dashboard", layout="wide")

st.title("📈 Professional Relative Rotation Graph")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    user_tickers = st.text_area("Tickers:", "AFM.V, NAK, A4N.AX, CSC.AX, IVN.TO, TGB")
    benchmark = st.text_input("Benchmark:", "COPX")
    timeframe = st.radio("Timeframe:", ["Daily", "Weekly"])
    tail_length = st.slider("Tail Length:", 5, 30, 15)

# --- Math & Smoothing Engine ---
def get_rrg_data(ticker_list, bench, tf, tail):
    interval = "1d" if tf == "Daily" else "1wk"
    tickers = [t.strip() for t in ticker_list.split(",") if t.strip()]
    all_list = list(set(tickers + [bench]))
    
    data_raw = yf.download(all_list, period="2y", interval=interval)
    data = data_raw['Close'] if isinstance(data_raw.columns, pd.MultiIndex) else data_raw[['Close']]
    data = data.dropna()
    
    rrg_results = {}
    bench_price = data[bench]
    
    for t in tickers:
        if t not in data.columns or t == bench: continue
        
        rel_price = (data[t] / bench_price) * 100
        sma = rel_price.rolling(14).mean()
        std = rel_price.rolling(14).std()
        rs_ratio = 100 + ((rel_price - sma) / std)
        
        roc = rs_ratio.pct_change(1) * 100
        roc_sma = roc.rolling(14).mean()
        roc_std = roc.rolling(14).std()
        rs_momentum = 100 + ((roc - roc_sma) / roc_std)
        
        raw_tail = pd.DataFrame({'x': rs_ratio, 'y': rs_momentum}).dropna().tail(tail)
        
        if len(raw_tail) < 3: continue

        t_raw = np.arange(len(raw_tail))
        t_smooth = np.linspace(0, len(raw_tail)-1, len(raw_tail)*5)
        
        fx = interp1d(t_raw, raw_tail['x'], kind='cubic')
        fy = interp1d(t_raw, raw_tail['y'], kind='cubic')
        
        rrg_results[t] = pd.DataFrame({'x': fx(t_smooth), 'y': fy(t_smooth)})

    return rrg_results

# --- Plotting ---
try:
    results = get_rrg_data(user_tickers, benchmark, timeframe, tail_length)
    
    fig = go.Figure()

    # 1. Add Quadrant Lines (Solid Black for visibility)
    fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="black", width=2))
    fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="black", width=2))

    # 2. Add Quadrant Labels
    fig.add_annotation(x=102, y=102, text="LEADING", showarrow=False, font=dict(color="green", size=14))
    fig.add_annotation(x=98, y=102, text="IMPROVING", showarrow=False, font=dict(color="blue", size=14))
    fig.add_annotation(x=98, y=98, text="LAGGING", showarrow=False, font=dict(color="red", size=14))
    fig.add_annotation(x=102, y=98, text="WEAKENING", showarrow=False, font=dict(color="#FF8C00", size=14))

    # 3. Add smoothed tails and Directional Arrows
    for ticker, df in results.items():
        # Line color is picked automatically by Plotly
        line_color = px.colors.qualitative.Plotly[list(results.keys()).index(ticker) % 10]

        # Draw the line
        fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='lines', 
                                 name=ticker, line=dict(width=3, color=line_color), 
                                 showlegend=True))
        
        # Add the directional Arrow at the tip using Annotations
        # This points from the second-to-last point to the last point
        fig.add_annotation(
            x=df['x'].iloc[-1],
            y=df['y'].iloc[-1],
            ax=df['x'].iloc[-2],
            ay=df['y'].iloc[-2],
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=2,
            arrowwidth=3,
            arrowcolor=line_color
        )
        
        # Add the Ticker Label near the arrow
        fig.add_annotation(
            x=df['x'].iloc[-1],
            y=df['y'].iloc[-1],
            text=ticker,
            showarrow=False,
            yshift=15,
            font=dict(color=line_color, size=12)
        )

    fig.update_layout(
        template="plotly_white", 
        xaxis_title="RS-Ratio", 
        yaxis_title="RS-Momentum",
        xaxis=dict(range=[96, 104]), 
        yaxis=dict(range=[96, 104]),
        height=800,
        width=800
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Waiting for data... Error: {e}")
