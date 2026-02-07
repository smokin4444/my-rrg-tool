import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIGURATION & PRESETS ---
st.set_page_config(page_title="Alpha RRG Scanner", layout="wide")

RRG_PRESETS = {
    "Major Themes": {
        "benchmark": "SPY",
        "tickers": ["SPY", "QQQ", "DIA", "MAGS", "IWM", "IJR", "GLD", "SLV", "COPX", "XLE", "BTC-USD"],
        "description": "Broad market vs. Small-Cap Quality & Hard Assets"
    },
    "Sector Rotation": {
        "benchmark": "SPY",
        "tickers": ["XLK", "XLY", "XLC", "XBI", "XLF", "XLI", "XLE", "XLV", "XLP", "XLU", "XLB", "XLRE"],
        "description": "Institutional flow across S&P 500 sectors"
    }
}

# --- 2. SIDEBAR UI ---
st.sidebar.header("Scanner Settings")
group_choice = st.sidebar.selectbox("Select Preset Group", list(RRG_PRESETS.keys()))
selected_group = RRG_PRESETS[group_choice]

benchmark_ticker = st.sidebar.text_input("Benchmark", value=selected_group["benchmark"])
tickers = st.sidebar.text_area("Tickers (comma separated)", value=", ".join(selected_group["tickers"]))
ticker_list = [t.strip() for t in tickers.split(",")]

window_ratio = st.sidebar.slider("RS-Ratio Window", 5, 30, 14)
window_mom = st.sidebar.slider("RS-Mom Window", 5, 30, 14)
tail_length = st.sidebar.slider("Tail Length", 1, 20, 10)

# --- 3. DATA & CALCULATIONS ---
@st.cache_data(ttl=3600)
def get_rrg_data(tickers, benchmark, days=200):
    all_symbols = list(set(tickers + [benchmark]))
    data = yf.download(all_symbols, period="1y")['Close']
    
    # Calculate RS-Ratio and RS-Momentum
    rs_ratio_df = pd.DataFrame()
    rs_mom_df = pd.DataFrame()
    
    bench_series = data[benchmark]
    
    for t in tickers:
        if t not in data.columns: continue
        # Step 1: Raw RS (Price / Benchmark) * 100
        rs = (data[t] / bench_series) * 100
        
        # Step 2: RS-Ratio (Moving Average of RS normalized)
        # Simplified JdK implementation for Streamlit
        ratio = rs.rolling(window_ratio).mean()
        norm_ratio = ((ratio - ratio.mean()) / ratio.std()) + 100
        
        # Step 3: RS-Momentum (Rate of Change of RS-Ratio)
        mom = norm_ratio.diff(window_mom)
        norm_mom = ((mom - mom.mean()) / mom.std()) + 100
        
        rs_ratio_df[t] = norm_ratio
        rs_mom_df[t] = norm_mom
        
    return rs_ratio_df.tail(30), rs_mom_df.tail(30)

ratio_data, mom_data = get_rrg_data(ticker_list, benchmark_ticker)

# --- 4. THE ALPHA GRID LOGIC ---
def get_sync_status(r, m, prev_r, prev_m):
    if r > 100 and m > 100:
        return "POWER WALK" if m > prev_m else "LEADING"
    elif r > 100 and m < 100:
        return "WEAKENING"
    elif r < 100 and m < 100:
        return "LAGGING"
    else:
        return "LEAD-THROUGH" if m > prev_m else "IMPROVING"

grid_data = []
for t in ratio_data.columns:
    r = ratio_data[t].iloc[-1]
    m = mom_data[t].iloc[-1]
    prev_r = ratio_data[t].iloc[-2]
    prev_m = mom_data[t].iloc[-2]
    
    grid_data.append({
        "Ticker": t,
        "RS-Ratio": round(r, 2),
        "RS-Mom": round(m, 2),
        "Status": get_sync_status(r, m, prev_r, prev_m)
    })

df_grid = pd.DataFrame(grid_data).sort_values(by="RS-Ratio", ascending=False)

# --- 5. VISUALIZATION ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"RRG: {group_choice}")
    fig = go.Figure()

    # Quadrant Backgrounds
    fig.add_rect(x0=100, y0=100, x1=110, y1=110, fillcolor="rgba(0,255,0,0.1)", line_width=0) # Leading
    fig.add_rect(x0=100, y0=90, x1=110, y1=100, fillcolor="rgba(255,255,0,0.1)", line_width=0) # Weakening
    fig.add_rect(x0=90, y0=90, x1=100, y1=100, fillcolor="rgba(255,0,0,0.1)", line_width=0) # Lagging
    fig.add_rect(x0=90, y0=100, x1=100, y1=110, fillcolor="rgba(0,0,255,0.1)", line_width=0) # Improving

    for t in ratio_data.columns:
        # Plot Tails
        fig.add_trace(go.Scatter(
            x=ratio_data[t].tail(tail_length), 
            y=mom_data[t].tail(tail_length),
            mode='lines+markers',
            name=t,
            marker=dict(size=[2]*9 + [12], symbol="diamond")
        ))

    fig.update_layout(
        xaxis=dict(title="RS-Ratio", range=[95, 105]),
        yaxis=dict(title="RS-Momentum", range=[95, 105]),
        height=600,
        showlegend=True,
        shapes=[
            dict(type='line', x0=100, y0=90, x1=100, y1=110, line=dict(color="Black", width=2)),
            dict(type='line', x0=90, y0=100, x1=110, y1=100, line=dict(color="Black", width=2))
        ]
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Alpha Grid")
    st.dataframe(df_grid, use_container_width=True, hide_index=True)
    st.caption("Status Guide: POWER WALK (Leading + Rising Mom), LEAD-THROUGH (Improving + Rising Mom)")
