import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

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
tickers_input = st.sidebar.text_area("Tickers", value=", ".join(selected_group["tickers"]))
ticker_list = [t.strip() for t in tickers_input.split(",")]

window_ratio = st.sidebar.slider("RS-Ratio Window", 5, 30, 14)
window_mom = st.sidebar.slider("RS-Mom Window", 5, 30, 14)
tail_length = st.sidebar.slider("Tail Length", 1, 20, 10)

# --- 3. DATA & CALCULATIONS ---
@st.cache_data(ttl=3600)
def get_rrg_data(tickers, benchmark):
    all_symbols = list(set(tickers + [benchmark]))
    data = yf.download(all_symbols, period="1y")['Close']
    
    rs_ratio_df = pd.DataFrame()
    rs_mom_df = pd.DataFrame()
    bench_series = data[benchmark]
    
    for t in tickers:
        if t not in data.columns: continue
        rs = (data[t] / bench_series) * 100
        
        # Normalized Ratio (Simplified JdK)
        ratio = rs.rolling(window_ratio).mean()
        norm_ratio = ((ratio - ratio.mean()) / ratio.std()) + 100
        
        # Normalized Momentum
        mom = norm_ratio.diff(window_mom)
        norm_mom = ((mom - mom.mean()) / mom.std()) + 100
        
        rs_ratio_df[t] = norm_ratio
        rs_mom_df[t] = norm_mom
        
    return rs_ratio_df.tail(30), rs_mom_df.tail(30)

ratio_data, mom_data = get_rrg_data(ticker_list, benchmark_ticker)

# --- 4. ALPHA GRID LOGIC ---
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
    r, m = ratio_data[t].iloc[-1], mom_data[t].iloc[-1]
    pr, pm = ratio_data[t].iloc[-2], mom_data[t].iloc[-2]
    grid_data.append({
        "Ticker": t,
        "RS-Ratio": round(r, 2),
        "RS-Mom": round(m, 2),
        "Status": get_sync_status(r, m, pr, pm)
    })

df_grid = pd.DataFrame(grid_data).sort_values(by="RS-Ratio", ascending=False)

# --- 5. VISUALIZATION ---
col1, col2 = st.columns([2, 1])

with col1:
    fig = go.Figure()

    # Universal Quadrant Shading using add_shape (Avoids AttributeError)
    quadrants = [
        dict(x0=100, y0=100, x1=110, y1=110, color="rgba(0,255,0,0.1)"),  # Leading
        dict(x0=100, y0=90, x1=110, y1=100, color="rgba(255,255,0,0.1)"), # Weakening
        dict(x0=90, y0=90, x1=100, y1=100, color="rgba(255,0,0,0.1)"),   # Lagging
        dict(x0=90, y0=100, x1=100, y1=110, color="rgba(0,0,255,0.1)")   # Improving
    ]

    for q in quadrants:
        fig.add_shape(type="rect", x0=q["x0"], y0=q["y0"], x1=q["x1"], y1=q["y1"],
                      fillcolor=q["color"], line_width=0, layer="below")

    # Draw Crosshairs
    fig.add_shape(type="line", x0=100, y0=90, x1=100, y1=110, line=dict(color="Black", width=2))
    fig.add_shape(type="line", x0=90, y0=100, x1=110, y1=100, line=dict(color="Black", width=2))

    for t in ratio_data.columns:
        fig.add_trace(go.Scatter(
            x=ratio_data[t].tail(tail_length), 
            y=mom_data[t].tail(tail_length),
            mode='lines+markers', name=t,
            marker=dict(size=[2]*(tail_length-1) + [12], symbol="diamond")
        ))

    fig.update_layout(xaxis=dict(title="RS-Ratio", range=[96, 104]),
                      yaxis=dict(title="RS-Momentum", range=[96, 104]),
                      height=600, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Alpha Grid")
    st.dataframe(df_grid, use_container_width=True, hide_index=True)
    st.caption("Sorting: RS-Ratio (High to Low)")
