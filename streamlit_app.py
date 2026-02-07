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
    # Added auto_adjust=True to handle yfinance parsing issues
    data = yf.download(all_symbols, period="1y", auto_adjust=True)
    
    if data.empty:
        st.error("No data returned from Yahoo Finance. Check ticker symbols or connection.")
        return pd.DataFrame(), pd.DataFrame()

    # Handle multi-index columns if necessary
    close_data = data['Close'] if 'Close' in data else data
    
    rs_ratio_df = pd.DataFrame()
    rs_mom_df = pd.DataFrame()
    
    if benchmark not in close_data.columns:
        st.error(f"Benchmark {benchmark} not found in data.")
        return pd.DataFrame(), pd.DataFrame()
        
    bench_series = close_data[benchmark]
    
    for t in tickers:
        if t not in close_data.columns: continue
        rs = (close_data[t] / bench_series) * 100
        
        ratio = rs.rolling(window_ratio).mean()
        norm_ratio = ((ratio - ratio.mean()) / ratio.std()) + 100
        
        mom = norm_ratio.diff(window_mom)
        norm_mom = ((mom - mom.mean()) / mom.std()) + 100
        
        rs_ratio_df[t] = norm_ratio
        rs_mom_df[t] = norm_mom
        
    return rs_ratio_df.tail(30), rs_mom_df.tail(30)

ratio_data, mom_data = get_rrg_data(ticker_list, benchmark_ticker)

if not ratio_data.empty:
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

    # --- STYLE THE TABLE ---
    def style_status(val):
        color_map = {
            "POWER WALK": "background-color: #00FF00; color: black; font-weight: bold",
            "LEADING": "background-color: #CCFFCC; color: black",
            "WEAKENING": "background-color: #FFFF99; color: black",
            "LAGGING": "background-color: #FFCCCC; color: black",
            "LEAD-THROUGH": "background-color: #99CCFF; color: black; font-weight: bold",
            "IMPROVING": "background-color: #CCE5FF; color: black"
        }
        return color_map.get(val, "")

    # --- 5. VISUALIZATION ---
    st.subheader(f"Current Scanner: {group_choice}")
    
    # Chart on Top
    fig = go.Figure()

    # Universal Quadrant Shading
    quads = [
        (100, 100, 110, 110, "rgba(0,255,0,0.1)"),  # Leading
        (100, 90, 110, 100, "rgba(255,255,0,0.1)"), # Weakening
        (90, 90, 100, 100, "rgba(255,0,0,0.1)"),   # Lagging
        (90, 100, 100, 110, "rgba(0,0,255,0.1)")   # Improving
    ]

    for x0, y0, x1, y1, color in quads:
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, fillcolor=color, line_width=0, layer="below")

    fig.add_shape(type="line", x0=100, y0=90, x1=100, y1=110, line=dict(color="Black", width=1))
    fig.add_shape(type="line", x0=90, y0=100, x1=110, y1=100, line=dict(color="Black", width=1))

    for t in ratio_data.columns:
        fig.add_trace(go.Scatter(
            x=ratio_data[t].tail(tail_length), 
            y=mom_data[t].tail(tail_length),
            mode='lines+markers', name=t,
            marker=dict(size=[2]*(tail_length-1) + [12], symbol="diamond")
        ))

    fig.update_layout(xaxis=dict(title="RS-Ratio", range=[97, 103]),
                      yaxis=dict(title="RS-Momentum", range=[97, 103]),
                      height=650, template="plotly_white", showlegend=True)
    
    st.plotly_chart(fig, use_container_width=True)

    # Table BELOW Chart (for better visibility)
    st.divider()
    st.subheader("Alpha Grid Analysis")
    st.dataframe(df_grid.style.applymap(style_status, subset=['Status']), use_container_width=True, hide_index=True)
    
else:
    st.warning("Please verify your tickers in the sidebar.")
