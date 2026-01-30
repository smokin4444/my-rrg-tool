import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from datetime import timedelta

st.set_page_config(page_title="Alpha-Scanner: Time Machine", layout="wide")

# --- MASTER TICKER HEAP ---
MINERS = "AFM.V, NAK, A4N.AX, CSC.AX, IVN.TO, TGB"
THEMES = "URA, COPX, GDXJ, SILJ, IBIT, ITA, POWR, XME, SOXX, IGV, MAGS"
SECTORS = "XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU"

FUND_MAP = {
    "SOXX": "Semiconductors", "IGV": "Software", "XLP": "Cons. Staples",
    "MAGS": "Mag Seven", "URA": "Uranium", "COPX": "Copper", "GDXJ": "Junior Gold", 
    "SILJ": "Junior Silver", "IBIT": "Spot Bitcoin", "ITA": "Defense", "POWR": "Power Infra", 
    "XME": "Metals & Mining", "XLC": "Comm. Services", "XLY": "Cons. Disc.", "XLE": "Energy", 
    "XLF": "Financials", "XLV": "Health Care", "XLI": "Industrials", 
    "XLB": "Materials", "XLRE": "Real Estate", "XLK": "Technology", "XLU": "Utilities"
}

# --- Sidebar ---
with st.sidebar:
    st.header("🎯 Watchlist")
    heap_type = st.radio("Choose Group:", ["My Miners", "Thematic Heaps", "Sector ETFs", "Custom"])
    
    if heap_type == "My Miners": current_list = MINERS
    elif heap_type == "Thematic Heaps": current_list = THEMES
    elif heap_type == "Sector ETFs": current_list = SECTORS
    else: current_list = st.session_state.get('custom_list', MINERS)

    tickers_input = st.text_area("Ticker Heap:", value=current_list, height=100)
    benchmark = st.text_input("Benchmark:", value="SPY")
    
    st.markdown("---")
    st.header("🕰️ Time Machine")
    # Date slider added here
    lookback_days = st.slider("Lookback Window (Days):", 30, 365, 180)
    timeframe = st.radio("Chart Timeframe:", ["Daily", "Weekly"], index=0)
    tail_len = st.slider("Tail Length (Trails):", 5, 30, 15)

# --- Math Engine ---
def get_quadrant(ratio, mom):
    if ratio >= 100 and mom >= 100: return "LEADING"
    if ratio < 100 and mom >= 100: return "IMPROVING"
    if ratio < 100 and mom < 100: return "LAGGING"
    return "WEAKENING"

@st.cache_data(ttl=3600)
def get_full_analysis(ticker_str, bench):
    tickers = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    all_list = list(set(tickers + [bench.upper(), "^VIX"]))
    
    # Download more data to support historical slider
    d_raw = yf.download(all_list, period="2y", interval="1d")
    w_raw = yf.download(all_list, period="2y", interval="1wk")
    vix_series = d_raw['Close']['^VIX']
    
    # Store full history for RRG plotting
    full_history = {"Daily": {}, "Weekly": {}}

    for t in tickers:
        if t not in d_raw['Close'].columns: continue
        
        def calc_series(df):
            px = df['Close'][t].dropna()
            bx = df['Close'][bench.upper()].dropna()
            common = px.index.intersection(bx.index)
            rel = (px.loc[common] / bx.loc[common]) * 100
            ratio = 100 + ((rel - rel.rolling(14).mean()) / rel.rolling(14).std())
            roc = ratio.pct_change(1) * 100
            mom = 100 + ((roc - roc.rolling(14).mean()) / roc.rolling(14).std())
            # For table metrics (Volume and CH Score)
            v = np.sqrt((ratio.iloc[-1] - ratio.iloc[-5])**2 + (mom.iloc[-1] - mom.iloc[-5])**2)
            rv = (df['Volume'][t].iloc[-1] / df['Volume'][t].tail(20).mean())
            return pd.DataFrame({'x': ratio, 'y': mom}), (v * 0.7) + (rv * 0.3), rv

        d_hist, d_ch, d_rv = calc_series(d_raw)
        w_hist, w_ch, _ = calc_series(w_raw)
        
        full_history["Daily"][t] = {"df": d_hist, "ch": d_ch, "rv": d_rv}
        full_history["Weekly"][t] = {"df": w_hist, "ch": w_ch}

    return full_history, vix_series

# --- Execution ---
try:
    history, vix_data = get_full_analysis(tickers_input, benchmark)
    
    # Determine date range from data
    all_dates = history[timeframe][next(iter(history[timeframe]))]["df"].index
    min_date = all_dates[-(lookback_days)] if len(all_dates) > lookback_days else all_dates[0]
    
    # THE TIME SLIDER
    target_date = st.select_slider(
        "Select Historical Date to View RRG:",
        options=all_dates[all_dates >= min_date],
        value=all_dates[-1]
    )

    st.info(f"🛡️ **VIX on {target_date.date()}:** {vix_data.asof(target_date):.2f}")

    # 1. RRG CHART (Historical View)
    st.subheader(f"🌀 {timeframe} Rotation on {target_date.date()}")
    fig = go.Figure()
    fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="rgba(0,0,0,0.2)", width=2))
    fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="rgba(0,0,0,0.2)", width=2))
    
    table_rows = []
    
    for i, (t, data) in enumerate(history[timeframe].items()):
        df_full = data["df"]
        # Slice data up to the selected date
        df_sliced = df_full[df_full.index <= target_date].tail(tail_len)
        
        if len(df_sliced) < 3: continue
        
        color = px.colors.qualitative.Plotly[i % 10]
        
        # Smooth the trail
        tr = np.arange(len(df_sliced)); ts = np.linspace(0, len(df_sliced)-1, len(df_sliced)*5)
        x_smooth = interp1d(tr, df_sliced['x'], kind='cubic')(ts)
        y_smooth = interp1d(tr, df_sliced['y'], kind='cubic')(ts)

        fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines', name=t, line=dict(width=2.5, color=color), opacity=0.4, legendgroup=t))
        fig.add_trace(go.Scatter(x=[x_smooth[-1]], y=[y_smooth[-1]], mode='markers+text', marker=dict(symbol='diamond', size=14, color=color, line=dict(width=2, color='white')), text=[t], textposition="top center", legendgroup=t, showlegend=False))
        
        # Prepare table data for the selected date
        curr_x, curr_y = df_sliced['x'].iloc[-1], df_sliced['y'].iloc[-1]
        quad = get_quadrant(curr_x, curr_y)
        
        table_rows.append({
            "Ticker": t, "Name": FUND_MAP.get(t, "Stock"), "Quadrant": quad,
            "RS-Ratio": round(curr_x, 2), "Momentum": round(curr_y, 2)
        })

    fig.update_layout(template="plotly_white", height=700, xaxis=dict(range=[97, 103]), yaxis=dict(range=[97, 103]), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    # 2. ALIGNMENT GRID (Simplified for Time Travel)
    st.subheader(f"📊 Market Snapshot: {target_date.date()}")
    df_snap = pd.DataFrame(table_rows)
    
    def color_quad(val):
        color = '#2ECC71' if val == 'LEADING' else '#F1C40F' if val == 'IMPROVING' else '#E74C3C' if val == 'LAGGING' else '#FBFCFC'
        return f'background-color: {color}; color: black; font-weight: bold'

    st.dataframe(df_snap.style.map(color_quad, subset=['Quadrant']), use_container_width=True)

except Exception as e:
    st.error(f"Time Machine Error: {e}")
