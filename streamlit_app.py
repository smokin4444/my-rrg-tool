import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import interp1d
from datetime import datetime

st.set_page_config(page_title="Alpha-Scanner Pro", layout="wide")

# --- CONSOLIDATED TICKER GROUPS ---
MINERS = "AFM.V, NAK, A4N.AX, CSC.AX, IVN.TO, TGB"
# Merged Thematic and Sectors into one master list
MARKET_GROUPS = (
    "URA, COPX, GDXJ, SILJ, IBIT, ITA, POWR, XME, SOXX, IGV, MAGS, "
    "XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU"
)

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
    st.header("🎯 Watchlist Selection")
    # Consolidated radio buttons
    heap_type = st.radio("Choose Group:", ["My Miners", "Market Themes & Sectors", "Custom"])
    
    if heap_type == "My Miners": current_list = MINERS
    elif heap_type == "Market Themes & Sectors": current_list = MARKET_GROUPS
    else: current_list = st.session_state.get('custom_list', MINERS)

    tickers_input = st.text_area("Ticker Heap:", value=current_list, height=120)
    benchmark = st.text_input("Benchmark:", value="SPY")
    
    st.markdown("---")
    st.header("🕰️ Time Controls")
    timeframe = st.radio("Chart Timeframe:", ["Daily", "Weekly"], index=0)
    tail_len = st.slider("Tail Length (Dots):", 5, 30, 15)
    filter_setups = st.checkbox("Show Only Top Setups", value=False)

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
    
    d_raw = yf.download(all_list, period="2y", interval="1d")
    w_raw = yf.download(all_list, period="2y", interval="1wk")
    vix = d_raw['Close']['^VIX'].iloc[-1]
    
    history = {"Daily": {}, "Weekly": {}}
    table_data = []

    for t in tickers:
        if t not in d_raw['Close'].columns: continue
        
        def calc_metrics(df):
            px_data = df['Close'][t].dropna()
            bx_data = df['Close'][bench.upper()].dropna()
            common = px_data.index.intersection(bx_data.index)
            rel = (px_data.loc[common] / bx_data.loc[common]) * 100
            ratio = 100 + ((rel - rel.rolling(14).mean()) / rel.rolling(14).std())
            roc = ratio.pct_change(1) * 100
            mom = 100 + ((roc - roc.rolling(14).mean()) / roc.rolling(14).std())
            
            v = np.sqrt((ratio.iloc[-1] - ratio.iloc[-5])**2 + (mom.iloc[-1] - mom.iloc[-5])**2)
            rv = (df['Volume'][t].iloc[-1] / df['Volume'][t].tail(20).mean())
            return ratio, mom, round((v * 0.7) + (rv * 0.3), 2), rv

        d_rat, d_mom, d_ch, d_rv = calc_metrics(d_raw)
        w_rat, w_mom, w_ch, _ = calc_metrics(w_raw)
        
        history["Daily"][t] = pd.DataFrame({'x': d_rat, 'y': d_mom}).dropna()
        history["Weekly"][t] = pd.DataFrame({'x': w_rat, 'y': w_mom}).dropna()
        
        d_q = get_quadrant(d_rat.iloc[-1], d_mom.iloc[-1])
        w_q = get_quadrant(w_rat.iloc[-1], w_mom.iloc[-1])
        status = "BULLISH SYNC" if d_q == "LEADING" and w_q == "LEADING" else \
                 "DAILY PIVOT" if d_q == "IMPROVING" and w_q == "LAGGING" else \
                 "BEARISH SYNC" if d_q == "LAGGING" and w_q == "LAGGING" else "DIVERGED"
        
        table_data.append({
            "Ticker": t, "Name": FUND_MAP.get(t, "Stock"), "Sync Status": status,
            "Daily Quad": d_q, "Weekly Quad": w_q, "Daily CH": d_ch, "Weekly CH": w_ch, "Rel Vol": d_rv
        })

    return pd.DataFrame(table_data), history, vix

# --- Execution ---
try:
    df_main, history_data, vix_val = get_full_analysis(tickers_input, benchmark)
    
    st.info(f"🛡️ **VIX:** {vix_val:.2f} | **Target Benchmark:** {benchmark.upper()}")

    # 1. RRG CHART (With Dots & Diamonds)
    st.subheader(f"🌀 {timeframe} Rotation (Diamonds & Period Dots)")
    fig = go.Figure()
    fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="rgba(0,0,0,0.2)", width=2))
    fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="rgba(0,0,0,0.2)", width=2))
    
    for i, (t, df) in enumerate(history_data[timeframe].items()):
        if filter_setups and t not in df_main[df_main['Sync Status'].isin(["BULLISH SYNC", "DAILY PIVOT"])]['Ticker'].values:
            continue
            
        color = px.colors.qualitative.Plotly[i % 10]
        df_plot = df.tail(tail_len)
        
        # Line + Dots
        fig.add_trace(go.Scatter(
            x=df_plot['x'], y=df_plot['y'], 
            mode='lines+markers', name=t,
            line=dict(width=2.5, color=color),
            marker=dict(size=6, color=color, opacity=0.6, line=dict(width=1, color='white')),
            opacity=0.4, legendgroup=t
        ))
        
        # Diamond Head
        fig.add_trace(go.Scatter(
            x=[df_plot['x'].iloc[-1]], y=[df_plot['y'].iloc[-1]],
            mode='markers+text',
            marker=dict(symbol='diamond', size=14, color=color, line=dict(width=2, color='white')),
            text=[t], textposition="top center",
            legendgroup=t, showlegend=False
        ))

    fig.update_layout(template="plotly_white", height=750, xaxis=dict(range=[97, 103]), yaxis=dict(range=[97, 103]), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    # 2. ALIGNMENT GRID
    st.subheader("📊 Dual-Timeframe Alpha Grid")
    
    # Custom Sort: Bullish Sync first, then Daily Pivot
    df_main['sort_val'] = df_main['Sync Status'].map({"BULLISH SYNC": 0, "DAILY PIVOT": 1, "DIVERGED": 2, "BEARISH SYNC": 3})
    df_display = df_main.sort_values(by=['sort_val', 'Daily CH'], ascending=[True, False])
    
    if filter_setups:
        df_display = df_display[df_display['Sync Status'].isin(["BULLISH SYNC", "DAILY PIVOT"])]

    st.dataframe(
        df_display.drop(columns=['sort_val']).style.map(lambda x: f'background-color: {"#2ECC71" if "BULLISH" in x else "#F1C40F" if "PIVOT" in x else "#E74C3C" if "BEARISH" in x else "#FBFCFC"}; color: black; font-weight: bold', subset=['Sync Status'])
        .map(lambda x: 'font-weight: bold; color: #1E88E5', subset=['Daily CH', 'Weekly CH'])
        .map(lambda x: 'color: #D32F2F; font-weight: bold; background-color: #FFF9C4' if x > 2.5 else '', subset=['Rel Vol'])
        .format({"Rel Vol": "{:.2f}x"}), use_container_width=True
    )

except Exception as e:
    st.error(f"Dashboard Initialization Error: {e}")
