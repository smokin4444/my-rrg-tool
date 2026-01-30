import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d

st.set_page_config(page_title="Alpha-Scanner Pro", layout="wide")

# --- MASTER TICKER HEAP ---
MASTER_HEAPS = "SOXX, IGV, XLP, MAGS, URA, COPX, GDXJ, SILJ, IBIT, ITA, POWR, XME, " \
               "XLC, XLY, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU, AFM.V, NAK, A4N.AX, CSC.AX, IVN.TO, TGB"

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
    st.header("🎯 Filter Controls")
    filter_setups = st.checkbox("Show Only Top Setups (Syncs & Pivots)", value=False)
    
    st.markdown("---")
    st.header("⚙️ Settings")
    tickers_input = st.text_area("Ticker Heap:", value=MASTER_HEAPS, height=150)
    benchmark = st.text_input("Benchmark:", value="SPY")
    timeframe = st.radio("Chart Timeframe:", ["Daily", "Weekly"], index=0)
    tail_len = st.slider("Tail Length:", 5, 30, 15)

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
    
    # Dual-Download
    d_raw = yf.download(all_list, period="2y", interval="1d")
    w_raw = yf.download(all_list, period="2y", interval="1wk")
    vix = d_raw['Close']['^VIX'].iloc[-1]
    
    rrg_plot_data = {}
    table_data = []

    for t in tickers:
        if t not in d_raw['Close'].columns: continue
        
        def calc_metrics(df):
            px = df['Close'][t].dropna()
            bx = df['Close'][bench.upper()].dropna()
            common = px.index.intersection(bx.index)
            rel = (px.loc[common] / bx.loc[common]) * 100
            ratio = 100 + ((rel - rel.rolling(14).mean()) / rel.rolling(14).std())
            roc = ratio.pct_change(1) * 100
            mom = 100 + ((roc - roc.rolling(14).mean()) / roc.rolling(14).std())
            
            v = np.sqrt((ratio.iloc[-1] - ratio.iloc[-5])**2 + (mom.iloc[-1] - mom.iloc[-5])**2)
            rv = (df['Volume'][t].iloc[-1] / df['Volume'][t].tail(20).mean())
            ch = (v * 0.7) + (rv * 0.3)
            return ratio, mom, ch, rv

        # Daily Stats
        d_rat_s, d_mom_s, d_ch, d_rv = calc_metrics(d_raw)
        # Weekly Stats
        w_rat_s, w_mom_s, w_ch, _ = calc_metrics(w_raw)
        
        d_q, w_q = get_quadrant(d_rat_s.iloc[-1], d_mom_s.iloc[-1]), get_quadrant(w_rat_s.iloc[-1], w_mom_s.iloc[-1])
        
        # Sync Logic
        status = "BULLISH SYNC" if d_q == "LEADING" and w_q == "LEADING" else \
                 "BEARISH SYNC" if d_q == "LAGGING" and w_q == "LAGGING" else \
                 "DAILY PIVOT" if d_q == "IMPROVING" and w_q == "LAGGING" else "DIVERGED"
        
        table_data.append({
            "Ticker": t, "Name": FUND_MAP.get(t, "Stock"), "Sync Status": status,
            "Daily Quad": d_q, "Weekly Quad": w_q, "Daily CH": round(d_ch, 2), "Weekly CH": round(w_ch, 2), "Rel Vol": d_rv
        })

        # Chart Data (Based on Sidebar TF)
        target_s = (d_rat_s, d_mom_s) if timeframe == "Daily" else (w_rat_s, w_mom_s)
        rt = pd.DataFrame({'x': target_s[0], 'y': target_s[1]}).dropna().tail(tail_len)
        if len(rt) >= 3:
            tr = np.arange(len(rt)); ts = np.linspace(0, len(rt)-1, len(rt)*5)
            rrg_plot_data[t] = pd.DataFrame({'x': interp1d(tr, rt['x'], kind='cubic')(ts), 'y': interp1d(tr, rt['y'], kind='cubic')(ts)})

    return pd.DataFrame(table_data), rrg_plot_data, vix

# --- Execution ---
try:
    df_main, plot_results, vix_val = get_full_analysis(tickers_input, benchmark)
    
    # HEADER
    st.info(f"🛡️ **VIX:** {vix_val:.2f} | **Scanning:** {len(df_main)} Tickers")

    # 1. THE GRID
    st.subheader("📊 The Alignment Grid")
    if filter_setups:
        df_display = df_main[df_main['Sync Status'].isin(["BULLISH SYNC", "DAILY PIVOT"])]
    else:
        df_display = df_main

    def color_sync(val):
        color = '#2ECC71' if 'BULLISH' in val else '#E74C3C' if 'BEARISH' in val else '#F1C40F' if 'PIVOT' in val else '#FDFEFE'
        return f'background-color: {color}; color: black; font-weight: bold'

    st.dataframe(
        df_display.sort_values(by="Daily CH", ascending=False).style.map(color_sync, subset=['Sync Status'])
        .map(lambda x: 'font-weight: bold; color: #1E88E5', subset=['Daily CH', 'Weekly CH'])
        .map(lambda x: 'color: #D32F2F; font-weight: bold; background-color: #FFF9C4' if x > 2.5 else '', subset=['Rel Vol'])
        .format({"Rel Vol": "{:.2f}x"}), use_container_width=True
    )

    # 2. THE CHART
    st.markdown("---")
    st.subheader(f"🌀 {timeframe} Rotation (Diamonds & Trails)")
    fig = go.Figure()
    fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="rgba(0,0,0,0.2)", width=2))
    fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="rgba(0,0,0,0.2)", width=2))
    
    for i, (t, df) in enumerate(plot_results.items()):
        # Filter chart too if checkbox is on
        if filter_setups and t not in df_display['Ticker'].values: continue
        
        color = px.colors.qualitative.Plotly[i % 10]
        fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='lines', name=t, line=dict(width=2.5, color=color), opacity=0.4, legendgroup=t))
        fig.add_trace(go.Scatter(x=[df['x'].iloc[-1]], y=[df['y'].iloc[-1]], mode='markers+text', marker=dict(symbol='diamond', size=14, color=color, line=dict(width=2, color='white')), text=[t], textposition="top center", legendgroup=t, showlegend=False))
    
    fig.update_layout(template="plotly_white", height=800, xaxis=dict(range=[97, 103]), yaxis=dict(range=[97, 103]), legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
