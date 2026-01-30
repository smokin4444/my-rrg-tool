import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import interp1d

st.set_page_config(page_title="Alpha-Scanner Pro", layout="wide")

# --- THE STAPLE: MASTER THEMATIC HEAP ---
# Restoring the original list including Copper, IBIT, Uranium, etc.
MASTER_THEMES = (
    "SOXX, IGV, XLP, MAGS, URA, COPX, GDXJ, SILJ, IBIT, ITA, POWR, XME, "
    "XLC, XLY, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU"
)

MINERS = "AFM.V, NAK, A4N.AX, CSC.AX, IVN.TO, TGB"

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
    heap_type = st.radio("Choose Group:", ["Master Themes", "My Miners", "Elite Themes", "Custom"])
    
    if heap_type == "Master Themes": current_list = MASTER_THEMES
    elif heap_type == "My Miners": current_list = MINERS
    elif heap_type == "Elite Themes": current_list = st.session_state.get('elite_list', "AMD, NVDA, TSM, PLTR, RKLB, ASTS") # Placeholder for the big list
    else: current_list = st.session_state.get('custom_list', MASTER_THEMES)

    tickers_input = st.text_area("Ticker Heap:", value=current_list, height=150)
    benchmark = st.text_input("Benchmark:", value="SPY")
    
    st.markdown("---")
    st.header("🕰️ Time Controls")
    timeframe = st.radio("Chart Timeframe:", ["Daily", "Weekly"], index=0)
    tail_len = st.slider("Tail Length (Dots):", 5, 30, 15)
    filter_setups = st.checkbox("Show Only Top Setups", value=True)

# --- Math Engine ---
def get_quadrant(ratio, mom):
    if ratio >= 100 and mom >= 100: return "LEADING"
    if ratio < 100 and mom >= 100: return "IMPROVING"
    if ratio < 100 and mom < 100: return "LAGGING"
    return "WEAKENING"

@st.cache_data(ttl=3600)
def get_full_analysis(ticker_str, bench):
    tickers = [t.strip().upper() for t in ticker_str.replace("!", "").split(",") if t.strip()]
    all_list = list(set(tickers + [bench.upper(), "^VIX"]))
    
    d_raw = yf.download(all_list, period="2y", interval="1d", group_by='ticker', progress=False)
    w_raw = yf.download(all_list, period="2y", interval="1wk", group_by='ticker', progress=False)
    
    try:
        vix = d_raw['^VIX']['Close'].iloc[-1]
    except: vix = 0
    
    history = {"Daily": {}, "Weekly": {}}
    table_data = []

    for t in tickers:
        try:
            def calc_metrics(df_raw, ticker, bench_ticker):
                if ticker not in df_raw.columns.get_level_values(0): return None
                px = df_raw[ticker]['Close'].dropna()
                bx = df_raw[bench_ticker]['Close'].dropna()
                if len(px) < 35 or len(bx) < 35: return None
                
                common = px.index.intersection(bx.index)
                rel = (px.loc[common] / bx.loc[common]) * 100
                ratio = 100 + ((rel - rel.rolling(14).mean()) / rel.rolling(14).std())
                roc = ratio.pct_change(1) * 100
                mom = 100 + ((roc - roc.rolling(14).mean()) / roc.rolling(14).std())
                
                v = np.sqrt((ratio.iloc[-1] - ratio.iloc[-5])**2 + (mom.iloc[-1] - mom.iloc[-5])**2)
                rv = (df_raw[ticker]['Volume'].iloc[-1] / df_raw[ticker]['Volume'].tail(20).mean())
                return ratio, mom, round((v * 0.7) + (rv * 0.3), 2), rv

            d_res = calc_metrics(d_raw, t, bench.upper())
            w_res = calc_metrics(w_raw, t, bench.upper())
            
            if d_res and w_res:
                history["Daily"][t] = pd.DataFrame({'x': d_res[0], 'y': d_res[1]}).dropna()
                history["Weekly"][t] = pd.DataFrame({'x': w_res[0], 'y': w_res[1]}).dropna()
                
                d_q = get_quadrant(d_res[0].iloc[-1], d_res[1].iloc[-1])
                w_q = get_quadrant(w_res[0].iloc[-1], w_res[1].iloc[-1])
                
                status = "BULLISH SYNC" if d_q == "LEADING" and w_q == "LEADING" else \
                         "EARLY ACCEL" if d_q == "LEADING" and w_q == "IMPROVING" else \
                         "DAILY PIVOT" if d_q == "IMPROVING" and w_q == "LAGGING" else "DIVERGED"
                
                table_data.append({
                    "Ticker": t, "Name": FUND_MAP.get(t, ""), "Sync Status": status,
                    "Daily Quad": d_q, "Weekly Quad": w_q,
                    "Daily CH": d_res[2], "Weekly CH": w_res[2], "Rel Vol": d_res[3]
                })
        except: continue
    return pd.DataFrame(table_data), history, vix

# --- Execution ---
try:
    df_main, history_data, vix_val = get_full_analysis(tickers_input, benchmark)
    st.info(f"🛡️ **VIX:** {vix_val:.2f} | **Current Benchmark:** {benchmark.upper()}")

    # 1. RRG CHART
    st.subheader(f"🌀 {timeframe} Rotation")
    fig = go.Figure()
    fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="rgba(0,0,0,0.2)", width=2))
    fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="rgba(0,0,0,0.2)", width=2))
    
    for i, (t, df) in enumerate(history_data[timeframe].items()):
        if filter_setups and t not in df_main[df_main['Sync Status'].isin(["BULLISH SYNC", "EARLY ACCEL", "DAILY PIVOT"])]['Ticker'].values: continue
        color = px.colors.qualitative.Plotly[i % 10]
        df_p = df.tail(tail_len)
        fig.add_trace(go.Scatter(x=df_p['x'], y=df_p['y'], mode='lines+markers', name=t, line=dict(width=2, color=color), marker=dict(size=5, color=color, opacity=0.6, line=dict(width=1, color='white')), opacity=0.4, legendgroup=t))
        fig.add_trace(go.Scatter(x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], mode='markers+text', marker=dict(symbol='diamond', size=13, color=color, line=dict(width=1.5, color='white')), text=[t], textposition="top center", legendgroup=t, showlegend=False))

    fig.update_layout(template="plotly_white", height=700, xaxis=dict(range=[97, 103], title="Strength"), yaxis=dict(range=[97, 103], title="Momentum"))
    st.plotly_chart(fig, use_container_width=True)

    # 2. GRID
    st.subheader("📊 Master Alpha Grid")
    df_main['sort_val'] = df_main['Sync Status'].map({"BULLISH SYNC": 0, "EARLY ACCEL": 1, "DAILY PIVOT": 2, "DIVERGED": 3})
    df_display = df_main.sort_values(by=['sort_val', 'Daily CH'], ascending=[True, False]).copy()
    if filter_setups: df_display = df_display[df_display['Sync Status'] != "DIVERGED"]

    def style_sync(val):
        colors = {"BULLISH SYNC": "#2ECC71", "EARLY ACCEL": "#3498DB", "DAILY PIVOT": "#F1C40F", "BEARISH SYNC": "#E74C3C"}
        return f'background-color: {colors.get(val, "#FBFCFC")}; color: black; font-weight: bold'

    st.dataframe(
        df_display.drop(columns=['sort_val']).style.map(style_sync, subset=['Sync Status'])
        .format(formatter={('Rel Vol'): "{:.2f}x"})
    , use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
