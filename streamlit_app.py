import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
import time

# --- INITIAL CONFIG ---
st.set_page_config(page_title="Alpha-Scanner Pro", layout="wide")

# --- CONSTANTS ---
GAS_URL = "https://script.google.com/macros/s/AKfycbxfcQoQCWnlbfOX8jIJKLAuc8VuWknXYQ5WQSKZhXywoHQRub91tyS6gRPBKqFrn01bWg/exec"
RRG_CENTER = 100
EPSILON = 1e-8
Z_LIMITS = (80, 120)  
CHART_RANGE = [96.5, 103.5] 
POWER_WALK_LEVEL = 101.5

# --- SITA HUB SYNC ---
def load_from_hub():
    try:
        response = requests.get(GAS_URL, timeout=10)
        if response.status_code == 200:
            all_data = response.json()
            return json.loads(all_data.get('watchlists', "{}"))
        return {}
    except: return {}

# --- MASTER TICKER DICTIONARY ---
TICKER_NAMES = {
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100", "DIA": "Dow Jones", "IWF": "Growth Stocks", 
    "IWD": "Value Stocks", "MAGS": "Magnificent 7", "IWM": "Small Caps", 
    "GLD": "Gold ETF", "SLV": "Silver ETF", "COPX": "Global Copper Miners", "XLE": "Energy",
    "XLK": "Technology", "XLY": "Consumer Durables", "XLC": "Communications", 
    "XLF": "Finance", "XLI": "Producer Manufacturing", 
    "XLV": "Health Services", "XLP": "Cons Staples", "XLU": "Utilities", 
    "XLB": "Materials (Broad)", "IYT": "Transportation", "SMH": "Semiconductors (NVDA)", 
    "SOXX": "Memory/Broad Semi", "FTXL": "Memory Super-Cycle (MU/WDC)", "IGV": "Software", 
    "XHB": "Home Construction", "IBIT": "Bitcoin Trust", "XME": "S&P Metals & Mining",
    "BDRY": "Dry Bulk Shipping", "BOAT": "Global Shipping ETF", "MOO": "Agribusiness",
    "JEDI": "Modern Warfare & Drones", "DRNZ": "Drone Tech (REX)", "ITA": "Aerospace & Defense",
    "POWR": "U.S. Power/Grid Infra", "PAVE": "U.S. Infrastructure Dev", 
    "REMX": "Rare Earth/Strategic Metals", "URNM": "Uranium Miners (Nuclear)", "ALB": "Lithium",
    "OZEM": "GLP-1 & Weight Loss", "IHI": "Medical Devices", "XBI": "Biotechnology",
    "GEV": "GE Vernova (Grid/Power)", "KWEB": "China Internet"
}

# --- WATCHLISTS ---
INDUSTRY_THEMES = "SMH, GEV, COPX, URNM, BOAT, BDRY, POWR, PAVE, REMX, OZEM, JEDI, DRNZ, HACK, IGV, BOTZ, QTUM, IBIT, WGMI, GDX, SIL, XME, SLX, TAN, XBI, IDNA, IYT, JETS, XHB, KRE, ITA, KWEB, XLE, OIH, IHI"

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎯 Watchlist")
    group_choice = st.radio("Choose Group:", ["Industry Themes", "Sita Hub Manager"])
    
    if group_choice == "Sita Hub Manager":
        hub_data = load_from_hub()
        selected_list = st.selectbox("Saved in Sita Hub:", ["Create New..."] + list(hub_data.keys()))
        initial_val = hub_data.get(selected_list, "AAPL, MSFT, GOOGL")
        tickers_input = st.text_area("Edit Tickers:", value=initial_val, height=150)
    else:
        tickers_input = st.text_area("Ticker Heap:", value=INDUSTRY_THEMES, height=150)

    st.markdown("---")
    st.header("⚙️ Engine Settings")
    scanner_speed = st.select_slider("Scanner Speed:", options=["Fast (Swing)", "Agile (Standard)", "Structural (Macro)"], value="Agile (Standard)")
    main_timeframe = st.radio("Display Chart Timeframe:", ["Weekly", "Daily"], index=0)

    if scanner_speed == "Fast (Swing)": d_look, w_look = 5, 5
    elif scanner_speed == "Agile (Standard)": d_look, w_look = 6, 6
    else: d_look, w_look = 14, 14
        
    active_lookback = d_look if main_timeframe == "Daily" else w_look
    benchmark = st.text_input("Active Benchmark:", value="SPY")
    tail_len = st.slider("Tail Length:", 2, 30, active_lookback)
    
    if st.button("♻️ Reset Engine"):
        st.cache_data.clear()
        st.rerun()

# --- ANALYTICS ENGINE ---
@st.cache_data(ttl=600)
def download_data(tickers, interval):
    data = yf.download(tickers, period="2y", interval=interval, progress=False)
    return data['Close'] if not data.empty else None

def get_rrg_metrics(df_raw, ticker, bench_t, lookback_val):
    if df_raw is None or ticker not in df_raw.columns: return None
    try:
        px = df_raw[ticker].dropna()
        bx = df_raw[bench_t].dropna()
        common = px.index.intersection(bx.index)
        if len(common) < lookback_val + 5: return None
        rel = ((px.loc[common] / bx.loc[common]) * 100).ewm(span=3).mean() 
        def standardize(s): return RRG_CENTER + ((s - s.rolling(lookback_val).mean()) / s.rolling(lookback_val).std().replace(0, EPSILON))
        ratio, mom = standardize(rel).clip(*Z_LIMITS), standardize(rel.diff(1)).clip(*Z_LIMITS)
        df_res = pd.DataFrame({'x': ratio, 'y': mom, 'date': ratio.index}).dropna()
        return df_res
    except: return None

# --- UI TABS ---
tab1, tab2 = st.tabs(["🌀 Relative Rotation (RRG)", "🏦 Capital Flow Leaders"])

tickers_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
full_ticker_set = list(set(tickers_list + [benchmark.upper()]))
interval = "1d" if main_timeframe == "Daily" else "1wk"
data_all = download_data(full_ticker_set, interval)

with tab1:
    if data_all is not None:
        col_t1, col_t2 = st.columns([1, 4])
        with col_t1: show_all = st.checkbox("Show All Tickers", value=True)
        to_plot = st.multiselect("Active Plotters:", options=tickers_list, default=tickers_list if show_all else [])
        
        fig = go.Figure()
        fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="rgba(0,0,0,0.3)", dash="dot"))
        fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="rgba(0,0,0,0.3)", dash="dot"))
        fig.add_vrect(x0=POWER_WALK_LEVEL, x1=CHART_RANGE[1], fillcolor="#2ECC71", opacity=0.1, layer="below")
        
        for i, t in enumerate(to_plot):
            res = get_rrg_metrics(data_all, t, benchmark.upper(), active_lookback)
            if res is not None:
                df_p = res.iloc[-min(tail_len, len(res)):]
                color = px.colors.qualitative.Alphabet[i % 26]
                fig.add_trace(go.Scatter(x=df_p['x'], y=df_p['y'], mode='lines+markers', marker=dict(size=6, color=color, opacity=0.5), line=dict(color=color, width=2, shape='spline'), legendgroup=t, showlegend=False))
                fig.add_trace(go.Scatter(x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], mode='markers+text', marker=dict(symbol='diamond', size=14, color=color, line=dict(width=1.5, color='white')), text=[f"<b>{t}</b>"], textposition="top center", legendgroup=t, name=t))
        
        fig.update_layout(template="plotly_white", height=800, xaxis=dict(range=CHART_RANGE, title="RS-Ratio"), yaxis=dict(range=CHART_RANGE, title="RS-Momentum"))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🏦 Capital Flow Scorecard (Weekly)")
    if data_all is not None:
        flow_data = []
        for t in tickers_list:
            if t in data_all.columns:
                px = data_all[t].dropna()
                bx = data_all[benchmark.upper()].dropna()
                common = px.index.intersection(bx.index)
                
                # Formula: (Trend Distance + RS Momentum) normalized to 0-100
                sma20 = px.loc[common].rolling(20).mean()
                trend_dist = (px.loc[common].iloc[-1] / sma20.iloc[-1]) - 1
                rel_s = (px.loc[common] / bx.loc[common])
                rs_mom = (rel_s.iloc[-1] / rel_s.iloc[-5]) - 1
                
                score = int(100 / (1 + np.exp(-((trend_dist * 350) + (rs_mom * 450)))))
                flow_data.append({"Ticker": t, "Name": TICKER_NAMES.get(t, t), "Flow Score": score, "Trend %": round(trend_dist*100, 2), "Status": "🔥 ACCUMULATION" if score > 80 else "⚖️ HOLD" if score > 40 else "⚠️ DISTRIBUTION"})
        
        st.dataframe(pd.DataFrame(flow_data).sort_values("Flow Score", ascending=False), use_container_width=True)
