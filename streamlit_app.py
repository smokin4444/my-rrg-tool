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
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100", "SMH": "Semiconductors", "GEV": "GE Vernova (Power)",
    "COPX": "Copper Miners", "URNM": "Uranium Miners", "BOAT": "Global Shipping", "BDRY": "Dry Bulk",
    "POWR": "US Grid Infra", "PAVE": "US Infrastructure", "REMX": "Rare Earths", "OZEM": "Weight Loss",
    "JEDI": "Modern Warfare", "DRNZ": "Drones", "XLE": "Energy", "OIH": "Oil Services", "KWEB": "China Internet"
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
    main_timeframe = st.radio("Timeframe:", ["Weekly", "Daily"], index=0)
    
    d_look, w_look = (5, 5) if scanner_speed == "Fast (Swing)" else (6, 6) if scanner_speed == "Agile (Standard)" else (14, 14)
    active_lookback = d_look if main_timeframe == "Daily" else w_look
    benchmark = st.text_input("Benchmark:", value="SPY")

# --- DATA ENGINE ---
@st.cache_data(ttl=600)
def download_data(tickers, interval):
    data = yf.download(tickers, period="1y", interval=interval, progress=False)
    return data['Close'] if not data.empty else None

# --- FLOW SCORE CALCULATION ---
def calculate_flow_score(df_raw, ticker, bench_ticker):
    try:
        px = df_raw[ticker].dropna()
        bx = df_raw[bench_ticker].dropna()
        common = px.index.intersection(bx.index)
        
        # 1. Trend Score (Distance from SMA)
        sma20 = px.loc[common].rolling(20).mean()
        trend_val = (px.loc[common].iloc[-1] / sma20.iloc[-1]) - 1
        
        # 2. RS Score (Ticker vs Bench)
        rel_strength = (px.loc[common] / bx.loc[common])
        rs_mom = (rel_strength.iloc[-1] / rel_strength.iloc[-5]) - 1
        
        # 3. Composite Calculation (Scaled 0-100)
        # Using a sigmoid-style normalization to keep it 0-100
        raw_score = (trend_val * 400) + (rs_mom * 500)
        final_score = int(100 / (1 + np.exp(-(raw_score))))
        
        return final_score, round(trend_val * 100, 2), round(rs_mom * 100, 2)
    except: return 50, 0, 0

# --- MAIN UI TABS ---
tab1, tab2 = st.tabs(["🌀 Relative Rotation (RRG)", "🏦 Capital Flow Leaders"])

# Shared download
tickers_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
full_ticker_set = list(set(tickers_list + [benchmark.upper()]))
interval = "1d" if main_timeframe == "Daily" else "1wk"
df_main = download_data(full_ticker_set, interval)

with tab1:
    st.subheader(f"Relative Rotation: {scanner_speed}")
    # (Existing RRG Logic goes here - plotting markers and tails)
    st.info("The RRG helps you visualize the 'rotation' of these sectors.")
    
with tab2:
    st.subheader("Institutional Money Flow Scorecard")
    st.markdown("This score (0-100) measures where big money is currently concentrating. Scores above 80 indicate **Heavy Accumulation**.")
    
    if df_main is not None:
        flow_data = []
        for t in tickers_list:
            if t in df_main.columns:
                score, trend, rs = calculate_flow_score(df_main, t, benchmark.upper())
                flow_data.append({
                    "Ticker": t,
                    "Name": TICKER_NAMES.get(t, t),
                    "Flow Score": score,
                    "Trend Strength %": trend,
                    "Rel. Strength Δ": rs,
                    "Status": "🔥 ACCUMULATION" if score > 80 else "⚖️ HOLD" if score > 40 else "⚠️ DISTRIBUTION"
                })
        
        flow_df = pd.DataFrame(flow_data).sort_values("Flow Score", ascending=False)
        
        # Displaying with color coding
        def color_score(val):
            color = 'green' if val > 80 else 'red' if val < 40 else 'white'
            return f'color: {color}'

        st.dataframe(flow_df.style.applymap(color_score, subset=['Flow Score']), use_container_width=True)
        
        st.markdown("---")
        st.caption("Flow Score is a composite of Trend Distance and Relative Strength Momentum.")
