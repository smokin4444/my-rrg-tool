import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time

# --- CONFIGURATION & CONSTANTS ---
LOOKBACK = 14
RRG_CENTER = 100
EPSILON = 1e-8
Z_LIMITS = (80, 120)  
CHART_RANGE = [96.5, 103.5] 
POWER_WALK_LEVEL = 101.5

st.set_page_config(page_title="Alpha-Scanner Pro", layout="wide")

# --- TICKER DICTIONARY (Truncated for brevity, keep your full list) ---
TICKER_NAMES = {
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100", "XLP": "Cons Staples", "XLRE": "Real Estate",
    "UUP": "US Dollar Index", "TLT": "20+Y Treasury Bonds", "XME": "S&P Metals & Mining",
    "BDRY": "Dry Bulk Shipping", "MOO": "Agribusiness", "OIH": "Oilfield Services"
}

# --- WATCHLISTS (Ensure these match your full lists) ---
MAJOR_THEMES = "SPY, QQQ, DIA, IWF, IWD, MAGS, IWM, IJR, GLD, SLV, COPX, XLE, IBIT, IGV, XLP, XLRE, ARKK, TLT, UUP, XME"
# ... [Keep your other lists here]

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎯 Watchlist")
    heap_type = st.radio("Choose Group:", ["Major Themes", "Sector Rotation", "Energy Torque", "Startup", "Tech Themes", "Hard Assets (Live)", "TV Industries (Full)", "Income Stocks", "Single/Custom"])
    
    # Logic to populate current_list...
    current_list = MAJOR_THEMES # Simplified for this snippet
    auto_bench = "ONE" if heap_type in ["Hard Assets (Live)", "Income Stocks"] else "SPY"
    
    tickers_input = st.text_area("Ticker Heap:", value=current_list, height=150)
    benchmark = st.text_input("Active Benchmark:", value=auto_bench)
    
    st.markdown("---")
    main_timeframe = st.radio("Display Chart Timeframe:", ["Weekly", "Daily"], index=0)
    tail_len = st.slider("Tail Length:", 2, 30, 3)
    
    if st.button("♻️ Reset Engine"):
        st.cache_data.clear()
        st.rerun()

# --- REFINED TREND-SYNC ENGINE ---
@st.cache_data(ttl=600)
def download_data(tickers, interval):
    period = "2y"
    try:
        data = yf.download(tickers, period=period, interval=interval, progress=False, auto_adjust=True)
        return data
    except Exception as e:
        st.error(f"Download Error: {e}")
        return pd.DataFrame()

def get_metrics(df_raw, ticker, bench_t, is_absolute):
    try:
        if isinstance(df_raw.columns, pd.MultiIndex):
            px = df_raw['Close'][ticker].dropna()
            bx = pd.Series(1.0, index=px.index) if is_absolute else df_raw['Close'][bench_t].dropna()
        else:
            px = df_raw['Close'].dropna()
            bx = pd.Series(1.0, index=px.index)
            
        common = px.index.intersection(bx.index)
        px_a, bx_a = px.loc[common], bx.loc[common]
        
        # 1. Smoothed Ratio (The X Axis)
        rel = (px_a / bx_a) * 100
        # We use a 3-period EMA to match the "Weekly 3-bar" feel and reduce lag
        ratio_smoothed = rel.ewm(span=3).mean()
        
        # Standardize Ratio
        ratio = RRG_CENTER + ((ratio_smoothed - ratio_smoothed.rolling(LOOKBACK).mean()) / ratio_smoothed.rolling(LOOKBACK).std().replace(0, EPSILON))
        
        # 2. Pure Momentum (The Y Axis)
        # Instead of Z-Score of the ratio, we look at the SLOPE (Rate of Change)
        # This keeps the tail pointing NE as long as the ratio is rising
        mom_raw = ratio.diff(1)
        # Smooth the momentum so the tail doesn't "jitter"
        mom_smoothed = mom_raw.ewm(span=3).mean()
        
        # Standardize Momentum to the 100 center
        momentum = RRG_CENTER + (mom_smoothed * 5) # Multiplier to amplify the "Heading"
        
        df_res = pd.DataFrame({'x': ratio, 'y': momentum, 'date': ratio.index}).dropna()
        df_res['date_str'] = df_res['date'].dt.strftime('%b %d, %Y')
        df_res['full_name'] = TICKER_NAMES.get(ticker, ticker)
        return df_res.clip(lower=80, upper=120)
    except Exception: return None

# --- [The rest of the run_dual_analysis and display code remains the same] ---
# ... (run_dual_analysis, get_stage, and plotly display logic)
