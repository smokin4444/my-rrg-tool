import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as p_express 
import time

# --- CONFIG ---
LOOKBACK = 14
RRG_CENTER = 100
EPSILON = 1e-8
CHART_RANGE = [96.5, 103.5]
POWER_WALK_LEVEL = 101.5

st.set_page_config(page_title="Alpha-Scanner Pro", layout="wide")

# --- TICKER DICTIONARY ---
TICKER_NAMES = {
    "GC=F": "Gold Futures", "SI=F": "Silver Futures", "HG=F": "Copper Futures", "CL=F": "Crude Oil",
    "BZ=F": "Brent Oil", "NG=F": "Natural Gas", "PL=F": "Platinum", "PA=F": "Palladium",
    "TIO=F": "Iron Ore", "ALB": "Albemarle (Lithium)", "ZS=F": "Soybeans", "MOO": "Agribusiness",
    "OIH": "Oilfield Services", "URNM": "Uranium Miners", "SPY": "S&P 500", "QQQ": "Nasdaq 100"
}

# --- WATCHLISTS ---
HARD_ASSETS = "GC=F, SI=F, HG=F, CL=F, BZ=F, NG=F, PL=F, PA=F, TIO=F, ALB, URNM, ZS=F, MOO, OIH"
MAJOR_THEMES = "SPY, QQQ, DIA, IWF, IWD, MAGS, IWM, IJR, GLD, SLV, COPX, XLE, IBIT, IGV, XLP, XLRE, ARKK, TLT, UUP, XME"

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎯 Watchlist")
    group_choice = st.radio("Group:", ["Hard Assets", "Major Themes", "TV Industries", "Income Stocks", "Custom"])
    current_list = {"Hard Assets": HARD_ASSETS, "Major Themes": MAJOR_THEMES}.get(group_choice, "")
    auto_bench = "ONE" if group_choice in ["Hard Assets", "Income Stocks"] else "SPY"
    
    tickers_input = st.text_area("Tickers:", value=current_list, height=150)
    benchmark = st.text_input("Benchmark:", value=auto_bench)
    main_timeframe = st.radio("Timeframe:", ["Weekly", "Daily"], index=0)
    tail_len = st.slider("Tail Length:", 2, 30, 3)
    
    if st.button("♻️ Reset Engine"):
        st.cache_data.clear()
        st.rerun()

# --- REFINED FAIL-SAFE FETCH (FUTURES COMPATIBLE) ---
def fetch_safe(ticker, tf):
    try:
        # We download slightly more data to ensure ffill has context
        data = yf.download(ticker, period="3y", interval=("1wk" if tf == "Weekly" else "1d"), 
                           progress=False, auto_adjust=False)
        if data.empty: return None
        
        # Determine best column and forward-fill gaps (crucial for Futures)
        target_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        
        if isinstance(data.columns, pd.MultiIndex):
            series = data[target_col][ticker]
        else:
            series = data[target_col]
            
        return series.ffill().dropna() # Fill gaps then drop initial NAs
    except: return None

def calculate_rrg(price_s, bench_s, ticker):
    try:
        # Re-align series to ensure identical timestamps
        common = price_s.index.intersection(bench_s.index)
        if len(common) < 30: return None # Increased buffer for futures
        
        p, b = price_s.loc[common], bench_s.loc[common]
        rel = (p / b) * 100
        ratio_s = rel.ewm(span=3).mean()
        
        # X Axis
        x = RRG_CENTER + ((ratio_s - ratio_s.rolling(LOOKBACK).mean()) / ratio_s.rolling(LOOKBACK).std().replace(0, EPSILON))
        # Y Axis
        y_raw = x.diff(1).ewm(span=3).mean()
        y = RRG_CENTER + (y_raw * 5)
        
        df = pd.DataFrame({'x': x, 'y': y, 'date': x.index}).dropna()
        df['full_name'] = TICKER_NAMES.get(ticker, ticker)
        return df
    except: return None

# --- RUN ---
try:
    ticker_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    is_abs = benchmark.upper() == "ONE"
    
    with st.spinner("Fetching Macro Data..."):
        # Absolute ONE creates a 1.0 baseline
        bench_data = pd.Series(1.0, index=pd.date_range(end=pd.Timestamp.now(), periods=1000)) if is_abs else fetch_safe(benchmark, main_timeframe)
        # Required for the Sync Table
        bench_d = pd.Series(1.0, index=pd.date_range(end=pd.Timestamp.now(), periods=1000)) if is_abs else fetch_safe(benchmark, "Daily")
        bench_w = pd.Series(1.0, index=pd.date_range(end=pd.Timestamp.now(), periods=1000)) if is_abs else fetch_safe(benchmark, "Weekly")

    if bench_data is not None:
        results_map = {}
        prog = st.progress(0)
        
        for i, t in enumerate(ticker_list):
            p_display = fetch_safe(t, main_timeframe)
            if p_display is not None:
                m_display = calculate_rrg(p_display, bench_data, t)
                
                # Table Sync calculations
                p_d, p_w = fetch_safe(t, "Daily"), fetch_safe(t, "Weekly")
                m_daily = calculate_rrg(p_d, bench_d, t) if p_d is not None and bench_d is not None else None
                m_weekly = calculate_rrg(p_w, bench_w, t) if p_w is not None and bench_w is not None else None
                
                if m_display is not None:
                    results_map[t] = {'chart': m_display, 'daily': m_daily, 'weekly': m_weekly}
            prog.progress((i + 1) / len(ticker_list))
        prog.empty()

        if results_map:
            st.subheader(f"🌀 {main_timeframe} Rotation vs {benchmark}")
            fig = go.Figure()
            fig.add_vrect(x0=POWER_WALK_LEVEL, x1=CHART_RANGE[1], fillcolor="rgba(46, 204, 113, 0.1)", layer="below", line_width=0)
            
            for idx, (t, data) in enumerate(results_map.items()):
                df_p = data['chart'].iloc[-tail_len:]
                color = p_express.colors.qualitative.Alphabet[idx % 26]
                fig.add_trace(go.Scatter(x=df_p['x'], y=df_p['y'], mode='lines', line=dict(color=color, width=2.5), showlegend=False))
                fig.add_trace(go.Scatter(x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], mode='markers+text', 
                                         text=[f"<b>{t}</b>"], textposition="top center",
                                         marker=dict(symbol='diamond', size=14, color=color, line=dict(width=1, color='white')), name=t))

            fig.update_layout(template="plotly_white", height=750, xaxis=dict(range=CHART_RANGE), yaxis=dict(range=[96.5, 103.5]))
            st.plotly_chart(fig, use_container_width=True)
            
            # Simple results table to verify data is flowing
            table_rows = [{"Ticker": t, "Ratio": round(data['chart']['x'].iloc[-1], 2)} for t, data in results_map.items()]
            st.dataframe(pd.DataFrame(table_rows).sort_values("Ratio", ascending=False), use_container_width=True)
        else:
            st.warning("No data found. If this is a futures group, try hitting Reset Engine or wait 30 seconds.")
except Exception as e:
    st.error(f"Error: {e}")
