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
RRG_CENTER, EPSILON = 100, 1e-8
Z_LIMITS = (80, 120)  
CHART_RANGE = [96.5, 103.5] 

# --- TICKER DICTIONARY ---
TICKER_NAMES = {
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100", "DIA": "Dow Jones", "IWF": "Growth Stocks", 
    "IWD": "Value Stocks", "MAGS": "Magnificent 7", "IWM": "Small Caps", 
    "GLD": "Gold ETF", "SLV": "Silver ETF", "COPX": "Global Copper Miners", "XLE": "Energy",
    "XLK": "Technology", "XLY": "Consumer Durables", "XLC": "Communications", 
    "XLF": "Finance", "XLI": "Producer Manufacturing", "XLV": "Health Services", 
    "XLP": "Cons Staples", "XLU": "Utilities", "XLB": "Materials (Broad)", 
    "IYT": "Transportation", "SMH": "Semiconductors (NVDA)", "GEV": "GE Vernova (Power)",
    "BDRY": "Dry Bulk Shipping", "BOAT": "Global Shipping ETF", "JEDI": "Modern Warfare",
    "POWR": "U.S. Power/Grid Infra", "URNM": "Uranium Miners (Nuclear)", "KWEB": "China Internet"
}

# --- WATCHLIST GROUPS ---
MAJOR_THEMES = "SPY, QQQ, DIA, IWF, IWD, MAGS, IWM, GLD, SLV, COPX, XLE, IBIT, IGV, XLP, XLRE, ARKK, TLT, UUP, XME, SMH, SOXX, FTXL"
INDUSTRY_THEMES = "SMH, GEV, COPX, URNM, BOAT, BDRY, POWR, PAVE, REMX, OZEM, JEDI, DRNZ, HACK, IGV, BOTZ, QTUM, IBIT, WGMI, GDX, SIL, XME, SLX, TAN, XBI, IDNA, IYT, JETS, XHB, KRE, ITA, KWEB, XLE, OIH, IHI"
INTL_COUNTRIES = "THD, EWZ, EWY, EWT, EWG, EWJ, EWC, EWW, EPU, ECH, ARGT, EZA, EIDO, EWM, EWP, EWL, EWQ, EWU, EWH, INDA, EWA"
HARD_ASSETS = "GC=F, SI=F, HG=F, CL=F, BZ=F, NG=F, PL=F, PA=F, TIO=F, ALB, URNM, ZS=F, MOO, OIH"

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎯 Watchlist")
    group_choice = st.radio("Choose Group:", ["Major Themes", "Industry Themes", "International Countries", "Hard Assets"])
    tickers_input = {
        "Major Themes": MAJOR_THEMES, "Industry Themes": INDUSTRY_THEMES,
        "International Countries": INTL_COUNTRIES, "Hard Assets": HARD_ASSETS
    }.get(group_choice, "")
    tickers_input = st.text_area("Ticker Heap:", value=tickers_input, height=150)
    st.markdown("---")
    st.header("⚙️ Engine Settings")
    scanner_speed = st.select_slider("Scanner Speed:", options=["Fast (Swing)", "Agile (Standard)", "Structural (Macro)"], value="Agile (Standard)")
    main_timeframe = st.radio("Display Timeframe:", ["Weekly", "Daily"], index=0)
    if scanner_speed == "Fast (Swing)": d_look, w_look = 5, 5
    elif scanner_speed == "Agile (Standard)": d_look, w_look = 6, 6
    else: d_look, w_look = 14, 14
    active_lookback = d_look if main_timeframe == "Daily" else w_look
    auto_bench = "ONE" if group_choice == "Hard Assets" else "SPY"
    benchmark = st.text_input("Benchmark:", value=auto_bench)
    tail_len = st.slider("Tail Length:", 2, 30, active_lookback)

# --- ANALYTICS ENGINE ---
@st.cache_data(ttl=600)
def download_data(tickers, interval):
    data = yf.download(tickers, period="2y", interval=interval, progress=False)
    return data['Close'] if not data.empty else None

def get_rrg_metrics(df_raw, ticker, bench_t, lookback_val):
    if df_raw is None or ticker not in df_raw.columns: return None
    try:
        px = df_raw[ticker].dropna()
        bx = pd.Series(1.0, index=px.index) if bench_t.upper() == "ONE" else df_raw[bench_t].dropna()
        common = px.index.intersection(bx.index)
        if len(common) < lookback_val + 5: return None
        rel = ((px.loc[common] / bx.loc[common]) * 100).ewm(span=3).mean() 
        def stdz(s): return RRG_CENTER + ((s - s.rolling(lookback_val).mean()) / s.rolling(lookback_val).std().replace(0, EPSILON))
        ratio, mom = stdz(rel).clip(*Z_LIMITS), stdz(rel.diff(1)).clip(*Z_LIMITS)
        return pd.DataFrame({'x': ratio, 'y': mom, 'date': ratio.index}).dropna()
    except: return None

# --- UI LOGIC ---
tickers_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
full_ticker_set = list(set(tickers_list + ([benchmark.upper()] if benchmark.upper() != "ONE" else [])))
interval = "1d" if main_timeframe == "Daily" else "1wk"
data_all = download_data(full_ticker_set, interval)

tab1, tab2 = st.tabs(["🌀 Relative Rotation (RRG)", "🏦 Capital Flow Leaders"])

with tab1:
    if data_all is not None:
        to_plot = st.multiselect("Active Plotters:", options=tickers_list, default=tickers_list)
        fig = go.Figure()
        fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="rgba(0,0,0,0.3)", dash="dot"))
        fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="rgba(0,0,0,0.3)", dash="dot"))
        
        for i, t in enumerate(to_plot):
            res = get_rrg_metrics(data_all, t, benchmark.upper(), active_lookback)
            if res is not None:
                df_p = res.iloc[-min(tail_len, len(res)):]
                color = px.colors.qualitative.Alphabet[i % 26]
                fig.add_trace(go.Scatter(x=df_p['x'], y=df_p['y'], mode='lines+markers', marker=dict(size=6, color=color, opacity=0.5), line=dict(color=color, width=2, shape='spline'), legendgroup=t, showlegend=False))
                fig.add_trace(go.Scatter(x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], mode='markers+text', marker=dict(symbol='diamond', size=14, color=color, line=dict(width=1.5, color='white')), text=[f"<b>{t}</b>"], textposition="top center", legendgroup=t, name=t))
        fig.update_layout(template="plotly_white", height=800, xaxis=dict(range=CHART_RANGE, title="RS-Ratio"), yaxis=dict(range=CHART_RANGE, title="RS-Momentum"))
        st.plotly_chart(fig, use_container_width=True)

        # --- RESTORED MINI-GRID UNDER RRG ---
        st.subheader("📋 Snapshot Grid")
        snap_data = []
        for t in tickers_list:
            res = get_rrg_metrics(data_all, t, benchmark.upper(), active_lookback)
            if res is not None:
                x, y = res['x'].iloc[-1], res['y'].iloc[-1]
                stg = "LEADING" if x>=100 and y>=100 else "IMPROVING" if x<100 and y>=100 else "LAGGING" if x<100 and y<100 else "WEAKENING"
                snap_data.append({"Ticker": t, "Stage": stg, "Ratio": round(x, 2), "Momentum": round(y, 2)})
        st.dataframe(pd.DataFrame(snap_data), use_container_width=True)

with tab2:
    st.subheader("🏦 Capital Flow Scorecard")
    if data_all is not None:
        flow_data = []
        for t in tickers_list:
            if t in data_all.columns:
                px = data_all[t].dropna()
                bx = pd.Series(1.0, index=px.index) if benchmark.upper() == "ONE" else data_all[benchmark.upper()].dropna()
                common = px.index.intersection(bx.index)
                if len(common) > 20:
                    sma20 = px.loc[common].rolling(20).mean()
                    trend_dist = (px.loc[common].iloc[-1] / sma20.iloc[-1]) - 1
                    rel_s = (px.loc[common] / bx.loc[common])
                    rs_mom = (rel_s.iloc[-1] / rel_s.iloc[-5]) - 1
                    
                    # Anti-Crash + Stretching the Score
                    trend_dist = 0 if np.isnan(trend_dist) else trend_dist
                    rs_mom = 0 if np.isnan(rs_mom) else rs_mom
                    
                    # --- NEW NUANCED STRETCH MATH ---
                    raw_blend = (trend_dist * 1.5) + (rs_mom * 2.5)
                    # Increased denominator from 0.20 to 0.40 to stop the 100/0 clumping
                    score = int(np.clip((raw_blend + 0.20) / 0.40, 0, 1) * 100)
                    
                    flow_data.append({"Ticker": t, "Name": TICKER_NAMES.get(t, t), "Flow Score": score, "Trend %": round(trend_dist*100, 1), "RS Δ": round(rs_mom*100, 1), "Status": "🔥 ACCUMULATION" if score > 80 else "⚖️ HOLD" if score > 40 else "⚠️ DISTRIBUTION"})
        
        st.dataframe(pd.DataFrame(flow_data).sort_values("Flow Score", ascending=False), use_container_width=True)
