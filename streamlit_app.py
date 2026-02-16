import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# --- CONFIG ---
LOOKBACK = 14
RRG_CENTER = 100
EPSILON = 1e-8
CHART_RANGE = [96.5, 103.5]
POWER_WALK_LEVEL = 101.5

st.set_page_config(page_title="Alpha-Scanner Pro", layout="wide")

# --- TICKER DICTIONARY (Full) ---
TICKER_NAMES = {
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100", "DIA": "Dow Jones", "IWF": "Growth Stocks", 
    "IWD": "Value Stocks", "MAGS": "Magnificent 7", "IWM": "Small Caps", 
    "GLD": "Gold", "SLV": "Silver", "COPX": "Copper Miners", "XLE": "Energy",
    "XLK": "Technology", "XLY": "Consumer Durables", "XLC": "Communications", 
    "XLF": "Finance", "XLI": "Producer Manufacturing", 
    "XLV": "Health Services", "XLP": "Cons Staples", "XLU": "Utilities", 
    "XLB": "Materials (Broad)", "IYT": "Transportation", "PICK": "Non-Energy Minerals", 
    "URNM": "Energy Minerals", "OII": "Oilfield Services", "VAW": "Process Industries", 
    "SMH": "Electronic Tech", "IGV": "Software", "IBB": "Health Tech", "XHB": "Cons. Durables",
    "PEJ": "Consumer Services", "XRT": "Retail Trade", "IYZ": "Communications", "VNQ": "Invest Trusts",
    "VTI": "Misc/Broad", "IBIT": "Bitcoin Trust", "FAST": "Distribution", "IHE": "Pharma",
    "XES": "Contract Drilling", "FLR": "Eng. & Construction", "XME": "S&P Metals & Mining",
    "BDRY": "Dry Bulk Shipping", "MOO": "Agribusiness", "OIH": "Oilfield Services",
    "ARKK": "Innovation (High Beta)", "TLT": "20+Y Treasury Bonds", "UUP": "US Dollar Index",
    "QDVO": "YieldMax QQQ", "FEPI": "Rex FANG", "KGLD": "YieldMax Gold", "SOXY": "YieldMax Semi",
    "USCL.TO": "Horizon US Large Cap", "BANK.TO": "Evolve Cdn Banks"
}

# --- WATCHLISTS ---
MAJOR_THEMES = "SPY, QQQ, DIA, IWF, IWD, MAGS, IWM, IJR, GLD, SLV, COPX, XLE, IBIT, IGV, XLP, XLRE, ARKK, TLT, UUP, XME"
TV_INDUSTRIES = "XES, OIH, FLR, EVX, AMLP, VTI, TTD, VPP, SPGI, MAN, WSC, SYY, AVT, MCK, FI, ACN, IGV, FDN, UNH, THC, HCA, IQV, DIS, NXST, CHTR, NYT, EATZ, CRUZ, BETZ, PEJ, KR, CVS, M, WMT, NKE, HD, BBY, TSCO, ONLN, IYT, XLU, XLF, IYZ, XLI, VAW, SMH, IBB, XHB, XLP, XRT, BDRY"
INCOME_STOCKS = "QDVO, CEFS, MLPX, AMLP, PBDC, PFFA, RLTY, UTF, ARCC, MAIN, FEPI, BSK, STK, BTCI, MSTY, USCL.TO, BANK.TO, KGLD, CHPY, SOXY"
HARD_ASSETS = "GC=F, SI=F, HG=F, CL=F, BZ=F, NG=F, PL=F, PA=F, TIO=F, ALB, URNM, ZS=F, MOO, OIH"

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎯 Watchlist")
    heap_type = st.radio("Group:", ["Major Themes", "TV Industries", "Hard Assets", "Income Stocks", "Custom"])
    
    current_list = {"Major Themes": MAJOR_THEMES, "TV Industries": TV_INDUSTRIES, "Hard Assets": HARD_ASSETS, "Income Stocks": INCOME_STOCKS, "Custom": ""}.get(heap_type)
    auto_bench = "ONE" if heap_type in ["Hard Assets", "Income Stocks"] else "SPY"
    
    tickers_input = st.text_area("Tickers:", value=current_list, height=150)
    benchmark = st.text_input("Benchmark:", value=auto_bench)
    
    main_timeframe = st.radio("Timeframe:", ["Weekly", "Daily"], index=0)
    tail_len = st.slider("Tail Length:", 2, 30, 3)
    
    if st.button("♻️ Reset Engine"):
        st.cache_data.clear()
        st.rerun()

# --- HELPER: ROBUST FETCH ---
def fetch_clean(ticker, tf):
    try:
        data = yf.download(ticker, period="2y", interval=("1wk" if tf == "Weekly" else "1d"), progress=False, auto_adjust=True)
        if data.empty: return None
        # Extract only the Close column as a Series
        if isinstance(data.columns, pd.MultiIndex):
            return data['Close'][ticker]
        return data['Close']
    except: return None

# --- ENGINE ---
def get_rrg_metrics(px, bx, ticker):
    try:
        common = px.index.intersection(bx.index)
        p, b = px.loc[common], bx.loc[common]
        
        # JdK Smoothing logic
        rel = (p / b) * 100
        ratio_s = rel.ewm(span=3).mean()
        
        # RS-Ratio (X)
        x = RRG_CENTER + ((ratio_s - ratio_s.rolling(LOOKBACK).mean()) / ratio_s.rolling(LOOKBACK).std().replace(0, EPSILON))
        
        # RS-Momentum (Y) - Slope based for XLP Sync
        y_raw = x.diff(1).ewm(span=3).mean()
        y = RRG_CENTER + (y_raw * 5)
        
        df = pd.DataFrame({'x': x, 'y': y, 'date': x.index}).dropna()
        df['date_str'] = df['date'].dt.strftime('%b %d')
        df['full_name'] = TICKER_NAMES.get(ticker, ticker)
        return df.iloc[-50:] # Limit data for speed
    except: return None

# --- EXECUTION ---
try:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    is_abs = benchmark.upper() == "ONE"
    
    # 1. Get Benchmark
    with st.spinner("Fetching Benchmark..."):
        bx = pd.Series(1.0, index=pd.date_range(end=pd.Timestamp.now(), periods=500)) if is_abs else fetch_clean(benchmark, main_timeframe)
    
    if bx is not None:
        hist = {}
        prog = st.progress(0)
        
        # 2. Individual Fetch Loop
        for i, t in enumerate(tickers):
            px = fetch_clean(t, main_timeframe)
            if px is not None:
                m = get_rrg_metrics(px, bx, t)
                if m is not None: hist[t] = m
            prog.progress((i + 1) / len(tickers))
        prog.empty()

        # 3. Plotting
        if hist:
            st.subheader(f"🌀 {main_timeframe} Rotation vs {benchmark}")
            fig = go.Figure()
            
            # Shading & Lines
            fig.add_vrect(x0=POWER_WALK_LEVEL, x1=CHART_RANGE[1], fillcolor="green", opacity=0.05, layer="below", line_width=0)
            fig.add_shape(type="line", x0=100, y0=80, x1=100, y1=120, line=dict(color="gray", dash="dot", width=1))
            fig.add_shape(type="line", x0=80, y0=100, x1=120, y1=100, line=dict(color="gray", dash="dot", width=1))
            
            # Ticker Tails
            for t, df in hist.items():
                df_p = df.iloc[-tail_len:]
                color = px.colors.qualitative.Plotly[list(hist.keys()).index(t) % 10]
                
                # Trail
                fig.add_trace(go.Scatter(x=df_p['x'], y=df_p['y'], mode='lines', line=dict(color=color, width=2), showlegend=False))
                # Head
                fig.add_trace(go.Scatter(x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], 
                                         mode='markers+text', text=[f"<b>{t}</b>"], textposition="top center",
                                         marker=dict(symbol='diamond', size=12, color=color, line=dict(width=1, color='white')), name=t))

            fig.update_layout(template="plotly_white", height=700, xaxis=dict(range=CHART_RANGE, title="RS-Ratio"), yaxis=dict(range=[96.5, 103.5], title="RS-Momentum"))
            st.plotly_chart(fig, use_container_width=True)
            
            # Simple Grid Table
            grid_data = [{"Ticker": t, "Name": TICKER_NAMES.get(t, t), "Ratio": round(df['x'].iloc[-1], 2)} for t, df in hist.items()]
            st.dataframe(pd.DataFrame(grid_data).sort_values("Ratio", ascending=False), use_container_width=True)
        else:
            st.warning("No data found for the selected tickers. Try hitting Reset Engine.")
    else:
        st.error(f"Failed to load benchmark: {benchmark}")

except Exception as e:
    st.error(f"Critical System Error: {e}")
