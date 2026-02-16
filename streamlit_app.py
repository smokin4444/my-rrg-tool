import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as p_express  # Renamed to avoid collision with price data
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
    "CHPY": "YieldMax China", "MSTY": "YieldMax MSTR", "USCL.TO": "Horizon US Large Cap", "BANK.TO": "Evolve Cdn Banks"
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

# --- HELPER: DATA FETCH ---
def fetch_data(ticker, tf):
    try:
        data = yf.download(ticker, period="2y", interval=("1wk" if tf == "Weekly" else "1d"), progress=False, auto_adjust=True)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            return data['Close'][ticker]
        return data['Close']
    except: return None

# --- ENGINE ---
def get_metrics(price_series, bench_series, ticker):
    try:
        common = price_series.index.intersection(bench_series.index)
        p, b = price_series.loc[common], bench_series.loc[common]
        rel = (p / b) * 100
        ratio_s = rel.ewm(span=3).mean()
        x = RRG_CENTER + ((ratio_s - ratio_s.rolling(LOOKBACK).mean()) / ratio_s.rolling(LOOKBACK).std().replace(0, EPSILON))
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
    
    with st.spinner("Initializing Benchmark..."):
        benchmark_series = pd.Series(1.0, index=pd.date_range(end=pd.Timestamp.now(), periods=500)) if is_abs else fetch_data(benchmark, main_timeframe)
    
    if benchmark_series is not None:
        hist_d = {}
        hist_w = {}
        prog = st.progress(0)
        
        for i, t in enumerate(ticker_list):
            # Fetch Daily for Table Sync
            p_d = fetch_data(t, "Daily")
            b_d = pd.Series(1.0, index=pd.date_range(end=pd.Timestamp.now(), periods=500)) if is_abs else fetch_data(benchmark, "Daily")
            
            # Fetch Display Timeframe for Chart
            p_display = fetch_data(t, main_timeframe)
            
            if p_display is not None:
                metrics_display = get_metrics(p_display, benchmark_series, t)
                metrics_daily = get_metrics(p_d, b_d, t)
                metrics_weekly = metrics_display if main_timeframe == "Weekly" else get_metrics(fetch_data(t, "Weekly"), (pd.Series(1.0, index=pd.date_range(end=pd.Timestamp.now(), periods=500)) if is_abs else fetch_data(benchmark, "Weekly")), t)
                
                if metrics_display is not None:
                    hist_d[t] = {
                        'display': metrics_display,
                        'daily': metrics_daily,
                        'weekly': metrics_weekly
                    }
            prog.progress((i + 1) / len(ticker_list))
        prog.empty()

        if hist_d:
            st.subheader(f"🌀 {main_timeframe} Rotation vs {benchmark}")
            fig = go.Figure()
            fig.add_vrect(x0=POWER_WALK_LEVEL, x1=CHART_RANGE[1], fillcolor="rgba(46, 204, 113, 0.1)", layer="below", line_width=0)
            fig.add_shape(type="line", x0=100, y0=80, x1=100, y1=120, line=dict(color="rgba(0,0,0,0.2)", dash="dot"))
            fig.add_shape(type="line", x0=80, y0=100, x1=120, y1=100, line=dict(color="rgba(0,0,0,0.2)", dash="dot"))
            
            for idx, (t, data) in enumerate(hist_d.items()):
                df = data['display'].iloc[-tail_len:]
                color = p_express.colors.qualitative.Alphabet[idx % 26]
                fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='lines', line=dict(color=color, width=2.5), showlegend=False))
                fig.add_trace(go.Scatter(x=[df['x'].iloc[-1]], y=[df['y'].iloc[-1]], mode='markers+text', 
                                         text=[f"<b>{t}</b>"], textposition="top center",
                                         marker=dict(symbol='diamond', size=14, color=color, line=dict(width=1, color='white')), name=t))

            fig.update_layout(template="plotly_white", height=750, xaxis=dict(range=CHART_RANGE, title="RS-Ratio"), yaxis=dict(range=[96.5, 103.5], title="RS-Momentum"))
            st.plotly_chart(fig, use_container_width=True)

            # Table Logic
            table_rows = []
            for t, data in hist_d.items():
                d_x, d_y = data['daily']['x'].iloc[-1], data['daily']['y'].iloc[-1] if data['daily'] is not None else (0,0)
                w_x, w_y = data['weekly']['x'].iloc[-1], data['weekly']['y'].iloc[-1] if data['weekly'] is not None else (0,0)
                
                def get_q(x, y):
                    if x >= 100 and y >= 100: return "LEADING"
                    if x < 100 and y >= 100: return "IMPROVING"
                    if x < 100 and y < 100: return "LAGGING"
                    return "WEAKENING"
                
                q_d, q_w = get_q(d_x, d_y), get_q(w_x, w_y)
                sync = "💎 BULLISH SYNC" if q_d == "LEADING" and q_w == "LEADING" else "📈 PULLBACK BUY" if q_d == "IMPROVING" and q_w == "LEADING" else "---"
                
                table_rows.append({"Ticker": t, "Full Name": TICKER_NAMES.get(t, t), "Sync Status": sync, "Daily": q_d, "Weekly": q_w, "Score": round(data['display']['x'].iloc[-1], 2)})
            
            st.subheader("📊 Dual-Timeframe Strategy")
            df_final = pd.DataFrame(table_rows).sort_values("Score", ascending=False)
            st.dataframe(df_final.style.applymap(lambda v: "background-color: #2ecc71; color: white" if v == "💎 BULLISH SYNC" else "background-color: #3498db; color: white" if v == "📈 PULLBACK BUY" else "", subset=["Sync Status"]), use_container_width=True)
except Exception as e:
    st.error(f"Critical System Error: {e}")
