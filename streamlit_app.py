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
    "BDRY": "Dry Bulk Shipping", "MOO": "Agribusiness", "EVX": "Environmental Svcs", 
    "AMLP": "Pipelines (MLP)", "TTD": "Ad-Tech/Marketing", "VPP": "Commercial Printing", 
    "SPGI": "Financial Services", "MAN": "Personnel Services", "WSC": "Wholesale Dist.", 
    "SYY": "Food Distribution", "AVT": "Electronics Dist.", "MCK": "Medical Distribution", 
    "FI": "Data Processing", "ACN": "IT Services", "FDN": "Internet Services", 
    "UNH": "Managed Health", "THC": "Hospital Mgmt", "HCA": "Medical Services", 
    "IQV": "Health Industry Svcs", "DIS": "Media Conglomerate", "NXST": "Broadcasting", 
    "CHTR": "Cable/Satellite TV", "NYT": "Publishing", "EATZ": "Restaurants", 
    "CRUZ": "Hotels/Cruises", "BETZ": "Casinos/Gaming", "KR": "Food Retail", 
    "CVS": "Drugstore Chains", "M": "Department Stores", "WMT": "Discount Stores", 
    "NKE": "Apparel Retail", "HD": "Home Improvement", "BBY": "Electronics Stores", 
    "TSCO": "Specialty Stores", "ONLN": "Internet Retail", "PSCT": "Tech (Small Cap)", 
    "PSCD": "Cons Disc (Small Cap)", "PSCF": "Financials (Small Cap)", 
    "PSCI": "Industrials (Small Cap)", "PSCH": "Health Care (Small Cap)", 
    "PSCC": "Cons Staples (Small Cap)", "PSCU": "Utilities (Small Cap)", 
    "PSCM": "Materials (Small Cap)", "PSCE": "Energy (Small Cap)", "XLRE": "Real Estate", 
    "XBI": "Biotech (S&P)", "ARKK": "Innovation (High Beta)", "TLT": "20+Y Treasury Bonds", 
    "UUP": "US Dollar Index", "KGLD": "YieldMax Gold", "CHPY": "YieldMax China", "SOXY": "YieldMax Semi",
    "USCL.TO": "Horizon US Large Cap", "BANK.TO": "Evolve Cdn Banks"
}

# --- WATCHLISTS ---
MAJOR_THEMES = "SPY, QQQ, DIA, IWF, IWD, MAGS, IWM, IJR, GLD, SLV, COPX, XLE, IBIT, IGV, XLP, XLRE, ARKK, TLT, UUP, XME"
TV_INDUSTRIES_FULL = "XES, OIH, FLR, EVX, AMLP, VTI, TTD, VPP, SPGI, MAN, WSC, SYY, AVT, MCK, FI, ACN, IGV, FDN, UNH, THC, HCA, IQV, DIS, NXST, CHTR, NYT, EATZ, CRUZ, BETZ, PEJ, KR, CVS, M, WMT, NKE, HD, BBY, TSCO, ONLN, IYT, XLU, XLF, IYZ, XLI, VAW, SMH, IBB, XHB, XLP, XRT, BDRY"
HARD_ASSETS_LIVE = "GC=F, SI=F, HG=F, CL=F, BZ=F, NG=F, PL=F, PA=F, TIO=F, ALB, URNM, ZS=F, MOO, OIH"
INCOME_STOCKS = "QDVO, CEFS, MLPX, AMLP, PBDC, PFFA, RLTY, UTF, ARCC, MAIN, FEPI, BSK, STK, BTCI, MSTY, USCL.TO, BANK.TO, KGLD, CHPY, SOXY"

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎯 Watchlist")
    heap_type = st.radio("Choose Group:", ["Major Themes", "Sector Rotation", "Energy Torque", "Startup", "Tech Themes", "Hard Assets (Live)", "TV Industries (Full)", "Income Stocks", "Single/Custom"])
    
    current_list = {"Major Themes": MAJOR_THEMES, "Sector Rotation": "XLK, IGV, XLY, XLC, XBI, XLF, XLI, XLE, XLV, IHE, XLP, XLU, XLB, XLRE, PSCT, PSCD, PSCF, PSCI, PSCH, PSCC, PSCU, PSCM, PSCE", "Energy Torque": "AROC, KGS, LBRT, NE, SM, CRC, BTU, WHD, MGY, CNR, OII, INVX, LEU, VAL, CIVI, NINE, BORR, HP, STX, BHL", "Startup": "AMD, TSLA, NVDA, PLTR", "Tech Themes": "SMH, IGV, BOTZ", "Hard Assets (Live)": HARD_ASSETS_LIVE, "TV Industries (Full)": TV_INDUSTRIES_FULL, "Income Stocks": INCOME_STOCKS, "Single/Custom": ""}.get(heap_type)
    auto_bench = "ONE" if heap_type in ["Hard Assets (Live)", "Income Stocks"] else "SPY"
    
    tickers_input = st.text_area("Ticker Heap:", value=current_list, height=150)
    benchmark = st.text_input("Active Benchmark:", value=auto_bench)
    
    st.markdown("---")
    main_timeframe = st.radio("Display Chart Timeframe:", ["Weekly", "Daily"], index=0)
    tail_len = st.slider("Tail Length:", 2, 30, 3)
    
    if st.button("♻️ Reset Engine"):
        st.cache_data.clear()
        st.rerun()

# --- FAIL-SAFE DATA ENGINE ---
@st.cache_data(ttl=600)
def download_single_ticker(ticker, interval):
    try:
        data = yf.download(ticker, period="2y", interval=interval, progress=False, auto_adjust=True)
        return data['Close'] if not data.empty else None
    except:
        return None

def get_metrics(px_series, bx_series, ticker):
    try:
        common = px_series.index.intersection(bx_series.index)
        px_a, bx_a = px_series.loc[common], bx_series.loc[common]
        
        # Trend-Sync Ratio Calculation
        rel = (px_a / bx_a) * 100
        ratio_smoothed = rel.ewm(span=3).mean()
        
        ratio = RRG_CENTER + ((ratio_smoothed - ratio_smoothed.rolling(LOOKBACK).mean()) / ratio_smoothed.rolling(LOOKBACK).std().replace(0, EPSILON))
        
        # ROC-based Momentum to fix the XLP "Southwest" issue
        mom_raw = ratio.diff(1)
        mom_smoothed = mom_raw.ewm(span=3).mean()
        momentum = RRG_CENTER + (mom_smoothed * 5)
        
        df_res = pd.DataFrame({'x': ratio, 'y': momentum, 'date': ratio.index}).dropna()
        df_res['date_str'] = df_res['date'].dt.strftime('%b %d, %Y')
        df_res['full_name'] = TICKER_NAMES.get(ticker, ticker)
        return df_res.clip(lower=80, upper=120)
    except:
        return None

def run_analysis(ticker_str, bench, tf_display):
    tickers = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    bench_t = bench.strip().upper()
    is_absolute = bench_t == "ONE"
    interval = "1wk" if tf_display == "Weekly" else "1d"
    
    # Fetch Benchmark
    bx_series = pd.Series(1.0, index=pd.date_range(end=pd.Timestamp.now(), periods=500, freq='D')) if is_absolute else download_single_ticker(bench_t, interval)
    
    if bx_series is None:
        st.error(f"Could not load benchmark: {bench_t}")
        return pd.DataFrame(), {}

    history, table_data = {}, []
    
    # Progress bar for the fail-safe fetcher
    progress_bar = st.progress(0)
    for i, t in enumerate(tickers):
        px_series = download_single_ticker(t, interval)
        if px_series is not None:
            res = get_metrics(px_series, bx_series, t)
            if res is not None and not res.empty:
                history[t] = res
                dr, dm = res['x'].iloc[-1], res['y'].iloc[-1]
                table_data.append({
                    "Ticker": t, "Full Name": TICKER_NAMES.get(t, t), 
                    "Stage": "LEADING" if dr >= 100 and dm >= 100 else "IMPROVING" if dr < 100 and dm >= 100 else "LAGGING" if dr < 100 and dm < 100 else "WEAKENING",
                    "Score": round((dr * 0.5) + (dm * 0.2), 1), "RS-Ratio": round(dr, 2)
                })
        progress_bar.progress((i + 1) / len(tickers))
    progress_bar.empty()
    
    return pd.DataFrame(table_data), history

# --- DISPLAY ---
try:
    df_main, hist = run_analysis(tickers_input, benchmark, main_timeframe)
    if not df_main.empty:
        col_t1, col_t2 = st.columns([1, 4])
        with col_t1: show_all = st.checkbox("Show All Tickers", value=True)
        default_selection = list(hist.keys()) if show_all else []
        with col_t2: to_plot = st.multiselect("Active Plotters:", options=list(hist.keys()), default=default_selection)
        
        st.subheader(f"🌀 {main_timeframe} Chart Rotation vs {benchmark}")
        fig = go.Figure()
        fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="rgba(0,0,0,0.3)", dash="dot"))
        fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="rgba(0,0,0,0.3)", dash="dot"))
        fig.add_vrect(x0=POWER_WALK_LEVEL, x1=CHART_RANGE[1], fillcolor="#2ECC71", opacity=0.1, layer="below", line_width=0)
        
        for i, t in enumerate(to_plot):
            df = hist[t]
            color = px.colors.qualitative.Alphabet[i % 26]
            df_p = df.iloc[-min(tail_len, len(df)):]
            fig.add_trace(go.Scatter(x=df_p['x'], y=df_p['y'], mode='lines', line=dict(color=color, width=2.5, shape='spline'), showlegend=False))
            fig.add_trace(go.Scatter(x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], mode='markers+text', marker=dict(symbol='diamond', size=14, color=color, line=dict(width=1.5, color='white')), text=[f"<b>{t}</b>"], textposition="top center", name=t))
        
        fig.update_layout(template="plotly_white", height=800, xaxis=dict(range=CHART_RANGE, title="RS-Ratio"), yaxis=dict(range=CHART_RANGE, title="RS-Momentum"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_main.sort_values(by='Score', ascending=False), use_container_width=True)
except Exception as e: st.error(f"Critical Display Error: {e}")
