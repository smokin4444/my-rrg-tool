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
    "CHPY": "YieldMax China", "MSTY": "YieldMax MSTR", "USCL.TO": "Horizon US Large Cap", 
    "BANK.TO": "Evolve Cdn Banks", "PSCT": "Tech (Small Cap)", "PSCE": "Energy (Small Cap)",
    "XLRE": "Real Estate", "XBI": "Biotech (S&P)", "ARCC": "Ares Capital (BDC)", "MAIN": "Main Street Capital",
    "GC=F": "Gold Futures", "SI=F": "Silver Futures", "HG=F": "Copper Futures", "CL=F": "Crude Oil",
    "BZ=F": "Brent Oil", "NG=F": "Natural Gas", "PL=F": "Platinum", "PA=F": "Palladium",
    "TIO=F": "Iron Ore", "ALB": "Albemarle (Lithium)", "ZS=F": "Soybeans"
}

# --- WATCHLISTS ---
MAJOR_THEMES = "SPY, QQQ, DIA, IWF, IWD, MAGS, IWM, IJR, GLD, SLV, COPX, XLE, IBIT, IGV, XLP, XLRE, ARKK, TLT, UUP, XME"
TV_INDUSTRIES = "XES, OIH, FLR, EVX, AMLP, VTI, TTD, VPP, SPGI, MAN, WSC, SYY, AVT, MCK, FI, ACN, IGV, FDN, UNH, THC, HCA, IQV, DIS, NXST, CHTR, NYT, EATZ, CRUZ, BETZ, PEJ, KR, CVS, M, WMT, NKE, HD, BBY, TSCO, ONLN, IYT, XLU, XLF, IYZ, XLI, VAW, SMH, IBB, XHB, XLP, XRT, BDRY"
HARD_ASSETS = "GC=F, SI=F, HG=F, CL=F, BZ=F, NG=F, PL=F, PA=F, TIO=F, ALB, URNM, ZS=F, MOO, OIH"
INCOME_STOCKS = "QDVO, CEFS, MLPX, AMLP, PBDC, PFFA, RLTY, UTF, ARCC, MAIN, FEPI, BSK, STK, BTCI, MSTY, USCL.TO, BANK.TO, KGLD, CHPY, SOXY"
SECTOR_ROTATION = "XLK, IGV, XLY, XLC, XBI, XLF, XLI, XLE, XLV, IHE, XLP, XLU, XLB, XLRE, PSCT, PSCD, PSCF, PSCI, PSCH, PSCC, PSCU, PSCM, PSCE"

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎯 Watchlist")
    group_choice = st.radio("Group:", ["Major Themes", "Sector Rotation", "Hard Assets", "TV Industries", "Income Stocks", "Custom"])
    current_list = {"Major Themes": MAJOR_THEMES, "Sector Rotation": SECTOR_ROTATION, "Hard Assets": HARD_ASSETS, "TV Industries": TV_INDUSTRIES, "Income Stocks": INCOME_STOCKS, "Custom": ""}.get(group_choice)
    auto_bench = "ONE" if group_choice in ["Hard Assets", "Income Stocks"] else "SPY"
    tickers_input = st.text_area("Tickers:", value=current_list, height=150)
    benchmark = st.text_input("Benchmark:", value=auto_bench)
    main_timeframe = st.radio("Timeframe:", ["Weekly", "Daily"], index=0)
    tail_len = st.slider("Tail Length:", 2, 30, 3)
    if st.button("♻️ Reset Engine"):
        st.cache_data.clear()
        st.rerun()

# --- REFINED FAIL-SAFE FETCH ---
def fetch_safe(ticker, tf):
    try:
        # Period 2y is stable for futures
        data = yf.download(ticker, period="2y", interval=("1wk" if tf == "Weekly" else "1d"), progress=False)
        if data.empty: return None
        
        # Check for multiple possible column names (Yahoo can vary)
        for col in ['Adj Close', 'Close']:
            if col in data.columns:
                if isinstance(data.columns, pd.MultiIndex):
                    return data[col][ticker]
                return data[col]
        return None
    except: return None

def calculate_rrg(price_s, bench_s, ticker):
    try:
        common = price_s.index.intersection(bench_s.index)
        if len(common) < 20: return None
        p, b = price_s.loc[common], bench_s.loc[common]
        
        rel = (p / b) * 100
        ratio_s = rel.ewm(span=3).mean()
        x = RRG_CENTER + ((ratio_s - ratio_s.rolling(LOOKBACK).mean()) / ratio_s.rolling(LOOKBACK).std().replace(0, EPSILON))
        
        y_raw = x.diff(1).ewm(span=3).mean()
        y = RRG_CENTER + (y_raw * 5)
        
        df = pd.DataFrame({'x': x, 'y': y, 'date': x.index}).dropna()
        if df.empty: return None
        df['full_name'] = TICKER_NAMES.get(ticker, ticker)
        return df
    except: return None

# --- RUN ---
try:
    ticker_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    is_abs = benchmark.upper() == "ONE"
    
    with st.spinner("Loading Benchmark..."):
        # Absolute ONE generates a static series of 1.0
        bench_data = pd.Series(1.0, index=pd.date_range(end=pd.Timestamp.now(), periods=500)) if is_abs else fetch_safe(benchmark, main_timeframe)
        bench_d = pd.Series(1.0, index=pd.date_range(end=pd.Timestamp.now(), periods=500)) if is_abs else fetch_safe(benchmark, "Daily")
        bench_w = pd.Series(1.0, index=pd.date_range(end=pd.Timestamp.now(), periods=500)) if is_abs else fetch_safe(benchmark, "Weekly")

    if bench_data is not None:
        results_map = {}
        prog = st.progress(0)
        
        for i, t in enumerate(ticker_list):
            p_display = fetch_safe(t, main_timeframe)
            if p_display is not None:
                m_display = calculate_rrg(p_display, bench_data, t)
                
                # Table Sync Calcs
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
            # Power Walk Area
            fig.add_vrect(x0=POWER_WALK_LEVEL, x1=CHART_RANGE[1], fillcolor="rgba(46, 204, 113, 0.1)", layer="below", line_width=0)
            fig.add_shape(type="line", x0=100, y0=80, x1=100, y1=120, line=dict(color="rgba(0,0,0,0.2)", dash="dot"))
            fig.add_shape(type="line", x0=80, y0=100, x1=120, y1=100, line=dict(color="rgba(0,0,0,0.2)", dash="dot"))
            
            for idx, (t, data) in enumerate(results_map.items()):
                df_chart = data['chart']
                if len(df_chart) >= tail_len:
                    df_p = df_chart.iloc[-tail_len:]
                    color = p_express.colors.qualitative.Alphabet[idx % 26]
                    fig.add_trace(go.Scatter(x=df_p['x'], y=df_p['y'], mode='lines', line=dict(color=color, width=2.5), showlegend=False))
                    fig.add_trace(go.Scatter(x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], mode='markers+text', 
                                             text=[f"<b>{t}</b>"], textposition="top center",
                                             marker=dict(symbol='diamond', size=14, color=color, line=dict(width=1, color='white')), name=t))

            fig.update_layout(template="plotly_white", height=750, xaxis=dict(range=CHART_RANGE, title="RS-Ratio"), yaxis=dict(range=[96.5, 103.5], title="RS-Momentum"))
            st.plotly_chart(fig, use_container_width=True)

            # Sync Table
            st.subheader("📊 Dual-Timeframe Strategy")
            table_rows = []
            for t, data in results_map.items():
                def get_q(df):
                    if df is None or len(df) < 1: return "N/A"
                    lx, ly = df['x'].iloc[-1], df['y'].iloc[-1]
                    if lx >= 100 and ly >= 100: return "LEADING"
                    if lx < 100 and ly >= 100: return "IMPROVING"
                    if lx < 100 and ly < 100: return "LAGGING"
                    return "WEAKENING"
                
                q_d, q_w = get_q(data['daily']), get_q(data['weekly'])
                sync = "💎 BULLISH SYNC" if q_d == "LEADING" and q_w == "LEADING" else "📈 PULLBACK BUY" if q_d == "IMPROVING" and q_w == "LEADING" else "---"
                table_rows.append({"Ticker": t, "Full Name": TICKER_NAMES.get(t, t), "Sync Status": sync, "Daily": q_d, "Weekly": q_w, "Score": round(data['chart']['x'].iloc[-1], 2)})
            
            st.dataframe(pd.DataFrame(table_rows).sort_values("Score", ascending=False).style.applymap(lambda v: "background-color: #2ecc71; color: white" if v == "💎 BULLISH SYNC" else "background-color: #3498db; color: white" if v == "📈 PULLBACK BUY" else "", subset=["Sync Status"]), use_container_width=True)
        else: st.warning("No data found for the selected tickers.")
except Exception as e: st.error(f"Critical System Error: {e}")
