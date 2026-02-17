import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as p_express
import time

# --- CONFIGURATION & CONSTANTS ---
LOOKBACK = 14
RRG_CENTER = 100
EPSILON = 1e-8
Z_LIMITS = (80, 120)  
CHART_RANGE = [96.5, 103.5] 
POWER_WALK_LEVEL = 101.5

st.set_page_config(page_title="Alpha-Scanner Pro", layout="wide")

# --- EXPANDED DEEPVUE DICTIONARY ---
TICKER_NAMES = {
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100", "XLP": "Cons Staples", "XLK": "Technology",
    "XLU": "Utilities", "XLF": "Financials", "XLY": "Consumer Disc", "XLV": "Health Care",
    "HACK": "Cybersecurity", "BOTZ": "Robotics & AI", "QTUM": "Quantum Computing",
    "SMH": "Semiconductors", "GDX": "Gold Miners", "SIL": "Silver Miners",
    "SLX": "Steel", "TAN": "Solar", "XBI": "Biotechnology", "IDNA": "Genomics",
    "IYT": "Transports", "JETS": "Airlines", "XHB": "Home Construction",
    "KRE": "Regional Banks", "ITA": "Aerospace & Defense", "XME": "Metals & Mining",
    "IBIT": "Bitcoin", "WGMI": "Bitcoin Miners", "KWEB": "China Internet",
    "XLE": "Energy", "IHI": "Medical Devices", "IGV": "Software", "BDRY": "Dry Bulk",
    "ZS=F": "Soybean Futures", "GC=F": "Gold Futures", "CL=F": "Crude Oil",
    "USCL.TO": "Horizon US Large Cap", "BANK.TO": "Evolve Cdn Banks"
}

# --- WATCHLISTS ---
DEEPVUE_THEMES = "IBIT, WGMI, GDX, SIL, SMH, HACK, BOTZ, QTUM, TAN, XBI, IDNA, IYT, JETS, XHB, SLX, KRE, ITA, XME, KWEB, XLE, IHI, IGV"
MAJOR_THEMES = "SPY, QQQ, DIA, IWF, IWD, MAGS, IWM, IJR, GLD, SLV, COPX, XLE, IBIT, IGV, XLP, XLRE, ARKK, TLT, UUP, XME"
INCOME_STOCKS = "QDVO, CEFS, MLPX, AMLP, PBDC, PFFA, RLTY, UTF, ARCC, MAIN, FEPI, BSK, STK, BTCI, MSTY, USCL.TO, BANK.TO, KGLD, CHPY, SOXY"
HARD_ASSETS = "GC=F, SI=F, HG=F, CL=F, BZ=F, NG=F, PL=F, PA=F, TIO=F, ALB, URNM, ZS=F, MOO, OIH"

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎯 Watchlist")
    group_choice = st.radio("Choose Group:", ["Theme Tracker (Deepvue)", "Major Themes", "Hard Assets", "Income Stocks", "Custom"])
    
    current_list = {
        "Theme Tracker (Deepvue)": DEEPVUE_THEMES,
        "Major Themes": MAJOR_THEMES,
        "Hard Assets": HARD_ASSETS,
        "Income Stocks": INCOME_STOCKS,
        "Custom": ""
    }.get(group_choice)
    
    auto_bench = "ONE" if group_choice in ["Hard Assets", "Income Stocks"] else "SPY"
    tickers_input = st.text_area("Ticker Heap:", value=current_list, height=150)
    benchmark = st.text_input("Active Benchmark:", value=auto_bench)
    main_timeframe = st.radio("Timeframe:", ["Weekly", "Daily"], index=0)
    tail_len = st.slider("Tail Length:", 2, 30, 3)
    
    if st.button("♻️ Reset Engine"):
        st.cache_data.clear()
        st.rerun()

# --- ENGINES ---
@st.cache_data(ttl=600)
def download_data(tickers, interval):
    try:
        data = yf.download(tickers, period="2y", interval=interval, progress=False)
        return data if not data.empty else pd.DataFrame()
    except: return pd.DataFrame()

def get_metrics(df_raw, ticker, bench_t, is_absolute):
    try:
        if ticker not in df_raw['Close'].columns: return None
        px = df_raw['Close'][ticker].dropna()
        bx = pd.Series(1.0, index=px.index) if is_absolute else df_raw['Close'][bench_t].dropna()
        common = px.index.intersection(bx.index)
        if len(common) < LOOKBACK + 5: return None
        px_a, bx_a = px.loc[common], bx.loc[common]
        
        rel = ((px_a / bx_a) * 100).ewm(span=3).mean() 
        def standardize(series):
            return RRG_CENTER + ((series - series.rolling(LOOKBACK).mean()) / series.rolling(LOOKBACK).std().replace(0, EPSILON))
        
        ratio, mom = standardize(rel).clip(*Z_LIMITS), standardize(rel.diff(1)).clip(*Z_LIMITS)
        df_res = pd.DataFrame({'x': ratio, 'y': mom, 'date': ratio.index}).dropna()
        
        # Friday Label Fix
        if main_timeframe == "Weekly":
            df_res['display_date'] = df_res['date'] + pd.Timedelta(days=4)
        else:
            df_res['display_date'] = df_res['date']
            
        df_res['date_str'] = df_res['display_date'].dt.strftime('%b %d, %Y')
        df_res['full_name'] = TICKER_NAMES.get(ticker, ticker)
        return df_res
    except: return None

# --- RUN ANALYSIS ---
try:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    is_abs = benchmark.upper() == "ONE"
    data_d = download_data(list(set(tickers + ([benchmark] if not is_abs else []))), "1d")
    data_w = download_data(list(set(tickers + ([benchmark] if not is_abs else []))), "1wk")
    
    if not data_d.empty:
        hist, table_data = {}, []
        for t in tickers:
            res_d = get_metrics(data_d, t, benchmark, is_abs)
            res_w = get_metrics(data_w, t, benchmark, is_abs)
            res_display = res_d if main_timeframe == "Daily" else res_w
            
            if res_display is not None:
                hist[t] = res_display
                curr_x, curr_y = res_display['x'].iloc[-1], res_display['y'].iloc[-1]
                
                # Heat Score Logic: Is it in the Power Walk Zone?
                is_power = 1 if curr_x >= POWER_WALK_LEVEL else 0
                
                table_data.append({
                    "Ticker": t, "Name": TICKER_NAMES.get(t, t),
                    "RS-Ratio": round(curr_x, 2), "RS-Mom": round(curr_y, 2),
                    "Power": is_power
                })
        
        # --- DISPLAY ---
        st.subheader(f"🌀 {main_timeframe} Rotation vs {benchmark}")
        fig = go.Figure()
        fig.add_vrect(x0=POWER_WALK_LEVEL, x1=CHART_RANGE[1], fillcolor="#2ECC71", opacity=0.1, layer="below")
        
        for i, t in enumerate(hist.keys()):
            df_p = hist[t].iloc[-tail_len:]
            color = p_express.colors.qualitative.Alphabet[i % 26]
            fig.add_trace(go.Scatter(x=df_p['x'], y=df_p['y'], mode='lines+markers', name=t, line=dict(color=color, shape='spline')))
            
        fig.update_layout(template="plotly_white", height=700, xaxis=dict(range=CHART_RANGE), yaxis=dict(range=CHART_RANGE))
        st.plotly_chart(fig, use_container_width=True)

        # --- THEME HEAT SCORE TABLE ---
        st.markdown("---")
        st.subheader("🔥 Deepvue Theme Tracker & Heat Score")
        
        df_heat = pd.DataFrame(table_data)
        group_heat_score = (df_heat['Power'].sum() / len(df_heat)) * 100
        
        col1, col2 = st.columns([1, 4])
        col1.metric("Group Heat Score", f"{int(group_heat_score)}%", help="Percentage of group in Power Walk Zone")
        
        def style_heat(v):
            if v >= POWER_WALK_LEVEL: return 'background-color: #1e3d24; color: #2ecc71; font-weight: bold'
            return ''

        st.dataframe(df_heat.drop(columns=['Power']).style.applymap(style_heat, subset=['RS-Ratio']), use_container_width=True)

except Exception as e: st.error(f"Error: {e}")
