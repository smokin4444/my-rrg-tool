import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import glob

# --- CONFIGURATION & CONSTANTS ---
LOOKBACK = 14
RRG_CENTER = 100
EPSILON = 1e-8
Z_LIMITS = (80, 120)  
CHART_RANGE = [96.5, 103.5] 
POWER_WALK_LEVEL = 101.5
WATCHLIST_DIR = "my_watchlists"

st.set_page_config(page_title="Alpha-Scanner Pro", layout="wide")

if not os.path.exists(WATCHLIST_DIR):
    os.makedirs(WATCHLIST_DIR)

TICKER_NAMES = {
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100", "DIA": "Dow Jones", "IWF": "Growth Stocks", 
    "IWD": "Value Stocks", "MAGS": "Magnificent 7", "IWM": "Small Caps", 
    "GLD": "Gold", "SLV": "Silver", "COPX": "Copper Miners", "XLE": "Energy",
    "XLK": "Technology", "XLY": "Consumer Durables", "XLC": "Communications", 
    "XLF": "Finance", "XLI": "Producer Manufacturing", 
    "XLV": "Health Services", "XLP": "Cons Staples", "XLU": "Utilities", 
    "XLB": "Materials (Broad)", "IYT": "Transportation", "PICK": "Non-Energy Minerals", 
    "URNM": "Energy Minerals", "OII": "Oilfield Services", "VAW": "Process Industries", 
    "SMH": "Semiconductors (NVDA)", "SOXX": "Memory/Broad Semi", "FTXL": "Memory Super-Cycle",
    "IGV": "Software", "IBB": "Health Tech", "XHB": "Cons. Durables",
    "PEJ": "Consumer Services", "XRT": "Retail Trade", "IYZ": "Communications", "VNQ": "Invest Trusts",
    "VTI": "Misc/Broad", "IBIT": "Bitcoin Trust", "FAST": "Distribution", "IHE": "Pharma",
    "XES": "Contract Drilling", "OIH": "Oilfield Services", "FLR": "Eng. & Construction",
    "EVX": "Environmental Svcs", "AMLP": "Pipelines (MLP)", "TTD": "Ad-Tech/Marketing",
    "VPP": "Commercial Printing", "SPGI": "Financial Services", "MAN": "Personnel Services",
    "WSC": "Wholesale Dist.", "SYY": "Food Distribution", "AVT": "Electronics Dist.",
    "MCK": "Medical Distribution", "FI": "Data Processing", "ACN": "IT Services",
    "FDN": "Internet Services", "UNH": "Managed Health", "THC": "Hospital Mgmt",
    "HCA": "Medical Services", "IQV": "Health Industry Svcs", "DIS": "Media Conglomerate",
    "NXST": "Broadcasting", "CHTR": "Cable/Satellite TV", "NYT": "Publishing",
    "EATZ": "Restaurants", "CRUZ": "Hotels/Cruises", "BETZ": "Casinos/Gaming",
    "KR": "Food Retail", "CVS": "Drugstore Chains", "M": "Department Stores",
    "WMT": "Discount Stores", "NKE": "Apparel Retail", "HD": "Home Improvement",
    "BBY": "Electronics Stores", "TSCO": "Specialty Stores", "ONLN": "Internet Retail",
    "PSCT": "Tech (Small Cap)", "PSCD": "Cons Disc (Small Cap)", "PSCF": "Financials (Small Cap)", 
    "PSCI": "Industrials (Small Cap)", "PSCH": "Health Care (Small Cap)", "PSCC": "Cons Staples (Small Cap)", 
    "PSCU": "Utilities (Small Cap)", "PSCM": "Materials (Small Cap)", "PSCE": "Energy (Small Cap)",
    "XLRE": "Real Estate", "XBI": "Biotech (S&P)", "ARKK": "Innovation (High Beta)", 
    "TLT": "20+Y Treasury Bonds", "UUP": "US Dollar Index (Bullish)", "XME": "S&P Metals & Mining",
    "BDRY": "Dry Bulk Shipping", "MOO": "Agribusiness", "BOAT": "Global Shipping ETF",
    "QDVO": "YieldMax QQQ Option Income", "CEFS": "Saba Closed-End Funds", "MLPX": "Global X MLP", 
    "PBDC": "Putnam BDC Income", "PFFA": "Virtus Preferred Stock", "RLTY": "Cohen & Steers RE", 
    "UTF": "Cohen & Steers Infrastructure", "ARCC": "Ares Capital (BDC)", "MAIN": "Main Street Capital (BDC)", 
    "FEPI": "Rex FANG & Innovation", "BSK": "Baskervill (Proxy)", "STK": "Columbia Tech Income", 
    "BTCI": "YieldMax Bitcoin Option", "MSTY": "YieldMax MicroStrategy Option", 
    "USCL.TO": "Horizon US Large Cap (CAD)", "BANK.TO": "Evolve Cdn Banks (CAD)",
    "KGLD": "YieldMax Gold Option Income", "CHPY": "YieldMax China Option Income", "SOXY": "YieldMax Semi Option Income",
    "ZS=F": "Soybean Futures", "GC=F": "Gold Futures", "SI=F": "Silver Futures", "HG=F": "Copper Futures", "CL=F": "Crude Oil Futures",
    "WGMI": "Bitcoin Miners", "HACK": "Cybersecurity", "BOTZ": "Robotics & AI", "QTUM": "Quantum Computing",
    "TAN": "Solar", "IDNA": "Genomics", "JETS": "Airlines", "SLX": "Steel", "KRE": "Regional Banks",
    "ITA": "Aerospace & Defense", "KWEB": "China Internet", "IHI": "Medical Devices"
}

# --- WATCHLISTS ---
INCOME_STOCKS = "QDVO, CEFS, MLPX, AMLP, PBDC, PFFA, RLTY, UTF, ARCC, MAIN, FEPI, BSK, STK, BTCI, MSTY, USCL.TO, BANK.TO, KGLD, CHPY, SOXY"
MAJOR_THEMES = "SPY, QQQ, DIA, IWF, IWD, MAGS, IWM, IJR, GLD, SLV, COPX, XLE, IBIT, IGV, XLP, XLRE, ARKK, TLT, UUP, XME, SMH, SOXX, FTXL"
TV_INDUSTRIES = "BOAT, BDRY, XES, OIH, FLR, EVX, AMLP, VTI, TTD, VPP, SPGI, MAN, WSC, SYY, AVT, MCK, FI, ACN, IGV, FDN, UNH, THC, HCA, IQV, DIS, NXST, CHTR, NYT, EATZ, CRUZ, BETZ, PEJ, KR, CVS, M, WMT, NKE, HD, BBY, TSCO, ONLN, IYT, XLU, XLF, IYZ, XLI, VAW, SMH, IBB, XHB, XLP, XRT"
SECTOR_ROTATION = "XLK, IGV, XLY, XLC, XBI, XLF, XLI, XLE, XLV, IHE, XLP, XLU, XLB, XLRE, PSCT, PSCD, PSCF, PSCI, PSCH, PSCC, PSCU, PSCM, PSCE, SOXX, FTXL"
HARD_ASSETS_LIVE = "GC=F, SI=F, HG=F, CL=F, BZ=F, NG=F, PL=F, PA=F, TIO=F, ALB, URNM, ZS=F, MOO, OIH"
DEEPVUE_THEMES = "BOAT, IBIT, WGMI, GDX, SIL, SMH, SOXX, FTXL, HACK, BOTZ, QTUM, TAN, XBI, IDNA, IYT, JETS, XHB, SLX, KRE, ITA, XME, KWEB, XLE, IHI, IGV"

# --- SIDEBAR & CUSTOM LIST LOGIC ---
with st.sidebar:
    st.header("🎯 Watchlist")
    group_choice = st.radio("Choose Group:", ["Major Themes", "Sector Rotation", "Hard Assets (Live)", "TV Industries (Full)", "Income Stocks", "DeepVue Themes", "Custom Manager"])
    
    tickers_input = ""
    if group_choice == "Custom Manager":
        saved_files = glob.glob(os.path.join(WATCHLIST_DIR, "*.txt"))
        saved_names = [os.path.basename(f).replace(".txt", "") for f in saved_files]
        selected_custom = st.selectbox("Your Saved Lists:", ["Create New..."] + saved_names)
        
        initial_val = "AAPL, MSFT, GOOGL"
        if selected_custom != "Create New...":
            with open(os.path.join(WATCHLIST_DIR, f"{selected_custom}.txt"), "r") as f:
                initial_val = f.read()
        
        tickers_input = st.text_area("Edit Tickers:", value=initial_val, height=150)
        new_name = st.text_input("Save As (Name):", value="" if selected_custom == "Create New..." else selected_custom)
        if st.button("💾 Save List"):
            if new_name:
                with open(os.path.join(WATCHLIST_DIR, f"{new_name}.txt"), "w") as f:
                    f.write(tickers_input)
                st.success(f"Saved '{new_name}'")
                time.sleep(1)
                st.rerun()
    else:
        tickers_input = {
            "Major Themes": MAJOR_THEMES, 
            "Sector Rotation": SECTOR_ROTATION,
            "Hard Assets (Live)": HARD_ASSETS_LIVE, 
            "TV Industries (Full)": TV_INDUSTRIES,
            "Income Stocks": INCOME_STOCKS,
            "DeepVue Themes": DEEPVUE_THEMES
        }.get(group_choice, "")
        tickers_input = st.text_area("Ticker Heap:", value=tickers_input, height=150)

    st.markdown("---")
    auto_bench = "ONE" if group_choice in ["Hard Assets (Live)", "Income Stocks"] else "SPY"
    benchmark = st.text_input("Active Benchmark:", value=auto_bench)
    main_timeframe = st.radio("Display Chart Timeframe:", ["Weekly", "Daily"], index=0)
    tail_len = st.slider("Tail Length:", 2, 30, 3)
    
    if st.button("♻️ Reset Engine"):
        st.cache_data.clear()
        st.rerun()

# --- ANALYTICS ENGINES ---
def get_heading(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx > 0 and dy > 0: return "NE ↗️"
    if dx < 0 and dy > 0: return "NW ↖️"
    if dx < 0 and dy < 0: return "SW ↙️"
    if dx > 0 and dy < 0: return "SE ↘️"
    return "Neutral"

@st.cache_data(ttl=600)
def download_data(tickers, interval):
    period = "2y"
    chunk_size = 25
    dfs = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            data = yf.download(chunk, period=period, interval=interval, progress=False)
            if not data.empty: dfs.append(data)
            time.sleep(0.5)
        except: pass
    return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

def get_metrics(df_raw, ticker, bench_t, is_absolute):
    try:
        if ticker not in df_raw['Close'].columns: return None
        px = df_raw['Close'][ticker].dropna()
        bx = pd.Series(1.0, index=px.index) if is_absolute else df_raw['Close'][bench_t].dropna()
        common = px.index.intersection(bx.index)
        if len(common) < LOOKBACK + 5: return None
        px_a, bx_a = px.loc[common], bx.loc[common]
        rel_raw = (px_a / bx_a) * 100
        rel = rel_raw.ewm(span=3).mean() 
        def standardize(series):
            return RRG_CENTER + ((series - series.rolling(LOOKBACK).mean()) / series.rolling(LOOKBACK).std().replace(0, EPSILON))
        ratio, mom = standardize(rel).clip(*Z_LIMITS), standardize(rel.diff(1)).clip(*Z_LIMITS)
        df_res = pd.DataFrame({'x': ratio, 'y': mom, 'date': ratio.index}).dropna()
        if len(df_res) > 1:
            day_diff = (df_res['date'].iloc[1] - df_res['date'].iloc[0]).days
            df_res['display_date'] = df_res['date'] + pd.Timedelta(days=4) if day_diff >= 5 else df_res['date']
        else: df_res['display_date'] = df_res['date']
        df_res['date_str'] = df_res['display_date'].dt.strftime('%b %d, %Y')
        df_res['full_name'] = TICKER_NAMES.get(ticker, ticker)
        return df_res
    except: return None

def get_stage(x, y):
    if x >= 100 and y >= 100: return "LEADING"
    if x < 100 and y >= 100: return "IMPROVING"
    if x < 100 and y < 100: return "LAGGING"
    return "WEAKENING"

def run_dual_analysis(ticker_str, bench, tf_display):
    tickers = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    bench_t = bench.strip().upper()
    is_absolute = bench_t == "ONE"
    data_d = download_data(list(set(tickers + ([bench_t] if not is_absolute else []))), "1d")
    data_w = download_data(list(set(tickers + ([bench_t] if not is_absolute else []))), "1wk")
    if data_d.empty or data_w.empty: return pd.DataFrame(), {}
    history_display, table_data = {}, []
    for t in tickers:
        res_d = get_metrics(data_d, t, bench_t, is_absolute)
        res_w = get_metrics(data_w, t, bench_t, is_absolute)
        res_display = res_d if tf_display == "Daily" else res_w
        if res_display is not None and not res_display.empty:
            history_display[t] = res_display
            dr_d, dm_d = res_d['x'].iloc[-1], res_d['y'].iloc[-1]
            stg_d, stg_w = get_stage(dr_d, dm_d), get_stage(res_w['x'].iloc[-1], res_w['y'].iloc[-1])
            heading = get_heading(res_display['x'].iloc[-2], res_display['y'].iloc[-2], res_display['x'].iloc[-1], res_display['y'].iloc[-1])
            velocity = np.sqrt((res_display['x'].iloc[-1] - res_display['x'].iloc[-2])**2 + (res_display['y'].iloc[-1] - res_display['y'].iloc[-2])**2)
            sync = "💎 BULLISH SYNC" if stg_d == "LEADING" and stg_w == "LEADING" else "📈 PULLBACK BUY" if stg_d == "IMPROVING" and stg_w == "LEADING" else "---"
            table_data.append({"Ticker": t, "Full Name": TICKER_NAMES.get(t, t), "Sync Status": sync, "Daily Stage": stg_d, "Weekly Stage": stg_w, "Heading": heading, "Rotation Score": round((res_display['x'].iloc[-1] * 0.5) + (velocity * 2.0), 1)})
    return pd.DataFrame(table_data), history_display

# --- DISPLAY ---
try:
    df_main, hist = run_dual_analysis(tickers_input, benchmark, main_timeframe)
    if not df_main.empty:
        to_plot = st.multiselect("Active Plotters:", options=list(hist.keys()), default=list(hist.keys()))
        st.subheader(f"🌀 {main_timeframe} Chart Rotation vs {benchmark}")
        fig = go.Figure()
        fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="rgba(0,0,0,0.3)", dash="dot"))
        fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="rgba(0,0,0,0.3)", dash="dot"))
        fig.add_vrect(x0=POWER_WALK_LEVEL, x1=CHART_RANGE[1], fillcolor="#2ECC71", opacity=0.1, layer="below", annotation_text="POWER WALK")
        for i, t in enumerate(to_plot):
            df = hist[t]
            color = px.colors.qualitative.Alphabet[i % 26]
            df_p = df.iloc[-min(tail_len, len(df)):]
            fig.add_trace(go.Scatter(x=df_p['x'], y=df_p['y'], mode='lines', line=dict(color=color, width=2.5, shape='spline'), showlegend=False))
            fig.add_trace(go.Scatter(x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], mode='markers+text', marker=dict(symbol='diamond', size=14, color=color, line=dict(width=1.5, color='white')), text=[f"<b>{t}</b>"], textposition="top center", name=t))
        fig.update_layout(template="plotly_white", height=800, xaxis=dict(range=CHART_RANGE), yaxis=dict(range=CHART_RANGE))
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("📊 Dual-Timeframe Quant Grid")
        st.dataframe(df_main.sort_values(by='Rotation Score', ascending=False), use_container_width=True)
        
        # DeepVue Theme Tracker (Heat Score) at the very bottom
        st.markdown("---")
        st.subheader("🔥 DeepVue Theme Tracker (Heat Score)")
        theme_data = []
        total_p = 0
        for t, data in hist.items():
            cx = data['x'].iloc[-1]
            chg_1w = cx - data['x'].iloc[-2]
            is_p = 1 if cx >= POWER_WALK_LEVEL else 0
            total_p += is_p
            theme_data.append({"Theme": t, "Name": TICKER_NAMES.get(t, t), "RS": round(cx, 2), "1W Δ": round(chg_1w, 2), "Status": "🔥 ACCEL" if chg_1w > 0 else "🧊 COOL"})
        st.metric("Group Heat Score", f"{int((total_p/len(hist))*100)}%")
        st.dataframe(pd.DataFrame(theme_data).sort_values("1W Δ", ascending=False), use_container_width=True)
except Exception as e: st.error(f"Error: {e}")
