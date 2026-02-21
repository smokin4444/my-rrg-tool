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
POWER_WALK_LEVEL = 101.5

# --- SITA HUB SYNC ---
def load_from_hub():
    try:
        response = requests.get(GAS_URL, timeout=10)
        if response.status_code == 200:
            all_data = response.json()
            watchlist_data = all_data.get('watchlists', "{}")
            return json.loads(watchlist_data)
        return {}
    except: return {}

def save_to_hub(new_watchlists_dict):
    try:
        payload = {"watchlists": json.dumps(new_watchlists_dict)}
        response = requests.post(GAS_URL, data=json.dumps(payload), timeout=10)
        return response.status_code == 200
    except: return False

# --- MASTER TICKER DICTIONARY ---
TICKER_NAMES = {
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100", "DIA": "Dow Jones", "IWF": "Growth Stocks", 
    "IWD": "Value Stocks", "MAGS": "Magnificent 7", "IWM": "Small Caps", 
    "GLD": "Gold ETF", "SLV": "Silver ETF", "COPX": "Copper Miners", "XLE": "Energy",
    "XLK": "Technology", "XLY": "Consumer Durables", "XLC": "Communications", 
    "XLF": "Finance", "XLI": "Producer Manufacturing", 
    "XLV": "Health Services", "XLP": "Cons Staples", "XLU": "Utilities", 
    "XLB": "Materials (Broad)", "IYT": "Transportation", "SMH": "Semiconductors (NVDA)", 
    "SOXX": "Memory/Broad Semi", "FTXL": "Memory Super-Cycle (MU/WDC)", "IGV": "Software", 
    "XHB": "Home Construction", "IBIT": "Bitcoin Trust", "XME": "S&P Metals & Mining",
    "BDRY": "Dry Bulk Shipping", "BOAT": "Global Shipping ETF", "MOO": "Agribusiness",
    "JEDI": "Modern Warfare & Drones", "DRNZ": "Drone Tech (REX)", "ITA": "Aerospace & Defense",
    "POWR": "U.S. Power/Grid Infra", "PAVE": "U.S. Infrastructure Dev", 
    "REMX": "Rare Earth/Strategic Metals", "URNM": "Uranium Miners", "ALB": "Lithium",
    "OZEM": "GLP-1 & Weight Loss", "IHI": "Medical Devices", "XBI": "Biotechnology",
    "GC=F": "Gold Futures", "SI=F": "Silver Futures", "HG=F": "Copper Futures", 
    "CL=F": "Crude Oil Futures", "BZ=F": "Brent Oil Futures", "NG=F": "Natural Gas Futures", 
    "ZS=F": "Soybean Futures",
    "THD": "Thailand", "EWZ": "Brazil", "EWY": "South Korea", "EWT": "Taiwan", "EWG": "Germany",
    "EWJ": "Japan", "EWC": "Canada", "EWW": "Mexico", "EPU": "Peru", "ECH": "Chile",
    "ARGT": "Argentina", "EZA": "South Africa", "EIDO": "Indonesia", "EWM": "Malaysia",
    "EWP": "Spain", "EWL": "Switzerland", "EWQ": "France", "EWU": "United Kingdom",
    "EWH": "Hong Kong", "INDA": "India", "EWA": "Australia"
}

# --- WATCHLISTS ---
INDUSTRY_THEMES = "SMH, FTXL, HACK, IGV, BOTZ, JEDI, DRNZ, POWR, PAVE, REMX, OZEM, QTUM, IBIT, WGMI, GDX, SIL, XME, SLX, TAN, XBI, IDNA, IYT, JETS, XHB, BOAT, BDRY, KRE, ITA, KWEB, XLE, OIH, IHI"
INTL_COUNTRIES = "THD, EWZ, EWY, EWT, EWG, EWJ, EWC, EWW, EPU, ECH, ARGT, EZA, EIDO, EWM, EWP, EWL, EWQ, EWU, EWH, INDA, EWA"
MAJOR_THEMES = "SPY, QQQ, DIA, IWF, IWD, MAGS, IWM, GLD, SLV, COPX, XLE, IBIT, IGV, XLP, XLRE, ARKK, TLT, UUP, XME, SMH, SOXX, FTXL"
HARD_ASSETS = "GC=F, SI=F, HG=F, CL=F, BZ=F, NG=F, PL=F, PA=F, TIO=F, ALB, URNM, ZS=F, MOO, OIH"

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎯 Watchlist")
    group_choice = st.radio("Choose Group:", ["Major Themes", "Industry Themes", "International Countries", "Hard Assets", "Sita Hub Manager"])
    
    tickers_input = ""
    if group_choice == "Sita Hub Manager":
        hub_data = load_from_hub()
        list_names = list(hub_data.keys())
        selected_list = st.selectbox("Saved in Sita Hub:", ["Create New..."] + list_names)
        initial_val = "AAPL, MSFT, GOOGL"
        if selected_list != "Create New...": initial_val = hub_data[selected_list]
        tickers_input = st.text_area("Edit Tickers:", value=initial_val, height=150)
        new_name = st.text_input("List Name:", value="" if selected_list == "Create New..." else selected_list)
        if st.button("☁️ Sync to Sita Hub"):
            if new_name:
                hub_data[new_name] = tickers_input
                if save_to_hub(hub_data):
                    st.success(f"Synced to Cloud!")
                    time.sleep(1)
                    st.rerun()
    else:
        tickers_input = {
            "Major Themes": MAJOR_THEMES, "Industry Themes": INDUSTRY_THEMES,
            "International Countries": INTL_COUNTRIES, "Hard Assets": HARD_ASSETS
        }.get(group_choice, "")
        tickers_input = st.text_area("Ticker Heap:", value=tickers_input, height=150)

    st.markdown("---")
    auto_bench = "ONE" if group_choice in ["Hard Assets"] else "SPY"
    benchmark = st.text_input("Active Benchmark:", value=auto_bench)
    main_timeframe = st.radio("Display Chart Timeframe:", ["Weekly", "Daily"], index=0)
    tail_len = st.slider("Tail Length:", 2, 30, 3)
    if st.button("♻️ Reset Engine"):
        st.cache_data.clear()
        st.rerun()

# --- ANALYTICS ---
@st.cache_data(ttl=600)
def download_data(tickers, interval):
    period, chunk_size, dfs = "2y", 10, []
    tickers = list(set([t.strip().upper() for t in tickers if t.strip()]))
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            data = yf.download(chunk, period=period, interval=interval, progress=False)
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    dfs.append(data['Close'])
                else:
                    dfs.append(data[['Close']].rename(columns={'Close': chunk[0]}))
            time.sleep(1.0) 
        except: pass
    return pd.concat(dfs, axis=1) if dfs else None

def get_metrics(df_raw, ticker, bench_t, is_absolute, timeframe_choice):
    if df_raw is None or ticker not in df_raw.columns: return None
    
    # --- ADAPTIVE LOOKBACK ---
    # Shorter for Daily (8) to catch V-shapes; 10 for Weekly to catch Swing moves.
    current_lookback = 8 if timeframe_choice == "Daily" else 10
    
    try:
        px = df_raw[ticker].dropna()
        bx = pd.Series(1.0, index=px.index) if is_absolute else df_raw[bench_t].dropna()
        common = px.index.intersection(bx.index)
        if len(common) < current_lookback + 5: return None
        
        rel = ((px.loc[common] / bx.loc[common]) * 100).ewm(span=3).mean() 
        def standardize(s): return RRG_CENTER + ((s - s.rolling(current_lookback).mean()) / s.rolling(current_lookback).std().replace(0, EPSILON))
        ratio, mom = standardize(rel).clip(*Z_LIMITS), standardize(rel.diff(1)).clip(*Z_LIMITS)
        df_res = pd.DataFrame({'x': ratio, 'y': mom, 'date': ratio.index}).dropna()
        
        day_diff = (df_res['date'].iloc[1] - df_res['date'].iloc[0]).days if len(df_res) > 1 else 0
        df_res['display_date'] = df_res['date'] + pd.Timedelta(days=4) if day_diff >= 5 else df_res['date']
        df_res['date_str'] = df_res['display_date'].dt.strftime('%b %d, %Y')
        df_res['full_name'] = TICKER_NAMES.get(ticker, ticker)
        return df_res
    except: return None

def run_dual_analysis(ticker_str, bench, tf_display):
    tickers = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    bench_t, is_absolute = bench.strip().upper(), bench.strip().upper() == "ONE"
    interval = "1d" if tf_display == "Daily" else "1wk"
    data_all = download_data(list(set(tickers + ([bench_t] if not is_absolute else []))), interval)
    
    if data_all is None: return pd.DataFrame(), {}
    
    hist_out, table_data = {}, []
    for t in tickers:
        res = get_metrics(data_all, t, bench_t, is_absolute, tf_display)
        if res is not None:
            hist_out[t] = res
            x, y = res['x'].iloc[-1], res['y'].iloc[-1]
            stg = "LEADING" if x >= 100 and y >= 100 else "IMPROVING" if x < 100 and y >= 100 else "LAGGING" if x < 100 and y < 100 else "WEAKENING"
            velocity = np.sqrt((res['x'].iloc[-1] - res['x'].iloc[-2])**2 + (res['y'].iloc[-1] - res['y'].iloc[-2])**2)
            table_data.append({"Ticker": t, "Full Name": TICKER_NAMES.get(t, t), "Stage": stg, "Rotation Score": round((x * 0.5) + (velocity * 2.0), 1)})
    return pd.DataFrame(table_data), hist_out

# --- DISPLAY ---
try:
    df_main, hist = run_dual_analysis(tickers_input, benchmark, main_timeframe)
    if not df_main.empty:
        col_t1, col_t2 = st.columns([1, 4])
        with col_t1: show_all = st.checkbox("Show All Tickers", value=True)
        default_selection = list(hist.keys()) if show_all else []
        with col_t2: to_plot = st.multiselect("Active Plotters:", options=list(hist.keys()), default=default_selection)
        
        fig = go.Figure()
        fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="rgba(0,0,0,0.3)", dash="dot"))
        fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="rgba(0,0,0,0.3)", dash="dot"))
        fig.add_vrect(x0=POWER_WALK_LEVEL, x1=CHART_RANGE[1], fillcolor="#2ECC71", opacity=0.1, layer="below")
        
        for i, t in enumerate(to_plot):
            df_p = hist[t].iloc[-min(tail_len, len(hist[t])):]
            color = px.colors.qualitative.Alphabet[i % 26]
            # Fix: Grouping for legend isolation
            fig.add_trace(go.Scatter(x=df_p['x'], y=df_p['y'], mode='lines', line=dict(color=color, width=2.5, shape='spline'), legendgroup=t, showlegend=False))
            fig.add_trace(go.Scatter(x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], mode='markers+text', marker=dict(symbol='diamond', size=14, color=color, line=dict(width=1.5, color='white')), text=[f"<b>{t}</b>"], textposition="top center", legendgroup=t, name=t))
        
        fig.update_layout(template="plotly_white", height=800, xaxis=dict(range=CHART_RANGE, title="RS-Ratio"), yaxis=dict(range=CHART_RANGE, title="RS-Momentum"))
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 High-Velocity Rotation Grid")
        st.dataframe(df_main.sort_values(by='Rotation Score', ascending=False), use_container_width=True)
        
        st.markdown("---")
        st.subheader("🔥 Momentum Accelerator")
        theme_data = []
        for t, data in hist.items():
            cx = data['x'].iloc[-1]
            chg = cx - data['x'].iloc[-2]
            theme_data.append({"Ticker": t, "Theme": TICKER_NAMES.get(t, t), "RS Ratio": round(cx, 2), "Momentum Δ": round(chg, 2)})
        st.dataframe(pd.DataFrame(theme_data).sort_values("Momentum Δ", ascending=False), use_container_width=True)
except Exception as e:
    st.error(f"Engine Error: {e}")
