import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
import time

# --- CONFIGURATION ---
# Using your existing Sita Hub URL
GAS_URL = "https://script.google.com/macros/s/AKfycbxfcQoQCWnlbfOX8jIJKLAuc8VuWknXYQ5WQSKZhXywoHQRub91tyS6gRPBKqFrn01bWg/exec"
LOOKBACK = 14
RRG_CENTER = 100
EPSILON = 1e-8
Z_LIMITS = (80, 120)  
CHART_RANGE = [96.5, 103.5] 
POWER_WALK_LEVEL = 101.5

st.set_page_config(page_title="Alpha-Scanner Pro", layout="wide")

# --- SITA HUB CLOUD FUNCTIONS ---
def load_from_hub():
    try:
        # Your script returns all properties at once
        response = requests.get(GAS_URL)
        if response.status_code == 200:
            all_data = response.json()
            # We look specifically for the 'watchlists' key you've saved
            watchlist_data = all_data.get('watchlists', "{}")
            return json.loads(watchlist_data)
        return {}
    except Exception as e:
        st.error(f"Hub Load Error: {e}")
        return {}

def save_to_hub(new_watchlists_dict):
    try:
        # Since your script saves the top-level keys, we wrap our dict in a string
        payload = {"watchlists": json.dumps(new_watchlists_dict)}
        response = requests.post(GAS_URL, data=json.dumps(payload))
        return response.status_code == 200
    except Exception as e:
        st.error(f"Hub Save Error: {e}")
        return False

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
    "GC=F": "Gold Futures", "SI=F": "Silver Futures", "HG=F": "Copper Futures", 
    "CL=F": "Crude Oil Futures", "BZ=F": "Brent Oil Futures", "NG=F": "Natural Gas Futures", 
    "ZS=F": "Soybean Futures", "ALB": "Albemarle (Lithium)", "URNM": "Uranium Miners", 
    "THD": "Thailand", "EWZ": "Brazil", "EWY": "South Korea", "EWT": "Taiwan", "EWG": "Germany",
    "EWJ": "Japan", "EWC": "Canada", "EWW": "Mexico", "EPU": "Peru", "ECH": "Chile",
    "ARGT": "Argentina", "EZA": "South Africa", "EIDO": "Indonesia", "EWM": "Malaysia",
    "EWP": "Spain", "EWL": "Switzerland", "EWQ": "France", "EWU": "United Kingdom",
    "EWH": "Hong Kong", "INDA": "India", "EWA": "Australia"
}

# --- PRE-SET WATCHLISTS ---
INDUSTRY_THEMES = "SMH, FTXL, HACK, IGV, BOTZ, QTUM, IBIT, WGMI, GDX, SIL, XME, SLX, TAN, XBI, IDNA, IYT, JETS, XHB, BOAT, BDRY, KRE, ITA, KWEB, XLE, OIH, IHI"
INTL_COUNTRIES = "THD, EWZ, EWY, EWT, EWG, EWJ, EWC, EWW, EPU, ECH, ARGT, EZA, EIDO, EWM, EWP, EWL, EWQ, EWU, EWH, INDA, EWA"
MAJOR_THEMES = "SPY, QQQ, DIA, IWF, IWD, MAGS, IWM, GLD, SLV, COPX, XLE, IBIT, IGV, XLP, XLRE, ARKK, TLT, UUP, XME, SMH, SOXX, FTXL"
HARD_ASSETS = "GC=F, SI=F, HG=F, CL=F, BZ=F, NG=F, PL=F, PA=F, TIO=F, ALB, URNM, ZS=F, MOO, OIH"

# --- SIDEBAR & CLOUD MANAGER ---
with st.sidebar:
    st.header("🎯 Watchlist")
    group_choice = st.radio("Choose Group:", ["Major Themes", "Industry Themes", "International Countries", "Hard Assets", "Sita Hub Manager"])
    
    tickers_input = ""
    if group_choice == "Sita Hub Manager":
        hub_data = load_from_hub()
        list_names = list(hub_data.keys())
        
        selected_list = st.selectbox("Saved in Sita Hub:", ["Create New..."] + list_names)
        
        initial_val = "AAPL, MSFT, GOOGL"
        if selected_list != "Create New...":
            initial_val = hub_data[selected_list]
        
        tickers_input = st.text_area("Edit Tickers:", value=initial_val, height=150)
        new_name = st.text_input("List Name:", value="" if selected_list == "Create New..." else selected_list)
        
        if st.button("☁️ Sync to Sita Hub"):
            if new_name:
                hub_data[new_name] = tickers_input
                if save_to_hub(hub_data):
                    st.success(f"Saved '{new_name}' to Cloud!")
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
        sync_list = df_main[df_main['Sync Status'] == "💎 BULLISH SYNC"]['Ticker'].tolist()
        if sync_list:
            st.success(f"💎 **BULLISH SYNC ALERT:** {', '.join(sync_list)}")
        
        col_t1, col_t2 = st.columns([1, 4])
        with col_t1: show_all = st.checkbox("Show All Tickers", value=True)
        default_selection = list(hist.keys()) if show_all else []
        with col_t2: to_plot = st.multiselect("Active Plotters:", options=list(hist.keys()), default=default_selection)
        
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
        
        st.markdown("---")
        st.subheader("🔥 Industry/Asset Heat Score Tracker")
        theme_data = []
        total_p = 0
        for t, data in hist.items():
            cx = data['x'].iloc[-1]
            chg_1w = cx - data['x'].iloc[-2]
            is_p = 1 if cx >= POWER_WALK_LEVEL else 0
            total_p += is_p
            theme_data.append({"Ticker": t, "Name": TICKER_NAMES.get(t, t), "RS Ratio": round(cx, 2), "1W Δ": round(chg_1w, 2), "Status": "🔥 ACCEL" if chg_1w > 0 else "🧊 COOL"})
        
        st.metric("Group Heat Score", f"{int((total_p/len(hist))*100)}%")
        st.dataframe(pd.DataFrame(theme_data).sort_values("1W Δ", ascending=False), use_container_width=True)

except Exception as e: st.error(f"Error: {e}")
