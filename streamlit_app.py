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
CHART_RANGE = [96.5, 103.5] # Wider zoom
POWER_WALK_LEVEL = 101.5

st.set_page_config(page_title="Alpha-Scanner Pro", layout="wide")

TICKER_NAMES = {
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100", "DIA": "Dow Jones", "IWF": "Growth Stocks", 
    "IWD": "Value Stocks", "MAGS": "Magnificent 7", "IWM": "Small Caps", 
    "GLD": "Gold", "SLV": "Silver", "COPX": "Copper Miners", "XLE": "Energy",
    "XLK": "Technology", "XLY": "Consumer Durables", "XLC": "Communications", 
    "XLF": "Finance", "XLI": "Producer Manufacturing", 
    "XLV": "Health Services", "XLP": "Cons Staples", "XLU": "Utilities", 
    "XLB": "Materials (Broad)", "IYT": "Transportation", "PICK": "Non-Energy Minerals", 
    "URNM": "Energy Minerals", "OII": "Industrial Services", "VAW": "Process Industries", 
    "SMH": "Electronic Tech", "IGV": "Software", "IBB": "Health Tech", "XHB": "Cons. Durables",
    "PEJ": "Consumer Services", "XRT": "Retail Trade", "IYZ": "Communications", "VNQ": "Invest Trusts",
    "VTI": "Misc/Broad", "IBIT": "Bitcoin Trust", "FAST": "Distribution", "IHE": "Pharma",
    "XES": "Contract Drilling", "OIH": "Oilfield Services", "FLR": "Eng. & Construction",
    "EVX": "Environmental Svcs", "AMLP": "Pipelines", "TTD": "Ad-Tech/Marketing",
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
    "TLT": "20+Y Treasury Bonds", "UUP": "US Dollar Index (Bullish)"
}

# --- WATCHLISTS ---
MAJOR_THEMES = "SPY, QQQ, DIA, IWF, IWD, MAGS, IWM, IJR, GLD, SLV, COPX, XLE, IBIT, IGV, XLP, XLRE, ARKK, TLT, UUP"
TV_INDUSTRIES_FULL = "XES, OIH, FLR, EVX, AMLP, VTI, TTD, VPP, SPGI, MAN, WSC, SYY, AVT, MCK, FI, ACN, IGV, FDN, UNH, THC, HCA, IQV, DIS, NXST, CHTR, NYT, EATZ, CRUZ, BETZ, PEJ, KR, CVS, M, WMT, NKE, HD, BBY, TSCO, ONLN, IYT, XLU, XLF, IYZ, XLI, VAW, SMH, IBB, XHB, XLP, XRT"
SECTOR_ROTATION = "XLK, IGV, XLY, XLC, XBI, XLF, XLI, XLE, XLV, IHE, XLP, XLU, XLB, XLRE, PSCT, PSCD, PSCF, PSCI, PSCH, PSCC, PSCU, PSCM, PSCE"
ENERGY_TORQUE = "AROC, KGS, LBRT, NE, SM, CRC, BTU, WHD, MGY, CNR, OII, INVX, LEU, VAL, CIVI, NINE, BORR, HP, STX, BHL"
STARTUP_THEMES = "AMD, AMPX, BABA, BIDU, BITF, CIFR, CLSK, CORZ, CRWV, EOSE, GOOGL, HUT, IREN, LAES, NBIS, NUAI, NVDA, NVTS, PATH, POWL, RR, SERV, SNDK, TE, TSLA, TSM, WDC, ZETA, BHP, CMCL, COPX, CPER, ERO, FCX, HBM, HG=F, IE, RIO, SCCO, TGB, TMQ, AMTM, AVAV, BWXT, DPRO, ESLT, KRKNF, KRMN, KTOS, LPTH, MOB, MRCY, ONDS, OSS, PLTR, PRZO, RCAT, TDY, UMAC, CRDO, IBRX, IONQ, IONR, LAC, MP, NAK, NET, OPTT, PPTA, RZLT, SKYT, TMDX, UAMY, USAR, UUUU, WWR, ASTS, BKSY, FLY, GSAT, HEI, IRDM, KULR, LUNR, MNTS, PL, RDW, RKLB, SATL, SATS, SIDU, SPIR, UFO, VOYG, VSAT"
TECH_THEMES = "AIQ, SMH, SOXX, SETM, URNM, VST, BOTZ, HOOD, IBIT, LUNR, QTUM, AVAV, LIT, IGV"
HARD_ASSETS_LIVE = "GC=F, SI=F, HG=F, CL=F, BZ=F, NG=F, PL=F, PA=F, TIO=F, ALB, URNM, ZS=F"

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎯 Watchlist")
    heap_type = st.radio("Choose Group:", ["Major Themes", "Sector Rotation", "Energy Torque", "Startup", "Tech Themes", "Hard Assets (Live)", "TV Industries (Full)", "Single/Custom"])
    
    current_list = {"Major Themes": MAJOR_THEMES, "Sector Rotation": SECTOR_ROTATION, "Energy Torque": ENERGY_TORQUE, "Startup": STARTUP_THEMES, "Tech Themes": TECH_THEMES, "Hard Assets (Live)": HARD_ASSETS_LIVE, "TV Industries (Full)": TV_INDUSTRIES_FULL, "Single/Custom": ""}.get(heap_type)
    auto_bench = "ONE" if heap_type == "Hard Assets (Live)" else "SPY"
    
    tickers_input = st.text_area("Ticker Heap:", value=current_list, height=150)
    benchmark = st.text_input("Active Benchmark:", value=auto_bench)
    
    st.markdown("---")
    # UPDATED DEFAULT: Weekly 
    main_timeframe = st.radio("Display Chart Timeframe:", ["Weekly", "Daily"], index=0)
    # UPDATED DEFAULT: 3 bars
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
        except Exception as e: st.warning(f"Chunk error: {e}")
    return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

def get_metrics(df_raw, ticker, bench_t, is_absolute):
    try:
        px = df_raw['Close'][ticker].dropna()
        bx = pd.Series(1.0, index=px.index) if is_absolute else df_raw['Close'][bench_t].dropna()
        common = px.index.intersection(bx.index)
        if len(common) < LOOKBACK + 5: return None
        px_a, bx_a = px.loc[common], bx.loc[common]
        rel = (px_a / bx_a) * 100
        def standardize(series):
            return RRG_CENTER + ((series - series.rolling(LOOKBACK).mean()) / series.rolling(LOOKBACK).std().replace(0, EPSILON))
        ratio = standardize(rel).clip(*Z_LIMITS)
        mom = standardize(rel.diff(1)).clip(*Z_LIMITS)
        df_res = pd.DataFrame({'x': ratio, 'y': mom, 'date': ratio.index}).dropna()
        df_res['date_str'] = df_res['date'].dt.strftime('%b %d, %Y')
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
            stg_d = get_stage(dr_d, dm_d)
            stg_w = get_stage(res_w['x'].iloc[-1], res_w['y'].iloc[-1]) if res_w is not None and not res_w.empty else "N/A"
            heading = get_heading(res_display['x'].iloc[-2], res_display['y'].iloc[-2], res_display['x'].iloc[-1], res_display['y'].iloc[-1])
            velocity = np.sqrt((res_display['x'].iloc[-1] - res_display['x'].iloc[-2])**2 + (res_display['y'].iloc[-1] - res_display['y'].iloc[-2])**2)
            sync = "💎 BULLISH SYNC" if stg_d == "LEADING" and stg_w == "LEADING" else "📈 PULLBACK BUY" if stg_d == "IMPROVING" and stg_w == "LEADING" else "⚠️ TACTICAL" if stg_d == "LEADING" and stg_w == "LAGGING" else "---"
            table_data.append({"Ticker": t, "Full Name": TICKER_NAMES.get(t, t), "Sync Status": sync, "Daily Stage": stg_d, "Weekly Stage": stg_w, "Heading": heading, "Rotation Score": round((res_display['x'].iloc[-1] * 0.5) + (velocity * 2.0), 1)})
    return pd.DataFrame(table_data), history_display

# --- DISPLAY ---
try:
    df_main, hist = run_dual_analysis(tickers_input, benchmark, main_timeframe)
    if not df_main.empty:
        col_t1, col_t2 = st.columns([1, 4])
        with col_t1: show_all = st.checkbox("Show All Tickers", value=True)
        default_selection = list(hist.keys()) if show_all else []
        with col_t2: to_plot = st.multiselect("Active Plotters:", options=list(hist.keys()), default=default_selection)
        
        st.subheader(f"🌀 {main_timeframe} Chart Rotation vs {benchmark}")
        fig = go.Figure()
        fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="rgba(0,0,0,0.3)", dash="dot"))
        fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="rgba(0,0,0,0.3)", dash="dot"))
        
        # POWER WALK SHADING (Green shaded area for X > 101.5)
        fig.add_vrect(
            x0=POWER_WALK_LEVEL, x1=CHART_RANGE[1],
            fillcolor="#2ECC71", opacity=0.1, layer="below", line_width=0,
            annotation_text="POWER WALK ZONE", annotation_position="top left",
            annotation_font=dict(color="#27ae60", size=10)
        )
        # Vertical Border for Shading
        fig.add_shape(type="line", x0=POWER_WALK_LEVEL, y0=0, x1=POWER_WALK_LEVEL, y1=200, line=dict(color="#27ae60", dash="dash", width=1.5))
        
        for i, t in enumerate(to_plot):
            df = hist[t]
            color = px.colors.qualitative.Alphabet[i % 26]
            df_p = df.iloc[-min(tail_len, len(df)):]
            fig.add_trace(go.Scatter(x=df_p['x'], y=df_p['y'], mode='lines', line=dict(color=color, width=2.5, shape='spline'), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=df_p['x'], y=df_p['y'], mode='markers', marker=dict(size=4, color=color, opacity=0.4), name=t, customdata=np.stack((df_p['date_str'], df_p['full_name']), axis=-1), hovertemplate="<b>%{name} | %{customdata[1]}</b><br>%{customdata[0]}<br>Ratio: %{x:.2f}<br>Mom: %{y:.2f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], mode='markers+text', marker=dict(symbol='diamond', size=14, color=color, line=dict(width=1.5, color='white')), text=[f"<b>{t}</b>"], textposition="top center", name=t, customdata=np.stack(([df_p['date_str'].iloc[-1]], [df_p['full_name'].iloc[-1]]), axis=-1), hovertemplate="<b>%{name} | %{customdata[1]}</b><br>LATEST<br>Ratio: %{x:.2f}<br>Mom: %{y:.2f}<extra></extra>", showlegend=False))
        
        fig.update_layout(template="plotly_white", height=800, xaxis=dict(range=CHART_RANGE, title="RS-Ratio"), yaxis=dict(range=CHART_RANGE, title="RS-Momentum"))
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 Dual-Timeframe Quant Grid")
        def color_sync(val):
            if val == "💎 BULLISH SYNC": return "background-color: #2ECC71; color: white"
            if val == "📈 PULLBACK BUY": return "background-color: #3498DB; color: white"
            if val == "⚠️ TACTICAL": return "background-color: #F1C40F; color: black"
            return ""
        st.dataframe(df_main.sort_values(by='Rotation Score', ascending=False).style.applymap(color_sync, subset=['Sync Status']), use_container_width=True)
except Exception as e: st.error(f"Error: {e}")
