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
CHART_RANGE = [97.5, 102.5]

st.set_page_config(page_title="Alpha-Scanner Pro", layout="wide")

TICKER_NAMES = {
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100", "DIA": "Dow Jones", "IWF": "Growth Stocks", 
    "IWD": "Value Stocks", "MAGS": "Magnificent 7", "IWM": "Small Caps", 
    "GLD": "Gold", "SLV": "Silver", "COPX": "Copper Miners", "XLE": "Energy",
    "XLK": "Technology", "XLY": "Consumer Durables", "XLC": "Communications", 
    "XLF": "Finance", "XLI": "Producer Manufacturing", 
    "XLV": "Health Services", "XLP": "Cons Non-Durables", "XLU": "Utilities", 
    "XLB": "Materials (Broad)", "IYT": "Transportation", "PICK": "Non-Energy Minerals", 
    "URNM": "Energy Minerals", "OII": "Industrial Services", "VAW": "Process Industries", 
    "SMH": "Electronic Tech", "IGV": "Tech Services", "IBB": "Health Tech", "XHB": "Cons. Durables",
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
    "BBY": "Electronics Stores", "TSCO": "Specialty Stores", "ONLN": "Internet Retail"
}

# --- WATCHLISTS ---
TV_INDUSTRIES_FULL = "XES, OIH, FLR, EVX, AMLP, VTI, TTD, VPP, SPGI, MAN, WSC, SYY, AVT, MCK, FI, ACN, IGV, FDN, UNH, THC, HCA, IQV, DIS, NXST, CHTR, NYT, EATZ, CRUZ, BETZ, PEJ, KR, CVS, M, WMT, NKE, HD, BBY, TSCO, ONLN, IYT, XLU, XLF, IYZ, XLI, VAW, SMH, IBB, XHB, XLP, XRT"
MAJOR_THEMES = "SPY, QQQ, DIA, IWF, IWD, MAGS, IWM, IJR, GLD, SLV, COPX, XLE, IBIT"
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
    auto_bench = "ONE" if heap_type == "Hard Assets (Live)" else "QQQ" if heap_type == "Tech Themes" else "SPY"
    
    tickers_input = st.text_area("Ticker Heap:", value=current_list, height=150)
    benchmark = st.text_input("Active Benchmark:", value=auto_bench)
    
    st.markdown("---")
    timeframe = st.radio("Timeframe:", ["Daily", "Weekly", "Monthly"])
    tail_len = st.slider("Tail Length:", 2, 30, 12)
    
    if st.button("♻️ Reset Engine"):
        st.cache_data.clear()
        st.rerun()

# --- DATA ENGINE ---
@st.cache_data(ttl=600)
def download_data(tickers, interval):
    period = "10y" if interval == "1mo" else "2y"
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
        ratio, roc = standardize(rel).clip(*Z_LIMITS), standardize(rel.diff(1)).clip(*Z_LIMITS)
        df_res = pd.DataFrame({'x': ratio, 'y': roc, 'date': ratio.index}).dropna()
        df_res['date_str'] = df_res['date'].dt.strftime('%b %d, %Y')
        df_res['full_name'] = TICKER_NAMES.get(ticker, ticker)
        return df_res
    except: return None

def run_analysis(ticker_str, bench, tf):
    tickers = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    bench_t = bench.strip().upper()
    is_absolute = bench_t == "ONE"
    data = download_data(list(set(tickers + ([bench_t] if not is_absolute else []))), {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}[tf])
    if data.empty: return pd.DataFrame(), {}
    history, table_data = {}, []
    for t in tickers:
        res = get_metrics(data, t, bench_t, is_absolute)
        if res is not None and not res.empty:
            history[t] = res
            dr, dm = res['x'].iloc[-1], res['y'].iloc[-1]
            alert = "🔥 CROSSING" if (len(res) > 2 and res['x'].iloc[-2] < 100 and dr >= 100 and dm >= 100) else "---"
            table_data.append({"Ticker": t, "Name": TICKER_NAMES.get(t, t), "Quadrant": "LEADING" if dr >= 100 and dm >= 100 else "IMPROVING" if dr < 100 and dm >= 100 else "LAGGING" if dr < 100 and dm < 100 else "WEAKENING", "12 O'Clock Alert": alert, "RS-Ratio": round(dr, 2), "RS-Mom": round(dm, 2)})
    return pd.DataFrame(table_data), history

# --- DISPLAY ---
try:
    df_main, hist = run_analysis(tickers_input, benchmark, timeframe)
    if not df_main.empty:
        to_plot = st.multiselect("Active Plotters:", options=list(hist.keys()), default=list(hist.keys())[:20])
        st.subheader(f"🌀 {timeframe} Rotation vs {benchmark}")
        fig = go.Figure()
        fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="rgba(0,0,0,0.3)", dash="dot"))
        fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="rgba(0,0,0,0.3)", dash="dot"))
        for i, t in enumerate(to_plot):
            df = hist[t]
            color = px.colors.qualitative.Alphabet[i % 26]
            df_p = df.iloc[-min(tail_len, len(df)):]
            # The logic that brings back the full name:
            fig.add_trace(go.Scatter(x=df_p['x'], y=df_p['y'], mode='lines', line=dict(color=color, width=2, shape='spline'), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=df_p['x'], y=df_p['y'], mode='markers', marker=dict(size=4, color=color, opacity=0.4), name=t, customdata=np.stack((df_p['date_str'], df_p['full_name']), axis=-1), hovertemplate="<b>%{name} | %{customdata[1]}</b><br>%{customdata[0]}<br>Ratio: %{x:.2f}<br>Mom: %{y:.2f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], mode='markers+text', marker=dict(symbol='diamond', size=14, color=color, line=dict(width=1.5, color='white')), text=[t], textposition="top center", name=t, customdata=np.stack(([df_p['date_str'].iloc[-1]], [df_p['full_name'].iloc[-1]]), axis=-1), hovertemplate="<b>%{name} | %{customdata[1]}</b><br>LATEST<br>Ratio: %{x:.2f}<br>Mom: %{y:.2f}<extra></extra>", showlegend=False))
        fig.update_layout(template="plotly_white", height=800, xaxis=dict(range=CHART_RANGE, title="RS-Ratio"), yaxis=dict(range=CHART_RANGE, title="RS-Momentum"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_main.sort_values(by='RS-Ratio', ascending=False), use_container_width=True)
except Exception as e: st.error(f"Error: {e}")
