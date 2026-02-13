import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time

# --- CONFIG ---
st.set_page_config(page_title="Alpha-Scanner Pro", layout="wide")

# --- TICKER TRANSLATION MAP ---
TICKER_NAMES = {
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100", "DIA": "Dow Jones", "IWF": "Growth Stocks", 
    "IWD": "Value Stocks", "MAGS": "Magnificent 7", "IWM": "Small Caps", 
    "IJR": "Small Cap Core", "GLD": "Gold", "SLV": "Silver", "COPX": "Copper Miners", 
    "XLE": "Energy Sector", "IBIT": "iShares Bitcoin Trust", 
    "XLK": "Tech (Large)", "XLY": "Cons Disc (Large)", "XLC": "Comm (Large)", 
    "XBI": "Biotech", "XLF": "Financials (Large)", "XLI": "Industrials (Large)", 
    "XLV": "Health Care (Large)", "XLP": "Cons Staples (Large)", "XLU": "Utilities (Large)", 
    "XLB": "Materials (Large)", "XLRE": "Real Estate (Large)",
    "IHE": "Pharma (U.S.)", "IGV": "Software (Tech-Heavy)",
    "PSCT": "Tech (Small)", "PSCE": "Energy (Small)", "OII": "Oceaneering Intl",
    "GC=F": "GOLD (Live)", "SI=F": "SILVER (Live)", "HG=F": "COPPER (Live)", 
    "CL=F": "CRUDE OIL (Live)", "BZ=F": "BRENT OIL (Live)", "NG=F": "NAT GAS (Live)", 
    "PL=F": "PLATINUM (Live)", "PA=F": "PALLADIUM (Live)", "TIO=F": "IRON ORE (Live)",
    "ALB": "LITHIUM (Proxy)", "URNM": "URANIUM (Proxy)", "ZS=F": "SOYBEANS (Live)",
    "DBC": "Broad Commodities", "PICK": "Metal Miners", "DBB": "Base Metals"
}

# --- STATIC LISTS ---
MAJOR_THEMES = "SPY, QQQ, DIA, IWF, IWD, MAGS, IWM, IJR, GLD, SLV, COPX, XLE, IBIT"

# Added IHE and IGV to the main rotation flow
SECTOR_ROTATION = "XLK, IGV, XLY, XLC, XBI, XLF, XLI, XLE, XLV, IHE, XLP, XLU, XLB, XLRE, PSCT, PSCD, PSCF, PSCI, PSCH, PSCC, PSCU, PSCM, PSCE"

ENERGY_TORQUE = "AROC, KGS, LBRT, NE, SM, CRC, BTU, WHD, MGY, CNR, OII, INVX, LEU, VAL, CIVI, NINE, BORR, HP, STX, BHL"
STARTUP_THEMES = "AMD, AMPX, BABA, BIDU, BITF, CIFR, CLSK, CORZ, CRWV, EOSE, GOOGL, HUT, IREN, LAES, NBIS, NUAI, NVDA, NVTS, PATH, POWL, RR, SERV, SNDK, TE, TSLA, TSM, WDC, ZETA, BHP, CMCL, COPX, CPER, ERO, FCX, HBM, HG=F, IE, RIO, SCCO, TGB, TMQ, AMTM, AVAV, BWXT, DPRO, ESLT, KRKNF, KRMN, KTOS, LPTH, MOB, MRCY, ONDS, OSS, PLTR, PRZO, RCAT, TDY, UMAC, CRDO, IBRX, IONQ, IONR, LAC, MP, NAK, NET, OPTT, PPTA, RZLT, SKYT, TMDX, UAMY, USAR, UUUU, WWR, ASTS, BKSY, FLY, GSAT, HEI, IRDM, KULR, LUNR, MNTS, PL, RDW, RKLB, SATL, SATS, SIDU, SPIR, UFO, VOYG, VSAT"
TECH_THEMES = "AIQ, SMH, SOXX, SETM, URNM, VST, BOTZ, HOOD, IBIT, LUNR, QTUM, AVAV, LIT, IGV"
HARD_ASSETS_LIVE = "GC=F, SI=F, HG=F, CL=F, BZ=F, NG=F, PL=F, PA=F, TIO=F, ALB, URNM, ZS=F"
CUSTOM_LIST = ""

# --- SIDEBAR & ENGINE ---
# (Using the same robust logic from previous version...)
with st.sidebar:
    st.header("🎯 Watchlist")
    heap_type = st.radio("Choose Group:", ["Major Themes", "Sector Rotation", "Energy Torque", "Startup", "Tech Themes", "Hard Assets (Live)", "Single/Custom"])
    
    if heap_type == "Major Themes": current_list, auto_bench = MAJOR_THEMES, "SPY"
    elif heap_type == "Sector Rotation": current_list, auto_bench = SECTOR_ROTATION, "SPY"
    elif heap_type == "Energy Torque": current_list, auto_bench = ENERGY_TORQUE, "XLE"
    elif heap_type == "Startup": current_list, auto_bench = STARTUP_THEMES, "SPY"
    elif heap_type == "Tech Themes": current_list, auto_bench = TECH_THEMES, "QQQ"
    elif heap_type == "Hard Assets (Live)": current_list, auto_bench = HARD_ASSETS_LIVE, "ONE"
    elif heap_type == "Single/Custom": current_list, auto_bench = CUSTOM_LIST, "SPY"
    
    tickers_input = st.text_area("Ticker Heap:", value=current_list, height=150)
    benchmark = st.text_input("Active Benchmark (use 'ONE' for Absolute):", value=auto_bench)
    
    st.markdown("---")
    timeframe = st.radio("Timeframe:", ["Daily", "Weekly", "Monthly"])
    tail_len = st.slider("Tail Length:", 2, 30, 12)
    
    if st.button("♻️ Reset Engine"):
        st.cache_data.clear()
        st.rerun()

# --- ANALYSIS ENGINE ---
@st.cache_data(ttl=600)
def download_data(tickers, interval):
    period = "10y" if interval == "1mo" else "2y"
    chunk_size = 20
    all_df = pd.DataFrame()
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            data = yf.download(chunk, period=period, interval=interval, group_by='ticker', progress=False)
            all_df = data if all_df.empty else pd.concat([all_df, data], axis=1)
            time.sleep(0.5)
        except: continue
    return all_df

def get_metrics(df_raw, ticker, b_ticker, bench_val):
    try:
        if ticker not in df_raw.columns.get_level_values(0): return None
        px = df_raw[ticker]['Close'].dropna()
        if bench_val.upper() == "ONE": bx = pd.Series(1.0, index=px.index)
        else:
            if b_ticker not in df_raw.columns.get_level_values(0): return None
            bx = df_raw[b_ticker]['Close'].dropna()
        common = px.index.intersection(bx.index)
        if len(common) < 20: return None 
        px_a, bx_a = px.loc[common], bx.loc[common]
        rel = (px_a / bx_a) * 100
        ratio = 100 + ((rel - rel.rolling(14).mean()) / rel.rolling(14).std())
        roc = ratio.diff(1)
        mom = 100 + ((roc - roc.rolling(14).mean()) / roc.rolling(14).std())
        df_res = pd.DataFrame({'x': ratio, 'y': mom, 'date': ratio.index}).dropna()
        df_res['date_str'] = df_res['date'].dt.strftime('%b %d, %Y')
        return df_res
    except: return None

def get_quadrant(x, y):
    if x >= 100 and y >= 100: return "LEADING"
    if x < 100 and y >= 100: return "IMPROVING"
    if x < 100 and y < 100: return "LAGGING"
    return "WEAKENING"

def run_analysis(ticker_str, bench, tf):
    tickers = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    bench_t = bench.strip().upper()
    all_list = list(set(tickers + ([bench_t] if bench_t != "ONE" else [])))
    iv = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}[tf]
    data = download_data(all_list, iv)
    history, table_data = {}, []
    for t in tickers:
        res = get_metrics(data, t, bench_t, bench)
        if res is not None and not res.empty:
            history[t] = res
            dr, dm = res['x'].iloc[-1], res['y'].iloc[-1]
            dq = get_quadrant(dr, dm)
            cross = "---"
            if len(res) > 5:
                if (res['x'].iloc[-6:-1] < 100).any() and dr >= 100 and dm >= 100: cross = "🔥 CROSSING"
            table_data.append({"Ticker": t, "Name": TICKER_NAMES.get(t, t), "12 O'Clock Alert": cross, "RS-Ratio": round(dr, 2), "RS-Momentum": round(dm, 2)})
    return pd.DataFrame(table_data), history

# --- DISPLAY ---
try:
    df_main, hist = run_analysis(tickers_input, benchmark, timeframe)
    if not df_main.empty:
        st.subheader(f"🌀 {timeframe} Rotation vs {benchmark}")
        fig = go.Figure()
        fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="rgba(0,0,0,0.5)", width=2, dash="dot"))
        fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="rgba(0,0,0,0.5)", width=2, dash="dot"))
        for l, x, y, c in [("LEADING", 102.3, 102.3, "green"), ("IMPROVING", 97.7, 102.3, "blue"), ("LAGGING", 97.7, 97.7, "red"), ("WEAKENING", 102.3, 97.7, "orange")]:
            fig.add_annotation(x=x, y=y, text=f"<b>{l}</b>", showarrow=False, font=dict(color=c, size=14), opacity=0.4)
        for i, (t, df) in enumerate(hist.items()):
            color = px.colors.qualitative.Alphabet[i % 26]
            df_p = df.iloc[-min(tail_len, len(df)):]
            fig.add_trace(go.Scatter(x=df_p['x'], y=df_p['y'], mode='lines+markers', line=dict(color=color, width=3, shape='spline'), name=f"{t}", customdata=df_p['date_str'], hovertemplate="<b>%{name}</b><br>Date: %{customdata}<br>Ratio: %{x:.2f}<br>Mom: %{y:.2f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], mode='markers+text', marker=dict(symbol='diamond', size=18, color=color, line=dict(width=2, color='white')), text=[t], textposition="top center", showlegend=False))
        fig.update_layout(template="plotly_white", height=850, xaxis=dict(range=[97.5, 102.5], title="RS-Ratio"), yaxis=dict(range=[97.5, 102.5], title="RS-Momentum"), legend=dict(orientation="h", y=-0.12, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("📊 Alpha Grid")
        st.dataframe(df_main.sort_values(by='RS-Ratio', ascending=False), use_container_width=True)
except Exception as e:
    st.error(f"Engine Alert: {e}")
