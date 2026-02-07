import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- CONFIG ---
st.set_page_config(page_title="Alpha-Scanner Pro", layout="wide")

# --- TICKER TRANSLATION MAP ---
TICKER_NAMES = {
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100", "DIA": "Dow Jones", "IWF": "Growth Stocks", 
    "IWD": "Value Stocks", "MAGS": "Magnificent 7", "IWM": "Small Caps", 
    "IJR": "Small Cap Core", "GLD": "Gold", "SLV": "Silver", "COPX": "Copper Miners", 
    "XLE": "Energy Sector", "BTC-USD": "Bitcoin", "XLK": "Technology", "XLY": "Consumer Disc", 
    "XLC": "Communication", "XBI": "Biotech", "XLF": "Financials", "XLI": "Industrials", 
    "XLV": "Health Care", "XLP": "Consumer Staples", "XLU": "Utilities", "XLB": "Materials", 
    "XLRE": "Real Estate", "CBA.AX": "Commonwealth Bank", "BHP.AX": "BHP Group", 
    "CSL.AX": "CSL Limited", "NAB.AX": "National Australia Bank", "WBC.AX": "Westpac"
}

# --- LISTS ---
MAJOR_THEMES = "SPY, QQQ, DIA, IWF, IWD, MAGS, IWM, IJR, GLD, SLV, COPX, XLE, BTC-USD"
SECTOR_ROTATION = "XLK, XLY, XLC, XBI, XLF, XLI, XLE, XLV, XLP, XLU, XLB, XLRE"
STARTUP_THEMES = "AMD, AMPX, BABA, BIDU, BITF, CIFR, CLSK, CORZ, CRWV, EOSE, GOOGL, HUT, IREN, LAES, NBIS, NUAI, NVDA, NVTS, PATH, POWL, RR, SERV, SNDK, TE, TSLA, TSM, WDC, ZETA, BHP, CMCL, COPX, CPER, ERO, FCX, HBM, HG=F, IE, RIO, SCCO, TGB, TMQ, AMTM, AVAV, BWXT, DPRO, ESLT, KRKNF, KRMN, KTOS, LPTH, MOB, MRCY, ONDS, OSS, PLTR, PRZO, RCAT, TDY, UMAC, CRDO, IBRX, IONQ, IONR, LAC, MP, NAK, NET, OPTT, PPTA, RZLT, SKYT, TMDX, UAMY, USAR, UUUU, WWR, ASTS, BKSY, FLY, GSAT, HEI, IRDM, KULR, LUNR, MNTS, PL, RDW, RKLB, SATL, SATS, SIDU, SPIR, UFO, VOYG, VSAT"
ASX_LIST = "CBA.AX, BHP.AX, CSL.AX, NAB.AX, WBC.AX, ANZ.AX, MQG.AX, WES.AX, TLS.AX, WOW.AX, FMG.AX, RIO.AX, GMG.AX, TCL.AX, ALL.AX"
MINERS = "AFM.V, NAK, A4N.AX, CSC.AX, IVN.TO, TGB, MU, APLD"

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎯 Watchlist")
    heap_type = st.radio("Choose Group:", ["Major Themes", "Sector Rotation", "Startup", "ASX Blue Chips", "My Miners"])
    
    if heap_type == "Major Themes": current_list, auto_bench = MAJOR_THEMES, "SPY"
    elif heap_type == "Sector Rotation": current_list, auto_bench = SECTOR_ROTATION, "SPY"
    elif heap_type == "Startup": current_list, auto_bench = STARTUP_THEMES, "SPY"
    elif heap_type == "ASX Blue Chips": current_list, auto_bench = ASX_LIST, "VAS.AX"
    elif heap_type == "My Miners": current_list, auto_bench = MINERS, "SPY"
    
    tickers_input = st.text_area("Ticker Heap:", value=current_list, height=150).replace("BTC", "BTC-USD").replace("HG1!", "HG=F")
    
    st.header("⚖️ Benchmark")
    bench_preset = st.selectbox("Preset:", ["Auto-Detect", "SPY (S&P 500)", "VAS.AX (ASX 200)", "QQQ (Nasdaq)"])
    final_bench = auto_bench if bench_preset == "Auto-Detect" else bench_preset.split(" ")[0]
    benchmark = st.text_input("Active Benchmark:", value=final_bench)
    
    st.markdown("---")
    timeframe = st.radio("Timeframe:", ["Daily", "Weekly"])
    tail_len = st.slider("Tail Length:", 1, 30, 10)
    filter_setups = st.checkbox("Top Setups Only", value=False)
    
    if st.button("♻️ Reset Engine"):
        st.cache_data.clear()
        st.rerun()

# --- ENGINE ---
def get_rrg_metrics(df_raw, ticker, b_ticker, is_weekly=False):
    try:
        if ticker not in df_raw.columns.get_level_values(0): return None
        px, bx = df_raw[ticker]['Close'].dropna(), df_raw[b_ticker]['Close'].dropna()
        common = px.index.intersection(bx.index)
        if len(common) < 30: return None
        rel = (px.loc[common] / bx.loc[common]) * 100
        ratio = 100 + ((rel - rel.rolling(14).mean()) / rel.rolling(14).std())
        roc = ratio.diff(1)
        mom = 100 + ((roc - roc.rolling(14).mean()) / roc.rolling(14).std())
        df_res = pd.DataFrame({'x': ratio, 'y': mom, 'date': ratio.index}).dropna()
        if len(df_res) < 5: return None
        if is_weekly: df_res['date'] = df_res['date'] + pd.Timedelta(days=4)
        return df_res
    except: return None

def get_quadrant(x, y):
    if x >= 100 and y >= 100: return "LEADING"
    if x < 100 and y >= 100: return "IMPROVING"
    if x < 100 and y < 100: return "LAGGING"
    return "WEAKENING"

@st.cache_data(ttl=600)
def run_analysis(ticker_str, bench):
    tickers = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    bench_ticker = bench.strip().upper()
    all_list = list(set(tickers + [bench_ticker]))
    data = yf.download(all_list, period="2y", interval="1d", group_by='ticker', progress=False)
    w_data = yf.download(all_list, period="2y", interval="1wk", group_by='ticker', progress=False)
    history, table_data = {"Daily": {}, "Weekly": {}}, []
    for t in tickers:
        d_res = get_rrg_metrics(data, t, bench_ticker, False)
        w_res = get_rrg_metrics(w_data, t, bench_ticker, True)
        if d_res is not None and w_res is not None:
            history["Daily"][t], history["Weekly"][t] = d_res, w_res
            dq, wq = get_quadrant(d_res['x'].iloc[-1], d_res['y'].iloc[-1]), get_quadrant(w_res['x'].iloc[-1], w_res['y'].iloc[-1])
            is_synced = "✅ SYNCED" if dq == wq else "---"
            table_data.append({"Ticker": t, "Name": TICKER_NAMES.get(t, t), "Daily Quad": dq, "Weekly Quad": wq, "Sync Radar": is_synced, "RS-Ratio": round(d_res['x'].iloc[-1], 2)})
    return pd.DataFrame(table_data), history

# --- DISPLAY ---
try:
    df_main, history_data = run_analysis(tickers_input, benchmark)
    if not df_main.empty:
        st.subheader(f"🌀 {timeframe} Rotation vs {benchmark}")
        fig = go.Figure()
        
        # 1. Professional Background Colors & Shading
        # Power Zone (Lead-through Shading)
        fig.add_vrect(x0=101.5, x1=105, fillcolor="rgba(46, 204, 113, 0.15)", layer="below", line_width=0)
        
        # Quadrant Colors
        fig.add_vrect(x0=100, x1=110, y0=100, y1=110, fillcolor="rgba(0, 255, 0, 0.03)", layer="below", line_width=0) # Leading
        fig.add_vrect(x0=90, x1=100, y0=100, y1=110, fillcolor="rgba(0, 0, 255, 0.03)", layer="below", line_width=0) # Improving
        fig.add_vrect(x0=90, x1=100, y0=90, y1=100, fillcolor="rgba(255, 0, 0, 0.03)", layer="below", line_width=0) # Lagging
        fig.add_vrect(x0=100, x1=110, y0=90, y1=100, fillcolor="rgba(255, 165, 0, 0.03)", layer="below", line_width=0) # Weakening

        # Quadrant Crosshair
        fig.add_shape(type="line", x0=100, y0=80, x1=100, y1=120, line=dict(color="rgba(0,0,0,0.2)", width=2, dash="dot"))
        fig.add_shape(type="line", x0=80, y0=100, x1=120, y1=100, line=dict(color="rgba(0,0,0,0.2)", width=2, dash="dot"))

        # 2. Quadrant Labels (Small Professional Text)
        fig.add_annotation(x=101, y=101.5, text="LEADING", showarrow=False, font=dict(color="green", size=10))
        fig.add_annotation(x=99, y=101.5, text="IMPROVING", showarrow=False, font=dict(color="blue", size=10))
        fig.add_annotation(x=99, y=98.5, text="LAGGING", showarrow=False, font=dict(color="red", size=10))
        fig.add_annotation(x=101, y=98.5, text="WEAKENING", showarrow=False, font=dict(color="orange", size=10))

        # 3. Plot Tickers with Spline Rounding
        for i, (t, df) in enumerate(history_data[timeframe].items()):
            color = px.colors.qualitative.Alphabet[i % 26]
            full_name = TICKER_NAMES.get(t, t)
            df_p = df.tail(tail_len)
            
            # Tail (Spline smoothed)
            fig.add_trace(go.Scatter(
                x=df_p['x'], y=df_p['y'], mode='lines', name=f"{t}", 
                line=dict(color=color, width=3, shape='spline'), opacity=0.4, showlegend=True))
            
            # Head (Diamond)
            fig.add_trace(go.Scatter(
                x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], mode='markers+text', 
                marker=dict(symbol='diamond', size=14, color=color, line=dict(width=1.5, color='white')), 
                text=[t], textposition="top center", showlegend=False,
                hovertemplate=f"<b>{t} | {full_name}</b><br>Ratio: %{{x:.2f}}<br>Mom: %{{y:.2f}}<extra></extra>"))
            
        fig.update_layout(
            template="plotly_white", 
            height=850, 
            xaxis=dict(range=[98.2, 102], title="RS-Ratio (Trend)"),
            yaxis=dict(range=[98.2, 102], title="RS-Momentum (Energy)"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Alpha Grid")
        st.dataframe(df_main.sort_values(by='RS-Ratio', ascending=False), use_container_width=True)
except Exception as e:
    st.error(f"Dashboard Error: {e}")
