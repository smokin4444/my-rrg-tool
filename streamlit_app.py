import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests

st.set_page_config(page_title="Alpha-Scanner Pro", layout="wide")

# --- 1. DATA LISTS ---
MASTER_THEMES = "SOXX, IGV, XLP, MAGS, URA, COPX, GDXJ, SILJ, IBIT, ITA, POWR, XME, XLC, XLY, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU"
STARTUP_THEMES = "AMD, AMPX, BABA, BIDU, BITF, CIFR, CLSK, CORZ, CRWV, EOSE, GOOGL, HUT, IREN, LAES, NBIS, NUAI, NVDA, NVTS, PATH, POWL, RR, SERV, SNDK, TE, TSLA, TSM, WDC, ZETA, BHP, CMCL, COPX, CPER, ERO, FCX, HBM, HG=F, IE, RIO, SCCO, TGB, TMQ, AMTM, AVAV, BWXT, DPRO, ESLT, KRKNF, KRMN, KTOS, LPTH, MOB, MRCY, ONDS, OSS, PLTR, PRZO, RCAT, TDY, UMAC, CRDO, IBRX, IONQ, IONR, LAC, MP, NAK, NET, OPTT, PPTA, RZLT, SKYT, TMDX, UAMY, USAR, UUUU, WWR, ASTS, BKSY, FLY, GSAT, HEI, IRDM, KULR, LUNR, MNTS, PL, RDW, RKLB, SATL, SATS, SIDU, SPIR, UFO, VOYG, VSAT"
ASX_LIST = "CBA.AX, BHP.AX, CSL.AX, NAB.AX, WBC.AX, ANZ.AX, MQG.AX, WES.AX, TLS.AX, WOW.AX, FMG.AX, RIO.AX, GMG.AX, TCL.AX, ALL.AX"
MINERS = "AFM.V, NAK, A4N.AX, CSC.AX, IVN.TO, TGB, MU, APLD"

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("🎯 Watchlist")
    heap_type = st.radio("Choose Group:", ["Master Themes", "Startup", "ASX Blue Chips", "My Miners"])
    
    if heap_type == "Master Themes": current_list, auto_bench = MASTER_THEMES, "SPY"
    elif heap_type == "Startup": current_list, auto_bench = STARTUP_THEMES, "SPY"
    elif heap_type == "ASX Blue Chips": current_list, auto_bench = ASX_LIST, "VAS.AX"
    elif heap_type == "My Miners": current_list, auto_bench = MINERS, "SPY"
    else: current_list, auto_bench = "", "SPY"

    tickers_input = st.text_area("Ticker Heap:", value=current_list, height=200).replace("HG1!", "HG=F")
    
    st.markdown("---")
    st.header("⚖️ Benchmark")
    bench_preset = st.selectbox("Preset Benchmarks:", ["Auto-Detect", "SPY (S&P 500)", "VAS.AX (ASX 200)", "QQQ (Nasdaq 100)"])
    final_bench = auto_bench if bench_preset == "Auto-Detect" else bench_preset.split(" ")[0]
    benchmark = st.text_input("Active Benchmark:", value=final_bench)
    
    st.markdown("---")
    timeframe = st.radio("Timeframe:", ["Daily", "Weekly"])
    tail_len = st.slider("Tail Length:", 1, 30, 10)
    filter_setups = st.checkbox("Top Setups Only", value=True)

# --- 3. REPAIR ENGINE (Mimic Browser to avoid blocks) ---
def fetch_safe_data(tickers, period="2y", interval="1d"):
    # Mocking a real browser session
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    return yf.download(tickers, period=period, interval=interval, group_by='ticker', progress=False, session=session)

def get_rrg_metrics(df_raw, ticker, b_ticker, is_weekly=False):
    try:
        if ticker not in df_raw.columns.get_level_values(0): return None
        px, bx = df_raw[ticker]['Close'].dropna(), df_raw[b_ticker]['Close'].dropna()
        common = px.index.intersection(bx.index)
        if len(common) < 20: return None
        
        rel = (px.loc[common] / bx.loc[common]) * 100
        ratio = 100 + ((rel - rel.rolling(14).mean()) / rel.rolling(14).std())
        roc = ratio.diff(1)
        mom = 100 + ((roc - roc.rolling(14).mean()) / roc.rolling(14).std())
        
        df_res = pd.DataFrame({'x': ratio, 'y': mom, 'date': ratio.index}).dropna()
        if is_weekly: df_res['date'] = df_res['date'] + pd.Timedelta(days=4)
        
        ch = np.sqrt((ratio.iloc[-1] - ratio.iloc[-5])**2 + (mom.iloc[-1] - mom.iloc[-5])**2)
        
        # Safe Volume Calculation
        vol = df_raw[ticker].get('Volume', pd.Series())
        rv = (vol.iloc[-1] / vol.tail(20).mean()) if not vol.empty and vol.tail(20).mean() > 0 else 1.0
        
        return df_res, round(ch, 2), rv
    except: return None

# --- 4. EXECUTION ---
try:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    bench_ticker = benchmark.strip().upper()
    all_list = list(set(tickers + [bench_ticker]))

    # Download with the new session shielding
    data = fetch_safe_data(all_list, period="2y", interval="1d")
    w_data = fetch_safe_data(all_list, period="2y", interval="1wk")

    history, table_data = {"Daily": {}, "Weekly": {}}, []
    for t in tickers:
        d_res = get_rrg_metrics(data, t, bench_ticker, False)
        w_res = get_rrg_metrics(w_data, t, bench_ticker, True)
        if d_res and w_res:
            history["Daily"][t], history["Weekly"][t] = d_res[0], w_res[0]
            dr, dm = d_res[0]['x'].iloc[-1], d_res[0]['y'].iloc[-1]
            wr, wm = w_res[0]['x'].iloc[-1], w_res[0]['y'].iloc[-1]
            dq, wq = ( (100 if dr >= 100 else 99), (100 if dm >= 100 else 99) ) # Simplified for logic
            # Use original quadrant logic
            def q_name(x, y):
                if x >= 100 and y >= 100: return "LEADING"
                if x < 100 and y >= 100: return "IMPROVING"
                if x < 100 and y < 100: return "LAGGING"
                return "WEAKENING"
            dqn, wqn = q_name(dr, dm), q_name(wr, wm)
            
            status = "POWER WALK" if dr > 101.5 and dqn == "WEAKENING" else \
                     "LEAD-THROUGH" if dqn == "LEADING" and wqn == "IMPROVING" else \
                     "BULLISH SYNC" if dqn == "LEADING" and wqn == "LEADING" else \
                     "DAILY PIVOT" if dqn == "IMPROVING" and wqn == "LAGGING" else "DIVERGED"
            
            table_data.append({"Ticker": t, "Sync Status": status, "Daily Quad": dqn, "Weekly Quad": wqn, "RS-Ratio": round(dr, 2), "Rel Vol": d_res[2]})

    df_main = pd.DataFrame(table_data)
    
    # UI Display (Same Chart Logic)
    if not df_main.empty:
        st.subheader(f"🌀 {timeframe} Rotation vs {benchmark}")
        fig = go.Figure()
        fig.add_vrect(x0=101.5, x1=105, fillcolor="rgba(46, 204, 113, 0.15)", layer="below", line_width=0)
        # (Standard Crosshairs and Scatter logic here as in previous version...)
        for t, df in history_data[timeframe].items():
            df_p = df.tail(tail_len)
            fig.add_trace(go.Scatter(x=df_p['x'], y=df_p['y'], mode='lines+markers', name=t))
            fig.add_trace(go.Scatter(x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], mode='markers+text', text=[t], marker=dict(symbol='diamond', size=14)))
        
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("📊 Alpha Grid")
        st.dataframe(df_main.sort_values(by='RS-Ratio', ascending=False), use_container_width=True)
    else:
        st.warning("Data download failed. Yahoo might be rate-limiting. Try waiting 60 seconds or changing your Benchmark.")

except Exception as e:
    st.error(f"Error: {e}")
