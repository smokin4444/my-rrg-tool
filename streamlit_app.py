import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import interp1d

st.set_page_config(page_title="Alpha-Scanner Pro", layout="wide")

# --- TICKER GROUPS ---
MINERS = "AFM.V, NAK, A4N.AX, CSC.AX, IVN.TO, TGB"
ELITE_THEMES = (
    "AMD, AMPX, BABA, BIDU, BITF, CIFR, CLSK, CORZ, CRWV, EOSE, GOOGL, HUT, IREN, "
    "LAES, NBIS, NUAI, NVDA, NVTS, PATH, POWL, RR, SERV, SNDK, TE, TSLA, TSM, WDC, ZETA, "
    "BHP, CMCL, COPX, CPER, ERO, FCX, HBM, RIO, SCCO, TGB, TMQ, "
    "AMTM, AVAV, BWXT, DPRO, ESLT, KRKNF, KRMN, KTOS, LPTH, MOB, MRCY, ONDS, OSS, "
    "PLTR, PRZO, RCAT, TDY, UMAC, CRDO, IBRX, IONQ, IONR, LAC, MP, NAK, NET, OPTT, "
    "PPTA, RZLT, SKYT, TMDX, UAMY, USAR, UUUU, WWR, ASTS, BKSY, FLY, GSAT, HEI, "
    "IRDM, KULR, LUNR, MNTS, PL, RDW, RKLB, SATL, SATS, SIDU, SPIR, UFO, VOYG, VSAT"
)

# --- Sidebar ---
with st.sidebar:
    st.header("🎯 Watchlist Selection")
    heap_type = st.radio("Choose Group:", ["My Miners", "Elite Themes", "Custom"])
    
    if heap_type == "My Miners": current_list = MINERS
    elif heap_type == "Elite Themes": current_list = ELITE_THEMES
    else: current_list = st.session_state.get('custom_list', ELITE_THEMES)

    tickers_input = st.text_area("Ticker Heap:", value=current_list, height=150)
    benchmark = st.text_input("Benchmark:", value="SPY")
    
    st.markdown("---")
    st.header("🕰️ Time Controls")
    timeframe = st.radio("Chart Timeframe:", ["Daily", "Weekly"], index=0)
    tail_len = st.slider("Tail Length (Dots):", 5, 30, 15)
    filter_setups = st.checkbox("Show Only Top Setups", value=True)

# --- Math Engine ---
def get_quadrant(ratio, mom):
    if ratio >= 100 and mom >= 100: return "LEADING"
    if ratio < 100 and mom >= 100: return "IMPROVING"
    if ratio < 100 and mom < 100: return "LAGGING"
    return "WEAKENING"

@st.cache_data(ttl=3600)
def get_full_analysis(ticker_str, bench):
    # Clean tickers: removing special chars that might break yfinance like '!'
    tickers = [t.strip().upper() for t in ticker_str.replace("!", "").split(",") if t.strip()]
    all_list = list(set(tickers + [bench.upper(), "^VIX"]))
    
    d_raw = yf.download(all_list, period="2y", interval="1d", group_by='ticker')
    w_raw = yf.download(all_list, period="2y", interval="1wk", group_by='ticker')
    
    vix = yf.download("^VIX", period="1d")['Close'].iloc[-1]
    
    history = {"Daily": {}, "Weekly": {}}
    table_data = []

    for t in tickers:
        try:
            def calc_metrics(df_raw, ticker, benchmark):
                # Safety Check: Does ticker exist in the download?
                if ticker not in df_raw.columns.get_level_values(0): return None
                
                px_data = df_raw[ticker]['Close'].dropna()
                bx_data = df_raw[benchmark]['Close'].dropna()
                
                # Safety Check: Is there actually data in the series?
                if len(px_data) < 30 or len(bx_data) < 30: return None
                
                common = px_data.index.intersection(bx_data.index)
                rel = (px_data.loc[common] / bx_data.loc[common]) * 100
                ratio = 100 + ((rel - rel.rolling(14).mean()) / rel.rolling(14).std())
                roc = ratio.pct_change(1) * 100
                mom = 100 + ((roc - roc.rolling(14).mean()) / roc.rolling(14).std())
                
                v = np.sqrt((ratio.iloc[-1] - ratio.iloc[-5])**2 + (mom.iloc[-1] - mom.iloc[-5])**2)
                rv = (df_raw[ticker]['Volume'].iloc[-1] / df_raw[ticker]['Volume'].tail(20).mean())
                return ratio, mom, round((v * 0.7) + (rv * 0.3), 2), rv

            d_res = calc_metrics(d_raw, t, bench.upper())
            w_res = calc_metrics(w_raw, t, bench.upper())
            
            if d_res and w_res:
                d_rat, d_mom, d_ch, d_rv = d_res
                w_rat, w_mom, w_ch, _ = w_res
                
                history["Daily"][t] = pd.DataFrame({'x': d_rat, 'y': d_mom}).dropna()
                history["Weekly"][t] = pd.DataFrame({'x': w_rat, 'y': w_mom}).dropna()
                
                d_q = get_quadrant(d_rat.iloc[-1], d_mom.iloc[-1])
                w_q = get_quadrant(w_rat.iloc[-1], w_mom.iloc[-1])
                
                status = "BULLISH SYNC" if d_q == "LEADING" and w_q == "LEADING" else \
                         "EARLY ACCEL" if d_q == "LEADING" and w_q == "IMPROVING" else \
                         "DAILY PIVOT" if d_q == "IMPROVING" and w_q == "LAGGING" else "DIVERGED"
                
                table_data.append({
                    "Ticker": t, "Sync Status": status,
                    "Daily Quad": d_q, "Weekly Quad": w_q,
                    "Daily CH": d_ch, "Weekly CH": w_ch, "Rel Vol": d_rv
                })
        except:
            continue # Skip tickers that cause errors

    return pd.DataFrame(table_data), history, vix

# --- Execution ---
try:
    df_main, history_data, vix_val = get_full_analysis(tickers_input, benchmark)
    
    st.info(f"🛡️ **VIX:** {vix_val:.2f} | **Benchmark:** {benchmark.upper()}")

    # 1. RRG CHART
    st.subheader(f"🌀 {timeframe} Rotation")
    fig = go.Figure()
    fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="rgba(0,0,0,0.2)", width=2))
    fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="rgba(0,0,0,0.2)", width=2))
    
    for i, (t, df) in enumerate(history_data[timeframe].items()):
        if filter_setups and t not in df_main[df_main['Sync Status'].isin(["BULLISH SYNC", "EARLY ACCEL", "DAILY PIVOT"])]['Ticker'].values:
            continue
            
        color = px.colors.qualitative.Plotly[i % 10]
        df_p = df.tail(tail_len)
        fig.add_trace(go.Scatter(x=df_p['x'], y=df_p['y'], mode='lines+markers', name=t, line=dict(width=2, color=color), marker=dict(size=4, color=color), opacity=0.4))
        fig.add_trace(go.Scatter(x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], mode='markers+text', marker=dict(symbol='diamond', size=12, color=color, line=dict(width=1, color='white')), text=[t], textposition="top center", showlegend=False))

    fig.update_layout(template="plotly_white", height=700, xaxis=dict(range=[97, 103]), yaxis=dict(range=[97, 103]))
    st.plotly_chart(fig, use_container_width=True)

    # 2. ALIGNMENT GRID
    st.subheader("📊 Elite Alpha Grid")
    df_main['sort_val'] = df_main['Sync Status'].map({"BULLISH SYNC": 0, "EARLY ACCEL": 1, "DAILY PIVOT": 2, "DIVERGED": 3})
    df_display = df_main.sort_values(by=['sort_val', 'Daily CH'], ascending=[True, False])
    
    if filter_setups:
        df_display = df_display[df_display['Sync Status'] != "DIVERGED"]

    st.dataframe(
        df_display.drop(columns=['sort_val']).style.map(lambda x: f'background-color: {"#2ECC71" if "BULLISH" in x else "#3498DB" if "ACCEL" in x else "#F1C40F" if "PIVOT" in x else "#FBFCFC"}; color: black; font-weight: bold', subset=['Sync Status'])
        .format({"Rel Vol": "{:.2f}x"}), use_container_width=True
    )

except Exception as e:
    st.error(f"Error: {e}")
