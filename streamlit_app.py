import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Alpha-Scanner Pro", layout="wide")

# --- 1. THE STAPLE LISTS (LOCKED) ---
MASTER_THEMES = "SOXX, IGV, XLP, MAGS, URA, COPX, GDXJ, SILJ, IBIT, ITA, POWR, XME, XLC, XLY, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU"

STARTUP_THEMES = (
    "AMD, AMPX, BABA, BIDU, BITF, CIFR, CLSK, CORZ, CRWV, EOSE, GOOGL, HUT, IREN, LAES, NBIS, NUAI, NVDA, NVTS, "
    "PATH, POWL, RR, SERV, SNDK, TE, TSLA, TSM, WDC, ZETA, BHP, CMCL, COPX, CPER, ERO, FCX, HBM, HG=F, IE, RIO, "
    "SCCO, TGB, TMQ, AMTM, AVAV, BWXT, DPRO, ESLT, KRKNF, KRMN, KTOS, LPTH, MOB, MRCY, ONDS, OSS, PLTR, PRZO, "
    "RCAT, TDY, UMAC, CRDO, IBRX, IONQ, IONR, LAC, MP, NAK, NET, OPTT, PPTA, RZLT, SKYT, TMDX, UAMY, USAR, "
    "UUUU, WWR, ASTS, BKSY, FLY, GSAT, HEI, IRDM, KULR, LUNR, MNTS, PL, RDW, RKLB, SATL, SATS, SIDU, SPIR, UFO, VOYG, VSAT"
)

MINERS = "AFM.V, NAK, A4N.AX, CSC.AX, IVN.TO, TGB, MU, APLD"

# Name Mapping for the Master List
FUND_MAP = {
    "SOXX": "Semiconductors", "IGV": "Software", "XLP": "Cons. Staples",
    "MAGS": "Mag Seven", "URA": "Uranium", "COPX": "Copper", "GDXJ": "Junior Gold", 
    "SILJ": "Junior Silver", "IBIT": "Spot Bitcoin", "ITA": "Defense", "POWR": "Power Infra", 
    "XME": "Metals & Mining", "XLC": "Comm. Services", "XLY": "Cons. Disc.", "XLE": "Energy", 
    "XLF": "Financials", "XLV": "Health Care", "XLI": "Industrials", 
    "XLB": "Materials", "XLRE": "Real Estate", "XLK": "Technology", "XLU": "Utilities",
    "HG=F": "Copper Futures", "MU": "Micron", "APLD": "Applied Digital"
}

# --- Sidebar ---
with st.sidebar:
    st.header("🎯 Watchlist Selection")
    heap_type = st.radio("Choose Group:", ["Master Themes", "Startup", "My Miners", "Custom"])
    
    if heap_type == "Master Themes": current_list = MASTER_THEMES
    elif heap_type == "Startup": current_list = STARTUP_THEMES
    elif heap_type == "My Miners": current_list = MINERS
    else: current_list = st.session_state.get('custom_list', MASTER_THEMES)

    tickers_input = st.text_area("Ticker Heap:", value=current_list, height=200).replace("HG1!", "HG=F")
    benchmark = st.text_input("Benchmark:", value="SPY")
    
    st.markdown("---")
    timeframe = st.radio("Chart Timeframe:", ["Daily", "Weekly"])
    tail_len = st.slider("Tail Length:", 5, 30, 15)
    filter_setups = st.checkbox("Show Only Top Setups", value=True)

# --- Math Engine ---
def get_quadrant(ratio, mom):
    if ratio >= 100 and mom >= 100: return "LEADING"
    if ratio < 100 and mom >= 100: return "IMPROVING"
    if ratio < 100 and mom < 100: return "LAGGING"
    return "WEAKENING"

def get_sync_status(d_q, w_q, d_ratio):
    if d_ratio > 101.5 and d_q == "WEAKENING": return "POWER WALK"
    if d_q == "LEADING" and w_q == "LEADING": return "BULLISH SYNC"
    if d_q == "LEADING" and w_q == "IMPROVING": return "EARLY ACCEL"
    if d_q == "IMPROVING" and w_q == "LAGGING": return "DAILY PIVOT"
    return "DIVERGED"

@st.cache_data(ttl=3600)
def get_full_analysis(ticker_str, bench):
    tickers = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    all_list = list(set(tickers + [bench.upper(), "^VIX"]))
    
    d_raw = yf.download(all_list, period="2y", interval="1d", group_by='ticker', progress=False)
    w_raw = yf.download(all_list, period="2y", interval="1wk", group_by='ticker', progress=False)
    
    history, table_data = {"Daily": {}, "Weekly": {}}, []

    for t in tickers:
        try:
            if t not in d_raw.columns.get_level_values(0): continue
            
            def calc(df_raw, ticker, b_ticker):
                px, bx = df_raw[ticker]['Close'].dropna(), df_raw[b_ticker]['Close'].dropna()
                if len(px) < 40: return None
                common = px.index.intersection(bx.index)
                rel = (px.loc[common] / bx.loc[common]) * 100
                ratio = 100 + ((rel - rel.rolling(14).mean()) / rel.rolling(14).std())
                roc = ratio.pct_change(1)*100
                mom = 100 + ((roc - roc.rolling(14).mean()) / roc.rolling(14).std())
                ch_score = np.sqrt((ratio.iloc[-1] - ratio.iloc[-5])**2 + (mom.iloc[-1] - mom.iloc[-5])**2)
                rv = (df_raw[ticker]['Volume'].iloc[-1] / df_raw[ticker]['Volume'].tail(20).mean())
                return ratio, mom, round(ch_score, 2), rv

            d_res, w_res = calc(d_raw, t, bench.upper()), calc(w_raw, t, bench.upper())
            
            if d_res and w_res:
                history["Daily"][t] = pd.DataFrame({'x': d_res[0], 'y': d_res[1]}).dropna()
                history["Weekly"][t] = pd.DataFrame({'x': w_res[0], 'y': w_res[1]}).dropna()
                d_q, w_q = get_quadrant(d_res[0].iloc[-1], d_res[1].iloc[-1]), get_quadrant(w_res[0].iloc[-1], w_res[1].iloc[-1])
                status = get_sync_status(d_q, w_q, d_res[0].iloc[-1])
                
                table_data.append({
                    "Ticker": t, "Name": FUND_MAP.get(t, ""), "Sync Status": status,
                    "Daily Quad": d_q, "Weekly Quad": w_q,
                    "Daily CH": d_res[2], "Weekly CH": w_res[2],
                    "RS-Ratio": round(d_res[0].iloc[-1], 2), "Rel Vol": d_res[3]
                })
        except: continue
    return pd.DataFrame(table_data), history

# --- Execution ---
try:
    df_main, history_data = get_full_analysis(tickers_input, benchmark)

    # 1. RRG CHART
    st.subheader(f"🌀 {timeframe} Rotation (Dots: Periodic Movement)")
    fig = go.Figure()
    fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="gray", width=1, dash="dash"))
    fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="gray", width=1, dash="dash"))
    
    for i, (t, df) in enumerate(history_data[timeframe].items()):
        if filter_setups and t not in df_main[df_main['Sync Status'] != "DIVERGED"]['Ticker'].values: continue
        color = px.colors.qualitative.Plotly[i % 10]
        df_p = df.tail(tail_len)
        fig.add_trace(go.Scatter(x=df_p['x'], y=df_p['y'], mode='lines+markers', name=t, line=dict(color=color), marker=dict(size=4), legendgroup=t))
        fig.add_trace(go.Scatter(x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], mode='markers+text', marker=dict(symbol='diamond', size=12, color=color, line=dict(width=1, color='white')), text=[t], textposition="top center", showlegend=False, legendgroup=t))

    fig.update_layout(template="plotly_white", height=700, xaxis=dict(range=[97, 103], title="RS-Ratio"), yaxis=dict(range=[97, 103], title="RS-Momentum"))
    st.plotly_chart(fig, use_container_width=True)

    # 2. GRID
    st.subheader("📊 Dual-Timeframe Alpha Grid")
    df_main['sort_val'] = df_main['Sync Status'].map({"POWER WALK": 0, "BULLISH SYNC": 1, "EARLY ACCEL": 2, "DAILY PIVOT": 3, "DIVERGED": 4})
    df_display = df_main.sort_values(by=['sort_val', 'Daily CH'], ascending=[True, False]).copy()
    if filter_setups: df_display = df_display[df_display['Sync Status'] != "DIVERGED"]

    def style_sync(val):
        colors = {"POWER WALK": "#9B59B6", "BULLISH SYNC": "#2ECC71", "EARLY ACCEL": "#3498DB", "DAILY PIVOT": "#F1C40F"}
        bg = colors.get(val, "#FBFCFC")
        txt = "white" if val == "POWER WALK" else "black"
        return f'background-color: {bg}; color: {txt}; font-weight: bold'

    st.dataframe(df_display.drop(columns=['sort_val']).style.map(style_sync, subset=['Sync Status']).format({"Rel Vol": "{:.2f}x"}), use_container_width=True)

except Exception as e:
    st.error(f"Dashboard Error: {e}")
