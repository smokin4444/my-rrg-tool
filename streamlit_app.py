import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Alpha-Scanner Pro", layout="wide")

# --- DATA LISTS ---
MASTER_THEMES = "SOXX, IGV, XLP, MAGS, URA, COPX, GDXJ, SILJ, IBIT, ITA, POWR, XME, XLC, XLY, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU"
STARTUP_THEMES = "AMD, AMPX, BABA, BIDU, BITF, CIFR, CLSK, CORZ, CRWV, EOSE, GOOGL, HUT, IREN, LAES, NBIS, NUAI, NVDA, NVTS, PATH, POWL, RR, SERV, SNDK, TE, TSLA, TSM, WDC, ZETA, BHP, CMCL, COPX, CPER, ERO, FCX, HBM, HG=F, IE, RIO, SCCO, TGB, TMQ, AMTM, AVAV, BWXT, DPRO, ESLT, KRKNF, KRMN, KTOS, LPTH, MOB, MRCY, ONDS, OSS, PLTR, PRZO, RCAT, TDY, UMAC, CRDO, IBRX, IONQ, IONR, LAC, MP, NAK, NET, OPTT, PPTA, RZLT, SKYT, TMDX, UAMY, USAR, UUUU, WWR, ASTS, BKSY, FLY, GSAT, HEI, IRDM, KULR, LUNR, MNTS, PL, RDW, RKLB, SATL, SATS, SIDU, SPIR, UFO, VOYG, VSAT"
ASX_LIST = "CBA.AX, BHP.AX, CSL.AX, NAB.AX, WBC.AX, ANZ.AX, MQG.AX, WES.AX, TLS.AX, WOW.AX, FMG.AX, RIO.AX, GMG.AX, TCL.AX, ALL.AX"
MINERS = "AFM.V, NAK, A4N.AX, CSC.AX, IVN.TO, TGB, MU, APLD"

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎯 Watchlist")
    heap_type = st.radio("Choose Group:", ["Master Themes", "Startup", "ASX Blue Chips", "My Miners"])
    
    # Auto-detect logic
    if heap_type == "Master Themes": current_list, auto_bench = MASTER_THEMES, "SPY"
    elif heap_type == "Startup": current_list, auto_bench = STARTUP_THEMES, "SPY"
    elif heap_type == "ASX Blue Chips": current_list, auto_bench = ASX_LIST, "VAS.AX"
    elif heap_type == "My Miners": current_list, auto_bench = MINERS, "SPY"
    else: current_list, auto_bench = "", "SPY"

    tickers_input = st.text_area("Ticker Heap:", value=current_list, height=200).replace("HG1!", "HG=F")
    
    st.markdown("---")
    st.header("⚖️ Benchmark")
    bench_preset = st.selectbox("Preset Benchmarks:", ["Auto-Detect", "SPY (S&P 500)", "VAS.AX (ASX 200)", "QQQ (Nasdaq 100)", "NDQ.AX (ASX Nasdaq)"])
    
    # Benchmark Selector
    if bench_preset == "Auto-Detect": final_bench = auto_bench
    elif "SPY" in bench_preset: final_bench = "SPY"
    elif "VAS.AX" in bench_preset: final_bench = "VAS.AX"
    elif "QQQ" in bench_preset: final_bench = "QQQ"
    else: final_bench = "NDQ.AX"
    
    benchmark = st.text_input("Active Benchmark:", value=final_bench)
    st.markdown("---")
    timeframe = st.radio("Timeframe:", ["Daily", "Weekly"])
    tail_len = st.slider("Tail Length:", 1, 30, 10)
    filter_setups = st.checkbox("Top Setups Only", value=True)

# --- ENGINE ---
def get_rrg_metrics(df_raw, ticker, b_ticker, is_weekly=False):
    try:
        px, bx = df_raw[ticker]['Close'].dropna(), df_raw[b_ticker]['Close'].dropna()
        common = px.index.intersection(bx.index)
        if len(common) < 30: return None
        rel = (px.loc[common] / bx.loc[common]) * 100
        ratio = 100 + ((rel - rel.rolling(14).mean()) / rel.rolling(14).std())
        roc = ratio.diff(1)
        mom = 100 + ((roc - roc.rolling(14).mean()) / roc.rolling(14).std())
        df_res = pd.DataFrame({'x': ratio, 'y': mom, 'date': ratio.index}).dropna()
        if is_weekly: df_res['date'] = df_res['date'] + pd.Timedelta(days=4)
        ch = np.sqrt((ratio.iloc[-1] - ratio.iloc[-5])**2 + (mom.iloc[-1] - mom.iloc[-5])**2)
        rv = (df_raw[ticker]['Volume'].iloc[-1] / df_raw[ticker]['Volume'].tail(20).mean())
        return df_res, round(ch, 2), rv
    except: return None

def get_quadrant(x, y):
    if x >= 100 and y >= 100: return "LEADING"
    if x < 100 and y >= 100: return "IMPROVING"
    if x < 100 and y < 100: return "LAGGING"
    return "WEAKENING"

@st.cache_data(ttl=3600)
def run_analysis(ticker_str, bench):
    tickers = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    all_list = list(set(tickers + [bench.strip().upper()]))
    data = yf.download(all_list, period="2y", interval="1d", group_by='ticker', progress=False)
    w_data = yf.download(all_list, period="2y", interval="1wk", group_by='ticker', progress=False)
    history, table_data = {"Daily": {}, "Weekly": {}}, []
    for t in tickers:
        d_res = get_rrg_metrics(data, t, bench.upper(), False)
        w_res = get_rrg_metrics(w_data, t, bench.upper(), True)
        if d_res and w_res:
            history["Daily"][t], history["Weekly"][t] = d_res[0], w_res[0]
            dq, wq = get_quadrant(d_res[0]['x'].iloc[-1], d_res[0]['y'].iloc[-1]), get_quadrant(w_res[0]['x'].iloc[-1], w_res[0]['y'].iloc[-1])
            dr = d_res[0]['x'].iloc[-1]
            status = "POWER WALK" if dr > 101.5 and dq == "WEAKENING" else \
                     "LEAD-THROUGH" if dq == "LEADING" and wq == "IMPROVING" else \
                     "BULLISH SYNC" if dq == "LEADING" and wq == "LEADING" else \
                     "DAILY PIVOT" if dq == "IMPROVING" and wq == "LAGGING" else "DIVERGED"
            table_data.append({"Ticker": t, "Sync Status": status, "Daily Quad": dq, "Weekly Quad": wq, "Daily CH": d_res[1], "RS-Ratio": round(dr, 2), "Rel Vol": d_res[2]})
    return pd.DataFrame(table_data), history

# --- UI ---
try:
    df_main, history_data = run_analysis(tickers_input, benchmark)
    st.subheader(f"🌀 {timeframe} Rotation vs {benchmark}")
    fig = go.Figure()
    fig.add_vrect(x0=101.5, x1=105, fillcolor="rgba(46, 204, 113, 0.15)", layer="below", line_width=0)
    fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="gray", width=1, dash="dash"))
    fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="gray", width=1, dash="dash"))
    for i, (t, df) in enumerate(history_data[timeframe].items()):
        if filter_setups and t not in df_main[df_main['Sync Status'] != "DIVERGED"]['Ticker'].values: continue
        color = px.colors.qualitative.Plotly[i % 10]
        df_p = df.tail(tail_len).copy()
        df_p['q'] = df_p.apply(lambda r: get_quadrant(r['x'], r['y']), axis=1)
        df_p['d'] = df_p['date'].dt.strftime('%b %d')
        if tail_len > 1:
            fig.add_trace(go.Scatter(x=df_p['x'], y=df_p['y'], mode='lines+markers', name=t, line=dict(color=color, width=2, shape='spline'), marker=dict(size=8, opacity=0.7), customdata=np.stack((df_p['d'], df_p['q']), axis=-1), hovertemplate="<b>"+t+"</b><br>Date: %{customdata[0]}<br>Quad: %{customdata[1]}<extra></extra>", opacity=0.4, legendgroup=t))
        fig.add_trace(go.Scatter(x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], mode='markers+text', marker=dict(symbol='diamond', size=16, color=color, line=dict(width=2, color='white')), text=[t], textposition="top center", showlegend=(tail_len == 1), legendgroup=t))
    fig.update_layout(template="plotly_white", height=750, xaxis=dict(range=[98, 102.5]), yaxis=dict(range=[98, 102.5]))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Alpha Grid")
    def style_sync(val):
        colors = {"POWER WALK": "#9B59B6", "LEAD-THROUGH": "#E67E22", "BULLISH SYNC": "#2ECC71", "DAILY PIVOT": "#F1C40F"}
        return f'background-color: {colors.get(val, "#FBFCFC")}; color: {"white" if val in ["POWER WALK", "LEAD-THROUGH"] else "black"}; font-weight: bold'
    df_disp = df_main.sort_values(by=['RS-Ratio'], ascending=False)
    if filter_setups: df_disp = df_disp[df_disp['Sync Status'] != "DIVERGED"]
    st.dataframe(df_disp.style.map(style_sync, subset=['Sync Status']).format({"Rel Vol": "{:.2f}x"}), use_container_width=True)
except Exception as e:
    st.error(f"Error: {e}")
