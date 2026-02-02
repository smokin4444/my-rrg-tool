import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- APP CONFIG ---
st.set_page_config(page_title="Alpha-Scanner Pro", layout="wide")

# --- 1. PERMANENT DATA REPOSITORY ---
MASTER_THEMES = "SOXX, IGV, XLP, MAGS, URA, COPX, GDXJ, SILJ, IBIT, ITA, POWR, XME, XLC, XLY, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU"
STARTUP_THEMES = "AMD, AMPX, BABA, BIDU, BITF, CIFR, CLSK, CORZ, CRWV, EOSE, GOOGL, HUT, IREN, LAES, NBIS, NUAI, NVDA, NVTS, PATH, POWL, RR, SERV, SNDK, TE, TSLA, TSM, WDC, ZETA, BHP, CMCL, COPX, CPER, ERO, FCX, HBM, HG=F, IE, RIO, SCCO, TGB, TMQ, AMTM, AVAV, BWXT, DPRO, ESLT, KRKNF, KRMN, KTOS, LPTH, MOB, MRCY, ONDS, OSS, PLTR, PRZO, RCAT, TDY, UMAC, CRDO, IBRX, IONQ, IONR, LAC, MP, NAK, NET, OPTT, PPTA, RZLT, SKYT, TMDX, UAMY, USAR, UUUU, WWR, ASTS, BKSY, FLY, GSAT, HEI, IRDM, KULR, LUNR, MNTS, PL, RDW, RKLB, SATL, SATS, SIDU, SPIR, UFO, VOYG, VSAT"
SMALL_CAP_SEMIS = "LSCC, AMKR, ACLS, RMBS, DIOD, INDI, MTSI, POWI, SLAB, CRUS, PI, CAMT, PDFS, NVTS, CEVA, QUOT, MOSY, SKYT, IONQ, VRT"
MINERS = "AFM.V, NAK, A4N.AX, CSC.AX, IVN.TO, TGB, MU, APLD"

FUND_MAP = {
    "SOXX": "Semiconductors", "IGV": "Software", "XLP": "Cons. Staples", "MAGS": "Mag Seven",
    "URA": "Uranium", "COPX": "Copper", "GDXJ": "Junior Gold", "SILJ": "Junior Silver",
    "IBIT": "Spot Bitcoin", "ITA": "Defense", "POWR": "Power Infra", "XME": "Metals & Mining",
    "XLC": "Comm. Services", "XLY": "Cons. Disc.", "XLE": "Energy", "XLF": "Financials",
    "XLV": "Health Care", "XLI": "Industrials", "XLB": "Materials", "XLRE": "Real Estate",
    "XLK": "Technology", "XLU": "Utilities", "HG=F": "Copper Futures", "MU": "Micron", 
    "APLD": "Applied Digital", "ASTS": "AST SpaceMobile", "VRT": "Vertiv (AI Infra)"
}

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("🎯 Watchlist")
    heap_type = st.radio("Choose Group:", ["Master Themes", "Startup", "Small Cap Semis", "My Miners"])
    current_list = MASTER_THEMES if heap_type == "Master Themes" else STARTUP_THEMES if heap_type == "Startup" else SMALL_CAP_SEMIS if heap_type == "Small Cap Semis" else MINERS
    tickers_input = st.text_area("Ticker Heap:", value=current_list, height=200).replace("HG1!", "HG=F")
    benchmark = st.text_input("Benchmark (vs):", value="SPY")
    st.markdown("---")
    timeframe = st.radio("Timeframe:", ["Daily", "Weekly"])
    tail_len = st.slider("Tail Length (1 = Snapshot):", 1, 30, 10)
    filter_setups = st.checkbox("Top Setups Only", value=True)

# --- 3. MATH ENGINE ---
def get_rrg_metrics(df_raw, ticker, b_ticker, is_weekly=False):
    try:
        px = df_raw[ticker]['Close'].dropna()
        bx = df_raw[b_ticker]['Close'].dropna()
        common = px.index.intersection(bx.index)
        if len(common) < 60: return None
        
        rel_strength = (px.loc[common] / bx.loc[common]) * 100
        ratio = 100 + ((rel_strength - rel_strength.rolling(14).mean()) / rel_strength.rolling(14).std())
        roc = ratio.diff(1)
        mom = 100 + ((roc - roc.rolling(14).mean()) / roc.rolling(14).std())
        ch = np.sqrt((ratio.iloc[-1] - ratio.iloc[-5])**2 + (mom.iloc[-1] - mom.iloc[-5])**2)
        rv = (df_raw[ticker]['Volume'].iloc[-1] / df_raw[ticker]['Volume'].tail(20).mean())
        
        df_res = pd.DataFrame({'x': ratio, 'y': mom, 'date': ratio.index}).dropna()
        
        # Shift weekly dates to show Friday COB
        if is_weekly:
            df_res['date'] = df_res['date'] + pd.Timedelta(days=4)
            
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
    all_list = list(set(tickers + [bench.upper()]))
    data = yf.download(all_list, period="2y", interval="1d", group_by='ticker', progress=False)
    w_data = yf.download(all_list, period="2y", interval="1wk", group_by='ticker', progress=False)
    
    history, table_data = {"Daily": {}, "Weekly": {}}, []
    for t in tickers:
        d_res = get_rrg_metrics(data, t, bench.upper(), is_weekly=False)
        w_res = get_rrg_metrics(w_data, t, bench.upper(), is_weekly=True)
        if d_res and w_res:
            history["Daily"][t], history["Weekly"][t] = d_res[0], w_res[0]
            dq, wq = get_quadrant(d_res[0]['x'].iloc[-1], d_res[0]['y'].iloc[-1]), get_quadrant(w_res[0]['x'].iloc[-1], w_res[0]['y'].iloc[-1])
            dr = d_res[0]['x'].iloc[-1]
            
            # POWER WALK priority
            status = "POWER WALK" if dr > 101.5 and dq == "WEAKENING" else \
                     "LEAD-THROUGH" if dq == "LEADING" and wq == "IMPROVING" else \
                     "BULLISH SYNC" if dq == "LEADING" and wq == "LEADING" else \
                     "DAILY PIVOT" if dq == "IMPROVING" and wq == "LAGGING" else "DIVERGED"
            
            table_data.append({
                "Ticker": t, "Name": FUND_MAP.get(t, ""), "Sync Status": status,
                "Daily Quad": dq, "Weekly Quad": wq, "Daily CH": d_res[1],
                "RS-Ratio": round(dr, 2), "Rel Vol": d_res[2]
            })
    return pd.DataFrame(table_data), history

# --- 4. UI ---
try:
    df_main, history_data = run_analysis(tickers_input, benchmark)
    
    st.subheader(f"🌀 {timeframe} Rotation (Properly Scaled)")
    fig = go.Figure()
    fig.add_vrect(x0=101.5, x1=105, fillcolor="rgba(46, 204, 113, 0.15)", layer="below", line_width=0, annotation_text="POWER ZONE")
    fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="gray", width=1, dash="dash"))
    fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="gray", width=1, dash="dash"))
    
    for i, (t, df) in enumerate(history_data[timeframe].items()):
        if filter_setups and t not in df_main[df_main['Sync Status'] != "DIVERGED"]['Ticker'].values: continue
        color = px.colors.qualitative.Plotly[i % 10]
        df_p = df.tail(tail_len).copy()
        df_p['quad'] = df_p.apply(lambda row: get_quadrant(row['x'], row['y']), axis=1)
        df_p['date_label'] = df_p['date'].dt.strftime('%b %d')

        if tail_len > 1:
            fig.add_trace(go.Scatter(
                x=df_p['x'], y=df_p['y'], mode='lines+markers', name=t,
                line=dict(color=color, width=2, shape='spline'),
                marker=dict(size=8, opacity=0.7, line=dict(width=1, color='white')),
                customdata=np.stack((df_p['date_label'], df_p['quad']), axis=-1),
                hovertemplate="<b>" + t + "</b><br>Date: %{customdata[0]}<br>Quad: %{customdata[1]}<extra></extra>",
                opacity=0.4, legendgroup=t
            ))
        
        fig.add_trace(go.Scatter(
            x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], mode='markers+text',
            marker=dict(symbol='diamond', size=16, color=color, line=dict(width=2, color='white')),
            text=[t], textposition="top center", showlegend=(tail_len == 1), legendgroup=t,
            customdata=[[df_p['date_label'].iloc[-1], df_p['quad'].iloc[-1]]],
            hovertemplate="<b>" + t + "</b> (Latest)<br>Date: %{customdata[0]}<br>Quad: %{customdata[1]}<extra></extra>"
        ))

    fig.update_layout(template="plotly_white", height=750, xaxis=dict(range=[98, 102.5]), yaxis=dict(range=[98, 102.5]))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Alpha Grid")
    def style_sync(val):
        colors = {"POWER WALK": "#9B59B6", "LEAD-THROUGH": "#E67E22", "BULLISH SYNC": "#2ECC71", "DAILY PIVOT": "#F1C40F"}
        bg = colors.get(val, "#FBFCFC")
        txt = "white" if val in ["POWER WALK", "LEAD-THROUGH"] else "black"
        return f'background-color: {bg}; color: {txt}; font-weight: bold'

    df_disp = df_main.sort_values(by=['RS-Ratio'], ascending=False)
    if filter_setups: df_disp = df_disp[df_disp['Sync Status'] != "DIVERGED"]
    st.dataframe(df_disp.style.map(style_sync, subset=['Sync Status']).format({"Rel Vol": "{:.2f}x"}), use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
