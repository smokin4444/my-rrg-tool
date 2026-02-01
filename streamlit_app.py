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

STARTUP_THEMES = (
    "AMD, AMPX, BABA, BIDU, BITF, CIFR, CLSK, CORZ, CRWV, EOSE, GOOGL, HUT, IREN, LAES, NBIS, NUAI, NVDA, NVTS, "
    "PATH, POWL, RR, SERV, SNDK, TE, TSLA, TSM, WDC, ZETA, BHP, CMCL, COPX, CPER, ERO, FCX, HBM, HG=F, IE, RIO, "
    "SCCO, TGB, TMQ, AMTM, AVAV, BWXT, DPRO, ESLT, KRKNF, KRMN, KTOS, LPTH, MOB, MRCY, ONDS, OSS, PLTR, PRZO, "
    "RCAT, TDY, UMAC, CRDO, IBRX, IONQ, IONR, LAC, MP, NAK, NET, OPTT, PPTA, RZLT, SKYT, TMDX, UAMY, USAR, "
    "UUUU, WWR, ASTS, BKSY, FLY, GSAT, HEI, IRDM, KULR, LUNR, MNTS, PL, RDW, RKLB, SATL, SATS, SIDU, SPIR, UFO, VOYG, VSAT"
)

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

# --- 2. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("🎯 Watchlist Selection")
    heap_type = st.radio("Choose Group:", ["Master Themes", "Startup", "Small Cap Semis", "My Miners", "Custom"])
    
    if heap_type == "Master Themes": current_list = MASTER_THEMES
    elif heap_type == "Startup": current_list = STARTUP_THEMES
    elif heap_type == "Small Cap Semis": current_list = SMALL_CAP_SEMIS
    elif heap_type == "My Miners": current_list = MINERS
    else: current_list = st.session_state.get('custom_list', MASTER_THEMES)

    tickers_input = st.text_area("Ticker Heap:", value=current_list, height=200).replace("HG1!", "HG=F")
    benchmark = st.text_input("Benchmark (vs):", value="SPY")
    
    st.markdown("---")
    st.header("🕰️ Time Controls")
    timeframe = st.radio("Chart Timeframe:", ["Daily", "Weekly"])
    tail_len = st.slider("Tail Length (1 = Snapshot):", 1, 30, 10)
    filter_setups = st.checkbox("Show Only Top Setups", value=True)

# --- 3. MATH ENGINE (Normalized RRG) ---
def get_rrg_metrics(df_raw, ticker, b_ticker):
    try:
        px = df_raw[ticker]['Close'].dropna()
        bx = df_raw[b_ticker]['Close'].dropna()
        common = px.index.intersection(bx.index)
        if len(common) < 60: return None
        
        # RS Ratio (X-Axis)
        rel_strength = (px.loc[common] / bx.loc[common]) * 100
        rs_ratio = 100 + ((rel_strength - rel_strength.rolling(14).mean()) / rel_strength.rolling(14).std())
        
        # RS Momentum (Y-Axis) - Normalized to prevent "Long Snakes"
        roc = rs_ratio.diff(1)
        rs_mom = 100 + ((roc - roc.rolling(14).mean()) / roc.rolling(14).std())
        
        # Cross-Hair Velocity (Magnitude of recent move)
        ch_score = np.sqrt((rs_ratio.iloc[-1] - rs_ratio.iloc[-5])**2 + (rs_mom.iloc[-1] - rs_mom.iloc[-5])**2)
        # Relative Volume
        rel_vol = (df_raw[ticker]['Volume'].iloc[-1] / df_raw[ticker]['Volume'].tail(20).mean())
        
        return rs_ratio.dropna(), rs_mom.dropna(), round(ch_score, 2), rel_vol
    except:
        return None

def get_quadrant(ratio, mom):
    if ratio >= 100 and mom >= 100: return "LEADING"
    if ratio < 100 and mom >= 100: return "IMPROVING"
    if ratio < 100 and mom < 100: return "LAGGING"
    return "WEAKENING"

def get_sync_status(d_q, w_q, d_ratio):
    # Priority 1: Elite Power Walk (The Green Room)
    if d_ratio > 101.5 and d_q == "WEAKENING": return "POWER WALK"
    # Priority 2: Lead-Through (The Inflection)
    if d_q == "LEADING" and w_q == "IMPROVING": return "LEAD-THROUGH"
    # Priority 3: Standard Syncs
    if d_q == "LEADING" and w_q == "LEADING": return "BULLISH SYNC"
    if d_q == "IMPROVING" and w_q == "LAGGING": return "DAILY PIVOT"
    return "DIVERGED"

@st.cache_data(ttl=3600)
def run_full_analysis(ticker_str, bench):
    tickers = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    all_needed = list(set(tickers + [bench.upper()]))
    
    d_raw = yf.download(all_needed, period="2y", interval="1d", group_by='ticker', progress=False)
    w_raw = yf.download(all_needed, period="2y", interval="1wk", group_by='ticker', progress=False)
    
    history, table_data = {"Daily": {}, "Weekly": {}}, []
    
    for t in tickers:
        d_res = get_rrg_metrics(d_raw, t, bench.upper())
        w_res = get_rrg_metrics(w_raw, t, bench.upper())
        
        if d_res and w_res:
            # Store histories for the chart
            history["Daily"][t] = pd.DataFrame({'x': d_res[0], 'y': d_res[1]}).tail(35)
            history["Weekly"][t] = pd.DataFrame({'x': w_res[0], 'y': w_res[1]}).tail(35)
            
            # Latest readings
            dr, dm = d_res[0].iloc[-1], d_res[1].iloc[-1]
            wr, wm = w_res[0].iloc[-1], w_res[1].iloc[-1]
            dq, wq = get_quadrant(dr, dm), get_quadrant(wr, wm)
            
            status = get_sync_status(dq, wq, dr)
            
            table_data.append({
                "Ticker": t, "Name": FUND_MAP.get(t, ""), "Sync Status": status,
                "Daily Quad": dq, "Weekly Quad": wq,
                "Daily CH": d_res[2], "Weekly CH": w_res[2],
                "RS-Ratio": round(dr, 2), "Rel Vol": d_res[3]
            })
            
    return pd.DataFrame(table_data), history

# --- 4. UI EXECUTION ---
try:
    df_main, history_data = run_full_analysis(tickers_input, benchmark)

    # A. RRG PLOT
    st.subheader(f"🌀 {timeframe} Rotation (Properly Scaled)")
    fig = go.Figure()
    
    # Power Zone Highlight (The Green Room)
    fig.add_vrect(x0=101.5, x1=105, fillcolor="rgba(46, 204, 113, 0.15)", layer="below", line_width=0, annotation_text="POWER ZONE")
    
    # Crosshairs
    fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="gray", width=1, dash="dash"))
    fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="gray", width=1, dash="dash"))
    
    for i, (t, df) in enumerate(history_data[timeframe].items()):
        if filter_setups and t not in df_main[df_main['Sync Status'] != "DIVERGED"]['Ticker'].values: continue
        
        color = px.colors.qualitative.Plotly[i % 10]
        df_p = df.tail(tail_len)
        
        if tail_len > 1:
            fig.add_trace(go.Scatter(x=df_p['x'], y=df_p['y'], mode='lines', name=t, line=dict(color=color, width=2, shape='spline'), opacity=0.4, legendgroup=t))
            
        fig.add_trace(go.Scatter(x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], mode='markers+text', 
                                 marker=dict(symbol='diamond', size=13, color=color, line=dict(width=1, color='white')), 
                                 text=[t], textposition="top center", showlegend=(tail_len == 1), legendgroup=t))

    fig.update_layout(template="plotly_white", height=750, xaxis=dict(range=[98, 102.5], title="RS-Ratio"), yaxis=dict(range=[98, 102.5], title="RS-Momentum"))
    st.plotly_chart(fig, use_container_width=True)

    # B. ALPHA GRID
    st.subheader("📊 Dual-Timeframe Alpha Grid")
    
    def style_sync(val):
        colors = {"POWER WALK": "#9B59B6", "LEAD-THROUGH": "#E67E22", "BULLISH SYNC": "#2ECC71", "DAILY PIVOT": "#F1C40F"}
        bg = colors.get(val, "#FBFCFC")
        txt = "white" if val in ["POWER WALK", "LEAD-THROUGH"] else "black"
        return f'background-color: {bg}; color: {txt}; font-weight: bold'

    # Custom Sorting
    df_main['sort_val'] = df_main['Sync Status'].map({"POWER WALK": 0, "LEAD-THROUGH": 1, "BULLISH SYNC": 2, "DAILY PIVOT": 3, "DIVERGED": 4})
    df_disp = df_main.sort_values(by=['sort_val', 'Daily CH'], ascending=[True, False]).copy()
    
    if filter_setups:
        df_disp = df_disp[df_disp['Sync Status'] != "DIVERGED"]

    st.dataframe(df_disp.drop(columns=['sort_val']).style.map(style_sync, subset=['Sync Status']).format({"Rel Vol": "{:.2f}x"}), use_container_width=True)

except Exception as e:
    st.error(f"Critical Dashboard Error: {e}")
