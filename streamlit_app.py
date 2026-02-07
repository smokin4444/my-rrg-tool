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
    "IWD": "Value Stocks", "MAGS": "Magnificent 7", "IWM": "Small Caps (Russell 2000)", 
    "IJR": "Small Cap Core", "GLD": "Gold", "SLV": "Silver", "COPX": "Copper Miners", 
    "XLE": "Energy Sector", "BTC-USD": "Bitcoin", "XLK": "Technology", "XLY": "Consumer Disc", 
    "XLC": "Communication", "XBI": "Biotech", "XLF": "Financials", "XLI": "Industrials", 
    "XLV": "Health Care", "XLP": "Consumer Staples", "XLU": "Utilities", "XLB": "Materials", 
    "XLRE": "Real Estate", "BTC": "Bitcoin"
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

# --- HARDENED ENGINE ---
def get_rrg_metrics(df_raw, ticker, b_ticker, is_weekly=False):
    try:
        if ticker not in df_raw.columns.get_level_values(0): return None
        px = df_raw[ticker]['Close'].dropna()
        bx = df_raw[b_ticker]['Close'].dropna()
        common = px.index.intersection(bx.index)
        if len(common) < 30: return None
        
        rel = (px.loc[common] / bx.loc[common]) * 100
        ratio = 100 + ((rel - rel.rolling(14).mean()) / rel.rolling(14).std())
        roc = ratio.diff(1)
        mom = 100 + ((roc - roc.rolling(14).mean()) / roc.rolling(14).std())
        
        df_res = pd.DataFrame({'x': ratio, 'y': mom, 'date': ratio.index}).dropna()
        if len(df_res) < 5: return None
        if is_weekly: df_res['date'] = df_res['date'] + pd.Timedelta(days=4)
        
        ch = np.sqrt((df_res['x'].iloc[-1] - df_res['x'].iloc[-5])**2 + (df_res['y'].iloc[-1] - df_res['y'].iloc[-5])**2)
        vol_data = df_raw[ticker].get('Volume', None)
        rv = (vol_data.iloc[-1] / vol_data.tail(20).mean()) if vol_data is not None and not vol_data.empty else 1.0
        
        return df_res, round(ch, 2), rv
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
        if d_res and w_res:
            history["Daily"][t], history["Weekly"][t] = d_res[0], w_res[0]
            dr, dm = d_res[0]['x'].iloc[-1], d_res[0]['y'].iloc[-1]
            wr, wm = w_res[0]['x'].iloc[-1], w_res[0]['y'].iloc[-1]
            dq, wq = get_quadrant(dr, dm), get_quadrant(wr, wm)
            status = "POWER WALK" if dr > 101.5 and dq == "WEAKENING" else \
                     "LEAD-THROUGH" if dq == "LEADING" and wq == "IMPROVING" else \
                     "BULLISH SYNC" if dq == "LEADING" and wq == "LEADING" else \
                     "DAILY PIVOT" if dq == "IMPROVING" and wq == "LAGGING" else "DIVERGED"
            is_synced = "✅ SYNCED" if dq == wq else "---"
            table_data.append({"Ticker": t, "Name": TICKER_NAMES.get(t, t), "Sync Status": status, "Daily Quad": dq, "Weekly Quad": wq, "Sync Radar": is_synced, "RS-Ratio": round(dr, 2), "Rel Vol": d_res[2]})
    return pd.DataFrame(table_data), history

# --- DISPLAY ---
try:
    df_main, history_data = run_analysis(tickers_input, benchmark)
    if not df_main.empty:
        st.subheader(f"🌀 {timeframe} Rotation vs {benchmark}")
        fig = go.Figure()
        fig.add_vrect(x0=101.5, x1=105, fillcolor="rgba(46, 204, 113, 0.15)", layer="below", line_width=0)
        fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="gray", width=1, dash="dash"))
        fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="gray", width=1, dash="dash"))
        
        for i, (t, df) in enumerate(history_data[timeframe].items()):
            if filter_setups and t not in df_main[df_main['Sync Status'] != "DIVERGED"]['Ticker'].values: continue
            
            color = px.colors.qualitative.Plotly[i % 10]
            full_name = TICKER_NAMES.get(t, t)
            df_p = df.tail(tail_len)
            df_p['d_label'] = df_p['date'].dt.strftime('%b %d')
            
            # Hover customization for both tail and diamond
            if tail_len > 1:
                fig.add_trace(go.Scatter(
                    x=df_p['x'], y=df_p['y'], mode='lines+markers', name=f"{t} ({full_name})", 
                    line=dict(color=color, shape='spline'), marker=dict(size=8), 
                    customdata=df_p['d_label'], 
                    hovertemplate=f"<b>{t} | {full_name}</b><br>Date: %{{customdata}}<br>Ratio: %{{x:.2f}}<br>Momentum: %{{y:.2f}}<extra></extra>",
                    opacity=0.4))
            
            fig.add_trace(go.Scatter(
                x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], mode='markers+text', 
                marker=dict(symbol='diamond', size=16, color=color, line=dict(width=2, color='white')), 
                text=[t], textposition="top center", showlegend=False,
                customdata=[df_p['d_label'].iloc[-1]],
                hovertemplate=f"<b>{t} | {full_name}</b><br>LATEST<br>Ratio: %{{x:.2f}}<br>Momentum: %{{y:.2f}}<extra></extra>"))
            
        fig.update_layout(template="plotly_white", height=750, xaxis=dict(range=[98, 102.5]), yaxis=dict(range=[98, 102.5]))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Alpha Grid")
        def style_sync(val):
            colors = {"POWER WALK": "#9B59B6", "LEAD-THROUGH": "#E67E22", "BULLISH SYNC": "#2ECC71", "DAILY PIVOT": "#F1C40F"}
            return f'background-color: {colors.get(val, "#FBFCFC")}; color: {"white" if val in ["POWER WALK", "LEAD-THROUGH"] else "black"}; font-weight: bold'
        
        df_disp = df_main.sort_values(by='RS-Ratio', ascending=False)
        if filter_setups: df_disp = df_disp[df_disp['Sync Status'] != "DIVERGED"]
        
        # Displaying 'Name' column in table too
        st.dataframe(df_disp.style.map(style_sync, subset=['Sync Status']).format({"Rel Vol": "{:.2f}x"}), use_container_width=True)
except Exception as e:
    st.error(f"Dashboard Error: {e}")
