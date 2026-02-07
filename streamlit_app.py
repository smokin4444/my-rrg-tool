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
    "XLRE": "Real Estate", "PSCE": "Small Cap Energy", "PSCT": "Small Cap Tech",
    "PSCH": "Small Cap Health", "PSCF": "Small Cap Finance", "PSCI": "Small Cap Industrials",
    "PSCD": "Small Cap Disc", "PSCC": "Small Cap Staples", "PSCM": "Small Cap Materials", "PSCU": "Small Cap Utils"
}

# --- LISTS ---
MAJOR_THEMES = "SPY, QQQ, DIA, IWF, IWD, MAGS, IWM, IJR, GLD, SLV, COPX, XLE, BTC-USD"
SECTOR_ROTATION = "XLK, XLY, XLC, XBI, XLF, XLI, XLE, XLV, XLP, XLU, XLB, XLRE"
SMALL_CAP_SECTORS = "PSCE, PSCT, PSCH, PSCF, PSCI, PSCD, PSCC, PSCM, PSCU"
STARTUP_THEMES = "AMD, AMPX, BABA, BIDU, BITF, CIFR, CLSK, CORZ, CRWV, EOSE, GOOGL, HUT, IREN, LAES, NBIS, NUAI, NVDA, NVTS, PATH, POWL, RR, SERV, SNDK, TE, TSLA, TSM, WDC, ZETA, BHP, CMCL, COPX, CPER, ERO, FCX, HBM, HG=F, IE, RIO, SCCO, TGB, TMQ, AMTM, AVAV, BWXT, DPRO, ESLT, KRKNF, KRMN, KTOS, LPTH, MOB, MRCY, ONDS, OSS, PLTR, PRZO, RCAT, TDY, UMAC, CRDO, IBRX, IONQ, IONR, LAC, MP, NAK, NET, OPTT, PPTA, RZLT, SKYT, TMDX, UAMY, USAR, UUUU, WWR, ASTS, BKSY, FLY, GSAT, HEI, IRDM, KULR, LUNR, MNTS, PL, RDW, RKLB, SATL, SATS, SIDU, SPIR, UFO, VOYG, VSAT"

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎯 Watchlist")
    heap_type = st.radio("Choose Group:", ["Major Themes", "Sector Rotation", "Small Cap Sectors", "Startup"])
    
    if heap_type == "Major Themes": current_list, auto_bench = MAJOR_THEMES, "SPY"
    elif heap_type == "Sector Rotation": current_list, auto_bench = SECTOR_ROTATION, "SPY"
    elif heap_type == "Small Cap Sectors": current_list, auto_bench = SMALL_CAP_SECTORS, "IWM"
    elif heap_type == "Startup": current_list, auto_bench = STARTUP_THEMES, "SPY"
    
    tickers_input = st.text_area("Ticker Heap:", value=current_list, height=150).replace("BTC", "BTC-USD")
    
    st.header("⚖️ Benchmark")
    benchmark = st.text_input("Active Benchmark:", value=auto_bench)
    
    st.markdown("---")
    timeframe = st.radio("Timeframe:", ["Daily", "Weekly"])
    tail_len = st.slider("Tail Length:", 1, 30, 10)
    
    if st.button("♻️ Reset Engine"):
        st.cache_data.clear()
        st.rerun()

# --- HARDENED ENGINE ---
def get_rrg_metrics(df_raw, ticker, b_ticker, is_weekly=False):
    try:
        if ticker not in df_raw.columns.get_level_values(0): return None
        px, bx = df_raw[ticker]['Close'].dropna(), df_raw[b_ticker]['Close'].dropna()
        common = px.index.intersection(bx.index)
        
        # Requirement: At least 30 days of data
        if len(common) < 30: return None
        
        rel = (px.loc[common] / bx.loc[common]) * 100
        # Rolling stats require valid windows
        ratio = 100 + ((rel - rel.rolling(14).mean()) / rel.rolling(14).std())
        roc = ratio.diff(1)
        mom = 100 + ((roc - roc.rolling(14).mean()) / roc.rolling(14).std())
        
        df_res = pd.DataFrame({'x': ratio, 'y': mom, 'date': ratio.index}).dropna()
        if is_weekly: df_res['date'] = df_res['date'] + pd.Timedelta(days=4)
        
        return df_res if not df_res.empty else None
    except: return None

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
            table_data.append({"Ticker": t, "Name": TICKER_NAMES.get(t, t), "RS-Ratio": round(d_res['x'].iloc[-1], 2)})
    return pd.DataFrame(table_data), history

# --- DISPLAY ---
try:
    df_main, history_data = run_analysis(tickers_input, benchmark)
    if not df_main.empty:
        st.subheader(f"🌀 {timeframe} Rotation vs {benchmark}")
        fig = go.Figure()
        
        # Quadrant Annotations
        fig.add_annotation(x=102, y=102, text="<b>LEADING</b>", showarrow=False, font=dict(color="rgba(0,100,0,0.3)", size=14))
        fig.add_annotation(x=98, y=102, text="<b>IMPROVING</b>", showarrow=False, font=dict(color="rgba(0,0,100,0.3)", size=14))
        fig.add_annotation(x=98, y=98, text="<b>LAGGING</b>", showarrow=False, font=dict(color="rgba(100,0,0,0.3)", size=14))
        fig.add_annotation(x=102, y=98, text="<b>WEAKENING</b>", showarrow=False, font=dict(color="rgba(100,50,0,0.3)", size=14))

        for i, (t, df) in enumerate(history_data[timeframe].items()):
            color = px.colors.qualitative.Alphabet[i % 26]
            # Robust Slicing: Ensure we don't exceed existing data points
            avail_len = len(df)
            actual_tail = min(tail_len, avail_len)
            if actual_tail < 2: continue # Skip if not enough points to draw a line
            
            df_p = df.iloc[-actual_tail:]
            
            # Draw faded tail segments safely
            for j in range(len(df_p)-1):
                opacity = (j + 1) / len(df_p) * 0.4
                fig.add_trace(go.Scatter(
                    x=df_p['x'].iloc[j:j+2], y=df_p['y'].iloc[j:j+2], 
                    mode='lines', line=dict(color=color, width=3, shape='spline'),
                    opacity=opacity, showlegend=False, hoverinfo='skip'))
            
            # Head (Diamond)
            fig.add_trace(go.Scatter(
                x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], mode='markers+text', 
                marker=dict(symbol='diamond', size=15, color=color, line=dict(width=1.5, color='white')), 
                text=[t], textposition="top center", name=f"{t}",
                hovertemplate=f"<b>{t}</b><br>Ratio: %{{x:.2f}}<br>Mom: %{{y:.2f}}<extra></extra>"))
            
        fig.update_layout(template="plotly_white", height=850, 
                          xaxis=dict(range=[97.5, 102.5], title="RS-Ratio"),
                          yaxis=dict(range=[97.5, 102.5], title="RS-Momentum"),
                          legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_main.sort_values(by='RS-Ratio', ascending=False), use_container_width=True)
    else:
        st.info("Wait for data sync. If it persists, try another watchlist group.")
except Exception as e:
    st.error(f"Engine Debug Alert: {e}. Check if a specific ticker in the input is causing the gap.")
