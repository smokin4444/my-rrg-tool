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
    "AROC": "Archrock", "KGS": "Kodiak Gas", "LBRT": "Liberty Energy", "NE": "Noble Corp", "LEU": "Centrus Energy"
}

# --- LISTS ---
MAJOR_THEMES = "SPY, QQQ, DIA, IWF, IWD, MAGS, IWM, IJR, GLD, SLV, COPX, XLE, BTC-USD"
SECTOR_ROTATION = "XLK, XLY, XLC, XBI, XLF, XLI, XLE, XLV, XLP, XLU, XLB, XLRE"
ENERGY_TORQUE = "AROC, KGS, LBRT, NE, SM, CRC, BTU, WHD, MGY, CNR, OII, INVX, LEU, VAL, CIVI, NINE, BORR, HP, STX, BHL"
STARTUP_THEMES = "AMD, AMPX, BABA, BIDU, BITF, CIFR, CLSK, CORZ, CRWV, EOSE, GOOGL, HUT, IREN, LAES, NBIS, NUAI, NVDA, NVTS, PATH, POWL, RR, SERV, SNDK, TE, TSLA, TSM, WDC, ZETA, BHP, CMCL, COPX, CPER, ERO, FCX, HBM, HG=F, IE, RIO, SCCO, TGB, TMQ, AMTM, AVAV, BWXT, DPRO, ESLT, KRKNF, KRMN, KTOS, LPTH, MOB, MRCY, ONDS, OSS, PLTR, PRZO, RCAT, TDY, UMAC, CRDO, IBRX, IONQ, IONR, LAC, MP, NAK, NET, OPTT, PPTA, RZLT, SKYT, TMDX, UAMY, USAR, UUUU, WWR, ASTS, BKSY, FLY, GSAT, HEI, IRDM, KULR, LUNR, MNTS, PL, RDW, RKLB, SATL, SATS, SIDU, SPIR, UFO, VOYG, VSAT"

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎯 Watchlist")
    heap_type = st.radio("Choose Group:", ["Major Themes", "Sector Rotation", "Energy Torque", "Startup"])
    
    if heap_type == "Major Themes": current_list, auto_bench = MAJOR_THEMES, "SPY"
    elif heap_type == "Sector Rotation": current_list, auto_bench = SECTOR_ROTATION, "SPY"
    elif heap_type == "Energy Torque": current_list, auto_bench = ENERGY_TORQUE, "XLE"
    elif heap_type == "Startup": current_list, auto_bench = STARTUP_THEMES, "SPY"
    
    tickers_input = st.text_area("Ticker Heap:", value=current_list, height=150).replace("BTC", "BTC-USD")
    benchmark = st.text_input("Active Benchmark:", value=auto_bench)
    
    st.markdown("---")
    timeframe = st.radio("Timeframe:", ["Daily", "Weekly"])
    tail_len = st.slider("Tail Length:", 2, 30, 10) # Minimum 2 to prevent out-of-bounds
    
    if st.button("♻️ Reset Engine"):
        st.cache_data.clear()
        st.rerun()

# --- ENGINE ---
def get_metrics(df_raw, ticker, b_ticker, is_weekly):
    try:
        px = df_raw[ticker]['Close'].dropna()
        bx = df_raw[b_ticker]['Close'].dropna()
        common = px.index.intersection(bx.index)
        if len(common) < 20: return None # Guard for insufficient history
        
        rel = (px.loc[common] / bx.loc[common]) * 100
        ratio = 100 + ((rel - rel.rolling(14).mean()) / rel.rolling(14).std())
        roc = ratio.diff(1)
        mom = 100 + ((roc - roc.rolling(14).mean()) / roc.rolling(14).std())
        df_res = pd.DataFrame({'x': ratio, 'y': mom, 'date': ratio.index}).dropna()
        
        if is_weekly: df_res['date'] = df_res['date'] + pd.Timedelta(days=4)
        return df_res if len(df_res) >= 2 else None
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
        d_res = get_metrics(data, t, bench_ticker, False)
        w_res = get_metrics(w_data, t, bench_ticker, True)
        if d_res is not None and w_res is not None:
            history["Daily"][t], history["Weekly"][t] = d_res, w_res
            table_data.append({"Ticker": t, "RS-Ratio": round(d_res['x'].iloc[-1], 2)})
    return pd.DataFrame(table_data), history

# --- DISPLAY ---
try:
    df_main, history_data = run_analysis(tickers_input, benchmark)
    if not df_main.empty:
        st.subheader(f"🌀 {timeframe} Rotation vs {benchmark}")
        fig = go.Figure()
        
        # Quadrant Lines
        fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="rgba(0,0,0,0.4)", width=2, dash="dot"))
        fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="rgba(0,0,0,0.4)", width=2, dash="dot"))

        # Labels
        for label, x, y, col in [("LEADING", 102.3, 102.3, "green"), ("IMPROVING", 97.7, 102.3, "blue"), ("LAGGING", 97.7, 97.7, "red"), ("WEAKENING", 102.3, 97.7, "orange")]:
            fig.add_annotation(x=x, y=y, text=f"<b>{label}</b>", showarrow=False, font=dict(color=col, size=14), opacity=0.3)

        for i, (t, df) in enumerate(history_data[timeframe].items()):
            color = px.colors.qualitative.Alphabet[i % 26]
            avail_len = len(df)
            actual_tail = min(tail_len, avail_len)
            
            # COMET TAIL - Safe Loop
            if actual_tail >= 2:
                df_p = df.iloc[-actual_tail:]
                for j in range(len(df_p) - 1):
                    opacity = (j + 1) / len(df_p) * 0.5
                    fig.add_trace(go.Scatter(
                        x=df_p['x'].iloc[j:j+2], y=df_p['y'].iloc[j:j+2], 
                        mode='lines+markers', line=dict(color=color, width=3, shape='spline'),
                        marker=dict(size=4, opacity=opacity), showlegend=False, hoverinfo='skip'))
            
            # HEAD DIAMOND
            head_x = df['x'].iloc[-1]
            head_y = df['y'].iloc[-1]
            fig.add_trace(go.Scatter(
                x=[head_x], y=[head_y], mode='markers+text', 
                marker=dict(symbol='diamond', size=16, color=color, line=dict(width=1.5, color='white')), 
                text=[t], textposition="top center", name=f"{t}",
                customdata=[TICKER_NAMES.get(t, t)],
                hovertemplate=f"<b>{t} | %{{customdata}}</b><br>Ratio: %{{x:.2f}}<br>Mom: %{{y:.2f}}<extra></extra>"))
            
        fig.update_layout(template="plotly_white", height=850, 
                          xaxis=dict(range=[97.5, 102.5], title="RS-Ratio"),
                          yaxis=dict(range=[97.5, 102.5], title="RS-Momentum"),
                          legend=dict(orientation="h", y=-0.12, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_main.sort_values(by='RS-Ratio', ascending=False), use_container_width=True)
except Exception as e:
    st.error(f"Engine Error: {e}. Try hitting Reset or check for bad tickers.")
