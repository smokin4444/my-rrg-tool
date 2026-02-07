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
            dr, dm = d_res['x'].iloc[-1], d_res['y'].iloc[-1]
            status = "POWER WALK" if dr > 101.5 and dq == "WEAKENING" else \
                     "LEAD-THROUGH" if dq == "LEADING" and wq == "IMPROVING" else \
                     "BULLISH SYNC" if dq == "LEADING" and wq == "LEADING" else \
                     "DAILY PIVOT" if dq == "IMPROVING" and wq == "LAGGING" else "DIVERGED"
            table_data.append({"Ticker": t, "Name": TICKER_NAMES.get(t, t), "Sync Status": status, "Daily Quad": dq, "Weekly Quad": wq, "Sync Radar": "✅ SYNCED" if dq == wq else "---", "RS-Ratio": round(dr, 2)})
    return pd.DataFrame(table_data), history

# --- DISPLAY ---
try:
    df_main, history_data = run_analysis(tickers_input, benchmark)
    if not df_main.empty:
        st.subheader(f"🌀 {timeframe} Rotation vs {benchmark}")
        fig = go.Figure()
        
        # 1. Background Logic & Quadrant Crosshair
        fig.add_vrect(x0=101.5, x1=105, fillcolor="rgba(46, 204, 113, 0.15)", layer="below", line_width=0)
        fig.add_shape(type="line", x0=100, y0=80, x1=100, y1=120, line=dict(color="rgba(0,0,0,0.4)", width=2, dash="dot"))
        fig.add_shape(type="line", x0=80, y0=100, x1=120, y1=100, line=dict(color="rgba(0,0,0,0.4)", width=2, dash="dot"))

        # 2. Quadrant Labels (Pushed out to extremes)
        fig.add_annotation(x=102.3, y=102.3, text="<b>LEADING</b>", showarrow=False, font=dict(color="rgba(0,100,0,0.25)", size=16))
        fig.add_annotation(x=97.7, y=102.3, text="<b>IMPROVING</b>", showarrow=False, font=dict(color="rgba(0,0,100,0.25)", size=16))
        fig.add_annotation(x=97.7, y=97.7, text="<b>LAGGING</b>", showarrow=False, font=dict(color="rgba(100,0,0,0.25)", size=16))
        fig.add_annotation(x=102.3, y=97.7, text="<b>WEAKENING</b>", showarrow=False, font=dict(color="rgba(100,50,0,0.25)", size=16))

        # 3. Comet Trails & Hovers
        for i, (t, df) in enumerate(history_data[timeframe].items()):
            color = px.colors.qualitative.Alphabet[i % 26]
            full_name = TICKER_NAMES.get(t, t)
            df_p = df.iloc[-min(tail_len, len(df)):]
            
            for j in range(len(df_p)-1):
                opacity = (j + 1) / len(df_p) * 0.4
                fig.add_trace(go.Scatter(x=df_p['x'].iloc[j:j+2], y=df_p['y'].iloc[j:j+2], mode='lines', 
                                         line=dict(color=color, width=3, shape='spline'), opacity=opacity, showlegend=False, hoverinfo='skip'))
            
            fig.add_trace(go.Scatter(x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], mode='markers+text', 
                                     marker=dict(symbol='diamond', size=15, color=color, line=dict(width=1.5, color='white')), 
                                     text=[t], textposition="top center", name=f"{t}", customdata=[full_name],
                                     hovertemplate=f"<b>{t} | %{{customdata}}</b><br>Ratio: %{{x:.2f}}<br>Mom: %{{y:.2f}}<extra></extra>"))
            
        fig.update_layout(template="plotly_white", height=850, xaxis=dict(range=[97.5, 102.5], title="RS-Ratio (Trend)"),
                          yaxis=dict(range=[97.5, 102.5], title="RS-Momentum (Energy)"), legend=dict(orientation="h", y=-0.12, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True)

        # 4. Alpha Grid with Shaded Status Areas
        st.subheader("📊 Alpha Grid")
        def style_status(val):
            color_map = {
                "POWER WALK": "background-color: #9B59B6; color: white; font-weight: bold;",   # Purple
                "LEAD-THROUGH": "background-color: #E67E22; color: white; font-weight: bold;", # Orange
                "BULLISH SYNC": "background-color: #2ECC71; color: white; font-weight: bold;", # Green
                "DAILY PIVOT": "background-color: #F1C40F; color: black; font-weight: bold;"    # Yellow
            }
            return color_map.get(val, "")

        df_styled = df_main.sort_values(by='RS-Ratio', ascending=False)
        st.dataframe(df_styled.style.applymap(style_status, subset=['Sync Status']), use_container_width=True)

except Exception as e:
    st.error(f"Engine Debug Alert: {e}")
