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
    "XLE": "Energy Sector", "IBIT": "iShares Bitcoin Trust", 
    "XLK": "Tech (Large)", "XLY": "Cons Disc (Large)", "XLC": "Comm (Large)", 
    "XBI": "Biotech", "XLF": "Financials (Large)", "XLI": "Industrials (Large)", 
    "XLV": "Health Care (Large)", "XLP": "Cons Staples (Large)", "XLU": "Utilities (Large)", 
    "XLB": "Materials (Large)", "XLRE": "Real Estate (Large)",
    "PSCT": "Tech (Small)", "PSCD": "Cons Disc (Small)", "PSCF": "Financials (Small)", 
    "PSCI": "Industrials (Small)", "PSCH": "Health Care (Small)", "PSCC": "Cons Staples (Small)", 
    "PSCU": "Utilities (Small)", "PSCM": "Materials (Small)", "PSCE": "Energy (Small)",
    "AROC": "Archrock", "KGS": "Kodiak Gas", "LBRT": "Liberty Energy", "NE": "Noble Corp", "OII": "Oceaneering Intl"
}

# --- STATIC LISTS ---
MAJOR_THEMES = "SPY, QQQ, DIA, IWF, IWD, MAGS, IWM, IJR, GLD, SLV, COPX, XLE, IBIT"
SECTOR_ROTATION = "XLK, XLY, XLC, XBI, XLF, XLI, XLE, XLV, XLP, XLU, XLB, XLRE, PSCT, PSCD, PSCF, PSCI, PSCH, PSCC, PSCU, PSCM, PSCE"
ENERGY_TORQUE = "AROC, KGS, LBRT, NE, SM, CRC, BTU, WHD, MGY, CNR, OII, INVX, LEU, VAL, CIVI, NINE, BORR, HP, STX, BHL"
STARTUP_THEMES = "AMD, AMPX, BABA, BIDU, BITF, CIFR, CLSK, CORZ, CRWV, EOSE, GOOGL, HUT, IREN, LAES, NBIS, NUAI, NVDA, NVTS, PATH, POWL, RR, SERV, SNDK, TE, TSLA, TSM, WDC, ZETA, BHP, CMCL, COPX, CPER, ERO, FCX, HBM, HG=F, IE, RIO, SCCO, TGB, TMQ, AMTM, AVAV, BWXT, DPRO, ESLT, KRKNF, KRMN, KTOS, LPTH, MOB, MRCY, ONDS, OSS, PLTR, PRZO, RCAT, TDY, UMAC, CRDO, IBRX, IONQ, IONR, LAC, MP, NAK, NET, OPTT, PPTA, RZLT, SKYT, TMDX, UAMY, USAR, UUUU, WWR, ASTS, BKSY, FLY, GSAT, HEI, IRDM, KULR, LUNR, MNTS, PL, RDW, RKLB, SATL, SATS, SIDU, SPIR, UFO, VOYG, VSAT"
CUSTOM_LIST = ""

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎯 Watchlist")
    heap_type = st.radio("Choose Group:", ["Major Themes", "Sector Rotation", "Energy Torque", "Startup", "Single/Custom"])
    
    if heap_type == "Major Themes": current_list, auto_bench = MAJOR_THEMES, "SPY"
    elif heap_type == "Sector Rotation": current_list, auto_bench = SECTOR_ROTATION, "SPY"
    elif heap_type == "Energy Torque": current_list, auto_bench = ENERGY_TORQUE, "XLE"
    elif heap_type == "Startup": current_list, auto_bench = STARTUP_THEMES, "SPY"
    elif heap_type == "Single/Custom": current_list, auto_bench = CUSTOM_LIST, "SPY"
    
    tickers_input = st.text_area("Ticker Heap:", value=current_list, height=150)
    benchmark = st.text_input("Active Benchmark:", value=auto_bench)
    
    st.markdown("---")
    timeframe = st.radio("Timeframe:", ["Daily", "Weekly"])
    tail_len = st.slider("Tail Length:", 2, 30, 12)
    
    if st.button("♻️ Reset Engine"):
        st.cache_data.clear()
        st.rerun()

# --- ENGINE ---
def get_metrics(df_raw, ticker, b_ticker):
    try:
        # Get raw close prices
        px = df_raw[ticker]['Close'].dropna()
        bx = df_raw[b_ticker]['Close'].dropna()
        
        # Intersection to align dates exactly
        common = px.index.intersection(bx.index)
        if len(common) < 30: return None
        
        # Final safety truncation
        px_aligned = px.loc[common]
        bx_aligned = bx.loc[common]
        
        rel = (px_aligned / bx_aligned) * 100
        ratio = 100 + ((rel - rel.rolling(14).mean()) / rel.rolling(14).std())
        roc = ratio.diff(1)
        mom = 100 + ((roc - roc.rolling(14).mean()) / roc.rolling(14).std())
        
        df_res = pd.DataFrame({'x': ratio, 'y': mom, 'date': ratio.index}).dropna()
        df_res['date_str'] = df_res['date'].dt.strftime('%b %d, %Y')
        return df_res
    except:
        return None

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
    
    # Download data
    data = yf.download(all_list, period="2y", interval="1d", group_by='ticker', progress=False)
    w_data = yf.download(all_list, period="2y", interval="1wk", group_by='ticker', progress=False)
    
    history, table_data = {"Daily": {}, "Weekly": {}}, []
    for t in tickers:
        # Use only Daily for charts to maintain accuracy
        d_res = get_metrics(data, t, bench_ticker)
        w_res = get_metrics(w_data, t, bench_ticker)
        
        if d_res is not None and w_res is not None:
            history["Daily"][t], history["Weekly"][t] = d_res, w_res
            
            # Use current timeframe selection for table metrics
            active_df = d_res if timeframe == "Daily" else w_res
            if not active_df.empty:
                dr, dm = active_df['x'].iloc[-1], active_df['y'].iloc[-1]
                dq = get_quadrant(dr, dm)
                wq = get_quadrant(w_res['x'].iloc[-1], w_res['y'].iloc[-1])
                
                # Crossing Logic
                cross_alert = "---"
                if len(active_df) > 5:
                    was_below = (active_df['x'].iloc[-6:-1] < 100).any()
                    if was_below and dr >= 100 and dm >= 100:
                        cross_alert = "🔥 CROSSING"

                status = "POWER WALK" if dr > 101.5 and dq == "WEAKENING" else \
                         "LEAD-THROUGH" if dq == "LEADING" and wq == "IMPROVING" else \
                         "BULLISH SYNC" if dq == "LEADING" and wq == "LEADING" else \
                         "DAILY PIVOT" if dq == "IMPROVING" and wq == "LAGGING" else "DIVERGED"
                
                table_data.append({
                    "Ticker": t, "Name": TICKER_NAMES.get(t, t), "12 O'Clock Alert": cross_alert,
                    "Sync Status": status, "RS-Ratio": round(dr, 2)
                })
    return pd.DataFrame(table_data), history

# --- DISPLAY ---
try:
    df_main, history_data = run_analysis(tickers_input, benchmark)
    if not df_main.empty:
        st.subheader(f"🌀 {timeframe} Rotation vs {benchmark}")
        fig = go.Figure()
        
        # Quadrant Design
        fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="rgba(0,0,0,0.5)", width=2, dash="dot"))
        fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="rgba(0,0,0,0.5)", width=2, dash="dot"))
        fig.add_vrect(x0=101.5, x1=105, fillcolor="rgba(46, 204, 113, 0.12)", layer="below", line_width=0)

        for label, x, y, col in [("LEADING", 102.3, 102.3, "green"), ("IMPROVING", 97.7, 102.3, "blue"), ("LAGGING", 97.7, 97.7, "red"), ("WEAKENING", 102.3, 97.7, "orange")]:
            fig.add_annotation(x=x, y=y, text=f"<b>{label}</b>", showarrow=False, font=dict(color=col, size=14), opacity=0.4)

        for i, (t, df) in enumerate(history_data[timeframe].items()):
            color = px.colors.qualitative.Alphabet[i % 26]
            df_p = df.iloc[-min(tail_len, len(df)):]
            
            fig.add_trace(go.Scatter(
                x=df_p['x'], y=df_p['y'], mode='lines+markers',
                line=dict(color=color, width=3, shape='spline'),
                marker=dict(size=6, color=color, opacity=0.8, line=dict(width=1, color='white')), 
                name=f"{t}", customdata=df_p['date_str'],
                hovertemplate=f"<b>{t}</b><br>Date: %{{customdata}}<br>Ratio: %{{x:.2f}}<br>Mom: %{{y:.2f}}<extra></extra>"))
            
            fig.add_trace(go.Scatter(
                x=[df_p['x'].iloc[-1]], y=[df_p['y'].iloc[-1]], mode='markers+text', 
                marker=dict(symbol='diamond', size=18, color=color, line=dict(width=2, color='white')), 
                text=[t], textposition="top center", showlegend=False,
                customdata=[[TICKER_NAMES.get(t, t), df_p['date_str'].iloc[-1]]],
                hovertemplate=f"<b>{t} | %{{customdata[0]}}</b><br>LATEST: %{{customdata[1]}}<br>Ratio: %{{x:.2f}}<br>Mom: %{{y:.2f}}<extra></extra>"))
            
        fig.update_layout(template="plotly_white", height=850, xaxis=dict(range=[97.5, 102.5], title="RS-Ratio"),
                          yaxis=dict(range=[97.5, 102.5], title="RS-Momentum"), legend=dict(orientation="h", y=-0.12, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Alpha Grid")
        def style_status(val):
            color_map = {"POWER WALK": "background-color: #9B59B6; color: white;", "LEAD-THROUGH": "background-color: #E67E22; color: white;",
                         "BULLISH SYNC": "background-color: #2ECC71; color: white;", "DAILY PIVOT": "background-color: #F1C40F; color: black;"}
            return color_map.get(val, "")
        def style_alert(val):
            return "background-color: #E74C3C; color: white; font-weight: bold;" if val == "🔥 CROSSING" else ""

        st.dataframe(df_main.sort_values(by='RS-Ratio', ascending=False).style.applymap(style_status, subset=['Sync Status']).applymap(style_alert, subset=["12 O'Clock Alert"]), use_container_width=True)
except Exception as e:
    st.error(f"Engine Alert: {e}")
