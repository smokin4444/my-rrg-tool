import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d

st.set_page_config(page_title="Regime-Aware Alpha Scanner", layout="wide")

# --- Persistent Ticker Management ---
# Added XLP for Risk-Off and IGV for Software/AI Pressure
MY_MINERS = "AFM.V, NAK, A4N.AX, CSC.AX, IVN.TO, TGB"
THEMATIC_HEAPS = "SOXX, IGV, XLP, MAGS, URA, COPX, GDXJ, SILJ, BITO, ITA, POWR, XME"

FUND_MAP = {
    "SOXX": "Semiconductors", "IGV": "Software (AI Pressure)", "XLP": "Consumer Staples (Risk-Off)",
    "MAGS": "Mag Seven", "URA": "Uranium", "COPX": "Copper", "GDXJ": "Junior Gold", 
    "SILJ": "Junior Silver", "BITO": "Bitcoin", "ITA": "Defense", "POWR": "Power Infra", 
    "XME": "Metals", "XLE": "Energy", "XLB": "Materials", "XLK": "Technology"
}

# --- Sidebar ---
with st.sidebar:
    st.header("Dashboard Controls")
    heap_type = st.radio("Watchlist:", ["My Miners", "Thematic Heaps", "Custom"])
    current_list = MY_MINERS if heap_type == "My Miners" else THEMATIC_HEAPS if heap_type == "Thematic Heaps" else st.session_state.get('custom_list', MY_MINERS)
    tickers_input = st.text_area("Ticker Heap:", value=current_list)
    benchmark = st.text_input("Benchmark:", value="SPY")
    timeframe = st.radio("Timeframe:", ["Daily", "Weekly"], index=1)
    tail_len = st.slider("Tail Length:", 5, 30, 15)

# --- Math Engine ---
@st.cache_data(ttl=3600)
def get_rrg_data(ticker_list, bench, tf, tail):
    interval = "1d" if tf == "Daily" else "1wk"
    tickers = [t.strip().upper() for t in ticker_list.split(",") if t.strip()]
    all_list = list(set(tickers + [bench.upper(), "^VIX"]))
    data_raw = yf.download(all_list, period="2y", interval=interval)
    
    close_data = data_raw['Close'] if isinstance(data_raw.columns, pd.MultiIndex) else data_raw[['Close']]
    vol_raw = data_raw['Volume'] if isinstance(data_raw.columns, pd.MultiIndex) else data_raw[['Volume']]
    
    # RS Rating logic
    perf_scores = {}
    valid_tickers = [t for t in tickers if t in close_data.columns]
    for t in valid_tickers:
        px_curr = close_data[t].iloc[-1]
        px_3m = close_data[t].iloc[-63] if len(close_data) > 63 else close_data[t].iloc[0]
        perf_scores[t] = (px_curr / px_3m) * 100
    
    sorted_scores = sorted(perf_scores.values())
    rs_ratings = {t: int((sorted_scores.index(s) / len(sorted_scores)) * 99) if len(sorted_scores) > 0 else 0 for t, s in perf_scores.items()}

    rrg_results, table_data = {}, []
    for t in valid_tickers:
        if t == bench.upper(): continue
        rel = (close_data[t] / close_data[bench.upper()]) * 100
        rs_ratio = 100 + ((rel - rel.rolling(14).mean()) / rel.rolling(14).std())
        roc = rs_ratio.pct_change(1) * 100
        rs_mom = 100 + ((roc - roc.rolling(14).mean()) / roc.rolling(14).std())
        
        vel = np.sqrt((rs_ratio.iloc[-1] - rs_ratio.iloc[-5])**2 + (rs_mom.iloc[-1] - rs_mom.iloc[-5])**2)
        r_vol = (vol_raw[t].iloc[-1] / vol_raw[t].tail(20).mean())
        ch_score = (vel * 0.7) + (r_vol * 0.3)
        
        table_data.append({
            "Ticker": t, "Name": FUND_MAP.get(t, "Stock"), 
            "RS Rating": rs_ratings.get(t, 0), "CH Score": round(ch_score, 2),
            "Velocity": round(vel, 2), "Rel Vol": round(r_vol, 2)
        })

        rt = pd.DataFrame({'x': rs_ratio, 'y': rs_mom}).dropna().tail(tail)
        if len(rt) >= 3:
            tr = np.arange(len(rt)); ts = np.linspace(0, len(rt)-1, len(rt)*5)
            rrg_results[t] = pd.DataFrame({'x': interp1d(tr, rt['x'], kind='cubic')(ts), 'y': interp1d(tr, rt['y'], kind='cubic')(ts)})
            
    return rrg_results, table_data, close_data

# --- Execution ---
try:
    results, table_list, full_data = get_rrg_data(tickers_input, benchmark, timeframe, tail_len)
    
    # TOP HEADER: VIX REGIME & ALERTS
    current_vix = full_data["^VIX"].iloc[-1]
    vix_col, alert_col = st.columns([1, 2])
    
    with vix_col:
        if current_vix < 15: st.info(f"🛡️ VIX: {current_vix:.2f} (Low)")
        elif 15 <= current_vix <= 25: st.success(f"⚖️ VIX: {current_vix:.2f} (Normal)")
        else: st.error(f"⚠️ VIX: {current_vix:.2f} (Danger)")

    with alert_col:
        top_pick = sorted(table_list, key=lambda x: x['CH Score'], reverse=True)[0]
        st.warning(f"🎯 **TOP CONVICTION:** {top_pick['Ticker']} | CH Score: {top_pick['CH Score']}")

    # 1. RRG CHART
    fig = go.Figure()
    fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="black", width=2))
    fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="black", width=2))
    for i, (t, df) in enumerate(results.items()):
        color = px.colors.qualitative.Plotly[i % 10]
        fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='lines', name=t, line=dict(width=2, color=color)))
        fig.add_annotation(x=df['x'].iloc[-1], y=df['y'].iloc[-1], ax=df['x'].iloc[-2], ay=df['y'].iloc[-2], showarrow=True, arrowhead=2, arrowsize=2, arrowcolor=color)
    fig.update_layout(template="plotly_white", height=800, xaxis=dict(range=[96, 104]), yaxis=dict(range=[96, 104]), legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, use_container_width=True)

    # 2. THE SCANNER
    st.subheader("📊 The Alpha Scanner")
    st.dataframe(pd.DataFrame(table_list).sort_values(by="CH Score", ascending=False), use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
