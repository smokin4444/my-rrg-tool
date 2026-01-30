import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d

st.set_page_config(page_title="Alpha-Scanner Pro", layout="wide")

# --- MASTER TICKER HEAP ---
MY_MINERS = "AFM.V, NAK, A4N.AX, CSC.AX, IVN.TO, TGB"
THEMATIC_HEAPS = "SOXX, IGV, XLP, MAGS, URA, COPX, GDXJ, SILJ, IBIT, ITA, POWR, XME, " \
                 "XLC, XLY, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU"

FUND_MAP = {
    "SOXX": "Semiconductors", "IGV": "Software (AI Pressure)", "XLP": "Cons. Staples (Risk-Off)",
    "MAGS": "Mag Seven", "URA": "Uranium", "COPX": "Copper", "GDXJ": "Junior Gold", 
    "SILJ": "Junior Silver", "IBIT": "Spot Bitcoin", "ITA": "Defense", "POWR": "Power Infra", 
    "XME": "Metals & Mining", "XLC": "Comm. Services", "XLY": "Cons. Disc.", "XLE": "Energy", 
    "XLF": "Financials", "XLV": "Health Care", "XLI": "Industrials", 
    "XLB": "Materials", "XLRE": "Real Estate", "XLK": "Technology", "XLU": "Utilities"
}

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    heap_type = st.radio("Watchlist:", ["My Miners", "Master Heap", "Custom"])
    current_list = MY_MINERS if heap_type == "My Miners" else THEMATIC_HEAPS if heap_type == "Master Heap" else st.session_state.get('custom_list', MY_MINERS)
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
    data = close_data.dropna()
    
    rrg_results, table_data = {}, []
    perf_scores = {}
    valid_tickers = [t for t in tickers if t in data.columns]
    
    for t in valid_tickers:
        px_curr = data[t].iloc[-1]
        px_3m = data[t].iloc[-63] if len(data) > 63 else data[t].iloc[0]
        perf_scores[t] = (px_curr / px_3m) * 100
    sorted_scores = sorted(perf_scores.values())
    rs_ratings = {t: int((sorted_scores.index(s) / len(sorted_scores)) * 99) if len(sorted_scores) > 0 else 0 for t, s in perf_scores.items()}

    for t in valid_tickers:
        if t == bench.upper(): continue
        rel = (data[t] / data[bench.upper()]) * 100
        rs_ratio = 100 + ((rel - rel.rolling(14).mean()) / rel.rolling(14).std())
        roc = rs_ratio.pct_change(1) * 100
        rs_mom = 100 + ((roc - roc.rolling(14).mean()) / roc.rolling(14).std())
        
        vel = np.sqrt((rs_ratio.iloc[-1] - rs_ratio.iloc[-5])**2 + (rs_mom.iloc[-1] - rs_mom.iloc[-5])**2)
        r_vol_val = (vol_raw[t].iloc[-1] / vol_raw[t].tail(20).mean())
        ch_score = (vel * 0.7) + (r_vol_val * 0.3)
        
        table_data.append({
            "Ticker": t, "Name": FUND_MAP.get(t, "Stock"), 
            "RS Rating": rs_ratings.get(t, 0), "CH Score": round(ch_score, 2),
            "Velocity": round(vel, 2), "Rel Vol": r_vol_val
        })

        rt = pd.DataFrame({'x': rs_ratio, 'y': rs_mom}).dropna().tail(tail)
        if len(rt) >= 3:
            tr = np.arange(len(rt)); ts = np.linspace(0, len(rt)-1, len(rt)*5)
            rrg_results[t] = pd.DataFrame({'x': interp1d(tr, rt['x'], kind='cubic')(ts), 'y': interp1d(tr, rt['y'], kind='cubic')(ts)})
            
    return rrg_results, table_data, data

# --- Execution ---
try:
    results, table_list, full_data = get_rrg_data(tickers_input, benchmark, timeframe, tail_len)
    
    vix = full_data["^VIX"].iloc[-1]
    st.info(f"🛡️ **VIX Regime:** {vix:.2f}")

    # --- RRG CHART ---
    fig = go.Figure()
    # Thicker Quadrant Dividers for better orientation
    fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="rgba(0,0,0,0.3)", width=3))
    fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="rgba(0,0,0,0.3)", width=3))
    
    for i, (t, df) in enumerate(results.items()):
        color = px.colors.qualitative.Plotly[i % 10]
        
        # 1. THE TRAIL (Thicker, solid line)
        fig.add_trace(go.Scatter(
            x=df['x'], y=df['y'], mode='lines', name=t, 
            line=dict(width=2.5, color=color), # Increased width for better visibility
            opacity=0.5, legendgroup=t
        ))
        
        # 2. THE CURRENT HEAD (Large Diamond)
        fig.add_trace(go.Scatter(
            x=[df['x'].iloc[-1]], y=[df['y'].iloc[-1]],
            mode='markers+text',
            marker=dict(
                symbol='diamond', 
                size=14, # Slightly larger diamond
                color=color, 
                line=dict(width=2, color='white') # White border to make it pop
            ),
            text=[t], textposition="top center",
            textfont=dict(size=10, color="black"),
            legendgroup=t, showlegend=False
        ))
    
    fig.update_layout(
        template="plotly_white", 
        height=850, 
        xaxis=dict(title="RS-Ratio (Strength)", range=[96, 104], gridcolor='lightgrey'), 
        yaxis=dict(title="RS-Momentum (Velocity)", range=[96, 104], gridcolor='lightgrey'), 
        legend=dict(orientation="h", y=1.05, itemclick="toggle")
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- LEADERBOARD ---
    st.subheader("📊 The Alpha Scanner")
    df_table = pd.DataFrame(table_list).sort_values(by="CH Score", ascending=False)
    
    # Styled Table
    st.dataframe(
        df_table.style.map(lambda x: 'font-weight: bold; color: #1E88E5', subset=['CH Score'])
        .map(lambda x: 'color: #D32F2F; font-weight: bold; background-color: #FFF9C4' if x > 2.5 else '', subset=['Rel Vol'])
        .format({"Rel Vol": "{:.2f}x"}), 
        use_container_width=True
    )

except Exception as e:
    st.error(f"Dashboard Initialization Error: {e}")
