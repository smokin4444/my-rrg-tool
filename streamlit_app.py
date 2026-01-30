import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d

st.set_page_config(page_title="Fixed RRG Dashboard", layout="wide")

st.title("📈 Pro RRG: Fixed Toggles & Arrows")

# --- Persistent Ticker Management ---
MY_MINERS = "AFM.V, NAK, A4N.AX, CSC.AX, IVN.TO, TGB"
SECTOR_ETFS = "XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU"

FUND_MAP = {
    "XLC": "Comm. Services", "XLY": "Cons. Discretionary", "XLP": "Cons. Staples",
    "XLE": "Energy", "XLF": "Financials", "XLV": "Health Care",
    "XLI": "Industrials", "XLB": "Materials", "XLRE": "Real Estate",
    "XLK": "Technology", "XLU": "Utilities", "XBI": "Biotech",
    "XME": "Metals & Mining", "XSD": "Semiconductors", "XOP": "Oil & Gas Exploration",
    "GDXJ": "Junior Gold Miners", "COPX": "Copper Miners", "REMX": "Strategic Metals"
}

# --- Sidebar ---
with st.sidebar:
    st.header("1. Controls")
    heap_type = st.radio("Choose Watchlist:", ["My Miners", "Sector ETFs", "Custom List"])
    
    current_list = MY_MINERS if heap_type == "My Miners" else SECTOR_ETFS if heap_type == "Sector ETFs" else st.session_state.get('custom_list', MY_MINERS)

    tickers_input = st.text_area("Ticker Heap:", value=current_list)
    benchmark = st.text_input("Benchmark:", value="SPY")
    timeframe = st.radio("Timeframe:", ["Daily", "Weekly"], index=1)
    tail_len = st.slider("Tail Length:", 5, 30, 15)

# --- Calculations ---
@st.cache_data(ttl=3600)
def get_market_data(ticker_str, bench, tf):
    interval = "1d" if tf == "Daily" else "1wk"
    t_list = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    full_list = list(set(t_list + [bench.upper()]))
    raw = yf.download(full_list, period="2y", interval=interval)
    close_data = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']]
    vol_data = raw['Volume'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Volume']]
    return close_data.dropna(), vol_data, t_list

try:
    data, vol_data, t_list = get_market_data(tickers_input, benchmark, timeframe)
    bench_ticker = benchmark.upper()
    
    rrg_dict = {}
    table_rows = [] 
    
    for t in t_list:
        if t not in data.columns or t == bench_ticker: continue
        
        rel = (data[t] / data[bench_ticker]) * 100
        rs_ratio = 100 + ((rel - rel.rolling(14).mean()) / rel.rolling(14).std())
        mom = rs_ratio.pct_change(1) * 100
        rs_mom = 100 + ((mom - mom.rolling(14).mean()) / mom.rolling(14).std())
        
        current_vol = vol_data[t].iloc[-1]
        rel_vol = (current_vol / vol_data[t].tail(20).mean())
        
        r_val, m_val = rs_ratio.iloc[-1], rs_mom.iloc[-1]
        quad = "LEADING" if r_val > 100 and m_val > 100 else "IMPROVING" if r_val < 100 and m_val > 100 else "LAGGING" if r_val < 100 and m_val < 100 else "WEAKENING"
               
        table_rows.append({"Ticker": t, "Name": FUND_MAP.get(t, "Stock"), "Quadrant": quad, "RS-Ratio": round(r_val, 2), "Momentum": round(m_val, 2), "Rel Volume": f"{round(rel_vol, 2)}x"})

        rt = pd.DataFrame({'x': rs_ratio, 'y': rs_mom}).dropna().tail(tail_len)
        if len(rt) > 3:
            tr, ts = np.arange(len(rt)), np.linspace(0, len(rt)-1, len(rt)*5)
            rrg_dict[t] = pd.DataFrame({'x': interp1d(tr, rt['x'], kind='cubic')(ts), 'y': interp1d(tr, rt['y'], kind='cubic')(ts)})

    # --- FULL WIDTH RRG CHART ---
    fig = go.Figure()
    fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="black", width=2))
    fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="black", width=2))
    
    for i, (t, df) in enumerate(rrg_dict.items()):
        color = px.colors.qualitative.Plotly[i % 10]
        
        # Trace 1: The Snake Trail
        fig.add_trace(go.Scatter(
            x=df['x'], y=df['y'], mode='lines', name=t, 
            line=dict(width=1.5, color=color),
            legendgroup=t, showlegend=True
        ))
        
        # Trace 2: The Arrowhead (Marker at the end)
        # Using 'arrow-bar-up' or 'triangle-up' with angleref='previous'
        fig.add_trace(go.Scatter(
            x=[df['x'].iloc[-1]], y=[df['y'].iloc[-1]],
            mode='markers',
            marker=dict(
                symbol='arrow', # Using standard arrow symbol
                size=18, 
                color=color, 
                angleref='previous', # Points the arrowhead along the line
                standoff=0 # Keeps the arrow at the exact point
            ),
            legendgroup=t, showlegend=False, # Linked to the same legend item
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        height=850, template="plotly_white", 
        xaxis=dict(title="RS-Ratio", range=[95, 105]), 
        yaxis=dict(title="RS-Momentum", range=[95, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- LEADERBOARD & RS LINE ---
    st.markdown("---")
    st.subheader("📊 Sector Leaderboard & Conviction")
    st.dataframe(pd.DataFrame(table_rows).sort_values(by="RS-Ratio", ascending=False), use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Long-Term Relative Strength Trend")
    selected_stock = st.selectbox("Select a stock for RS Line:", t_list)
    if selected_stock:
        rs_line = (data[selected_stock] / data[bench_ticker])
        rs_line = (rs_line / rs_line.iloc[0]) * 100
        st.plotly_chart(px.line(rs_line, title=f"{selected_stock} vs {bench_ticker}").add_hline(y=100, line_dash="dash", line_color="red"), use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
