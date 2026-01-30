import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d

st.set_page_config(page_title="Pro-RRG Dashboard", layout="wide")

st.title("📈 Professional Relative Rotation Graph")

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
    st.header("Settings")
    heap_type = st.radio("Choose Watchlist:", ["My Miners", "Sector ETFs", "Custom List"])
    
    current_list = MY_MINERS if heap_type == "My Miners" else SECTOR_ETFS if heap_type == "Sector ETFs" else st.session_state.get('custom_list', MY_MINERS)

    tickers_input = st.text_area("Tickers:", value=current_list)
    benchmark = st.text_input("Benchmark:", value="SPY")
    timeframe = st.radio("Timeframe:", ["Daily", "Weekly"], index=1)
    tail_len = st.slider("Tail Length:", 5, 30, 15)

# --- Math & Smoothing Engine ---
@st.cache_data(ttl=3600)
def get_rrg_data(ticker_list, bench, tf, tail):
    interval = "1d" if tf == "Daily" else "1wk"
    tickers = [t.strip().upper() for t in ticker_list.split(",") if t.strip()]
    all_list = list(set(tickers + [bench.upper()]))
    
    data_raw = yf.download(all_list, period="2y", interval=interval)
    data = data_raw['Close'] if isinstance(data_raw.columns, pd.MultiIndex) else data_raw[['Close']]
    vol_raw = data_raw['Volume'] if isinstance(data_raw.columns, pd.MultiIndex) else data_raw[['Volume']]
    data = data.dropna()
    
    rrg_results = {}
    table_data = []
    bench_price = data[bench.upper()]
    
    for t in tickers:
        if t not in data.columns or t == bench.upper(): continue
        
        rel_price = (data[t] / bench_price) * 100
        sma = rel_price.rolling(14).mean()
        std = rel_price.rolling(14).std()
        rs_ratio = 100 + ((rel_price - sma) / std)
        
        roc = rs_ratio.pct_change(1) * 100
        rs_momentum = 100 + ((roc - roc.rolling(14).mean()) / roc.rolling(14).std())
        
        # Volume
        rel_vol = (vol_raw[t].iloc[-1] / vol_raw[t].tail(20).mean())
        
        r_val, m_val = rs_ratio.iloc[-1], rs_momentum.iloc[-1]
        quad = "LEADING" if r_val >= 100 and m_val >= 100 else "IMPROVING" if r_val < 100 and m_val >= 100 else "LAGGING" if r_val < 100 and m_val < 100 else "WEAKENING"
        
        table_data.append({"Ticker": t, "Name": FUND_MAP.get(t, "Stock"), "Quadrant": quad, "RS-Ratio": round(r_val, 2), "Momentum": round(m_val, 2), "Rel Volume": f"{round(rel_vol, 2)}x"})

        raw_tail = pd.DataFrame({'x': rs_ratio, 'y': rs_momentum}).dropna().tail(tail)
        if len(raw_tail) >= 3:
            t_raw = np.arange(len(raw_tail))
            t_smooth = np.linspace(0, len(raw_tail)-1, len(raw_tail)*5)
            rrg_results[t] = pd.DataFrame({'x': interp1d(t_raw, raw_tail['x'], kind='cubic')(t_smooth), 
                                           'y': interp1d(t_raw, raw_tail['y'], kind='cubic')(t_smooth)})

    return rrg_results, table_data, data, tickers

# --- Plotting ---
try:
    results, table_list, full_data, tickers_list = get_rrg_data(tickers_input, benchmark, timeframe, tail_len)
    
    fig = go.Figure()

    # Quadrant Lines
    fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="black", width=2))
    fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="black", width=2))

    # Add smoothed tails and directional Arrows
    for i, (ticker, df) in enumerate(results.items()):
        line_color = px.colors.qualitative.Plotly[i % 10]

        # 1. The Line
        fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='lines', 
                                 name=ticker, line=dict(width=2, color=line_color), 
                                 showlegend=True))
        
        # 2. The Arrow (Using Annotations - much more reliable)
        fig.add_annotation(
            x=df['x'].iloc[-1],
            y=df['y'].iloc[-1],
            ax=df['x'].iloc[-2],
            ay=df['y'].iloc[-2],
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=2,
            arrowwidth=2,
            arrowcolor=line_color,
            name=ticker # This helps with visibility grouping
        )

    fig.update_layout(
        template="plotly_white", 
        xaxis=dict(title="RS-Ratio", range=[96, 104]), 
        yaxis=dict(title="RS-Momentum", range=[96, 104]),
        height=800,
        legend=dict(itemclick="toggle", itemdoubleclick="toggleothers")
    )

    st.plotly_chart(fig, use_container_width=True)

    # Leaderboard
    st.markdown("---")
    st.subheader("📊 Leaderboard")
    st.dataframe(pd.DataFrame(table_list).sort_values(by="RS-Ratio", ascending=False), use_container_width=True)

    # RS Line
    st.markdown("---")
    st.subheader("📈 RS Line Trend")
    sel = st.selectbox("Select Ticker:", list(results.keys()))
    if sel:
        rs_line = (full_data[sel] / full_data[benchmark.upper()])
        rs_line = (rs_line / rs_line.iloc[0]) * 100
        st.plotly_chart(px.line(rs_line, title=f"{sel} vs {benchmark.upper()}").add_hline(y=100, line_dash="dash", line_color="red"), use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
