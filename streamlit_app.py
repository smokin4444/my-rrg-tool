import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d

st.set_page_config(page_title="Pro-Rotation Dashboard", layout="wide")

st.title("📈 Advanced Rotation & Relative Strength")

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
    
    if heap_type == "My Miners":
        current_list = MY_MINERS
    elif heap_type == "Sector ETFs":
        current_list = SECTOR_ETFS
    else:
        current_list = st.session_state.get('custom_list', MY_MINERS)

    tickers_input = st.text_area("Ticker Heap:", value=current_list)
    benchmark = st.text_input("Benchmark:", value="SPY")
    timeframe = st.radio("Timeframe:", ["Daily", "Weekly"], index=1)
    tail_len = st.slider("Tail Length:", 5, 30, 15)
    
    st.markdown("---")
    show_all = st.checkbox("Show All on Chart", value=True)

# --- Calculations ---
@st.cache_data(ttl=3600)
def get_market_data(ticker_str, bench, tf):
    interval = "1d" if tf == "Daily" else "1wk"
    t_list = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    full_list = list(set(t_list + [bench.upper()]))
    raw = yf.download(full_list, period="2y", interval=interval)
    data = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']]
    return data.dropna(), t_list

# --- Execution ---
try:
    data, t_list = get_market_data(tickers_input, benchmark, timeframe)
    bench_ticker = benchmark.upper()
    
    rrg_dict = {}
    table_rows = [] # Fix: Ensuring list is defined before the loop starts
    
    for t in t_list:
        if t not in data.columns or t == bench_ticker: continue
        
        rel = (data[t] / data[bench_ticker]) * 100
        rs_ratio = 100 + ((rel - rel.rolling(14).mean()) / rel.rolling(14).std())
        mom = rs_ratio.pct_change(1) * 100
        rs_mom = 100 + ((mom - mom.rolling(14).mean()) / mom.rolling(14).std())
        
        # Rankings Data logic moved inside the loop
        r_val = rs_ratio.iloc[-1]
        m_val = rs_mom.iloc[-1]
        quad = "LEADING" if r_val > 100 and m_val > 100 else "IMPROVING" if r_val < 100 and m_val > 100 else "LAGGING" if r_val < 100 and m_val < 100 else "WEAKENING"
               
        table_rows.append({
            "Ticker": t, "Fund Name": FUND_MAP.get(t, "Individual Stock"),
            "Quadrant": quad, "RS-Ratio": round(r_val, 2), "Momentum": round(m_val, 2)
        })

        rt = pd.DataFrame({'x': rs_ratio, 'y': rs_mom}).dropna().tail(tail_len)
        if len(rt) > 3:
            tr = np.arange(len(rt)); ts = np.linspace(0, len(rt)-1, len(rt)*5)
            rrg_dict[t] = pd.DataFrame({'x': interp1d(tr, rt['x'], kind='cubic')(ts), 'y': interp1d(tr, rt['y'], kind='cubic')(ts)})

    # Layout: Larger Chart
    col1, col2 = st.columns([2.5, 1])

    with col1:
        fig = go.Figure()
        fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="black", width=2))
        
        # Visibility logic for legend toggle
        vis_state = True if show_all else "legendonly"
        
        for i, (t, df) in enumerate(rrg_dict.items()):
            color = px.colors.qualitative.Plotly[i % 10]
            # Mode set to lines only for the snake, markers for the arrow
            fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='lines', name=t, 
                                     line=dict(width=4, color=color), visible=vis_state))
            
            # Arrow/Label visibility tied to the snake line
            fig.add_annotation(x=df['x'].iloc[-1], y=df['y'].iloc[-1], ax=df['x'].iloc[-2], ay=df['y'].iloc[-2],
                               showarrow=True, arrowhead=2, arrowcolor=color, text=f"<b> {t} </b>", 
                               bgcolor=color, font=dict(color="white"), yshift=15, visible=show_all)
        
        # Chart Size increase
        fig.update_layout(height=850, width=850, template="plotly_white", 
                          xaxis=dict(range=[95, 105]), yaxis=dict(range=[95, 105]),
                          legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Leaderboard")
        if table_rows:
            rank_df = pd.DataFrame(table_rows).sort_values(by="RS-Ratio", ascending=False)
            st.dataframe(rank_df, height=750)

    # --- THE RELATIVE STRENGTH LINE CHART ---
    st.markdown("---")
    st.subheader("📈 Long-Term Relative Strength Trend")
    selected_stock = st.selectbox("Select a stock to view its RS Line vs Benchmark:", t_list)
    
    if selected_stock:
        rs_line = (data[selected_stock] / data[bench_ticker])
        rs_line = (rs_line / rs_line.iloc[0]) * 100
        fig_rs = px.line(rs_line, title=f"{selected_stock} vs {bench_ticker} (Indexed to 100)")
        fig_rs.add_hline(y=100, line_dash="dash", line_color="red")
        fig_rs.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_rs, use_container_width=True)

except Exception as e:
    st.error(f"Waiting for inputs... Error: {e}")
