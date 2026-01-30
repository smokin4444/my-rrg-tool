import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d

# --- Page Configuration ---
st.set_page_config(page_title="Pro-RRG Dashboard", layout="wide")

st.title("📈 Advanced Rotation & Volume Conviction")

# --- Persistent Ticker Management ---
# Default lists for quick switching
MY_MINERS = "AFM.V, NAK, A4N.AX, CSC.AX, IVN.TO, TGB"
SECTOR_ETFS = "XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU"

# Dictionary for mapping tickers to names in the leaderboard
FUND_MAP = {
    "XLC": "Comm. Services", "XLY": "Cons. Discretionary", "XLP": "Cons. Staples",
    "XLE": "Energy", "XLF": "Financials", "XLV": "Health Care",
    "XLI": "Industrials", "XLB": "Materials", "XLRE": "Real Estate",
    "XLK": "Technology", "XLU": "Utilities", "XBI": "Biotech",
    "XME": "Metals & Mining", "XSD": "Semiconductors", "XOP": "Oil & Gas Exploration",
    "GDXJ": "Junior Gold Miners", "COPX": "Copper Miners", "REMX": "Strategic Metals"
}

# --- Sidebar Controls ---
with st.sidebar:
    st.header("1. Controls")
    heap_type = st.radio("Choose Watchlist:", ["My Miners", "Sector ETFs", "Custom List"])
    
    if heap_type == "My Miners":
        current_list = MY_MINERS
    elif heap_type == "Sector ETFs":
        current_list = SECTOR_ETFS
    else:
        current_list = st.session_state.get('custom_list', MY_MINERS)

    tickers_input = st.text_area("Ticker Heap (Comma Separated):", value=current_list)
    benchmark = st.text_input("Benchmark Ticker:", value="SPY")
    timeframe = st.radio("Timeframe:", ["Daily", "Weekly"], index=1)
    tail_len = st.slider("Tail Length (Periods):", 5, 30, 15)

# --- Data Engine ---
@st.cache_data(ttl=3600)
def get_market_data(ticker_str, bench, tf):
    interval = "1d" if tf == "Daily" else "1wk"
    t_list = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    full_list = list(set(t_list + [bench.upper()]))
    
    # Download Close Price and Volume
    raw = yf.download(full_list, period="2y", interval=interval)
    
    close_data = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']]
    vol_data = raw['Volume'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Volume']]
    
    return close_data.dropna(), vol_data, t_list

# --- Main Dashboard Logic ---
try:
    data, vol_data, t_list = get_market_data(tickers_input, benchmark, timeframe)
    bench_ticker = benchmark.upper()
    
    rrg_dict = {}
    table_rows = [] 
    
    for t in t_list:
        if t not in data.columns or t == bench_ticker: 
            continue
        
        # 1. RRG Math (14-period standard)
        rel_strength = (data[t] / data[bench_ticker]) * 100
        rs_ratio = 100 + ((rel_strength - rel_strength.rolling(14).mean()) / rel_strength.rolling(14).std())
        
        momentum_raw = rs_ratio.pct_change(1) * 100
        rs_momentum = 100 + ((momentum_raw - momentum_raw.rolling(14).mean()) / momentum_raw.rolling(14).std())
        
        # 2. Volume Conviction (Current vs 20-period Average)
        current_vol = vol_data[t].iloc[-1]
        avg_vol = vol_data[t].tail(20).mean()
        rel_vol = (current_vol / avg_vol)
        
        # 3. Quadrant Logic
        r_val = rs_ratio.iloc[-1]
        m_val = rs_momentum.iloc[-1]
        
        if r_val >= 100 and m_val >= 100: quad = "LEADING"
        elif r_val < 100 and m_val >= 100: quad = "IMPROVING"
        elif r_val < 100 and m_val < 100: quad = "LAGGING"
        else: quad = "WEAKENING"
               
        table_rows.append({
            "Ticker": t, 
            "Name": FUND_MAP.get(t, "Stock/Other"),
            "Quadrant": quad, 
            "RS-Ratio": round(r_val, 2), 
            "Momentum": round(m_val, 2), 
            "Rel Volume": f"{round(rel_vol, 2)}x"
        })

        # 4. Tail Smoothing for the Chart
        raw_tail = pd.DataFrame({'x': rs_ratio, 'y': rs_momentum}).dropna().tail(tail_len)
        if len(raw_tail) > 3:
            t_indices = np.arange(len(raw_tail))
            smooth_indices = np.linspace(0, len(raw_tail)-1, len(raw_tail)*5)
            
            fx = interp1d(t_indices, raw_tail['x'], kind='cubic')
            fy = interp1d(t_indices, raw_tail['y'], kind='cubic')
            
            rrg_dict[t] = pd.DataFrame({'x': fx(smooth_indices), 'y': fy(smooth_indices)})

    # --- 1. THE RRG CHART ---
    fig = go.Figure()
    
    # Quadrant Dividers
    fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="black", width=2))
    fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="black", width=2))
    
    for i, (t, df) in enumerate(rrg_dict.items()):
        color = px.colors.qualitative.Plotly[i % 10]
        
        # The Snake Trail
        fig.add_trace(go.Scatter(
            x=df['x'], y=df['y'], mode='lines', name=t, 
            line=dict(width=1.5, color=color),
            legendgroup=t
        ))
        
        # The Directional Arrow (Triangle pointing along the path)
        fig.add_trace(go.Scatter(
            x=[df['x'].iloc[-1]], y=[df['y'].iloc[-1]],
            mode='markers',
            marker=dict(symbol='triangle-up', size=14, color=color, angleref='previous'),
            name=t, legendgroup=t, showlegend=False,
            hoverinfo='text', 
            text=f"Ticker: {t}<br>Ratio: {df['x'].iloc[-1]:.2f}<br>Momentum: {df['y'].iloc[-1]:.2f}"
        ))
    
    fig.update_layout(
        height=800, template="plotly_white", 
        xaxis=dict(title="RS-Ratio (Strength)", range=[95, 105]), 
        yaxis=dict(title="RS-Momentum (Velocity)", range=[95, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- 2. THE LEADERBOARD ---
    st.markdown("---")
    st.subheader("📊 Sector Leaderboard & Conviction")
    if table_rows:
        rank_df = pd.DataFrame(table_rows).sort_values(by="RS-Ratio", ascending=False)
        
        # Quadrant color styling
        def color_quadrant(val):
            color = 'green' if val == 'LEADING' else 'blue' if val == 'IMPROVING' else 'red' if val == 'LAGGING' else 'orange'
            return f'color: {color}; font-weight: bold'

        st.dataframe(rank_df.style.map(color_quadrant, subset=['Quadrant']), use_container_width=True)

    # --- 3. RS LINE CHART ---
    st.markdown("---")
    st.subheader("📈 Long-Term Relative Strength Trend")
    selected_stock = st.selectbox("Analyze RS Line for:", t_list)
    
    if selected_stock:
        # Calculate Indexed RS Line
        rs_line = (data[selected_stock] / data[bench_ticker])
        rs_line = (rs_line / rs_line.iloc[0]) * 100
        
        fig_rs = px.line(rs_line, title=f"{selected_stock} vs {bench_ticker} (Performance Indexed to 100)")
        fig_rs.add_hline(y=100, line_dash="dash", line_color="red")
        fig_rs.update_layout(template="plotly_white", height=500, yaxis_title="Relative Performance")
        st.plotly_chart(fig_rs, use_container_width=True)

except Exception as e:
    st.error(f"Dashboard Initialization Error: {e}")
