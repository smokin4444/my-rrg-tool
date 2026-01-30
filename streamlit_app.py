import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="High-Beta RRG", layout="wide")

st.title("🚀 Custom Relative Rotation Graph")
st.write("Compare your high-beta stocks against a benchmark (like COPX or SPY).")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("1. Settings")
    # Ticker heap
    user_tickers = st.text_area("Enter Tickers (comma separated):", 
                                "AFM.V, NAK, A4N.AX, CSC.AX, IVN.TO, TGB, ERO.TO, HBM")
    
    # Benchmark
    benchmark = st.text_input("Benchmark Ticker:", "COPX")
    
    # Timeframe Toggle
    timeframe = st.radio("Timeframe:", ["Daily", "Weekly"])
    
    # History (Length of the tail)
    tail_length = st.slider("Tail Length (Periods):", 5, 20, 10)

# --- The RRG Math Engine ---
def calculate_rrg(ticker_list, bench, tf, tail):
    interval = "1d" if tf == "Daily" else "1wk"
    period = "2y"
    
    all_list = [t.strip() for t in ticker_list.split(",") if t.strip()] + [bench]
    data = yf.download(all_list, period=period, interval=interval)['Close']
    
    # Drop rows with missing data to prevent math errors
    data = data.dropna()
    
    rrg_results = []
    bench_price = data[bench]
    
    for t in all_list:
        if t == bench: continue
        
        # 1. Price Relative
        rel_price = (data[t] / bench_price) * 100
        
        # 2. RS-Ratio (14-period smoothed)
        sma = rel_price.rolling(14).mean()
        std = rel_price.rolling(14).std()
        rs_ratio = 100 + ((rel_price - sma) / std)
        
        # 3. RS-Momentum (Rate of Change of Ratio)
        # Using a simple 1-period ROC, smoothed over 14 periods
        roc = rs_ratio.pct_change(1) * 100
        roc_sma = roc.rolling(14).mean()
        roc_std = roc.rolling(14).std()
        rs_momentum = 100 + ((roc - roc_sma) / roc_std)
        
        # Grab the tail
        df = pd.DataFrame({'Ratio': rs_ratio, 'Momentum': rs_momentum}).tail(tail)
        df['Ticker'] = t
        rrg_results.append(df)
        
    return pd.concat(rrg_results)

# --- Generate and Display ---
try:
    with st.spinner('Fetching market data...'):
        plot_df = calculate_rrg(user_tickers, benchmark, timeframe, tail_length)
    
    # Plotly Scatter Plot
    fig = px.scatter(plot_df, x="Ratio", y="Momentum", color="Ticker",
                     text="Ticker", template="plotly_dark", height=700)
    
    # Add Quadrant Background Colors/Lines
    fig.add_hline(y=100, line_dash="dash", line_color="white")
    fig.add_vline(x=100, line_dash="dash", line_color="white")
    
    # Add the visual "Tails"
    for ticker in plot_df['Ticker'].unique():
        t_data = plot_df[plot_df['Ticker'] == ticker]
        fig.add_scatter(x=t_data['Ratio'], y=t_data['Momentum'], 
                        mode='lines', line=dict(width=2), showlegend=False)

    st.plotly_chart(fig, use_container_width=True)
    
    st.success(f"Showing {timeframe} rotation against {benchmark}")

except Exception as e:
    st.error(f"Waiting for data... Ensure tickers are correct. Error: {e}")
