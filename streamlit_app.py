import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# Set up the page
st.set_page_config(page_title="High-Beta RRG Dashboard", layout="wide")

st.title("🚀 Relative Rotation Graph (RRG)")
st.write("Compare your juniors and high-beta names against any benchmark.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("1. Settings")
    # Your "Heap" of tickers - You can add as many as you want here
    user_tickers = st.text_area("Enter Tickers (comma separated):", 
                                "AFM.V, NAK, A4N.AX, CSC.AX, IVN.TO, TGB, ERO.TO, HBM")
    
    # Benchmark - This is what you compare them against (e.g., COPX or SPY)
    benchmark = st.text_input("Benchmark Ticker:", "COPX")
    
    # Timeframe Toggle
    timeframe = st.radio("Timeframe:", ["Daily", "Weekly"])
    
    # History (Length of the tail)
    tail_length = st.slider("Tail Length (Periods):", 5, 20, 10)

# --- The RRG Math Engine ---
def calculate_rrg(ticker_list, bench, tf, tail):
    # Set interval based on toggle
    interval = "1d" if tf == "Daily" else "1wk"
    period = "2y" # We need enough data for the 14-period average
    
    # Clean the ticker list
    tickers = [t.strip() for t in ticker_list.split(",") if t.strip()]
    all_list = list(set(tickers + [bench])) # Remove duplicates
    
    # Download data
    data_raw = yf.download(all_list, period=period, interval=interval)
    
    # --- CRITICAL FIX: Flatten the Multi-Index Table ---
    # When yfinance downloads multiple tickers, it creates a "Double Header"
    # We only want the 'Close' prices
    if isinstance(data_raw.columns, pd.MultiIndex):
        data = data_raw['Close']
    else:
        # If it's a single ticker, it might not be a MultiIndex
        data = data_raw[['Close']]
        data.columns = all_list # Rename to the ticker name
        
    data = data.dropna()
    
    rrg_results = []
    
    # Safety Check: Ensure the benchmark exists in the data
    if bench not in data.columns:
        st.error(f"Error: Benchmark '{bench}' not found in data. Check the ticker name.")
        return pd.DataFrame()

    bench_price = data[bench]
    
    for t in tickers:
        if t not in data.columns or t == bench:
            continue
        
        # 1. Price Relative (Stock / Benchmark)
        rel_price = (data[t] / bench_price) * 100
        
        # 2. RS-Ratio (Moving Average & Standard Deviation)
        # Traditionally uses a 14-period smoothing
        sma = rel_price.rolling(14).mean()
        std = rel_price.rolling(14).std()
        rs_ratio = 100 + ((rel_price - sma) / std)
        
        # 3. RS-Momentum (Rate of Change of the Ratio)
        roc = rs_ratio.pct_change(1) * 100
        roc_sma = roc.rolling(14).mean()
        roc_std = roc.rolling(14).std()
        rs_momentum = 100 + ((roc - roc_sma) / roc_std)
        
        # Create a tiny table for just this stock's "tail"
        df = pd.DataFrame({'Ratio': rs_ratio, 'Momentum': rs_momentum}).tail(tail)
        df['Ticker'] = t
        rrg_results.append(df)
        
    if not rrg_results:
        return pd.DataFrame()
        
    return pd.concat(rrg_results)

# --- Generate and Display ---
try:
    with st.spinner('Calculating Rotation...'):
        plot_df = calculate_rrg(user_tickers, benchmark, timeframe, tail_length)
    
    if not plot_df.empty:
        # Create the Plotly Scatter Plot
        fig = px.scatter(plot_df, x="Ratio", y="Momentum", color="Ticker",
                         text="Ticker", template="plotly_dark", height=700)
        
        # Add Quadrant Lines (The crosshair at 100, 100)
        fig.add_hline(y=100, line_dash="dash", line_color="white")
        fig.add_vline(x=100, line_dash="dash", line_color="white")
        
        # Add the "Tails" (The lines showing where the stock came from)
        for ticker in plot_df['Ticker'].unique():
            t_data = plot_df[plot_df['Ticker'] == ticker]
            fig.add_scatter(x=t_data['Ratio'], y=t_data['Momentum'], 
                            mode='lines', line=dict(width=2), showlegend=False)

        # Labels for the quadrants
        fig.add_annotation(x=102, y=102, text="LEADING", showarrow=False, font=dict(color="green"))
        fig.add_annotation(x=98, y=102, text="IMPROVING", showarrow=False, font=dict(color="blue"))
        fig.add_annotation(x=98, y=98, text="LAGGING", showarrow=False, font=dict(color="red"))
        fig.add_annotation(x=102, y=98, text="WEAKENING", showarrow=False, font=dict(color="yellow"))

        st.plotly_chart(fig, use_container_width=True)
        st.success(f"Successfully comparing {len(plot_df['Ticker'].unique())} tickers vs {benchmark}")
    else:
        st.warning("No data found for these tickers. Check your spelling (e.g., AFM.V or A4N.AX).")

except Exception as e:
    st.error(f"Something went wrong: {e}")
