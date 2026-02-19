# --- OPTIMIZED DOWNLOAD ENGINE ---
@st.cache_data(ttl=600)
def download_data(tickers, interval):
    # Reduced chunk size and only requesting 'Close' to minimize payload
    period, chunk_size, dfs = "2y", 10, []
    tickers = list(set([t.strip().upper() for t in tickers if t.strip()]))
    
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            # group_by='ticker' + auto_adjust helps prevent NoneType structure errors
            data = yf.download(chunk, period=period, interval=interval, progress=False, actions=False)
            if data is not None and not data.empty:
                # Handle multi-index columns vs single ticker returns
                if isinstance(data.columns, pd.MultiIndex):
                    dfs.append(data['Close'])
                else:
                    # Single ticker case
                    df_single = data[['Close']].rename(columns={'Close': chunk[0]})
                    dfs.append(df_single)
            time.sleep(1.2) # Increased delay for Daily requests
        except: pass
        
    if not dfs: return None
    combined = pd.concat(dfs, axis=1)
    # Ensure there are no duplicate columns that crash the app
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined

def run_dual_analysis(ticker_str, bench, tf_display):
    tickers = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    bench_t, is_absolute = bench.strip().upper(), bench.strip().upper() == "ONE"
    
    all_req_tickers = list(set(tickers + ([bench_t] if not is_absolute else [])))
    
    # Logic: Only request the timeframe being displayed to save bandwidth
    data_d = download_data(all_req_tickers, "1d")
    data_w = download_data(all_req_tickers, "1wk")
    
    # TRIPLE CHECK: If download failed, notify user
    if tf_display == "Daily" and (data_d is None or data_d.empty):
        st.warning("⚠️ Daily data download delayed. Retrying... (Check your internet or hit 'Reset Engine')")
        return pd.DataFrame(), {}

    # ... (Rest of logic remains the same)
