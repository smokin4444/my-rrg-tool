# --- ADD THIS TO YOUR CONFIGURATION BLOCK ---
SMOOTH_LEN = 14  # Matches TradingView default smoothing

# --- UPDATED METRIC ENGINE (Replace your old get_metrics with this) ---
def get_metrics(df_raw, ticker, bench_t, is_absolute):
    try:
        if ticker not in df_raw['Close'].columns: return None
        px = df_raw['Close'][ticker].dropna()
        bx = pd.Series(1.0, index=px.index) if is_absolute else df_raw['Close'][bench_t].dropna()
        
        common = px.index.intersection(bx.index)
        if len(common) < (SMOOTH_LEN * 2): return None
        
        px_a, bx_a = px.loc[common], bx.loc[common]
        
        # 1. Raw Relative Strength Ratio
        rs_raw = (px_a / bx_a) * 100
        
        # 2. RS-Ratio: Smoothed relative strength centered at 100
        # Instead of Z-Score, we use the Ratio vs EMA method for stability
        ema_rs = rs_raw.ewm(span=SMOOTH_LEN).mean()
        std_rs = rs_raw.rolling(SMOOTH_LEN).std().replace(0, EPSILON)
        
        # Centering and standardizing for the X-Axis
        ratio = 100 + ((rs_raw - ema_rs) / std_rs)
        ratio = ratio.clip(*Z_LIMITS)
        
        # 3. RS-Momentum: Velocity of the RS-Ratio
        # We calculate this as the ROC of the smoothed ratio to eliminate "lag"
        mom_raw = ratio.diff(1)
        ema_mom = mom_raw.ewm(span=SMOOTH_LEN).mean()
        std_mom = mom_raw.rolling(SMOOTH_LEN).std().replace(0, EPSILON)
        
        momentum = 100 + ((mom_raw - ema_mom) / std_mom)
        momentum = momentum.clip(*Z_LIMITS)
        
        df_res = pd.DataFrame({'x': ratio, 'y': momentum, 'date': ratio.index}).dropna()
        df_res['date_str'] = df_res['date'].dt.strftime('%b %d, %Y')
        df_res['full_name'] = TICKER_NAMES.get(ticker, ticker)
        return df_res
    except:
        return None
