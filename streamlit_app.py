def get_metrics(df_raw, ticker, bench_t, is_absolute):
    try:
        if ticker not in df_raw['Close'].columns: return None
        px = df_raw['Close'][ticker].dropna()
        bx = pd.Series(1.0, index=px.index) if is_absolute else df_raw['Close'][bench_t].dropna()
        common = px.index.intersection(bx.index)
        if len(common) < LOOKBACK + 5: return None
        px_a, bx_a = px.loc[common], bx.loc[common]
        
        # --- THE ADDITION: Light EMA Smoothing ---
        # This smooths the raw ratio by the last 3 bars before Z-score calculation
        rel_raw = (px_a / bx_a) * 100
        rel = rel_raw.ewm(span=3).mean() 
        
        def standardize(series):
            return RRG_CENTER + ((series - series.rolling(LOOKBACK).mean()) / series.rolling(LOOKBACK).std().replace(0, EPSILON))
        
        ratio, mom = standardize(rel).clip(*Z_LIMITS), standardize(rel.diff(1)).clip(*Z_LIMITS)
        
        df_res = pd.DataFrame({'x': ratio, 'y': mom, 'date': ratio.index}).dropna()
        df_res['date_str'] = df_res['date'].dt.strftime('%b %d, %Y')
        df_res['full_name'] = TICKER_NAMES.get(ticker, ticker)
        return df_res
    except: return None
