# --- IMPROVED ERROR HANDLING FOR DAILY THEMES ---
def get_metrics(df_raw, ticker, bench_t, is_absolute):
    try:
        # SAFETY CHECK: Ensure df_raw exists and contains the ticker
        if df_raw is None or 'Close' not in df_raw or ticker not in df_raw['Close'].columns:
            return None
            
        px = df_raw['Close'][ticker].dropna()
        bx = pd.Series(1.0, index=px.index) if is_absolute else df_raw['Close'][bench_t].dropna()
        
        # Ensure we have enough data points for the 14-period RRG
        common = px.index.intersection(bx.index)
        if len(common) < LOOKBACK + 5: 
            return None
            
        px_a, bx_a = px.loc[common], bx.loc[common]
        rel = ((px_a / bx_a) * 100).ewm(span=3).mean() 
        
        def standardize(s): 
            return RRG_CENTER + ((s - s.rolling(LOOKBACK).mean()) / s.rolling(LOOKBACK).std().replace(0, EPSILON))
        
        ratio, mom = standardize(rel).clip(*Z_LIMITS), standardize(rel.diff(1)).clip(*Z_LIMITS)
        df_res = pd.DataFrame({'x': ratio, 'y': mom, 'date': ratio.index}).dropna()
        
        # Friday Fix Logic
        day_diff = (df_res['date'].iloc[1] - df_res['date'].iloc[0]).days if len(df_res) > 1 else 0
        df_res['display_date'] = df_res['date'] + pd.Timedelta(days=4) if day_diff >= 5 else df_res['date']
        df_res['date_str'] = df_res['display_date'].dt.strftime('%b %d, %Y')
        df_res['full_name'] = TICKER_NAMES.get(ticker, ticker)
        return df_res
    except Exception as e:
        # Silently fail for individual tickers to keep the app running
        return None
