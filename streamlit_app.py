# ... (Imports, Hub Sync, and Dictionary remain identical)

# --- ANALYTICS ENGINE UPDATE ---
with tab2:
    st.subheader("🏦 Capital Flow Scorecard")
    if data_all is not None:
        flow_data = []
        for t in tickers_list:
            if t in data_all.columns:
                px = data_all[t].dropna()
                bx = pd.Series(1.0, index=px.index) if benchmark.upper() == "ONE" else data_all[benchmark.upper()].dropna()
                common = px.index.intersection(bx.index)
                
                # 1. Trend Intensity
                sma20 = px.loc[common].rolling(20).mean()
                trend_dist = (px.loc[common].iloc[-1] / sma20.iloc[-1]) - 1
                
                # 2. RS Momentum (Relative to Benchmark)
                rel_s = (px.loc[common] / bx.loc[common])
                rs_mom = (rel_s.iloc[-1] / rel_s.iloc[-5]) - 1
                
                # --- NEW NUANCED SCORING ENGINE ---
                # We blend the two factors and use a wider divisor to avoid "clumping" at 100
                raw_blend = (trend_dist * 1.5) + (rs_mom * 2.5)
                
                # Map -0.15 to +0.15 range into 0 to 100
                # This creates the 90s, 80s, and 70s you're looking for
                score = int(np.clip((raw_blend + 0.10) / 0.20, 0, 1) * 100)
                
                flow_data.append({
                    "Ticker": t, 
                    "Name": TICKER_NAMES.get(t, t), 
                    "Flow Score": score, 
                    "Trend %": round(trend_dist*100, 1), 
                    "RS Δ": round(rs_mom*100, 1),
                    "Status": "🔥 ACCUMULATION" if score > 80 else "⚖️ HOLD" if score > 40 else "⚠️ DISTRIBUTION"
                })
        
        # Sort and Display
        df_flow = pd.DataFrame(flow_data).sort_values("Flow Score", ascending=False)
        st.dataframe(df_flow, use_container_width=True)
