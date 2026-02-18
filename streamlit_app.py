import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import os

# --- CONFIGURATION ---
LOOKBACK = 14
RRG_CENTER = 100
EPSILON = 1e-8
Z_LIMITS = (80, 120)  
CHART_RANGE = [96.5, 103.5] 
POWER_WALK_LEVEL = 101.5
CUSTOM_FILE = "custom_watchlist.txt"

st.set_page_config(page_title="Alpha-Scanner Pro", layout="wide")

# --- TICKER DICTIONARY ---
TICKER_NAMES = {
    "BOAT": "Global Shipping ETF", "BDRY": "Dry Bulk Shipping", "XME": "S&P Metals & Mining",
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100", "XLP": "Cons Staples", "SMH": "Semiconductors",
    "SOXX": "Memory/Broad Semi", "FTXL": "Memory Super-Cycle", "IBIT": "Bitcoin ETF",
    "WGMI": "Bitcoin Miners", "HACK": "Cybersecurity", "BOTZ": "Robotics & AI",
    "ZS=F": "Soybean Futures", "GC=F": "Gold Futures", "CL=F": "Crude Oil",
    "USCL.TO": "Horizon US Large Cap", "BANK.TO": "Evolve Cdn Banks"
}

# --- WATCHLISTS ---
MAJOR_THEMES = "SPY, QQQ, DIA, IWF, IWD, MAGS, IWM, IJR, GLD, SLV, COPX, XLE, IBIT, IGV, XLP, XLRE, ARKK, TLT, UUP, XME, SMH, SOXX, FTXL"
TV_INDUSTRIES_FULL = "BOAT, BDRY, XES, OIH, FLR, EVX, AMLP, VTI, TTD, VPP, SPGI, MAN, WSC, SYY, AVT, MCK, FI, ACN, IGV, FDN, UNH, THC, HCA, IQV, DIS, NXST, CHTR, NYT, EATZ, CRUZ, BETZ, PEJ, KR, CVS, M, WMT, NKE, HD, BBY, TSCO, ONLN, IYT, XLU, XLF, IYZ, XLI, VAW, SMH, IBB, XHB, XLP, XRT"
DEEPVUE_THEMES = "BOAT, IBIT, WGMI, GDX, SIL, SMH, SOXX, FTXL, HACK, BOTZ, QTUM, TAN, XBI, IDNA, IYT, JETS, XHB, SLX, KRE, ITA, XME, KWEB, XLE, IHI, IGV"

# --- PERSISTENCE LOGIC ---
def load_custom():
    if os.path.exists(CUSTOM_FILE):
        with open(CUSTOM_FILE, "r") as f:
            return f.read()
    return "AAPL, TSLA, NVDA"

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎯 Watchlist")
    group_choice = st.radio("Choose Group:", ["Major Themes", "Sector Rotation", "Hard Assets (Live)", "TV Industries (Full)", "Income Stocks", "DeepVue Themes", "Single/Custom"])
    
    # Logic to handle the Custom list persistence
    initial_custom = load_custom()
    
    current_list = {
        "Major Themes": MAJOR_THEMES, 
        "TV Industries (Full)": TV_INDUSTRIES_FULL,
        "DeepVue Themes": DEEPVUE_THEMES,
        "Single/Custom": initial_custom
    }.get(group_choice, "")
    
    tickers_input = st.text_area("Ticker Heap:", value=current_list, height=150)
    
    # NEW: Save button for the Custom group
    if group_choice == "Single/Custom":
        if st.button("💾 Save Custom List"):
            with open(CUSTOM_FILE, "w") as f:
                f.write(tickers_input)
            st.success("List saved locally!")

    auto_bench = "ONE" if group_choice in ["Hard Assets (Live)", "Income Stocks"] else "SPY"
    benchmark = st.text_input("Active Benchmark:", value=auto_bench)
    main_timeframe = st.radio("Display Chart Timeframe:", ["Weekly", "Daily"], index=0)
    tail_len = st.slider("Tail Length:", 2, 30, 3)

# --- ANALYTICS ENGINE & DISPLAY ---
# (Existing get_metrics, run_dual_analysis, and display_deepvue_theme_tracker logic remains here)
