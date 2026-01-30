import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d

st.set_page_config(page_title="Pro-RRG Dashboard", layout="wide")

st.title("📈 Advanced Rotation & Volume Conviction")

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

# --- Calculations Engine ---
@st.cache_data(ttl=3600)
def get_market_data(
