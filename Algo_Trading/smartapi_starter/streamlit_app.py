from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from config import get_settings
from main import build_signal_summary, fetch_recent_closes
from smartapi_client import FYERSClient

st.set_page_config(page_title="FYERS Signal Viewer", page_icon="📈", layout="wide")
st.title("FYERS Signal Dashboard")

AUTOFRESH_SECONDS = 10

with st.sidebar:
    st.header("Setup")
    st.caption("This UI is signal-only and does not place orders.")
    auth_code = st.text_input("FYERS Auth Code", help="Paste the auth code returned from the FYERS login flow.")

    stock_options = {
        "Infosys": "NSE:INFY-EQ",
        "TCS": "NSE:TCS-EQ",
        "Reliance": "NSE:RELIANCE-EQ",
        "HDFC Bank": "NSE:HDFCBANK-EQ",
        "ICICI Bank": "NSE:ICICIBANK-EQ",
        "IRFC": "NSE:IRFC-EQ",
        "SBI": "NSE:SBIN-EQ",
        "ITC": "NSE:ITC-EQ",
        "LT": "NSE:LT-EQ",
        "IRCTC": "NSE:IRCTC-EQ",
        "BPCL": "NSE:BPCL-EQ",
        "NSLNISP": "NSE:NSLNISP-EQ",
        "TATASTEEL": "NSE:TATASTEEL-EQ",
    }
    selected_stock = st.selectbox("Choose a stock", ["Custom", *list(stock_options.keys())], index=0)
    custom_symbol = st.text_input("Or enter symbol manually", value="NSE:IRFC-EQ")
    symbol = stock_options.get(selected_stock, custom_symbol) if selected_stock != "Custom" else custom_symbol
    refresh = st.button("Refresh signal")

if not auth_code:
    st.info("Paste your FYERS auth code to load the signal.")
    st.stop()

if refresh:
    st.rerun()

settings = get_settings()
settings = settings.__class__(**{**settings.__dict__, "symbol": symbol})
client = FYERSClient(settings)
client.login(auth_code=auth_code)

summary = build_signal_summary(client)

recommendation = summary["recommendation"]
color = {
    "BUY": "green",
    "SELL": "red",
    "HOLD": "orange",
}.get(recommendation, "gray")

st.markdown(f"## Signal: <span style='color:{color}'><b>{recommendation}</b></span>", unsafe_allow_html=True)
st.write(f"Stock: {summary['stock']}")

col1, col2 = st.columns(2)
col1.metric("Current Price", summary["current_price"] if summary["current_price"] is not None else "N/A")
col2.metric("Target", summary["target_price"] if summary["target_price"] is not None else "N/A")

col3, col4 = st.columns(2)
col3.metric("Stop Loss", summary["stop_loss"] if summary["stop_loss"] is not None else "N/A")
col4.metric("RSI Signal", summary["rsi_signal"])

st.markdown("---")

col5, col6 = st.columns(2)
col5.metric("EMA Signal", summary["ema_signal"])
col6.metric("Long MA", summary["long_ma"] if summary["long_ma"] is not None else "N/A")

st.info(summary["message"])

try:
    closes = fetch_recent_closes(client, minutes=90)
    if closes:
        df = pd.DataFrame({"close": closes})
        df.index = pd.RangeIndex(start=1, stop=len(df) + 1)
        st.subheader("Recent price trend")
        st.line_chart(df, y="close")
    else:
        st.warning("No chart data available yet.")
except Exception as exc:
    st.warning(f"Chart unavailable: {exc}")

st.caption(f"Auto-refreshing every {AUTOFRESH_SECONDS} seconds.")
time.sleep(AUTOFRESH_SECONDS)
st.rerun()
