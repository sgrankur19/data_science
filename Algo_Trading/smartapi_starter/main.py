from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta

from config import get_settings
from smartapi_client import FYERSClient
from strategy import calculate_trade_levels, evaluate_signal


def fetch_recent_closes(client: FYERSClient, minutes: int = 30) -> list[float]:
    now = datetime.now()
    start = now - timedelta(minutes=minutes)

    for offset_days in (0, 1, 2):
        from_date = (start - timedelta(days=offset_days)).date()
        to_date = now.date()
        candles = client.get_candles(
            symbol=client.settings.symbol,
            timeframe="1",
            from_date=from_date,
            to_date=to_date,
        )
        candles_data = candles.get("candles", []) if isinstance(candles, dict) else []
        if candles_data:
            break

    closes = []
    for candle in candles_data:
        if isinstance(candle, (list, tuple)) and len(candle) >= 5:
            close = candle[4]
            if close is not None:
                closes.append(float(close))
    return closes


def get_stock_label(client: FYERSClient, symbol: str | None = None) -> str:
    quote_symbol = symbol or client.settings.symbol
    stock_names = {
        "NSE:INFY-EQ": "Infosys Ltd.",
        "INFY-EQ": "Infosys Ltd.",
        "INFY": "Infosys Ltd.",
        "NSE:TCS-EQ": "Tata Consultancy Services Ltd.",
        "TCS-EQ": "Tata Consultancy Services Ltd.",
        "TCS": "Tata Consultancy Services Ltd.",
        "NSE:RELIANCE-EQ": "Reliance Industries Ltd.",
        "RELIANCE-EQ": "Reliance Industries Ltd.",
        "RELIANCE": "Reliance Industries Ltd.",
        "NSE:HDFCBANK-EQ": "HDFC Bank Ltd.",
        "HDFCBANK-EQ": "HDFC Bank Ltd.",
        "HDFCBANK": "HDFC Bank Ltd.",
        "NSE:ICICIBANK-EQ": "ICICI Bank Ltd.",
        "ICICIBANK-EQ": "ICICI Bank Ltd.",
        "ICICIBANK": "ICICI Bank Ltd.",
    }

    mapped_name = stock_names.get(quote_symbol) or stock_names.get(quote_symbol.replace("NSE:", ""))
    if mapped_name:
        return f"{mapped_name} ({quote_symbol})"

    try:
        quote = client.get_quote(symbol=quote_symbol)
        items = quote.get("d", []) if isinstance(quote, dict) else []
        if items:
            first = items[0].get("v", {}) if isinstance(items[0], dict) else {}
            short_name = first.get("short_name") or first.get("description") or first.get("symbol")
            if short_name:
                return f"{short_name} ({quote_symbol})"
    except Exception:
        pass
    return quote_symbol


def build_signal_summary(client: FYERSClient) -> dict:
    stock_label = get_stock_label(client)
    closes = fetch_recent_closes(client, minutes=60)
    if not closes:
        return {
            "stock": stock_label,
            "recommendation": "HOLD",
            "ema_signal": "HOLD",
            "rsi_signal": "HOLD",
            "short_ma": None,
            "long_ma": None,
            "current_price": None,
            "target_price": None,
            "stop_loss": None,
            "message": "No candle data received.",
            "action": "HOLD",
        }

    signal = evaluate_signal(closes)
    action = signal.get("action")
    recommendation = "BUY" if action == "BUY" else "SELL" if action == "SELL" else "HOLD"

    quote = client.get_quote(symbol=client.settings.symbol)
    price = None
    try:
        items = quote.get("d", []) if isinstance(quote, dict) else []
        if items:
            first = items[0].get("v", {}) if isinstance(items[0], dict) else {}
            if isinstance(first, dict):
                price = first.get("last_price") or first.get("lp") or first.get("ltp")
    except Exception:
        price = None

    levels = calculate_trade_levels(price, recommendation)

    return {
        "stock": stock_label,
        "recommendation": recommendation,
        "ema_signal": signal.get("ema_signal"),
        "rsi_signal": signal.get("rsi_signal"),
        "short_ma": signal.get("short_ma"),
        "long_ma": signal.get("long_ma"),
        "current_price": levels.get("current_price"),
        "target_price": levels.get("target_price"),
        "stop_loss": levels.get("stop_loss"),
        "message": "Signal only mode: no order will be placed.",
        "action": action,
    }


def process_signal(client: FYERSClient) -> None:
    summary = build_signal_summary(client)
    stock_label = summary["stock"]
    recommendation = summary["recommendation"]

    print("\n=== Trading Signal ===")
    print(f"Stock: {stock_label}")
    print(f"Recommendation: {recommendation}")
    print(f"EMA Signal: {summary.get('ema_signal')}")
    print(f"RSI Signal: {summary.get('rsi_signal')}")
    print(f"Short MA: {summary.get('short_ma')}")
    print(f"Long MA: {summary.get('long_ma')}")
    print("====================")

    if summary["action"] == "HOLD":
        print("No trade signal. Recommendation: HOLD")
        return

    print(f"Signal only mode: {recommendation} recommendation generated for {stock_label}, no order will be placed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="FYERS trading bot")
    parser.add_argument("--once", action="store_true", help="run one cycle only")
    parser.add_argument("--interval", type=int, default=None, help="seconds between cycles")
    parser.add_argument("--auth-code", type=str, default=None, help="FYERS auth code returned after login")
    args = parser.parse_args()

    settings = get_settings()
    if args.interval is not None:
        settings = settings.__class__(**{**settings.__dict__, "interval_seconds": args.interval})

    client = FYERSClient(settings)
    print("Logging in to FYERS...")
    if args.auth_code is None:
        print("Open this URL and paste the returned auth_code:")
        print(client.build_auth_url())
        return

    profile = client.login(auth_code=args.auth_code)
    print("Login successful.")
    print(profile)

    quote = client.get_quote(symbol=client.settings.symbol)
    print("Latest quote:")
    print(quote)

    if args.once:
        process_signal(client)
        return

    while True:
        process_signal(client)
        time.sleep(settings.interval_seconds)


if __name__ == "__main__":
    main()
