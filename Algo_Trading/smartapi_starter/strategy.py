from __future__ import annotations


def ema_signal(closes: list[float]) -> str:
    if len(closes) < 20:
        return "HOLD"

    short_ma = sum(closes[-5:]) / 5
    long_ma = sum(closes[-20:]) / 20

    if short_ma > long_ma:
        return "BUY"
    if short_ma < long_ma:
        return "SELL"
    return "HOLD"


def rsi_signal(closes: list[float], period: int = 14) -> str:
    if len(closes) < period + 2:
        return "HOLD"

    gains = []
    losses = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return "BUY"

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    if rsi < 30:
        return "BUY"
    if rsi > 70:
        return "SELL"
    return "HOLD"


def evaluate_signal(closes: list[float]) -> dict:
    ema = ema_signal(closes)
    rsi = rsi_signal(closes)

    if ema == "BUY" and rsi == "BUY":
        action = "BUY"
    elif ema == "SELL" and rsi == "SELL":
        action = "SELL"
    else:
        action = "HOLD"

    short_ma = sum(closes[-5:]) / 5 if len(closes) >= 5 else None
    long_ma = sum(closes[-20:]) / 20 if len(closes) >= 20 else None
    return {
        "action": action,
        "ema_signal": ema,
        "rsi_signal": rsi,
        "short_ma": round(short_ma, 2) if short_ma is not None else None,
        "long_ma": round(long_ma, 2) if long_ma is not None else None,
    }


def calculate_trade_levels(current_price: float | None, signal: str) -> dict:
    if current_price is None:
        return {
            "current_price": None,
            "target_price": None,
            "stop_loss": None,
        }

    if signal == "BUY":
        target_price = round(current_price * 1.02, 2)
        stop_loss = round(current_price * 0.98, 2)
    elif signal == "SELL":
        target_price = round(current_price * 0.98, 2)
        stop_loss = round(current_price * 1.02, 2)
    else:
        target_price = round(current_price, 2)
        stop_loss = round(current_price, 2)

    return {
        "current_price": round(current_price, 2),
        "target_price": target_price,
        "stop_loss": stop_loss,
    }


def strategy_summary(closes: list[float], current_price: float | None = None) -> dict:
    signal = evaluate_signal(closes)
    levels = calculate_trade_levels(current_price, signal.get("action", "HOLD"))
    signal.update(levels)
    return signal
