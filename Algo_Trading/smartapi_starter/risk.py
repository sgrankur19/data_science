from __future__ import annotations


class RiskManager:
    def __init__(self, max_daily_loss: float = 2000.0, max_open_positions: int = 2):
        self.max_daily_loss = max_daily_loss
        self.max_open_positions = max_open_positions
        self.daily_pnl = 0.0
        self.open_positions = 0

    def update_daily_pnl(self, pnl_value: float) -> None:
        self.daily_pnl += pnl_value

    def check_trade(self, action: str) -> bool:
        if self.daily_pnl <= -self.max_daily_loss:
            return False
        if action in {"BUY", "SELL"} and self.open_positions >= self.max_open_positions:
            return False
        return True

    def register_trade(self, action: str) -> None:
        if action in {"BUY", "SELL"}:
            self.open_positions += 1

    def close_trade(self) -> None:
        self.open_positions = max(0, self.open_positions - 1)
