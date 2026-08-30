from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class TradeLogger:
    def __init__(self, db_path: str = "trades.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    action TEXT,
                    quantity INTEGER,
                    price TEXT,
                    result TEXT,
                    status TEXT,
                    details TEXT
                )
                """
            )

    def log_trade(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: str,
        result: str,
        status: str,
        details: Any,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO trades (timestamp, symbol, action, quantity, price, result, status, details)
                VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    action,
                    quantity,
                    price,
                    result,
                    status,
                    str(details),
                ),
            )
            conn.commit()

    def get_recent_trades(self, limit: int = 10) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT timestamp, symbol, action, quantity, price, result, status, details
                FROM trades ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "timestamp": row[0],
                "symbol": row[1],
                "action": row[2],
                "quantity": row[3],
                "price": row[4],
                "result": row[5],
                "status": row[6],
                "details": row[7],
            }
            for row in rows
        ]
