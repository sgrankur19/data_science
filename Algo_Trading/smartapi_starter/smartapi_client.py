from __future__ import annotations

from datetime import date, datetime
from typing import Any

from fyers_apiv3.fyersModel import FyersModel, SessionModel

from config import Settings


class FYERSClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.session_model = SessionModel(
            client_id=settings.client_id,
            redirect_uri=settings.redirect_uri,
            response_type="code",
            state="fyers_trading_bot",
            secret_key=settings.secret_key,
            grant_type="authorization_code",
        )
        self.model: FyersModel | None = None
        self.access_token: str | None = None
        self.profile: dict[str, Any] | None = None

    def build_auth_url(self) -> str:
        return self.session_model.generate_authcode()

    def login(self, auth_code: str | None = None) -> dict[str, Any]:
        if not auth_code:
            raise RuntimeError(
                "FYERS auth_code is required. Open this URL in a browser, then paste the returned code: "
                + self.build_auth_url()
            )

        self.session_model.set_token(auth_code)
        token_response = self.session_model.generate_token()
        access_token = token_response.get("access_token")
        if not access_token:
            raise RuntimeError(f"FYERS login failed: {token_response}")

        self.access_token = access_token
        self.model = FyersModel(client_id=self.settings.client_id, token=self.access_token, log_path="/tmp")
        self.profile = token_response
        return token_response

    def get_quote(self, symbol: str | None = None) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("FYERS client is not logged in. Call login() first.")
        symbol = symbol or self.settings.symbol
        return self.model.quotes({"symbols": symbol})

    @staticmethod
    def _to_fyers_date(value: str | datetime | date | None) -> str:
        if value is None:
            return datetime.now().strftime("%Y-%m-%d")
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, date):
            return value.isoformat()
        text = str(value).strip()
        if not text:
            return datetime.now().strftime("%Y-%m-%d")
        return text.split(" ")[0].split("T")[0]

    def get_candles(
        self,
        symbol: str | None = None,
        timeframe: str = "1",
        from_date: str | datetime | date | None = None,
        to_date: str | datetime | date | None = None,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("FYERS client is not logged in. Call login() first.")
        symbol = symbol or self.settings.symbol
        params = {
            "symbol": symbol,
            "resolution": timeframe,
            "date_format": 1,
            "range_from": self._to_fyers_date(from_date),
            "range_to": self._to_fyers_date(to_date),
            "cont_flag": 1,
        }
        return self.model.history(params)

    def place_order(
        self,
        transaction_type: str,
        quantity: int | None = None,
        price: str | None = None,
        order_type: str = "LIMIT",
        product_type: str = "CNC",
        duration: str = "DAY",
        symbol: str | None = None,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("FYERS client is not logged in. Call login() first.")
        qty = quantity if quantity is not None else self.settings.quantity
        side = 1 if transaction_type.upper() == "BUY" else -1
        payload = {
            "symbol": symbol or self.settings.symbol,
            "qty": int(qty),
            "type": 2 if order_type.upper() == "LIMIT" else 1,
            "side": side,
            "productType": product_type.upper(),
            "limitPrice": float(price) if price else 0,
            "stopPrice": 0,
            "validity": duration.upper(),
            "disclosedQty": 0,
            "offlineOrder": False,
            "stopLoss": 0,
            "takeProfit": 0,
        }
        return self.model.place_order(payload)
