from datetime import datetime

from config import Settings
from smartapi_client import FYERSClient


class DummyModel:
    def __init__(self):
        self.last_params = None

    def history(self, params):
        self.last_params = params
        return {
            "code": 200,
            "message": "ok",
            "s": "ok",
            "candles": [
                [1700000000, 100.0, 101.0, 99.0, 100.5, 10],
                [1700000600, 100.5, 102.0, 100.0, 101.0, 20],
                [1700001200, 101.0, 103.0, 100.5, 102.0, 30],
            ],
        }


def test_get_candles_uses_date_only_ranges_for_fyers():
    settings = Settings(
        client_id="demo",
        secret_key="demo",
        redirect_uri="https://example.com",
    )
    client = FYERSClient(settings)
    client.model = DummyModel()

    from_date = datetime(2026, 8, 30, 9, 45)
    to_date = datetime(2026, 8, 30, 10, 15)

    client.get_candles(
        symbol="NSE:INFY-EQ",
        timeframe="1",
        from_date=from_date,
        to_date=to_date,
    )

    assert client.model.last_params["date_format"] == 1
    assert client.model.last_params["range_from"] == "2026-08-30"
    assert client.model.last_params["range_to"] == "2026-08-30"
