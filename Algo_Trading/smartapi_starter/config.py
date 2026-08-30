import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    client_id: str
    secret_key: str
    redirect_uri: str
    symbol: str = "NSE:INFY-EQ"
    quantity: int = 1
    paper_trading: bool = True
    enable_live_trading: bool = False
    interval_seconds: int = 60
    max_daily_loss: float = 2000.0
    max_open_positions: int = 2


def get_settings() -> Settings:
    required = {
        "FYERS_CLIENT_ID": os.getenv("FYERS_CLIENT_ID"),
        "FYERS_SECRET_KEY": os.getenv("FYERS_SECRET_KEY"),
        "FYERS_REDIRECT_URI": os.getenv("FYERS_REDIRECT_URI"),
    }

    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill in your FYERS app credentials."
        )

    return Settings(
        client_id=os.getenv("FYERS_CLIENT_ID"),
        secret_key=os.getenv("FYERS_SECRET_KEY"),
        redirect_uri=os.getenv("FYERS_REDIRECT_URI"),
        symbol=os.getenv("FYERS_SYMBOL", "NSE:INFY-EQ"),
        quantity=int(os.getenv("FYERS_QUANTITY", "1")),
        paper_trading=True,
        enable_live_trading=os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true",
        interval_seconds=int(os.getenv("BOT_INTERVAL_SECONDS", "60")),
        max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "2000")),
        max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "2")),
    )
