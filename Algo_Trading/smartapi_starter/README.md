# FYERS Trading Bot Starter

This project is set up for the FYERS API v3 app-based auth flow and is dry-run by default.

## Setup

```bash
cd /Users/ankur/Documents/GitHub/data_science/Algo_Trading/smartapi_starter
cp .env.example .env
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python main.py --auth-code YOUR_AUTH_CODE
```

## Environment

```env
FYERS_CLIENT_ID=YOUR_APP_ID
FYERS_SECRET_KEY=YOUR_SECRET_ID
FYERS_REDIRECT_URI=https://trade.fyers.in/api-login/redirect-uri/index.html
FYERS_SYMBOL=NSE:INFY-EQ
FYERS_QUANTITY=1
ENABLE_LIVE_TRADING=false
MAX_DAILY_LOSS=2000
MAX_OPEN_POSITIONS=2
```

## FYERS auth flow

1. Create a FYERS app in the FYERS dashboard.
2. Copy the App ID and Secret ID into `.env`.
3. Open the generated auth URL from the app to get an `auth_code`.
4. Run the bot with:

```bash
python main.py --auth-code YOUR_AUTH_CODE
```

The auth code is exchanged for an access token, and that token is used to make quote and order calls.

> Do not use username/password/TOTP fields in this configuration. FYERS authentication is app-based via `client_id`, `secret_key`, and `auth_code`.
