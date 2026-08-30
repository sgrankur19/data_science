from __future__ import annotations

import os
import webbrowser
from pathlib import Path

from config import get_settings
from smartapi_client import FYERSClient


def main() -> None:
    settings = get_settings()
    client = FYERSClient(settings)
    auth_url = client.build_auth_url()

    print("FYERS login URL:")
    print(auth_url)
    print()
    print("Opening the FYERS login page in your browser...")
    webbrowser.open(auth_url)

    print("After login, FYERS will redirect back with an auth code in the URL.")
    print("Copy the code parameter value and paste it into the app or run:")
    print("python main.py --auth-code 'PASTE_YOUR_CODE_HERE' --once")
    print()

    auth_code_file = Path(__file__).with_name(".fyers_auth_code.txt")
    print(f"The script also saves a placeholder file at: {auth_code_file}")
    auth_code_file.write_text("PASTE_YOUR_AUTH_CODE_HERE\n", encoding="utf-8")


if __name__ == "__main__":
    main()
