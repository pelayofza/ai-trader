import os
from dotenv import load_dotenv

from ai_trader.telegram_bot import build_application

load_dotenv()


def main():
    print("Argos is running")
    
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in .env")

    app = build_application(token)
    app.run_polling()


if __name__ == "__main__":
    main()