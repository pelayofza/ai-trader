import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

load_dotenv()

def main():
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")

    print("ALPACA_API_KEY loaded:", bool(key), "len:", (len(key) if key else None))
    print("ALPACA_SECRET_KEY loaded:", bool(secret), "len:", (len(secret) if secret else None))

    if not key or not secret:
        print("❌ No se cargan las credenciales. Revisa tu .env y que estás ejecutando desde la carpeta correcta.")
        return

    client = StockHistoricalDataClient(api_key=key, secret_key=secret)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)

    req = StockBarsRequest(
        symbol_or_symbols="AAPL",
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX,  # importante: IEX suele ser lo que funciona en planes free
    )

    try:
        bars = client.get_stock_bars(req)
        df = bars.df  # devuelve un pandas DF (multi-index si pides varios símbolos)
        print("Bars df shape:", df.shape)
        print(df.tail(5))
    except Exception as e:
        print("❌ Alpaca error:", repr(e))

if __name__ == "__main__":
    main()