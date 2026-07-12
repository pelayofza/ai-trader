from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from ai_trader.data.providers.base import MarketDataProvider
from ai_trader.shared.bars import OHLCV_COLUMNS, normalize_bars


class AlpacaProvider(MarketDataProvider):
    """
    Barras diarias de renta variable via Alpaca.

    El cliente se construye de forma perezosa: sin simbolos de stock en el universo
    la app debe arrancar aunque no haya credenciales de Alpaca configuradas.
    """

    def __init__(self) -> None:
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = self._build_client()
        return self._client

    @staticmethod
    def _build_client():
        from alpaca.data.historical import StockHistoricalDataClient

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise RuntimeError(
                "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY. "
                "They are only required when the universe contains stock symbols."
            )

        return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

    def get_daily_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame | None:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        normalized_symbol = symbol.strip().upper()

        request = StockBarsRequest(
            symbol_or_symbols=normalized_symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )

        df = self.client.get_stock_bars(request).df

        if df is None or df.empty:
            return None

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(normalized_symbol)

        df = normalize_bars(df)

        return df[list(OHLCV_COLUMNS)]
