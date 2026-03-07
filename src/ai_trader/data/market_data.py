from __future__ import annotations

from datetime import datetime

import pandas as pd

from ai_trader.data.cache import load_bars, save_bars
from ai_trader.data.providers.alpaca import AlpacaProvider
from ai_trader.data.providers.ccxt_crypto import CCXTCrypto


class MarketDataService:
    def __init__(self) -> None:
        self.stock_provider = AlpacaProvider()
        self.crypto_provider = CCXTCrypto()

    def get_daily_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame | None:
        normalized_symbol = symbol.strip().upper()
        asset_class = self._detect_asset_class(normalized_symbol)
        cache_symbol = self._build_cache_symbol(normalized_symbol, asset_class)

        # 1) Load cache
        cached = load_bars(cache_symbol, timeframe="1D")
        if cached is not None and not cached.empty:
            cmin = cached.index.min()
            cmax = cached.index.max()
            if cmin <= start and cmax >= end:
                return cached.loc[start:end]

        # 2) Fetch from provider
        df = self._fetch_daily_bars(
            symbol=normalized_symbol,
            start=start,
            end=end,
            asset_class=asset_class,
        )

        if df is None or df.empty:
            if cached is not None and not cached.empty:
                return cached.loc[start:end]
            return None

        # 3) Merge with cache (avoid duplicates) and save
        if cached is not None and not cached.empty:
            merged = pd.concat([cached, df]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
        else:
            merged = df

        save_bars(cache_symbol, merged, timeframe="1D")

        return merged.loc[start:end]

    def get_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame | None:
        normalized_symbol = symbol.strip().upper()
        asset_class = self._detect_asset_class(normalized_symbol)

        if asset_class == "crypto":
            return self.crypto_provider.get_ohlcv(
                symbol=normalized_symbol,
                start=start,
                end=end,
                timeframe=timeframe,
            )

        if timeframe != "1d":
            raise ValueError(
                f"Stock provider does not support timeframe '{timeframe}' in get_ohlcv yet."
            )

        return self.get_daily_bars(
            symbol=normalized_symbol,
            start=start,
            end=end,
        )

    def _fetch_daily_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        asset_class: str,
    ) -> pd.DataFrame | None:
        if asset_class == "crypto":
            return self.crypto_provider.get_daily_bars(symbol, start, end)

        return self.stock_provider.get_daily_bars(symbol, start, end)

    def _detect_asset_class(self, symbol: str) -> str:
        if "/" in symbol:
            return "crypto"

        if self.crypto_provider.can_handle_symbol(symbol):
            return "crypto"

        return "stock"

    @staticmethod
    def _build_cache_symbol(symbol: str, asset_class: str) -> str:
        if asset_class == "crypto":
            return f"crypto::{symbol}"
        return symbol