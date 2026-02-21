from datetime import datetime
import pandas as pd

from ai_trader.data.cache import load_bars, save_bars
from ai_trader.data.providers.alpaca import AlpacaProvider


class MarketDataService:
    def __init__(self):
        self.provider = AlpacaProvider()

    def get_daily_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame | None:
        # 1) Load cache
        cached = load_bars(symbol, timeframe="1D")
        if cached is not None and not cached.empty:
            # If cache covers requested range, return slice
            cmin = cached.index.min()
            cmax = cached.index.max()
            if cmin <= start and cmax >= end:
                return cached.loc[start:end]

        # 2) Fetch from provider
        df = self.provider.get_daily_bars(symbol, start, end)
        if df is None or df.empty:
            # If we had partial cache, return what we can
            if cached is not None and not cached.empty:
                return cached.loc[start:end]
            return None

        # 3) Merge with cache (avoid duplicates) and save
        if cached is not None and not cached.empty:
            merged = pd.concat([cached, df]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
        else:
            merged = df

        save_bars(symbol, merged, timeframe="1D")

        return merged.loc[start:end]