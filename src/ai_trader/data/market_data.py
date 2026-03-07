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

        slice_start, slice_end = self._build_daily_slice_bounds(start, end)
        fetch_start = slice_start.to_pydatetime()
        fetch_end = (slice_end + pd.Timedelta(days=1)).to_pydatetime()

        cached = load_bars(cache_symbol, timeframe="1D")
        cached = self._prepare_daily_bars(cached) if cached is not None and not cached.empty else None

        if cached is not None:
            cmin = cached.index.min()
            cmax = cached.index.max()

            if cmin <= slice_start and cmax >= slice_end:
                return self._slice_daily_bars(cached, slice_start, slice_end)

            frames: list[pd.DataFrame] = [cached]

            if slice_start < cmin:
                before_df = self._fetch_daily_bars(
                    symbol=normalized_symbol,
                    start=fetch_start,
                    end=(cmin + pd.Timedelta(days=1)).to_pydatetime(),
                    asset_class=asset_class,
                )
                if before_df is not None and not before_df.empty:
                    frames.append(before_df)

            if slice_end > cmax:
                after_df = self._fetch_daily_bars(
                    symbol=normalized_symbol,
                    start=cmax.to_pydatetime(),
                    end=fetch_end,
                    asset_class=asset_class,
                )
                if after_df is not None and not after_df.empty:
                    frames.append(after_df)

            merged = self._merge_daily_frames(frames)
            save_bars(cache_symbol, merged, timeframe="1D")
            return self._slice_daily_bars(merged, slice_start, slice_end)

        df = self._fetch_daily_bars(
            symbol=normalized_symbol,
            start=fetch_start,
            end=fetch_end,
            asset_class=asset_class,
        )
        if df is None or df.empty:
            return None

        prepared = self._prepare_daily_bars(df)
        save_bars(cache_symbol, prepared, timeframe="1D")
        return self._slice_daily_bars(prepared, slice_start, slice_end)

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

    @staticmethod
    def _to_utc_timestamp(value: datetime) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    @classmethod
    def _build_daily_slice_bounds(
        cls,
        start: datetime,
        end: datetime,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        start_ts = cls._to_utc_timestamp(start).normalize()

        # Para barras 1D pedimos hasta la última vela diaria ya cerrada.
        end_ts = cls._to_utc_timestamp(end).normalize() - pd.Timedelta(days=1)

        if end_ts < start_ts:
            end_ts = start_ts

        return start_ts, end_ts

    @staticmethod
    def _prepare_daily_bars(df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        prepared.index = pd.to_datetime(prepared.index, utc=True, errors="coerce")
        prepared = prepared[~prepared.index.isna()]
        prepared = prepared.sort_index()
        prepared = prepared[~prepared.index.duplicated(keep="last")]
        return prepared

    @classmethod
    def _merge_daily_frames(cls, frames: list[pd.DataFrame]) -> pd.DataFrame:
        prepared_frames = [
            cls._prepare_daily_bars(frame)
            for frame in frames
            if frame is not None and not frame.empty
        ]
        merged = pd.concat(prepared_frames).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        return merged

    @staticmethod
    def _slice_daily_bars(
        df: pd.DataFrame,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> pd.DataFrame:
        return df.loc[start_ts:end_ts]