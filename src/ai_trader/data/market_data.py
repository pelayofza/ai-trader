from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from ai_trader.data.cache import load_bars, save_bars
from ai_trader.data.providers.alpaca import AlpacaProvider
from ai_trader.data.providers.ccxt_crypto import CCXTCrypto
from ai_trader.data.providers.polymarket_clob import PolymarketClobProvider
from ai_trader.data.providers.polymarket_gamma import PolymarketGammaProvider
from ai_trader.shared.bars import normalize_bars
from ai_trader.shared.instruments import AssetClass, PredictionMarket, detect_asset_class

logger = logging.getLogger(__name__)


class MarketDataService:
    """Fachada unica de datos de mercado. Enruta por clase de activo y cachea barras."""

    def __init__(
        self,
        stock_provider: AlpacaProvider | None = None,
        crypto_provider: CCXTCrypto | None = None,
        polymarket_gamma: PolymarketGammaProvider | None = None,
        polymarket_clob: PolymarketClobProvider | None = None,
    ) -> None:
        self.stock_provider = stock_provider or AlpacaProvider()
        self.crypto_provider = crypto_provider or CCXTCrypto()
        self.polymarket_gamma = polymarket_gamma or PolymarketGammaProvider()
        self.polymarket_clob = polymarket_clob or PolymarketClobProvider()

    # --- barras OHLCV ---

    def get_daily_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame | None:
        normalized_symbol = symbol.strip().upper()
        asset_class = self.detect_asset_class(normalized_symbol)

        if asset_class == AssetClass.PREDICTION:
            # Los mercados de prediccion no tienen OHLCV. Antes esta rama caia
            # silenciosamente al proveedor de stocks, que recibia simbolos "PM::...".
            logger.debug("No OHLCV for prediction symbol=%s; use get_prediction_midpoint", symbol)
            return None

        cache_symbol = self._build_cache_symbol(normalized_symbol, asset_class)
        slice_start, slice_end = self._build_daily_slice_bounds(start, end)
        fetch_start = slice_start.to_pydatetime()
        fetch_end = (slice_end + pd.Timedelta(days=1)).to_pydatetime()

        cached = load_bars(cache_symbol, timeframe="1D")
        if cached is not None and cached.empty:
            cached = None

        if cached is None:
            fetched = self._fetch_daily_bars(normalized_symbol, fetch_start, fetch_end, asset_class)
            if fetched is None or fetched.empty:
                return None

            prepared = normalize_bars(fetched)
            save_bars(cache_symbol, prepared, timeframe="1D")
            return self._slice_daily_bars(prepared, slice_start, slice_end)

        cached_min = cached.index.min()
        cached_max = cached.index.max()

        if cached_min <= slice_start and cached_max >= slice_end:
            return self._slice_daily_bars(cached, slice_start, slice_end)

        frames = [cached]

        if slice_start < cached_min:
            before = self._fetch_daily_bars(
                normalized_symbol,
                fetch_start,
                (cached_min + pd.Timedelta(days=1)).to_pydatetime(),
                asset_class,
            )
            if before is not None and not before.empty:
                frames.append(before)

        if slice_end > cached_max:
            after = self._fetch_daily_bars(
                normalized_symbol,
                cached_max.to_pydatetime(),
                fetch_end,
                asset_class,
            )
            if after is not None and not after.empty:
                frames.append(after)

        merged = self._merge_daily_frames(frames)
        save_bars(cache_symbol, merged, timeframe="1D")
        return self._slice_daily_bars(merged, slice_start, slice_end)

    # --- mercados de prediccion ---

    def search_prediction_markets(
        self,
        query: str,
        limit: int = 20,
        active_only: bool = True,
    ) -> list[PredictionMarket]:
        return self.polymarket_gamma.search_markets(
            query=query,
            limit=limit,
            active_only=active_only,
        )

    def get_prediction_market(self, slug: str) -> PredictionMarket | None:
        return self.polymarket_gamma.get_market_by_slug(slug)

    def get_prediction_orderbook(self, token_id: str) -> dict:
        return self.polymarket_clob.get_orderbook(token_id)

    def get_prediction_midpoint(self, token_id: str) -> float | None:
        return self.polymarket_clob.get_midpoint(token_id)

    # --- enrutado ---

    def detect_asset_class(self, symbol: str) -> AssetClass:
        # Regla canonica por nombre (shared.instruments) + lo que solo sabe el proveedor:
        # un simbolo sin barra puede seguir siendo cripto si CCXT sabe servirlo.
        normalized = symbol.strip().upper()
        base = detect_asset_class(normalized)
        if base == AssetClass.STOCK and self.crypto_provider.can_handle_symbol(normalized):
            return AssetClass.CRYPTO
        return base

    def _fetch_daily_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        asset_class: AssetClass,
    ) -> pd.DataFrame | None:
        if asset_class == AssetClass.CRYPTO:
            return self.crypto_provider.get_daily_bars(symbol, start, end)
        if asset_class == AssetClass.STOCK:
            return self.stock_provider.get_daily_bars(symbol, start, end)

        raise ValueError(f"No OHLCV provider for asset class {asset_class.value}")

    @staticmethod
    def _build_cache_symbol(symbol: str, asset_class: AssetClass) -> str:
        if asset_class == AssetClass.CRYPTO:
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

        # Para barras 1D pedimos hasta la ultima vela diaria ya cerrada.
        end_ts = cls._to_utc_timestamp(end).normalize() - pd.Timedelta(days=1)

        if end_ts < start_ts:
            end_ts = start_ts

        return start_ts, end_ts

    @staticmethod
    def _merge_daily_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
        prepared = [normalize_bars(frame) for frame in frames if frame is not None and not frame.empty]
        merged = pd.concat(prepared).sort_index()
        return merged[~merged.index.duplicated(keep="last")]

    @staticmethod
    def _slice_daily_bars(
        df: pd.DataFrame,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> pd.DataFrame:
        return df.loc[start_ts:end_ts]
