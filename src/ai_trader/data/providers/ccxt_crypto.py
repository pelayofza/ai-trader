from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import ccxt  # type: ignore
import pandas as pd

from ai_trader.shared.instruments import split_pair


def to_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def datetime_to_milliseconds(value: datetime) -> int:
    utc_value = to_utc_datetime(value)
    return int(utc_value.timestamp() * 1000)


@dataclass(slots=True)
class CCXTCryptoConfig:
    exchange_id: str = "binance"
    rate_limit: bool = True
    timeout_ms: int = 10_000
    max_batch_size: int = 1_000

    def __post_init__(self) -> None:
        if not self.exchange_id:
            raise ValueError("exchange_id cannot be empty")
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be greater than 0")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than 0")


class CCXTCrypto:
    """
    Simple OHLCV reader for crypto symbols using ccxt.

    Returns pandas DataFrames with these columns:
    - open
    - high
    - low
    - close
    - volume

    Index:
    - UTC DatetimeIndex named 'timestamp'
    """

    TIMEFRAME_MAP: dict[str, str] = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }

    def __init__(self, config: CCXTCryptoConfig | None = None) -> None:
        self.config = config or CCXTCryptoConfig()
        self._exchange = None

    @property
    def exchange(self):
        # Perezoso: load_markets() es una llamada de red y no debe dispararse
        # solo por construir el servicio de datos.
        if self._exchange is None:
            self._exchange = self._build_exchange()
        return self._exchange

    def get_daily_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        return self.get_ohlcv(
            symbol=symbol,
            start=start,
            end=end,
            timeframe="1d",
        )

    def get_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        normalized_symbol = self.normalize_symbol(symbol)
        ccxt_timeframe = self._resolve_timeframe(timeframe)

        start_ms = datetime_to_milliseconds(start)
        end_ms = datetime_to_milliseconds(end)

        all_rows: list[list[Any]] = []
        since_ms = start_ms

        while since_ms < end_ms:
            batch = self.exchange.fetch_ohlcv(
                normalized_symbol,
                timeframe=ccxt_timeframe,
                since=since_ms,
                limit=self.config.max_batch_size,
            )

            if not batch:
                break

            all_rows.extend(batch)

            last_batch_ts = int(batch[-1][0])
            next_since_ms = self._next_since_ms(last_batch_ts, ccxt_timeframe)

            if next_since_ms <= since_ms:
                break

            since_ms = next_since_ms

            if last_batch_ts >= end_ms:
                break

        df = self._build_dataframe(all_rows)

        if df.empty:
            return df

        df = df[(df.index >= to_utc_datetime(start)) & (df.index <= to_utc_datetime(end))]
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        return df

    def normalize_symbol(self, symbol: str) -> str:
        # La regla de corte (y la tupla de monedas de cotizacion) vive en
        # shared/instruments.py: la comparten este proveedor y la resolucion de entidad
        # de las senales, y duplicarla era la via directa a que divergieran.
        pair = split_pair(symbol)
        if pair is None:
            raise ValueError(
                f"Could not normalize crypto symbol '{symbol}'. "
                "Use format like BTC/USDT or ETH/USD."
            )
        return f"{pair[0]}/{pair[1]}"

    def can_handle_symbol(self, symbol: str) -> bool:
        try:
            normalized = self.normalize_symbol(symbol)
        except ValueError:
            return False

        return normalized in self.exchange.markets

    def _build_exchange(self):
        exchange_class = getattr(ccxt, self.config.exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Unsupported ccxt exchange_id: {self.config.exchange_id}")

        exchange = exchange_class(
            {
                "enableRateLimit": self.config.rate_limit,
                "timeout": self.config.timeout_ms,
            }
        )
        exchange.load_markets()
        return exchange

    def _resolve_timeframe(self, timeframe: str) -> str:
        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return self.TIMEFRAME_MAP[timeframe]

    def _build_dataframe(self, rows: list[list[Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(
            rows,
            columns=["timestamp_ms", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df = df.drop(columns=["timestamp_ms"])
        df = df.set_index("timestamp")
        df.index.name = "timestamp"

        numeric_columns = ["open", "high", "low", "close", "volume"]
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        return df

    def _next_since_ms(self, last_timestamp_ms: int, timeframe: str) -> int:
        timeframe_ms_map = {
            "1m": 60_000,
            "5m": 300_000,
            "15m": 900_000,
            "1h": 3_600_000,
            "4h": 14_400_000,
            "1d": 86_400_000,
        }
        return last_timestamp_ms + timeframe_ms_map[timeframe]