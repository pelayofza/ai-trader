from __future__ import annotations

import logging
from datetime import datetime
from typing import Protocol

import pandas as pd

from ai_trader.shared.bars import Bar, bar_from_row, normalize_bars
from ai_trader.shared.instruments import (
    PREDICTION_PREFIX,
    AssetClass,
    PredictionMarket,
    detect_asset_class,
)

logger = logging.getLogger(__name__)


class BarProvider(Protocol):
    """
    Fuente de barras OHLCV para el backtest.

    Es el unico contrato que el motor necesita de los datos, y lo cumplen por igual:
    - MarketDataService (datos reales de Binance/Alpaca, con cache).
    - El futuro generador de datos sinteticos (series simuladas para RL).

    Devuelve un DataFrame normalizado o None si no hay datos para el simbolo.
    """

    def get_daily_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame | None:
        ...


class HistoricalDataSource:
    """
    Fuente de datos para backtest. Implementa el mismo contrato que MarketDataService
    (MarketDataReader), pero sirviendo desde OHLCV precargado en memoria.

    Dos garantias:
    - Anti look-ahead: get_daily_bars solo devuelve barras ya cerradas, es decir
      estrictamente anteriores al dia del reloj. La estrategia decide con el pasado.
    - bar_on() da la barra del dia actual (open/high/low/close) para que el modelo de
      mercado la use en fills y salidas: informacion que la estrategia NO vio.
    """

    def __init__(self, bars: dict[str, pd.DataFrame], clock) -> None:
        self.clock = clock
        self._bars: dict[str, pd.DataFrame] = {}
        for symbol, df in bars.items():
            if df is None or df.empty:
                continue
            self._bars[symbol.strip().upper()] = normalize_bars(df)

    @staticmethod
    def fetch_bars(
        service,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> dict[str, pd.DataFrame]:
        """Precarga el historico completo de cada simbolo una sola vez (usa la cache
        parquet del servicio). Los simbolos de prediccion se ignoran: no hay OHLCV.

        Se separa de la construccion para poder cargar una vez y crear varias fuentes
        (una por ventana train/test), cada una con su propio reloj."""
        loaded: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            if symbol.strip().upper().startswith(PREDICTION_PREFIX):
                logger.info("Skipping prediction symbol in backtest (no OHLCV): %s", symbol)
                continue
            df = service.get_daily_bars(symbol, start, end)
            if df is None or df.empty:
                logger.warning("No historical bars for %s in backtest range", symbol)
                continue
            loaded[symbol.strip().upper()] = df
        return loaded

    @classmethod
    def from_service(
        cls,
        service,
        symbols: list[str],
        start: datetime,
        end: datetime,
        clock,
    ) -> HistoricalDataSource:
        return cls(cls.fetch_bars(service, symbols, start, end), clock)

    # --- contrato MarketDataReader ---

    def get_daily_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame | None:
        df = self._bars.get(symbol.strip().upper())
        if df is None:
            return None

        start_ts = _to_utc(start)
        # Cierre del look-ahead: solo barras estrictamente anteriores a hoy. La barra
        # de hoy aun no ha cerrado desde el punto de vista de la decision.
        today = _to_utc(self.clock.now()).normalize()

        window = df[(df.index >= start_ts) & (df.index < today)]
        return window if not window.empty else None

    def get_prediction_market(self, slug: str) -> PredictionMarket | None:
        return None

    def get_prediction_midpoint(self, token_id: str) -> float | None:
        return None

    def detect_asset_class(self, symbol: str) -> AssetClass:
        return detect_asset_class(symbol)

    # --- acceso para el modelo de mercado ---

    def bar_on(self, symbol: str, day: datetime) -> Bar | None:
        df = self._bars.get(symbol.strip().upper())
        if df is None:
            return None

        target = _to_utc(day).normalize()
        same_day = df[df.index.normalize() == target]
        if same_day.empty:
            return None

        return bar_from_row(same_day.index[0], same_day.iloc[0])

    # --- utilidades para el motor ---

    def trading_days(self, start: datetime, end: datetime) -> list[pd.Timestamp]:
        """Union ordenada de fechas con barra en el rango [start, end], entre todos
        los simbolos. Es el calendario sobre el que avanza el backtest."""
        start_ts = _to_utc(start).normalize()
        end_ts = _to_utc(end).normalize()

        days: set[pd.Timestamp] = set()
        for df in self._bars.values():
            normalized = df.index.normalize()
            mask = (normalized >= start_ts) & (normalized <= end_ts)
            days.update(normalized[mask])

        return sorted(days)

    @property
    def symbols(self) -> list[str]:
        return list(self._bars)

    @property
    def raw_bars(self) -> dict[str, pd.DataFrame]:
        """Series completas normalizadas por simbolo (sin recorte de reloj). Lo consume
        el ensamblador cross-sectional, que aplica su propio anti look-ahead via reloj."""
        return self._bars


def _to_utc(value: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")
