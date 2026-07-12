from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class MarketDataProvider(ABC):
    """
    Proveedor de barras OHLCV.

    Contrato: devuelve un DataFrame ya normalizado (ver `shared.bars.normalize_bars`),
    con columnas en minuscula [open, high, low, close, volume] e indice datetime UTC,
    o None si no hay datos. Los consumidores no deben sondear variantes del nombre
    de columna.
    """

    @abstractmethod
    def get_daily_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame | None:
        ...
