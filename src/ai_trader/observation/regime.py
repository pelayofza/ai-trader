from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

from ai_trader.shared import bars as bar_schema
from ai_trader.shared.clock import Clock


def _visible_cutoff(now: datetime) -> pd.Timestamp:
    """Frontera anti look-ahead IDENTICA a HistoricalDataSource.get_daily_bars: solo
    barras estrictamente anteriores a hoy. La barra de hoy aun no ha cerrado desde el
    punto de vista de la decision, asi que el regimen tampoco puede verla."""
    ts = pd.Timestamp(now)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.normalize()

# Orden canonico del bloque cross-sectional / regimen del vector de observacion.
# breadth y agg_vol son de mercado (iguales para todos los simbolos ese dia);
# relative_strength y corr_to_market son especificos del simbolo.
REGIME_FEATURES: tuple[str, ...] = (
    "relative_strength",
    "corr_to_market",
    "breadth",
    "agg_vol",
)

TRADING_DAYS_PER_YEAR = 365  # coherente con features.py y backtest/metrics.py

DEFAULT_BREADTH_SMA = 50
DEFAULT_RS_WINDOW = 20
DEFAULT_CORR_WINDOW = 60


def _sanitize(value: float) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    return float(value)


class MarketRegimeProvider:
    """
    Ensamblador de la observacion CROSS-SECTIONAL: mira el universo entero para
    situar a cada activo en su regimen de mercado.

    Es la costura que el contrato de estrategia (un solo simbolo por decision) no
    provee: se inyecta como colaborador OPCIONAL en las primitivas. Con el, calculan
    filtros de regimen; sin el, degradan a features de un solo activo. En vivo se
    inyectaria un provider real; aqui, en backtest, se construye sobre el bars_dict.

    Anti look-ahead: cada consulta recorta las series al 'ahora' del reloj, exactamente
    como HistoricalDataSource recorta el contexto que ve la estrategia. Nada de futuro.
    """

    def __init__(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        clock: Clock,
        *,
        breadth_sma: int = DEFAULT_BREADTH_SMA,
        rs_window: int = DEFAULT_RS_WINDOW,
        corr_window: int = DEFAULT_CORR_WINDOW,
    ) -> None:
        self._bars = bars_by_symbol
        self._clock = clock
        self._breadth_sma = breadth_sma
        self._rs_window = rs_window
        self._corr_window = corr_window
        # Memo de agregados de mercado, cacheados por timestamp 'ahora': breadth,
        # agg_vol y la serie de retornos de mercado se calculan una vez por dia y se
        # reutilizan para los 35 simbolos.
        self._cache_asof: datetime | None = None
        self._market_returns: pd.Series | None = None
        self._breadth: float = 0.0
        self._agg_vol: float = 0.0

    def features(self, symbol: str) -> dict[str, float]:
        """Bloque de regimen para `symbol` en el 'ahora' del reloj."""
        self._refresh()
        return {
            "relative_strength": self._relative_strength(symbol),
            "corr_to_market": self._corr_to_market(symbol),
            "breadth": self._breadth,
            "agg_vol": self._agg_vol,
        }

    # --- helpers de acceso anti look-ahead ---------------------------------

    def _close_asof(self, symbol: str, as_of: datetime) -> pd.Series:
        df = self._bars.get(symbol)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        close = bar_schema.series(df, bar_schema.CLOSE)
        return close[close.index < _visible_cutoff(as_of)].dropna()

    # --- agregados de mercado (cacheados por dia) --------------------------

    def _refresh(self) -> None:
        as_of = self._clock.now()
        if self._cache_asof == as_of:
            return

        returns_by_symbol: dict[str, pd.Series] = {}
        above_sma = 0
        counted = 0
        vols: list[float] = []

        for symbol in self._bars:
            close = self._close_asof(symbol, as_of)
            if len(close) < 2:
                continue

            log_ret = np.log(close / close.shift(1)).dropna()
            if not log_ret.empty:
                returns_by_symbol[symbol] = log_ret

            # Amplitud: fraccion del universo por encima de su SMA.
            if len(close) >= self._breadth_sma:
                sma = close.rolling(self._breadth_sma, min_periods=self._breadth_sma).mean().iloc[-1]
                if not pd.isna(sma):
                    counted += 1
                    if close.iloc[-1] > sma:
                        above_sma += 1

            # Volatilidad agregada: mediana de la vol realizada por activo.
            if len(log_ret) >= self._rs_window:
                std = log_ret.iloc[-self._rs_window :].std(ddof=1)
                if not pd.isna(std):
                    vols.append(std * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)

        self._market_returns = (
            pd.DataFrame(returns_by_symbol).mean(axis=1) if returns_by_symbol else None
        )
        self._breadth = _sanitize(above_sma / counted) if counted else 0.0
        self._agg_vol = _sanitize(float(np.median(vols))) if vols else 0.0
        self._cache_asof = as_of

    def _relative_strength(self, symbol: str) -> float:
        as_of = self._clock.now()
        close = self._close_asof(symbol, as_of)
        if len(close) <= self._rs_window or self._market_returns is None:
            return 0.0

        # Retorno del activo menos el retorno de mercado (media equiponderada del
        # universo), ambos acumulados sobre la ventana. Positivo = mas fuerte que el mercado.
        asset_ret = math.log(close.iloc[-1] / close.iloc[-1 - self._rs_window])
        market_ret = float(self._market_returns.iloc[-self._rs_window :].sum())
        return _sanitize(asset_ret - market_ret)

    def _corr_to_market(self, symbol: str) -> float:
        as_of = self._clock.now()
        close = self._close_asof(symbol, as_of)
        if len(close) < self._corr_window + 1 or self._market_returns is None:
            return 0.0

        asset_ret = np.log(close / close.shift(1)).dropna()
        joined = pd.concat(
            [asset_ret.rename("a"), self._market_returns.rename("m")], axis=1
        ).dropna()
        if len(joined) < self._corr_window:
            return 0.0

        tail = joined.iloc[-self._corr_window :]
        if tail["a"].std(ddof=1) == 0 or tail["m"].std(ddof=1) == 0:
            return 0.0
        return _sanitize(float(tail["a"].corr(tail["m"])))
