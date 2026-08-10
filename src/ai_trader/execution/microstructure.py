from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Protocol

import numpy as np
import pandas as pd

from ai_trader.shared import bars as bar_schema
from ai_trader.shared.clock import Clock
from ai_trader.shared.instruments import AssetClass

# --- spreads base por simbolo -------------------------------------------------
#
# Suelo del coste de ejecucion: lo que cuesta cruzar el libro con un tamano
# irrelevante, antes de volatilidad y de impacto. Es la parte del coste que depende
# SOLO del simbolo, y por eso vive en una tabla explicita y auditable en vez de salir
# de una formula: el spread de BTC y el de un altcoin de segunda fila no se diferencian
# en un factor calculable, se diferencian en un hecho de mercado.
#
# Los valores son ordenes de magnitud tipicos del spread medio cotizado en horario
# normal (bps sobre el midpoint), no cotizaciones de un dia concreto.

SPREAD_BPS_BY_SYMBOL: dict[str, float] = {
    # --- cripto de primer nivel ---
    "BTC/USDT": 1.0,
    "ETH/USDT": 1.5,
    # --- cripto de gran capitalizacion ---
    "BNB/USDT": 3.0,
    "SOL/USDT": 3.0,
    "XRP/USDT": 3.0,
    "ADA/USDT": 4.0,
    "DOGE/USDT": 4.0,
    "LTC/USDT": 4.0,
    "AVAX/USDT": 5.0,
    "LINK/USDT": 5.0,
    "DOT/USDT": 6.0,
    "ETC/USDT": 6.0,
    "MATIC/USDT": 6.0,
    "UNI/USDT": 6.0,
    "ATOM/USDT": 7.0,
    # --- cripto de capitalizacion media: aqui es donde 5 bps planos eran ficcion ---
    "AAVE/USDT": 8.0,
    "FIL/USDT": 8.0,
    "NEAR/USDT": 8.0,
    "ARB/USDT": 9.0,
    "OP/USDT": 9.0,
    "APT/USDT": 10.0,
    "SUI/USDT": 10.0,
    "INJ/USDT": 14.0,
    "TIA/USDT": 16.0,
    "SEI/USDT": 20.0,
    # --- renta variable: indices y megacaps ---
    "SPY": 0.4,
    "QQQ": 0.5,
    "AAPL": 0.8,
    "MSFT": 0.8,
    "NVDA": 1.0,
    "GOOGL": 1.0,
    "AMZN": 1.0,
    "META": 1.2,
    "TSLA": 1.2,
    "JPM": 1.2,
    "XOM": 1.2,
    "KO": 1.2,
    "PG": 1.2,
    "JNJ": 1.2,
    "AMD": 1.5,
    "BAC": 1.5,
    "CVX": 1.5,
    "GS": 2.0,
    "UNH": 2.0,
    "CAT": 2.0,
    # --- macro / refugio ---
    "GLD": 0.8,
    "TLT": 1.0,
    "UUP": 2.0,
}

# Spread de un simbolo que NO esta en la tabla, por clase de activo. Deliberadamente
# caro: un simbolo desconocido se trata como poco liquido, no como si fuera BTC. Que el
# desconocimiento salga caro es la direccion prudente del error.
SPREAD_BPS_BY_ASSET_CLASS: dict[AssetClass, float] = {
    AssetClass.CRYPTO: 25.0,
    AssetClass.STOCK: 8.0,
    AssetClass.PREDICTION: 60.0,
}
UNKNOWN_SPREAD_BPS = 25.0

BPS = 10_000.0

# Ventana (barras) sobre la que se miden volumen tipico y volatilidad reciente.
DEFAULT_WINDOW = 20
# Minimo de retornos para que una desviacion tipica signifique algo.
MIN_BARS_FOR_VOLATILITY = 5


@dataclass(frozen=True, slots=True)
class LiquiditySnapshot:
    """
    Estado de liquidez de un simbolo en el instante de ejecutar.

    Los dos campos pueden faltar (simbolo sin barras: un mercado de prediccion, un
    simbolo recien anadido). Cuando faltan, el modelo degrada de forma explicita: sin
    volatilidad usa la de referencia, y sin volumen no hay impacto ni techo de capacidad.
    """

    # Volumen negociado en una barra tipica, en UNIDADES del activo (no en USD).
    bar_volume: float | None = None
    # Desviacion tipica diaria de los log-retornos recientes (0.03 = 3% diario).
    recent_volatility: float | None = None


EMPTY_SNAPSHOT = LiquiditySnapshot()


@dataclass(slots=True)
class SlippageModel:
    """
    Deslizamiento como funcion del simbolo, de su volatilidad y del tamano relativo:

        slippage_bps = medio_spread(simbolo)
                     + vol_coef    * sigma_bps
                     + impact_coef * sigma_bps * participacion ** impact_exponent

        participacion = tamano_orden / volumen_de_la_barra

    Los tres terminos responden a tres preguntas distintas:

    - **medio spread**: que cuesta cruzar el libro por existir, aunque la orden sea
      minuscula. Depende solo del simbolo (tabla de arriba). Se cobra la MITAD del
      spread porque el precio de referencia es el midpoint, no el otro lado.
    - **volatilidad**: los creadores de mercado ensanchan la horquilla cuando el activo
      se mueve. Es el termino que hace que el mismo simbolo cueste mas en panico.
    - **impacto**: mover el precio con el propio tamano. Sigue la ley de raiz cuadrada
      (Almgren-Chriss y la evidencia empirica): el coste crece con la RAIZ de la
      fraccion del volumen consumida, no linealmente, y escala con la volatilidad.

    El resultado se limita a `max_slippage_bps`: mas alla de ahi la extrapolacion del
    modelo no significa nada, y es preferible un techo declarado a un numero inventado.

    Es determinista y sin estado: mismos argumentos, mismo resultado.
    """

    vol_coef: float = 0.02
    impact_coef: float = 0.60
    impact_exponent: float = 0.5
    # Volatilidad diaria que se asume cuando no hay barras para medirla.
    reference_volatility: float = 0.02
    max_slippage_bps: float = 500.0
    # Sobrescrituras de spread por simbolo (config). Tienen prioridad sobre la tabla.
    spread_bps: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vol_coef < 0:
            raise ValueError("vol_coef cannot be negative")
        if self.impact_coef < 0:
            raise ValueError("impact_coef cannot be negative")
        if not 0 < self.impact_exponent <= 1:
            raise ValueError("impact_exponent must be between 0 and 1")
        if self.reference_volatility <= 0:
            raise ValueError("reference_volatility must be greater than 0")
        if self.max_slippage_bps <= 0:
            raise ValueError("max_slippage_bps must be greater than 0")
        self.spread_bps = {
            str(symbol).strip().upper(): float(value)
            for symbol, value in self.spread_bps.items()
        }
        if any(value < 0 for value in self.spread_bps.values()):
            raise ValueError("spread_bps cannot be negative")

    def base_spread_bps(self, symbol: str, asset_class: AssetClass | None = None) -> float:
        """Spread cotizado del simbolo. Config > tabla > suelo por clase de activo."""
        key = (symbol or "").strip().upper()
        override = self.spread_bps.get(key)
        if override is not None:
            return override

        listed = SPREAD_BPS_BY_SYMBOL.get(key)
        if listed is not None:
            return listed

        return SPREAD_BPS_BY_ASSET_CLASS.get(asset_class, UNKNOWN_SPREAD_BPS)

    def slippage_bps(
        self,
        symbol: str,
        size: float,
        snapshot: LiquiditySnapshot | None = None,
        asset_class: AssetClass | None = None,
    ) -> float:
        """Deslizamiento en puntos basicos para una orden de `size` unidades."""
        snap = snapshot or EMPTY_SNAPSHOT

        sigma = snap.recent_volatility
        if sigma is None or not math.isfinite(sigma) or sigma <= 0:
            sigma = self.reference_volatility
        sigma_bps = sigma * BPS

        participation = self._participation(size, snap)

        half_spread = 0.5 * self.base_spread_bps(symbol, asset_class)
        vol_term = self.vol_coef * sigma_bps
        impact = self.impact_coef * sigma_bps * participation**self.impact_exponent

        return round(min(half_spread + vol_term + impact, self.max_slippage_bps), 6)

    @staticmethod
    def _participation(size: float, snapshot: LiquiditySnapshot) -> float:
        """Fraccion del volumen de la barra que consume la orden. 0 si no hay volumen:
        sin dato de liquidez no se inventa impacto, se cobra solo spread y volatilidad."""
        volume = snapshot.bar_volume
        if volume is None or not math.isfinite(volume) or volume <= 0:
            return 0.0
        return max(size, 0.0) / volume


class LiquidityProvider(Protocol):
    """Costura de liquidez: resuelve el estado del simbolo en el dia del reloj."""

    def snapshot(self, symbol: str) -> LiquiditySnapshot:
        ...


class _ReaderLike(Protocol):
    def get_daily_bars(self, symbol: str, start, end) -> pd.DataFrame | None:
        ...


class BarLiquidityProvider:
    """
    Liquidez derivada de las barras ya cerradas del propio simbolo.

    - **Volumen de barra**: MEDIANA del volumen de las ultimas `window` barras. Mediana
      y no ultima barra a proposito: el volumen se dispara los dias de movimiento fuerte,
      y tomar ese pico como capacidad regalaria liquidez justo en los dias en que de
      verdad escasea.
    - **Volatilidad reciente**: desviacion tipica de los log-retornos de esa ventana.

    Anti look-ahead: lee por `get_daily_bars`, que ya excluye la barra de hoy (la que
    aun no ha cerrado). La liquidez con la que se cobra el fill de hoy se estima, por
    tanto, con informacion que la estrategia tambien tenia.

    Cachea por (simbolo, dia del reloj): dentro de un mismo dia simulado la respuesta no
    puede cambiar, y el backtest pregunta una vez por orden.
    """

    def __init__(
        self,
        reader: _ReaderLike,
        clock: Clock,
        *,
        window: int = DEFAULT_WINDOW,
    ) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        self.reader = reader
        self.clock = clock
        self.window = window
        self._cache: dict[tuple[str, object], LiquiditySnapshot] = {}

    def snapshot(self, symbol: str) -> LiquiditySnapshot:
        now = self.clock.now()
        key = ((symbol or "").strip().upper(), now.date())
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        snapshot = self._compute(symbol, now)
        self._cache[key] = snapshot
        return snapshot

    def _compute(self, symbol: str, now) -> LiquiditySnapshot:
        # Holgura amplia sobre la ventana: en renta variable hay fines de semana y
        # festivos, asi que 20 barras no son 20 dias naturales.
        start = now - timedelta(days=self.window * 4)
        df = self.reader.get_daily_bars(symbol, start, now)
        if df is None or df.empty:
            return EMPTY_SNAPSHOT

        return LiquiditySnapshot(
            bar_volume=_typical_volume(df, self.window),
            recent_volatility=_recent_volatility(df, self.window),
        )


def _typical_volume(df: pd.DataFrame, window: int) -> float | None:
    if bar_schema.VOLUME not in df.columns:
        return None

    volume = bar_schema.series(df, bar_schema.VOLUME).dropna().iloc[-window:]
    if volume.empty:
        return None

    median = float(volume.median())
    return median if math.isfinite(median) and median > 0 else None


def _recent_volatility(df: pd.DataFrame, window: int) -> float | None:
    if bar_schema.CLOSE not in df.columns:
        return None

    close = bar_schema.series(df, bar_schema.CLOSE).dropna()
    close = close[close > 0].iloc[-(window + 1) :]
    if len(close) < MIN_BARS_FOR_VOLATILITY:
        return None

    returns = np.diff(np.log(close.to_numpy(dtype=float)))
    if len(returns) < 2:
        return None

    sigma = float(np.std(returns, ddof=1))
    return sigma if math.isfinite(sigma) and sigma > 0 else None
