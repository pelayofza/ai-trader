"""
CASCADA DE LIQUIDACIONES: comprar la capitulacion, salvo que quede combustible debajo.

LA TESIS
--------
Cripto no se mueve, se descuelga. El apalancamiento esta del lado del vendedor forzado: el
precio entra en un cumulo de precios de liquidacion, el flujo obligado acelera el movimiento
y, cuando el combustible se agota, el precio retrocede. Esa ultima parte —el agotamiento— es
lo unico que el precio por si solo puede ver, y lo ve como una BARRA DE CAPITULACION: rango
verdadero muy por encima del ATR, cierre pegado al extremo del dia y precio muy estirado
respecto de su media.

Lo que el precio NO puede decir es cuanto combustible queda, y ahi entra el tema
`liquidation`: el mapa de Hyperliquid dice a que distancia de precio revienta cuanto
notional, el apalancamiento (p90, cuota por encima de 10x, concentracion del OI en cinco
cuentas) dice cuanta fragilidad hay detras, el funding dice si los largos estan hacinados y
DVOL es el termometro del mismo mercado que se esta rompiendo. La diferencia entre comprar
una capitulacion que ya reventó y comprar la PRIMERA de tres.

Por eso el modo natural de esta primitiva es el VETO y no la direccion: el precio propone la
capitulacion, y la senal la cancela cuando el mapa dice que hay doscientos millones un cuatro
por ciento mas abajo.

LO QUE HOY NO SE PUEDE MEDIR HACIA ATRAS
-----------------------------------------
De las cuatro fuentes del tema solo `deribit_volatility` tiene historia medida, asi que la
cobertura del tema en cualquier backtest historico es 1/6 = 0,167, por debajo del minimo. En
el ranking sobre 2018-2025 esta primitiva se mide CIEGA: lo que se rankea es su nucleo de
precio. No es un fallo del diseno, es la profundidad que hay; ver el criterio de repeticion
en la documentacion del radar tematico.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import uuid4

import pandas as pd

from ai_trader.observation.signal_radar import INERT_MAX_INTENSITY, INERT_MIN_TONE
from ai_trader.observation.signal_themes import theme_reading
from ai_trader.shared import bars as bar_schema
from ai_trader.shared.clock import utc_now
from ai_trader.shared.indicators import atr, close_location, sma, true_range
from ai_trader.shared.schemas import Side, Signal
from ai_trader.strategies.signal_layer import (
    INERT_SIGNAL_WEIGHT,
    SIDE_CORE,
    atr_bracket,
    resolve_confidence,
    resolve_side,
    side_gate_reason,
    signal_features,
    validate_signal_fields,
)

logger = logging.getLogger(__name__)

THEME = "liquidation"


@dataclass(slots=True)
class LiquidationCascadeConfig:
    timeframe: str = "1d"
    # Ventana de la media contra la que se mide el estiramiento.
    mean_window: int = 20
    atr_window: int = 14
    # |cierre - media| en ATRs para considerar el precio estirado.
    entry_stretch_atr: float = 2.0
    # Rango verdadero de hoy en ATRs: lo que convierte un dia malo en una capitulacion.
    entry_range_atr: float = 1.5
    # El cierre tiene que quedar en el 25% inferior del rango del dia (o superior, para el
    # corto). Por encima de 0,5 el "extremo" deja de serlo y la primitiva se invierte en
    # silencio, asi que el constructor lo rechaza.
    close_location_max: float = 0.25
    min_atr_pct: float = 0.5
    stop_atr_mult: float = 1.5
    target_atr_mult: float = 2.0
    min_bars: int = 40
    # Lo fija `finalize`, no es una dimension del espacio de busqueda.
    allow_short: bool = True
    # --- capa de senal (tema 'liquidation'), inerte por construccion --------------------
    # Los cuatro defaults estan en el borde exacto de su rango; ver strategies/signal_layer.
    # NINGUNO entra en `scoring/search_space.py`: la cobertura de un tema crece con el
    # calendario de captura, asi que un umbral sorteable dejaria al optimizador elegir en que
    # tramo de historia se le permite operar a la estrategia.
    min_signal_tone: float = INERT_MIN_TONE
    max_signal_intensity: float = INERT_MAX_INTENSITY
    signal_side_mode: str = SIDE_CORE
    signal_tone_threshold: float = 0.0
    signal_weight: float = INERT_SIGNAL_WEIGHT

    def __post_init__(self) -> None:
        if self.mean_window <= 1:
            raise ValueError("mean_window must be > 1")
        if self.atr_window <= 0:
            raise ValueError("atr_window must be > 0")
        if self.entry_stretch_atr <= 0:
            raise ValueError("entry_stretch_atr must be > 0")
        if self.entry_range_atr <= 0:
            raise ValueError("entry_range_atr must be > 0")
        if not 0.0 < self.close_location_max < 0.5:
            raise ValueError("close_location_max must be in (0, 0.5)")
        if self.min_atr_pct <= 0:
            raise ValueError("min_atr_pct must be > 0")
        if self.stop_atr_mult <= 0:
            raise ValueError("stop_atr_mult must be > 0")
        if self.target_atr_mult <= 0:
            raise ValueError("target_atr_mult must be > 0")
        if self.min_bars < self.mean_window + self.atr_window:
            raise ValueError("min_bars must cover mean_window + atr_window")
        validate_signal_fields(
            min_tone=self.min_signal_tone,
            max_intensity=self.max_signal_intensity,
            side_mode=self.signal_side_mode,
            weight=self.signal_weight,
        )


class LiquidationCascadeStrategy:
    strategy_id = "liquidation_cascade_v1"
    theme = THEME

    def __init__(self, config: LiquidationCascadeConfig | None = None) -> None:
        self.config = config or LiquidationCascadeConfig()
        self._signals = None

    def attach_signal_provider(self, provider) -> None:
        """Mismo patron duck-typed que el regimen: quien no lo llame se queda sin capa."""
        self._signals = provider

    def _signals_active(self) -> bool:
        cfg = self.config
        return (
            cfg.min_signal_tone > INERT_MIN_TONE
            or cfg.max_signal_intensity < INERT_MAX_INTENSITY
            or cfg.signal_side_mode != SIDE_CORE
            or cfg.signal_weight > INERT_SIGNAL_WEIGHT
        )

    def supports_symbol(self, symbol: str) -> bool:
        return not symbol.strip().upper().startswith("PM::")

    def generate_signal(self, symbol: str, bars: pd.DataFrame) -> Signal | None:
        cfg = self.config
        if bars is None or bars.empty or len(bars) < cfg.min_bars:
            return None

        close = bar_schema.series(bars, bar_schema.CLOSE)
        mean = sma(close, cfg.mean_window)
        atr_series = atr(bars, cfg.atr_window)
        range_series = true_range(bars)
        location = close_location(bars)

        latest_close = float(close.iloc[-1])
        latest_mean = mean.iloc[-1]
        latest_atr = atr_series.iloc[-1]
        latest_range = range_series.iloc[-1]
        latest_loc = location.iloc[-1]
        if pd.isna(latest_mean) or pd.isna(latest_atr) or pd.isna(latest_range):
            return None
        if pd.isna(latest_loc) or latest_close <= 0 or float(latest_atr) <= 0:
            return None

        latest_mean = float(latest_mean)
        latest_atr = float(latest_atr)
        atr_pct = latest_atr / latest_close * 100.0
        stretch = (latest_close - latest_mean) / latest_atr
        range_ratio = float(latest_range) / latest_atr
        latest_loc = float(latest_loc)

        capitulation = range_ratio >= cfg.entry_range_atr and atr_pct >= cfg.min_atr_pct
        core_side: Side | None = None
        if capitulation and stretch <= -cfg.entry_stretch_atr and latest_loc <= cfg.close_location_max:
            core_side = Side.BUY
        elif (
            capitulation
            and cfg.allow_short
            and stretch >= cfg.entry_stretch_atr
            and latest_loc >= 1.0 - cfg.close_location_max
        ):
            core_side = Side.SELL

        logger.info(
            "Liquidation check | symbol=%s | close=%.6f | stretch=%.2f | range_ratio=%.2f | "
            "loc=%.2f | atr_pct=%.2f | core_side=%s",
            symbol, latest_close, stretch, range_ratio, latest_loc, atr_pct, core_side,
        )
        if core_side is None:
            return None

        reading = None
        if self._signals is not None and self._signals_active():
            features = self._signals.features(symbol)
            reading = theme_reading(features, THEME)
            core_side = resolve_side(
                core_side,
                reading,
                side_mode=cfg.signal_side_mode,
                threshold=cfg.signal_tone_threshold,
            )
            if core_side is None:
                logger.info("Signal layer cancelled symbol=%s (tema %s)", symbol, THEME)
                return None
            reason = side_gate_reason(
                features,
                THEME,
                core_side,
                min_tone=cfg.min_signal_tone,
                max_intensity=cfg.max_signal_intensity,
            )
            if reason is not None:
                logger.info("Signal gate blocked symbol=%s (%s)", symbol, reason)
                return None

        # El stop cuelga del EXTREMO del dia, no del cierre: en una capitulacion la mecha es
        # el hecho, y un stop al otro lado del cierre lo tocaria la propia barra siguiente.
        extreme = float(
            bar_schema.series(bars, bar_schema.LOW if core_side is Side.BUY else bar_schema.HIGH)
            .iloc[-1]
        )
        bracket = atr_bracket(
            latest_close,
            latest_atr,
            core_side,
            stop_mult=cfg.stop_atr_mult,
            target_mult=cfg.target_atr_mult,
            stop_anchor=extreme,
        )
        if bracket is None:
            logger.info("Degenerate SL/TP for symbol=%s; skipping", symbol)
            return None
        stop_loss, take_profit = bracket

        base = _confidence(stretch=stretch, range_ratio=range_ratio, config=cfg)
        confidence = resolve_confidence(base, reading, core_side, weight=cfg.signal_weight)

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            timeframe=cfg.timeframe,
            timestamp=utc_now(),
            side=core_side,
            confidence=confidence,
            entry_price=latest_close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=(
                f"Capitulacion: estiramiento={stretch:+.2f} ATR, rango={range_ratio:.2f} ATR, "
                f"cierre en {latest_loc:.2f} del rango"
            ),
            features={
                "close": latest_close,
                "mean": latest_mean,
                "atr": latest_atr,
                "atr_pct": atr_pct,
                "stretch_atr": stretch,
                "range_ratio": range_ratio,
                "close_location": latest_loc,
                **signal_features(reading),
            },
            signal_id=f"{self.strategy_id}:{symbol}:{uuid4().hex[:12]}",
        )


def _confidence(*, stretch: float, range_ratio: float, config: LiquidationCascadeConfig) -> float:
    """Misma forma que momentum y reversion: 0,55 + score*0,35, saturando pronto."""
    excess = max(abs(stretch) - config.entry_stretch_atr, 0.0)
    stretch_strength = min(excess / config.entry_stretch_atr, 1.0)
    range_strength = min(
        max(range_ratio - config.entry_range_atr, 0.0) / config.entry_range_atr, 1.0
    )
    raw = 0.6 * stretch_strength + 0.4 * range_strength
    return round(min(max(0.55 + raw * 0.35, 0.55), 0.90), 2)
