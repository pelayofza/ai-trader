"""
COMPUESTO DE SENALES: la unica primitiva que ve los cinco temas a la vez.

POR QUE EXISTE, Y POR QUE NO ES UNA SEXTA VARIANTE DE LO MISMO
---------------------------------------------------------------
Por la ley fundamental del gestor activo, K apuestas poco correlacionadas de capacidad
predictiva pequena valen raiz de K veces una sola. Con los canales declarados en la libreria
sintetica, el IC agregado de los cinco temas es ~0,074 mientras que el mejor tema por
separado vale ~0,048 y el peor ~0,010: la breadth vale aproximadamente x1,55, y es toda la
diferencia entre estar cerca del frente y no estarlo.

Esa raiz solo se cobra si UNA decision ve los cinco. Cinco estrategias que leen un tema cada
una la reparten: cada una gatea sobre un IC por debajo incluso de la celda rho=0,05 que el
barrido publicado ya midio perdiendo. Esta primitiva existe para que esa comparacion —cinco
apuestas separadas contra una agregada— este MEDIDA y no argumentada.

EL NUCLEO DE PRECIO ES DELIBERADAMENTE POBRE, Y HAY QUE DECIRLO
----------------------------------------------------------------
Aqui la senal ES la tesis, asi que el precio solo aporta dos cosas: que el activo sea
operable (piso de ATR) y CUANDO actuar (un giro reciente de la media corta, sin perseguir un
precio ya estirado). Con la capa apagada —y con los defaults lo esta— esto es un seguidor de
tendencia corriente y moliente: **ciego, el compuesto no aporta nada sobre `crypto_momentum`**.
Su puesto en un ranking sin senal no dice nada de su tesis, y es la fila de la tabla que hay
que leer con mas cuidado.

LA REGLA DE LOS DOS TEMAS, QUE SALE GRATIS
-------------------------------------------
`composite_reading` promedia solo los temas LEGIBLES y publica cobertura
`temas_legibles / 5`. Con `MIN_SIGNAL_COVERAGE = 0,25` eso exige DOS temas legibles
(1/5 = 0,20 < 0,25; 2/5 = 0,40), asi que el minimo aparece de la aritmetica que ya existia y
no de una constante nueva. En historico real son legibles `macro`, `attention` y `flow`: tres
de cinco, cobertura 0,60. Es la unica de las seis familias nuevas cuya capa se puede evaluar
contra senal capturada de verdad.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import uuid4

import pandas as pd

from ai_trader.observation.signal_radar import (
    INERT_MAX_INTENSITY,
    INERT_MIN_INTENSITY,
    INERT_MIN_TONE,
)
from ai_trader.shared import bars as bar_schema
from ai_trader.shared.clock import utc_now
from ai_trader.shared.indicators import atr, sma
from ai_trader.shared.schemas import Side, Signal
from ai_trader.strategies.signal_layer import (
    INERT_SIGNAL_WEIGHT,
    SIDE_CORE,
    atr_bracket,
    composite_gate_reason,
    composite_reading,
    resolve_confidence,
    resolve_side,
    signal_features,
    validate_signal_fields,
)

logger = logging.getLogger(__name__)

THEME = "composite"


@dataclass(slots=True)
class SignalCompositeConfig:
    timeframe: str = "1d"
    # Media corta cuyo giro marca el CUANDO.
    trigger_window: int = 20
    # Media larga que dice en que lado del ciclo estamos.
    trend_window: int = 100
    # Cuantas barras atras se admite que ocurriera el giro.
    cross_lookback: int = 3
    # No perseguir: si el precio ya se alejo mas de esto de la media corta, se deja pasar.
    max_stretch_atr: float = 2.5
    atr_window: int = 14
    min_atr_pct: float = 0.2
    stop_atr_mult: float = 2.0
    reward_atr_mult: float = 3.0
    min_bars: int = 130
    allow_short: bool = True
    # --- capa de senal (los CINCO temas), inerte por construccion -----------------------
    min_signal_tone: float = INERT_MIN_TONE
    min_signal_intensity: float = INERT_MIN_INTENSITY
    max_signal_intensity: float = INERT_MAX_INTENSITY
    signal_side_mode: str = SIDE_CORE
    signal_tone_threshold: float = 0.0
    signal_weight: float = INERT_SIGNAL_WEIGHT

    def __post_init__(self) -> None:
        if self.trigger_window <= 1:
            raise ValueError("trigger_window must be > 1")
        if self.trend_window <= self.trigger_window:
            raise ValueError("trend_window must be > trigger_window")
        if self.cross_lookback < 1:
            raise ValueError("cross_lookback must be >= 1")
        if self.max_stretch_atr <= 0:
            raise ValueError("max_stretch_atr must be > 0")
        if self.atr_window <= 0:
            raise ValueError("atr_window must be > 0")
        if self.min_atr_pct <= 0:
            raise ValueError("min_atr_pct must be > 0")
        if self.stop_atr_mult <= 0:
            raise ValueError("stop_atr_mult must be > 0")
        if self.reward_atr_mult <= 0:
            raise ValueError("reward_atr_mult must be > 0")
        if self.min_bars < self.trend_window + self.cross_lookback + self.atr_window:
            raise ValueError("min_bars must cover trend_window + cross_lookback + atr_window")
        validate_signal_fields(
            min_tone=self.min_signal_tone,
            min_intensity=self.min_signal_intensity,
            max_intensity=self.max_signal_intensity,
            side_mode=self.signal_side_mode,
            weight=self.signal_weight,
        )


class SignalCompositeStrategy:
    strategy_id = "signal_composite_v1"
    theme = THEME

    def __init__(self, config: SignalCompositeConfig | None = None) -> None:
        self.config = config or SignalCompositeConfig()
        self._signals = None

    def attach_signal_provider(self, provider) -> None:
        self._signals = provider

    def _signals_active(self) -> bool:
        cfg = self.config
        return (
            cfg.min_signal_tone > INERT_MIN_TONE
            or cfg.min_signal_intensity > INERT_MIN_INTENSITY
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
        trigger = sma(close, cfg.trigger_window)
        trend = sma(close, cfg.trend_window)
        atr_series = atr(bars, cfg.atr_window)

        if len(close) <= cfg.cross_lookback:
            return None
        latest_close = float(close.iloc[-1])
        latest_trigger = trigger.iloc[-1]
        latest_trend = trend.iloc[-1]
        latest_atr = atr_series.iloc[-1]
        past_close = close.iloc[-1 - cfg.cross_lookback]
        past_trigger = trigger.iloc[-1 - cfg.cross_lookback]
        if pd.isna(latest_trigger) or pd.isna(latest_trend) or pd.isna(latest_atr):
            return None
        if pd.isna(past_trigger) or pd.isna(past_close):
            return None
        if latest_close <= 0 or float(latest_atr) <= 0:
            return None

        latest_trigger = float(latest_trigger)
        latest_trend = float(latest_trend)
        latest_atr = float(latest_atr)
        atr_pct = latest_atr / latest_close * 100.0
        stretch = abs(latest_close - latest_trigger) / latest_atr

        above_now = latest_close > latest_trigger
        above_then = float(past_close) > float(past_trigger)
        turned_up = above_now and not above_then
        turned_down = (not above_now) and above_then

        tradable = atr_pct >= cfg.min_atr_pct and stretch <= cfg.max_stretch_atr
        core_side: Side | None = None
        if tradable and turned_up and latest_close > latest_trend:
            core_side = Side.BUY
        elif tradable and cfg.allow_short and turned_down and latest_close < latest_trend:
            core_side = Side.SELL

        logger.info(
            "Composite check | symbol=%s | close=%.6f | trigger=%.6f | trend=%.6f | "
            "stretch=%.2f | atr_pct=%.2f | core_side=%s",
            symbol, latest_close, latest_trigger, latest_trend, stretch, atr_pct, core_side,
        )
        if core_side is None:
            return None

        reading = None
        if self._signals is not None and self._signals_active():
            reading = composite_reading(self._signals.features(symbol))
            core_side = resolve_side(
                core_side,
                reading,
                side_mode=cfg.signal_side_mode,
                threshold=cfg.signal_tone_threshold,
            )
            if core_side is None:
                logger.info("Signal layer cancelled symbol=%s (compuesto)", symbol)
                return None
            if core_side is Side.SELL and not cfg.allow_short:
                return None
            reason = composite_gate_reason(
                reading,
                core_side,
                min_tone=cfg.min_signal_tone,
                min_intensity=cfg.min_signal_intensity,
                max_intensity=cfg.max_signal_intensity,
            )
            if reason is not None:
                logger.info("Signal gate blocked symbol=%s (%s)", symbol, reason)
                return None

        bracket = atr_bracket(
            latest_close,
            latest_atr,
            core_side,
            stop_mult=cfg.stop_atr_mult,
            target_mult=cfg.reward_atr_mult,
        )
        if bracket is None:
            logger.info("Degenerate SL/TP for symbol=%s; skipping", symbol)
            return None
        stop_loss, take_profit = bracket

        base = _confidence(stretch=stretch, atr_pct=atr_pct, config=cfg)
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
                f"Giro de la media de {cfg.trigger_window}d con {stretch:.2f} ATR de "
                f"estiramiento, ATR%={atr_pct:.2f}"
            ),
            features={
                "close": latest_close,
                "trigger": latest_trigger,
                "trend": latest_trend,
                "stretch_atr": stretch,
                "atr": latest_atr,
                "atr_pct": atr_pct,
                **signal_features(reading),
            },
            signal_id=f"{self.strategy_id}:{symbol}:{uuid4().hex[:12]}",
        )


def _confidence(*, stretch: float, atr_pct: float, config: SignalCompositeConfig) -> float:
    """Cuanto mas cerca de la media entra, mas margen queda antes del objetivo."""
    proximity = min(max(config.max_stretch_atr - stretch, 0.0) / config.max_stretch_atr, 1.0)
    volatility_strength = min(max(atr_pct / 4.0, 0.0), 1.0)
    raw = 0.7 * proximity + 0.3 * volatility_strength
    return round(min(max(0.55 + raw * 0.35, 0.55), 0.90), 2)
