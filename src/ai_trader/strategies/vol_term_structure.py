"""
ESTRUCTURA TEMPORAL DE VOLATILIDAD: romper la compresion, en la direccion que se esta pagando.

LA TESIS
--------
La superficie de opciones es lo unico del catalogo que COTIZA el futuro en vez de resumir el
pasado. De ahi salen tres hechos que no se pueden leer en una vela:

1. La volatilidad realizada se comprime antes de expandirse. La compresion es, con los
   costes de por medio, la unica regularidad de volatilidad que sobrevive en diario: no dice
   hacia donde, dice que se acerca un movimiento que paga el spread.
2. La DIRECCION de la expansion es lo que el skew esta pagando. `skew_25d` es literalmente
   cuanto mas cara esta la proteccion que la apuesta, medida en la misma unidad, y es el
   unico numero de opciones con direccion defendible sin una hipotesis nueva. Por eso es el
   unico del tema con polaridad declarada en el radar.
3. Un vencimiento que concentra una cuarta parte del interes abierto es un dia con gamma, y
   el precio tiende a quedarse clavado hasta que pasa. Eso no es alcista ni bajista: es
   INTENSIDAD, y por eso el techo de intensidad es el mando util aqui.

El nucleo de precio es el gemelo pobre de (1): compresion de la volatilidad realizada mas
rotura de un canal de Donchian. La capa aporta (2) y (3).

LO QUE HOY NO SE PUEDE MEDIR HACIA ATRAS
-----------------------------------------
`deribit_expiries` no tiene historia (su primer dia medido esta en el FUTURO: es un
calendario de vencimientos, no un archivo), asi que el tema se queda en 1/6 de cobertura y
esta primitiva se mide CIEGA en el ranking historico. Igual que `liquidation_cascade`.
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
from ai_trader.shared.indicators import atr, donchian_high, donchian_low, realized_vol
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

THEME = "vol_surface"


@dataclass(slots=True)
class VolTermStructureConfig:
    timeframe: str = "1d"
    rv_fast_window: int = 10
    rv_slow_window: int = 60
    # rv_corta / rv_larga por debajo de esto = mercado comprimido, primitiva armada.
    max_compression: float = 0.85
    breakout_lookback: int = 20
    atr_window: int = 14
    min_atr_pct: float = 0.1
    stop_atr_mult: float = 2.0
    reward_atr_mult: float = 3.0
    min_bars: int = 90
    allow_short: bool = True
    # --- capa de senal (tema 'vol_surface'), inerte por construccion --------------------
    min_signal_tone: float = INERT_MIN_TONE
    max_signal_intensity: float = INERT_MAX_INTENSITY
    signal_side_mode: str = SIDE_CORE
    signal_tone_threshold: float = 0.0
    signal_weight: float = INERT_SIGNAL_WEIGHT

    def __post_init__(self) -> None:
        if self.rv_fast_window <= 1:
            raise ValueError("rv_fast_window must be > 1")
        # Se RECHAZA en vez de repararse: una ventana rapida mas larga que la lenta no es una
        # configuracion exigente, es un error, y el cociente mediria lo contrario en silencio.
        if self.rv_slow_window <= self.rv_fast_window:
            raise ValueError("rv_slow_window must be > rv_fast_window")
        # +1 porque la compresion se lee en la barra anterior (ver generate_signal).
        if self.min_bars < self.rv_slow_window + 1:
            raise ValueError("min_bars must leave a bar before the breakout")
        if self.max_compression <= 0:
            raise ValueError("max_compression must be > 0")
        if self.breakout_lookback <= 1:
            raise ValueError("breakout_lookback must be > 1")
        if self.atr_window <= 0:
            raise ValueError("atr_window must be > 0")
        if self.min_atr_pct <= 0:
            raise ValueError("min_atr_pct must be > 0")
        if self.stop_atr_mult <= 0:
            raise ValueError("stop_atr_mult must be > 0")
        if self.reward_atr_mult <= 0:
            raise ValueError("reward_atr_mult must be > 0")
        if self.min_bars < self.rv_slow_window + self.breakout_lookback:
            raise ValueError("min_bars must cover rv_slow_window + breakout_lookback")
        validate_signal_fields(
            min_tone=self.min_signal_tone,
            max_intensity=self.max_signal_intensity,
            side_mode=self.signal_side_mode,
            weight=self.signal_weight,
        )


class VolTermStructureStrategy:
    strategy_id = "vol_term_structure_v1"
    theme = THEME

    def __init__(self, config: VolTermStructureConfig | None = None) -> None:
        self.config = config or VolTermStructureConfig()
        self._signals = None

    def attach_signal_provider(self, provider) -> None:
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
        rv_fast = realized_vol(close, cfg.rv_fast_window)
        rv_slow = realized_vol(close, cfg.rv_slow_window)
        atr_series = atr(bars, cfg.atr_window)
        upper = donchian_high(bars, cfg.breakout_lookback)
        lower = donchian_low(bars, cfg.breakout_lookback)

        latest_close = float(close.iloc[-1])
        # La compresion se mide con la barra ANTERIOR, y no es un detalle de implementacion:
        # la propia rotura es un dia de rango grande, asi que incluirla en `rv_fast` destruye
        # la compresion que arma la primitiva. Cuanto mas limpia la compresion previa, mas
        # violenta la rotura y mas alto el cociente medido con ella dentro: la condicion se
        # negaria a si misma. Lo que la tesis dice es "el mercado ESTABA comprimido y hoy
        # rompe", y eso es exactamente la penultima barra.
        latest_fast = rv_fast.iloc[-2]
        latest_slow = rv_slow.iloc[-2]
        latest_atr = atr_series.iloc[-1]
        latest_upper = upper.iloc[-1]
        latest_lower = lower.iloc[-1]
        if pd.isna(latest_fast) or pd.isna(latest_slow) or pd.isna(latest_atr):
            return None
        if pd.isna(latest_upper) or pd.isna(latest_lower):
            return None
        if latest_close <= 0 or float(latest_slow) <= 0 or float(latest_atr) <= 0:
            return None

        latest_atr = float(latest_atr)
        atr_pct = latest_atr / latest_close * 100.0
        compression = float(latest_fast) / float(latest_slow)

        armed = compression <= cfg.max_compression and atr_pct >= cfg.min_atr_pct
        core_side: Side | None = None
        if armed and latest_close > float(latest_upper):
            core_side = Side.BUY
        elif armed and cfg.allow_short and latest_close < float(latest_lower):
            core_side = Side.SELL

        logger.info(
            "VolTerm check | symbol=%s | close=%.6f | rv_fast=%.2f | rv_slow=%.2f | "
            "compression=%.3f | atr_pct=%.2f | core_side=%s",
            symbol, latest_close, float(latest_fast), float(latest_slow),
            compression, atr_pct, core_side,
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

        base = _confidence(compression=compression, atr_pct=atr_pct, config=cfg)
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
                f"Rotura tras compresion: rv_fast/rv_slow={compression:.2f} "
                f"<= {cfg.max_compression:.2f}, ATR%={atr_pct:.2f}"
            ),
            features={
                "close": latest_close,
                "rv_fast": float(latest_fast),
                "rv_slow": float(latest_slow),
                "compression": compression,
                "atr": latest_atr,
                "atr_pct": atr_pct,
                "donchian_high": float(latest_upper),
                "donchian_low": float(latest_lower),
                **signal_features(reading),
            },
            signal_id=f"{self.strategy_id}:{symbol}:{uuid4().hex[:12]}",
        )


def _confidence(*, compression: float, atr_pct: float, config: VolTermStructureConfig) -> float:
    """Cuanto mas comprimido estaba el mercado, mas conviccion en la expansion."""
    room = max(config.max_compression - compression, 0.0)
    compression_strength = min(room / config.max_compression, 1.0)
    volatility_strength = min(max(atr_pct / 4.0, 0.0), 1.0)
    raw = 0.7 * compression_strength + 0.3 * volatility_strength
    return round(min(max(0.55 + raw * 0.35, 0.55), 0.90), 2)
