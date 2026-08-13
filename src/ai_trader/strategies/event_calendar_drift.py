"""
DERIVA DE CALENDARIO: seguir el movimiento, dosificando por lo que hay en la agenda.

LA TESIS
--------
Hay dias que se saben con meses de antelacion —un FOMC, un dato de IPC— y el mercado se
coloca ANTES: reduce riesgo entrando y lo vuelve a poner despues. Y hay un flujo regulatorio
y judicial que llega a rachas —un 8-K esta en EDGAR horas antes de que nadie lo escriba, una
regla propuesta aparece en el Federal Register— que reprecia la clase de activo entera a la
vez. Entre esos hitos, el precio deriva; alrededor de ellos, no.

EL TEMA NO DICE HACIA DONDE, Y ESO NO ES UN DEFECTO
----------------------------------------------------
De las seis fuentes de `macro` solo `ofac_sdn` tiene polaridad declarada en el radar, asi
que el TONO del tema es ~0 por construccion y ademas las cinco restantes lo DILUYEN (entran
con tono 0 en la media, no se saltan). Y esta bien que sea asi, porque el catalogo lo razono
una a una: un recuento de dockets no distingue una aprobacion de una demanda, el recuento de
13F sube el 14 de febrero por el calendario de presentacion y no por el mercado, y un FOMC no
es bueno ni malo antes de ocurrir.

La consecuencia es de diseno y esta escrita en la ausencia: **esta config no tiene
`min_signal_tone` ni `signal_side_mode`**. La direccion la pone la deriva del precio; el tema
solo decide SI se opera y CUANTO. Es la misma clase de asimetria deliberada que hace que
`mean_reversion` no tenga `min_signal_intensity`.

Un piso de intensidad ("solo opero con catalizador cerca") y un techo ("me aparto cuando el
calendario esta ardiendo") son decisiones, no ausencias de datos: solo se evaluan cuando el
tema tiene cobertura, asi que el invariante de fallo abierto se respeta.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import uuid4

import pandas as pd

from ai_trader.observation.signal_radar import INERT_MAX_INTENSITY, INERT_MIN_INTENSITY
from ai_trader.observation.signal_themes import theme_reading, themed_gate_reason
from ai_trader.shared import bars as bar_schema
from ai_trader.shared.clock import utc_now
from ai_trader.shared.indicators import atr, roc
from ai_trader.shared.schemas import Side, Signal
from ai_trader.strategies.signal_layer import (
    INERT_SIGNAL_WEIGHT,
    atr_bracket,
    resolve_confidence,
    signal_features,
    validate_signal_fields,
)

logger = logging.getLogger(__name__)

THEME = "macro"


@dataclass(slots=True)
class EventCalendarDriftConfig:
    timeframe: str = "1d"
    drift_window: int = 5
    # Ventana corta que tiene que ir en el MISMO sentido: distingue una deriva de un rebote.
    confirm_window: int = 2
    min_drift_pct: float = 2.0
    # Por encima de esto ya no es deriva: es el evento, y llegamos tarde.
    max_drift_pct: float = 15.0
    atr_window: int = 14
    min_atr_pct: float = 0.2
    stop_atr_mult: float = 2.0
    reward_atr_mult: float = 2.0
    min_bars: int = 40
    allow_short: bool = True
    # --- capa de senal: SOLO intensidad. Ver el docstring del modulo --------------------
    min_signal_intensity: float = INERT_MIN_INTENSITY
    max_signal_intensity: float = INERT_MAX_INTENSITY
    signal_weight: float = INERT_SIGNAL_WEIGHT

    def __post_init__(self) -> None:
        if self.drift_window <= 1:
            raise ValueError("drift_window must be > 1")
        if not 0 < self.confirm_window < self.drift_window:
            raise ValueError("confirm_window must be in (0, drift_window)")
        if self.min_drift_pct <= 0:
            raise ValueError("min_drift_pct must be > 0")
        if self.max_drift_pct <= self.min_drift_pct:
            raise ValueError("max_drift_pct must be > min_drift_pct")
        if self.atr_window <= 0:
            raise ValueError("atr_window must be > 0")
        if self.min_atr_pct <= 0:
            raise ValueError("min_atr_pct must be > 0")
        if self.stop_atr_mult <= 0:
            raise ValueError("stop_atr_mult must be > 0")
        if self.reward_atr_mult <= 0:
            raise ValueError("reward_atr_mult must be > 0")
        if self.min_bars < self.drift_window + self.atr_window:
            raise ValueError("min_bars must cover drift_window + atr_window")
        validate_signal_fields(
            min_intensity=self.min_signal_intensity,
            max_intensity=self.max_signal_intensity,
            weight=self.signal_weight,
        )


class EventCalendarDriftStrategy:
    strategy_id = "event_calendar_drift_v1"
    theme = THEME

    def __init__(self, config: EventCalendarDriftConfig | None = None) -> None:
        self.config = config or EventCalendarDriftConfig()
        self._signals = None

    def attach_signal_provider(self, provider) -> None:
        self._signals = provider

    def _signals_active(self) -> bool:
        cfg = self.config
        return (
            cfg.min_signal_intensity > INERT_MIN_INTENSITY
            or cfg.max_signal_intensity < INERT_MAX_INTENSITY
            or cfg.signal_weight > INERT_SIGNAL_WEIGHT
        )

    def supports_symbol(self, symbol: str) -> bool:
        return not symbol.strip().upper().startswith("PM::")

    def generate_signal(self, symbol: str, bars: pd.DataFrame) -> Signal | None:
        cfg = self.config
        if bars is None or bars.empty or len(bars) < cfg.min_bars:
            return None

        close = bar_schema.series(bars, bar_schema.CLOSE)
        drift = roc(close, cfg.drift_window)
        confirm = roc(close, cfg.confirm_window)
        atr_series = atr(bars, cfg.atr_window)

        latest_close = float(close.iloc[-1])
        latest_drift = drift.iloc[-1]
        latest_confirm = confirm.iloc[-1]
        latest_atr = atr_series.iloc[-1]
        if pd.isna(latest_drift) or pd.isna(latest_confirm) or pd.isna(latest_atr):
            return None
        if latest_close <= 0 or float(latest_atr) <= 0:
            return None

        latest_atr = float(latest_atr)
        atr_pct = latest_atr / latest_close * 100.0
        drift_pct = float(latest_drift) * 100.0
        confirm_pct = float(latest_confirm) * 100.0

        magnitude_ok = cfg.min_drift_pct <= abs(drift_pct) <= cfg.max_drift_pct
        aligned = (drift_pct > 0 and confirm_pct > 0) or (drift_pct < 0 and confirm_pct < 0)
        core_side: Side | None = None
        if magnitude_ok and aligned and atr_pct >= cfg.min_atr_pct:
            core_side = Side.BUY if drift_pct > 0 else Side.SELL
        if core_side is Side.SELL and not cfg.allow_short:
            core_side = None

        logger.info(
            "Calendar drift check | symbol=%s | close=%.6f | drift=%.2f%% | confirm=%.2f%% | "
            "atr_pct=%.2f | core_side=%s",
            symbol, latest_close, drift_pct, confirm_pct, atr_pct, core_side,
        )
        if core_side is None:
            return None

        reading = None
        if self._signals is not None and self._signals_active():
            features = self._signals.features(symbol)
            reading = theme_reading(features, THEME)
            # `themed_gate_reason` y no `side_gate_reason`: sin tono que voltear, el lado no
            # cambia nada de lo que se comprueba, y pedirlo sugeriria una direccionalidad
            # que este tema no tiene.
            reason = themed_gate_reason(
                features,
                THEME,
                min_tone=-INERT_MAX_INTENSITY,  # el piso inerte: el tono aqui no decide
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

        base = _confidence(drift_pct=drift_pct, config=cfg)
        # La intensidad del calendario mueve el TAMANO, no el lado: se le pasa el lado que
        # ya decidio el precio para que la inclinacion tenga signo, pero el tono es ~0 por
        # construccion, asi que en la practica esto solo actua si el tema gana polaridad.
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
                f"Deriva de {cfg.drift_window}d: {drift_pct:+.2f}% confirmada por "
                f"{cfg.confirm_window}d ({confirm_pct:+.2f}%)"
            ),
            features={
                "close": latest_close,
                "drift_pct": drift_pct,
                "confirm_pct": confirm_pct,
                "atr": latest_atr,
                "atr_pct": atr_pct,
                **signal_features(reading),
            },
            signal_id=f"{self.strategy_id}:{symbol}:{uuid4().hex[:12]}",
        )


def _confidence(*, drift_pct: float, config: EventCalendarDriftConfig) -> float:
    """La conviccion crece con la deriva DENTRO de la banda y se apaga al acercarse al techo,
    que es donde deja de ser deriva y pasa a ser el evento ya ocurrido."""
    span = max(config.max_drift_pct - config.min_drift_pct, 1e-9)
    position = min(max(abs(drift_pct) - config.min_drift_pct, 0.0) / span, 1.0)
    # Triangular: maximo en el centro de la banda.
    raw = 1.0 - abs(position - 0.5) * 2.0
    return round(min(max(0.55 + raw * 0.35, 0.55), 0.90), 2)
