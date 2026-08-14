"""
IGNICION DE ATENCION: comprar el dia en que el minorista se entera, no el dia despues.

LA TESIS
--------
La atencion minorista es la demanda de ultimo recurso de cripto, y tiene tres propiedades que
la hacen operable en diario: llega TARDE (cuando el movimiento ya empezo), llega LENTA (dura
dias, no horas) y es INSENSIBLE al precio (no compra barato, compra lo que sale en la lista).
Ese perfil produce continuacion, no reversion, y por eso esta primitiva persigue en vez de
desvanecer.

El tema `attention` mide la ignicion desde cuatro angulos que casi nadie mira juntos: el
listado en Upbit —el evento mas limpio del catalogo, con signo: alta es demanda nueva, baja
es oferta forzada—, el diferencial de visibilidad Corea-EEUU en la App Store (Corea se
calienta ANTES), las busquedas en Naver y Yandex, las visitas a Wikipedia por idioma y la
prima P2P de quien compra por necesidad monetaria y no por especulacion.

El gemelo de precio es una BARRA DE IGNICION: volumen multiplo de su mediana movil, cierre
pegado al maximo del dia y precio por encima de su media larga.

SOLO LARGO, Y ESTA DECLARADO
-----------------------------
La atencion minorista llega COMPRANDO; su unico caso bajista es el deslistado, y eso vive en
activos cuya mediana mueve 247.645 dolares al dia (`cex_listings.typical_adv_usd`), donde no
cabe tamano que pague el viaje. Vender la ausencia de atencion no es lo mismo que vender la
atencion negativa, y esta primitiva no sabe hacer lo segundo. `allow_short = False` no es un
default olvidado: es la tesis.

TECHO DE INTENSIDAD, NO PISO
-----------------------------
Es la misma polaridad que `mean_reversion` y por el mismo motivo: la atencion SATURADA es el
techo, no el arranque. El piso de tono, en cambio, sigue siendo un piso —un deslistado o un
sentimiento roto cancelan la compra—, igual que en las dos primitivas de precio.
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
from ai_trader.shared.indicators import atr, close_location, sma, volume_ratio
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

THEME = "attention"


@dataclass(slots=True)
class AttentionIgnitionConfig:
    timeframe: str = "1d"
    volume_lookback: int = 20
    # Volumen de hoy dividido por su mediana movil. Un multiplo <= 1 no es ignicion.
    volume_mult: float = 2.5
    close_location_min: float = 0.70
    trend_window: int = 50
    atr_window: int = 14
    min_atr_pct: float = 0.2
    stop_atr_mult: float = 2.0
    reward_atr_mult: float = 3.0
    min_bars: int = 70
    # DECLARADO, no olvidado: ver el docstring del modulo.
    allow_short: bool = False
    # --- capa de senal (tema 'attention'), inerte por construccion ----------------------
    min_signal_tone: float = INERT_MIN_TONE
    max_signal_intensity: float = INERT_MAX_INTENSITY
    signal_side_mode: str = SIDE_CORE
    signal_tone_threshold: float = 0.0
    signal_weight: float = INERT_SIGNAL_WEIGHT

    def __post_init__(self) -> None:
        if self.volume_lookback <= 1:
            raise ValueError("volume_lookback must be > 1")
        if self.volume_mult <= 1.0:
            raise ValueError("volume_mult must be > 1")
        if not 0.5 < self.close_location_min < 1.0:
            raise ValueError("close_location_min must be in (0.5, 1)")
        if self.trend_window <= 1:
            raise ValueError("trend_window must be > 1")
        if self.atr_window <= 0:
            raise ValueError("atr_window must be > 0")
        if self.min_atr_pct <= 0:
            raise ValueError("min_atr_pct must be > 0")
        if self.stop_atr_mult <= 0:
            raise ValueError("stop_atr_mult must be > 0")
        if self.reward_atr_mult <= 0:
            raise ValueError("reward_atr_mult must be > 0")
        if self.min_bars < max(self.trend_window, self.volume_lookback) + self.atr_window:
            raise ValueError("min_bars must cover max(trend_window, volume_lookback) + atr_window")
        validate_signal_fields(
            min_tone=self.min_signal_tone,
            max_intensity=self.max_signal_intensity,
            side_mode=self.signal_side_mode,
            weight=self.signal_weight,
        )


class AttentionIgnitionStrategy:
    strategy_id = "attention_ignition_v1"
    theme = THEME

    def __init__(self, config: AttentionIgnitionConfig | None = None) -> None:
        self.config = config or AttentionIgnitionConfig()
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
        # Sin columna de volumen no hay ignicion que medir. Devolver None y no reventar: hay
        # proveedores sin volumen fiable, y esta primitiva no puede fingir que lo tiene.
        ratio = volume_ratio(bars, cfg.volume_lookback)
        location = close_location(bars)
        trend = sma(close, cfg.trend_window)
        atr_series = atr(bars, cfg.atr_window)

        latest_close = float(close.iloc[-1])
        latest_ratio = ratio.iloc[-1]
        latest_loc = location.iloc[-1]
        latest_trend = trend.iloc[-1]
        latest_atr = atr_series.iloc[-1]
        if pd.isna(latest_ratio) or pd.isna(latest_loc) or pd.isna(latest_trend):
            return None
        if pd.isna(latest_atr) or latest_close <= 0 or float(latest_atr) <= 0:
            return None

        latest_ratio = float(latest_ratio)
        latest_loc = float(latest_loc)
        latest_trend = float(latest_trend)
        latest_atr = float(latest_atr)
        atr_pct = latest_atr / latest_close * 100.0

        ignition = (
            latest_ratio >= cfg.volume_mult
            and latest_loc >= cfg.close_location_min
            and latest_close > latest_trend
            and atr_pct >= cfg.min_atr_pct
        )

        logger.info(
            "Attention check | symbol=%s | close=%.6f | vol_ratio=%.2f | loc=%.2f | "
            "trend=%.6f | atr_pct=%.2f | ignition=%s",
            symbol, latest_close, latest_ratio, latest_loc, latest_trend, atr_pct, ignition,
        )
        if not ignition:
            return None

        core_side: Side | None = Side.BUY
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
            # Una primitiva de un solo lado no puede aceptar que el tono la invierta: si el
            # tema pide vender, lo correcto es no operar, no operar al reves de la tesis.
            if core_side is not Side.BUY:
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
            Side.BUY,
            stop_mult=cfg.stop_atr_mult,
            target_mult=cfg.reward_atr_mult,
        )
        if bracket is None:
            logger.info("Degenerate SL/TP for symbol=%s; skipping", symbol)
            return None
        stop_loss, take_profit = bracket

        base = _confidence(volume_ratio_value=latest_ratio, location=latest_loc, config=cfg)
        confidence = resolve_confidence(base, reading, Side.BUY, weight=cfg.signal_weight)

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            timeframe=cfg.timeframe,
            timestamp=utc_now(),
            side=Side.BUY,
            confidence=confidence,
            entry_price=latest_close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=(
                f"Ignicion de atencion: volumen x{latest_ratio:.2f} sobre su mediana, "
                f"cierre en {latest_loc:.2f} del rango"
            ),
            features={
                "close": latest_close,
                "volume_ratio": latest_ratio,
                "close_location": latest_loc,
                "trend": latest_trend,
                "atr": latest_atr,
                "atr_pct": atr_pct,
                **signal_features(reading),
            },
            signal_id=f"{self.strategy_id}:{symbol}:{uuid4().hex[:12]}",
        )


# Que fraccion del umbral de volumen, POR ENCIMA de el, satura la conviccion; y que parte
# del rango que queda sobre el suelo de cierre la satura a ella.
#
# Los dos se bajaron despues de medir: con la saturacion en el doble del umbral (x5 el
# volumen mediano) y en el maximo exacto del dia, la confianza se quedaba tan pegada al suelo
# que el motor de riesgo rechazaba la MITAD de las senales. El listado de referencia son las
# dos primitivas de precio, que rechazan el 3% y el 12%. Saturar en valores inalcanzables no
# hace una estrategia mas exigente: hace que no opere, y entonces su puesto en un ranking
# mide esta funcion en vez de su tesis.
VOLUME_SATURATION = 0.5
LOCATION_SATURATION = 0.6


def _confidence(
    *, volume_ratio_value: float, location: float, config: AttentionIgnitionConfig
) -> float:
    excess = max(volume_ratio_value - config.volume_mult, 0.0)
    volume_strength = min(excess / (config.volume_mult * VOLUME_SATURATION), 1.0)
    # Cuanto mas arriba del rango cierra, mas limpia es la ignicion. Se reescala desde el
    # propio umbral para que "justo en el minimo" valga 0 y no un pedazo gratis de score.
    room = max((1.0 - config.close_location_min) * LOCATION_SATURATION, 1e-9)
    location_strength = min(max(location - config.close_location_min, 0.0) / room, 1.0)
    raw = 0.6 * volume_strength + 0.4 * location_strength
    return round(min(max(0.55 + raw * 0.35, 0.55), 0.90), 2)
