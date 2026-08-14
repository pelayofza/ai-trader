"""
PERSISTENCIA DE FLUJO: comprar la pausa mientras el dinero sigue entrando.

LA TESIS
--------
`flow` es el tema con mejor materia prima del catalogo: doce fuentes, ocho con historia
medida —la mas antigua desde 2011— y ONCE con polaridad declarada. Es el unico tema cuyo
tono se puede leer como direccion sin pedir prestada ninguna hipotesis nueva.

Y lo que mide tiene una propiedad que casi ninguna senal tiene: PERSISTENCIA. Una creacion de
ETF, una emision de stablecoins, un aumento de comisiones o de TVL, una rotacion de fondos
apalancados en CME —nada de eso se agota en un dia; son procesos con inercia de semanas—. La
contraparte de oferta es igual de lenta: un desbloqueo con fecha, una cola de salida de
staking, un tesoro cotizando por debajo de su NAV.

Por eso el nucleo de precio correcto NO es una rotura —eso ya lo tiene `crypto_momentum`—
sino el RETROCESO dentro de una tendencia persistente: una tendencia cuya pendiente apunta en
un sentido, con una fraccion alta de dias a favor, y un precio que ha vuelto a tocar su media
sin perderla. Comprar la pausa, no el arranque.

La capa hace aqui lo que no puede hacer en ningun otro tema: `SIDE_TONE` es defendible,
porque el flujo decide el lado y el precio solo decide cuando. Y el piso de intensidad es un
PISO —"no operes cuando no esta pasando nada"—, que es lo contrario del techo de la reversion
y de la ignicion de atencion.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import uuid4

import pandas as pd

from ai_trader.observation.signal_radar import INERT_MIN_INTENSITY, INERT_MIN_TONE
from ai_trader.observation.signal_themes import theme_reading
from ai_trader.shared import bars as bar_schema
from ai_trader.shared.clock import utc_now
from ai_trader.shared.indicators import atr, sma, up_share
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

THEME = "flow"


@dataclass(slots=True)
class FlowPersistenceConfig:
    timeframe: str = "1d"
    trend_window: int = 50
    # Sobre cuantos dias se mide la pendiente de la media.
    slope_window: int = 10
    persistence_window: int = 20
    # Fraccion de dias en el sentido de la pendiente. Por debajo de 0,5 "persistencia"
    # significaria lo contrario de lo que dice, asi que el constructor lo rechaza.
    min_persistence: float = 0.55
    # Cuanto se deja retroceder el precio hacia la media, en ATRs, sin perderla.
    pullback_atr: float = 1.0
    atr_window: int = 14
    stop_atr_mult: float = 2.0
    reward_atr_mult: float = 3.0
    min_bars: int = 80
    allow_short: bool = True
    # --- capa de senal (tema 'flow'), inerte por construccion ---------------------------
    min_signal_tone: float = INERT_MIN_TONE
    min_signal_intensity: float = INERT_MIN_INTENSITY
    signal_side_mode: str = SIDE_CORE
    signal_tone_threshold: float = 0.0
    signal_weight: float = INERT_SIGNAL_WEIGHT

    def __post_init__(self) -> None:
        if self.trend_window <= 1:
            raise ValueError("trend_window must be > 1")
        if not 0 < self.slope_window < self.trend_window:
            raise ValueError("slope_window must be in (0, trend_window)")
        if self.persistence_window <= 1:
            raise ValueError("persistence_window must be > 1")
        if not 0.5 <= self.min_persistence < 1.0:
            raise ValueError("min_persistence must be in [0.5, 1)")
        if self.pullback_atr <= 0:
            raise ValueError("pullback_atr must be > 0")
        if self.atr_window <= 0:
            raise ValueError("atr_window must be > 0")
        if self.stop_atr_mult <= 0:
            raise ValueError("stop_atr_mult must be > 0")
        if self.reward_atr_mult <= 0:
            raise ValueError("reward_atr_mult must be > 0")
        if self.min_bars < self.trend_window + max(self.persistence_window, self.atr_window):
            raise ValueError(
                "min_bars must cover trend_window + max(persistence_window, atr_window)"
            )
        validate_signal_fields(
            min_tone=self.min_signal_tone,
            min_intensity=self.min_signal_intensity,
            side_mode=self.signal_side_mode,
            weight=self.signal_weight,
        )


class FlowPersistenceStrategy:
    strategy_id = "flow_persistence_v1"
    theme = THEME

    def __init__(self, config: FlowPersistenceConfig | None = None) -> None:
        self.config = config or FlowPersistenceConfig()
        self._signals = None

    def attach_signal_provider(self, provider) -> None:
        self._signals = provider

    def _signals_active(self) -> bool:
        cfg = self.config
        return (
            cfg.min_signal_tone > INERT_MIN_TONE
            or cfg.min_signal_intensity > INERT_MIN_INTENSITY
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
        trend = sma(close, cfg.trend_window)
        persistence = up_share(close, cfg.persistence_window)
        atr_series = atr(bars, cfg.atr_window)

        latest_close = float(close.iloc[-1])
        latest_trend = trend.iloc[-1]
        past_trend = trend.iloc[-1 - cfg.slope_window] if len(trend) > cfg.slope_window else None
        latest_persistence = persistence.iloc[-1]
        latest_atr = atr_series.iloc[-1]
        if latest_trend is None or past_trend is None:
            return None
        if pd.isna(latest_trend) or pd.isna(past_trend) or pd.isna(latest_persistence):
            return None
        if pd.isna(latest_atr) or latest_close <= 0 or float(latest_atr) <= 0:
            return None

        latest_trend = float(latest_trend)
        latest_atr = float(latest_atr)
        slope = latest_trend - float(past_trend)
        up_fraction = float(latest_persistence)
        # Distancia a la media EN ATRs: positiva por encima, negativa por debajo.
        distance = (latest_close - latest_trend) / latest_atr

        core_side: Side | None = None
        if (
            slope > 0
            and up_fraction >= cfg.min_persistence
            and 0.0 <= distance <= cfg.pullback_atr
        ):
            core_side = Side.BUY
        elif (
            cfg.allow_short
            and slope < 0
            and (1.0 - up_fraction) >= cfg.min_persistence
            and -cfg.pullback_atr <= distance <= 0.0
        ):
            core_side = Side.SELL

        logger.info(
            "Flow persistence check | symbol=%s | close=%.6f | slope=%.6f | up_share=%.2f | "
            "distance=%.2f ATR | core_side=%s",
            symbol, latest_close, slope, up_fraction, distance, core_side,
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
            if core_side is Side.SELL and not cfg.allow_short:
                return None
            reason = side_gate_reason(
                features,
                THEME,
                core_side,
                min_tone=cfg.min_signal_tone,
                min_intensity=cfg.min_signal_intensity,
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

        base = _confidence(
            up_fraction=up_fraction if core_side is Side.BUY else 1.0 - up_fraction,
            distance=distance,
            atr_pct=latest_atr / latest_close * 100.0,
            config=cfg,
        )
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
                f"Retroceso en tendencia persistente: pendiente {slope:+.6f}, "
                f"dias a favor {up_fraction:.2f}, distancia {distance:+.2f} ATR"
            ),
            features={
                "close": latest_close,
                "trend": latest_trend,
                "slope": slope,
                "up_share": up_fraction,
                "distance_atr": distance,
                "atr": latest_atr,
                "atr_pct": latest_atr / latest_close * 100.0,
                **signal_features(reading),
            },
            signal_id=f"{self.strategy_id}:{symbol}:{uuid4().hex[:12]}",
        )


# Cuanto por encima del umbral hace falta para dar por SATURADA la persistencia.
#
# El primer intento saturaba en `up_fraction = 1.0` —veinte dias seguidos al alza— y eso no
# es exigente, es inalcanzable: con el umbral en 0,55 y una fraccion tipica de 0,60, el
# termino aportaba 0,11 de un maximo de 1, la confianza se quedaba clavada cerca del suelo y
# el motor de riesgo rechazaba el 71% de las senales por `min_confidence_per_trade`. Una
# familia que no llega al liston de riesgo no es una familia estricta: es una que no opera,
# y su puesto en un ranking mediria la calibracion de esta funcion y no su tesis.
#
# Es el mismo criterio con el que momentum satura su fuerza de tendencia en un 5% y su
# ruptura en un 3%: valores ALCANZABLES, no extremos teoricos. 0,15 sobre el umbral son
# tres dias de veinte por encima de lo exigido.
PERSISTENCE_SATURATION = 0.15


def _confidence(
    *, up_fraction: float, distance: float, atr_pct: float, config: FlowPersistenceConfig
) -> float:
    """
    Mas conviccion cuanto mas persistente la tendencia, mas profundo el retroceso y mas
    volatilidad haya que capturar.

    LOS TRES TERMINOS, Y POR QUE HACEN FALTA TRES. La primera version tenia solo dos, y los
    dos se median DESDE LA PROPIA PUERTA de entrada: en el margen —que es donde cae la mayoria
    de las entradas, porque la puerta esta puesta justo ahi— los dos valen casi cero, la
    confianza se quedaba pegada a su suelo de 0,55 y el motor de riesgo rechazaba el 71% de
    las senales por `min_confidence_per_trade`.

    Las dos primitivas publicadas no tienen ese problema, y al mirarlas se ve por que: las dos
    llevan un termino de VOLATILIDAD que no se mide desde ninguna puerta —`atr_pct/4` en
    momentum, `std_pct/4` en reversion— y que por tanto levanta la base incluso en una entrada
    marginal. Este es el mismo termino, con el mismo peso que en momentum.
    """
    room = max(min(PERSISTENCE_SATURATION, 1.0 - config.min_persistence), 1e-9)
    persistence_strength = min(max(up_fraction - config.min_persistence, 0.0) / room, 1.0)
    # Un retroceso que toca la media vale mas que uno que apenas se aparto de maximos.
    pullback_strength = min(max(config.pullback_atr - abs(distance), 0.0) / config.pullback_atr, 1.0)
    volatility_strength = min(max(atr_pct / 4.0, 0.0), 1.0)
    raw = 0.45 * persistence_strength + 0.35 * pullback_strength + 0.20 * volatility_strength
    return round(min(max(0.55 + raw * 0.35, 0.55), 0.90), 2)
