from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import uuid4

import pandas as pd

from ai_trader.observation.signal_radar import (
    INERT_MAX_INTENSITY,
    INERT_MIN_INTENSITY,
    INERT_MIN_TONE,
    signal_gate_reason,
)
from ai_trader.shared import bars as bar_schema
from ai_trader.shared.clock import utc_now
# El ATR vive en `shared/indicators.py` porque este modulo y `mean_reversion.py` tenian
# la MISMA funcion duplicada. Se importa con el nombre local de siempre para que ninguna
# llamada de aqui abajo cambie (y porque `atr` a secas chocaria con la variable local).
from ai_trader.shared.indicators import atr as _atr
from ai_trader.shared.schemas import Side, Signal

logger = logging.getLogger(__name__)


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


@dataclass(slots=True)
class CryptoMomentumConfig:
    timeframe: str = "1d"
    fast_sma_window: int = 10
    slow_sma_window: int = 20
    atr_window: int = 14
    breakout_lookback: int = 5
    min_atr_pct: float = 0.2
    risk_atr_multiple: float = 2.0
    reward_atr_multiple: float = 3.0
    min_bars: int = 30
    # El filtro de ruptura estaba anulado a mano (`breakout_ok = True`), asi que la
    # estrategia era en realidad un cruce de medias pese a llamarse "breakout".
    # Se restaura como puerta real y se deja configurable.
    require_breakout: bool = True
    # Filtros de regimen cross-sectional (opcionales, los aplica solo si hay un
    # MarketRegimeProvider inyectado). Con los defaults permisivos no filtran nada, asi
    # que el comportamiento en vivo es identico. El CEM los optimiza como params.
    min_breadth: float = 0.0  # 0 = sin filtro; exige fraccion minima del universo > SMA
    min_relative_strength: float = -1.0  # -1 = sin filtro; exige fuerza vs mercado
    # Filtros de SENALES EXTERNAS (opcionales, solo con SignalRadarProvider inyectado).
    #
    # Los defaults son los valores INERTES por construccion: el tono esta acotado a
    # [-Z_CLIP, +Z_CLIP] y la intensidad a [0, Z_CLIP], asi que un piso en el borde exacto
    # no puede bloquear ninguna lectura posible. No es "permisivo": es imposible que filtre.
    #
    # POLARIDAD. Momentum quiere las dos cosas ALTAS y por motivos distintos: el tono como
    # CONFIRMACION —comprar fuerza mientras el flujo, la actividad y la atencion acompanan—
    # y la intensidad porque una ruptura sin nada detras es ruido. La intensidad es el
    # unico eje en el que esta primitiva y la reversion a la media quieren cosas opuestas;
    # en el tono las dos ponen un PISO (ver mean_reversion.py).
    #
    # NINGUNO ENTRA EN `scoring/search_space.py`: son constantes de configuracion, no
    # dimensiones sorteables. Ver el test que lo congela en tests/test_transfer.py.
    min_signal_tone: float = INERT_MIN_TONE
    min_signal_intensity: float = INERT_MIN_INTENSITY

    def __post_init__(self) -> None:
        if self.fast_sma_window <= 0:
            raise ValueError("fast_sma_window must be > 0")
        if self.slow_sma_window <= 0:
            raise ValueError("slow_sma_window must be > 0")
        if self.atr_window <= 0:
            raise ValueError("atr_window must be > 0")
        if self.breakout_lookback <= 1:
            raise ValueError("breakout_lookback must be > 1")
        if self.min_atr_pct <= 0:
            raise ValueError("min_atr_pct must be > 0")
        if self.risk_atr_multiple <= 0:
            raise ValueError("risk_atr_multiple must be > 0")
        if self.reward_atr_multiple <= 0:
            raise ValueError("reward_atr_multiple must be > 0")
        if self.min_bars < self.slow_sma_window:
            raise ValueError("min_bars must be >= slow_sma_window")
        if not 0.0 <= self.min_breadth <= 1.0:
            raise ValueError("min_breadth must be between 0 and 1")
        # Un umbral fuera del rango alcanzable no es "muy exigente": es una puerta cerrada
        # a cal y canto que nadie escribio queriendo. Se rechaza al construir.
        if not INERT_MIN_TONE <= self.min_signal_tone <= -INERT_MIN_TONE:
            raise ValueError(
                f"min_signal_tone must be between {INERT_MIN_TONE} and {-INERT_MIN_TONE}"
            )
        if not INERT_MIN_INTENSITY <= self.min_signal_intensity <= INERT_MAX_INTENSITY:
            raise ValueError(
                f"min_signal_intensity must be between {INERT_MIN_INTENSITY} "
                f"and {INERT_MAX_INTENSITY}"
            )


class CryptoMomentumStrategy:
    strategy_id = "crypto_momentum_v1"

    def __init__(self, config: CryptoMomentumConfig | None = None) -> None:
        self.config = config or CryptoMomentumConfig()
        # Colaboradores opcionales; los inyecta quien construye (backtest o proceso vivo).
        self._regime = None
        self._signals = None

    def attach_regime_provider(self, provider) -> None:
        self._regime = provider

    def attach_signal_provider(self, provider) -> None:
        """Mismo patron duck-typed que el regimen: quien no lo llame se queda sin puerta."""
        self._signals = provider

    def _regime_active(self) -> bool:
        return self.config.min_breadth > 0.0 or self.config.min_relative_strength > -1.0

    def _signals_active(self) -> bool:
        return (
            self.config.min_signal_tone > INERT_MIN_TONE
            or self.config.min_signal_intensity > INERT_MIN_INTENSITY
        )

    def supports_symbol(self, symbol: str) -> bool:
        # Opera cualquier simbolo con barras OHLCV; los de prediccion no las tienen.
        return not symbol.strip().upper().startswith("PM::")

    def generate_signal(self, symbol: str, bars: pd.DataFrame) -> Signal | None:
        if bars is None or bars.empty:
            return None

        if len(bars) < self.config.min_bars:
            return None

        close = bar_schema.series(bars, bar_schema.CLOSE)
        high = bar_schema.series(bars, bar_schema.HIGH)

        fast_sma = _sma(close, self.config.fast_sma_window)
        slow_sma = _sma(close, self.config.slow_sma_window)
        atr = _atr(bars, self.config.atr_window)

        latest_close = float(close.iloc[-1])
        latest_fast_sma = fast_sma.iloc[-1]
        latest_slow_sma = slow_sma.iloc[-1]
        latest_atr = atr.iloc[-1]

        if pd.isna(latest_fast_sma) or pd.isna(latest_slow_sma) or pd.isna(latest_atr):
            return None

        if latest_close <= 0:
            return None

        atr_pct = float(latest_atr / latest_close * 100.0)

        recent_high = high.shift(1).rolling(
            window=self.config.breakout_lookback,
            min_periods=self.config.breakout_lookback,
        ).max()
        breakout_level = recent_high.iloc[-1]

        if pd.isna(breakout_level):
            return None

        trend_ok = bool(latest_fast_sma > latest_slow_sma)
        volatility_ok = bool(atr_pct >= self.config.min_atr_pct)
        breakout_ok = (
            bool(latest_close > float(breakout_level))
            if self.config.require_breakout
            else True
        )

        logger.info(
            (
                "Momentum check | symbol=%s | close=%.6f | fast_sma=%.6f | slow_sma=%.6f | "
                "atr_pct=%.2f | trend_ok=%s | breakout_ok=%s | volatility_ok=%s"
            ),
            symbol,
            latest_close,
            float(latest_fast_sma),
            float(latest_slow_sma),
            atr_pct,
            trend_ok,
            breakout_ok,
            volatility_ok,
        )

        if not (trend_ok and breakout_ok and volatility_ok):
            logger.info("Strategy produced no signal for symbol=%s", symbol)
            return None

        # Puerta de regimen: momentum quiere activos fuertes en un mercado amplio.
        if self._regime is not None and self._regime_active():
            regime = self._regime.features(symbol)
            if regime["breadth"] < self.config.min_breadth:
                logger.info("Regime breadth gate blocked symbol=%s", symbol)
                return None
            if regime["relative_strength"] < self.config.min_relative_strength:
                logger.info("Regime relative-strength gate blocked symbol=%s", symbol)
                return None

        # Puerta de senales externas, DESPUES de la de regimen: es la mas cara de las dos
        # (toca treinta fuentes) y la que menos veces esta activa, asi que preguntarla
        # la ultima es lo mismo que preguntarla nunca en la configuracion por defecto.
        # Nunca bloquea por falta de datos: `signal_gate_reason` se salta el bloque cuya
        # cobertura no llega al minimo declarado.
        if self._signals is not None and self._signals_active():
            reason = signal_gate_reason(
                self._signals.features(symbol),
                min_tone=self.config.min_signal_tone,
                min_intensity=self.config.min_signal_intensity,
            )
            if reason is not None:
                logger.info("Signal gate blocked symbol=%s (%s)", symbol, reason)
                return None

        stop_loss = float(latest_close - (latest_atr * self.config.risk_atr_multiple))
        take_profit = float(latest_close + (latest_atr * self.config.reward_atr_multiple))

        confidence = self._build_confidence(
            latest_close=latest_close,
            fast_sma=float(latest_fast_sma),
            slow_sma=float(latest_slow_sma),
            atr_pct=atr_pct,
            breakout_level=float(breakout_level),
        )

        logger.info("Strategy produced BUY signal for symbol=%s", symbol)

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            timeframe=self.config.timeframe,
            timestamp=utc_now(),
            side=Side.BUY,
            confidence=confidence,
            entry_price=latest_close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=(
                "Momentum breakout: fast_sma > slow_sma, "
                f"ATR%={atr_pct:.2f}"
            ),
            features={
                "close": latest_close,
                "sma_fast": float(latest_fast_sma),
                "sma_slow": float(latest_slow_sma),
                "atr": float(latest_atr),
                "atr_pct": atr_pct,
                "breakout_level": float(breakout_level),
            },
            signal_id=f"{self.strategy_id}:{symbol}:{uuid4().hex[:12]}",
        )

    def _build_confidence(
        self,
        *,
        latest_close: float,
        fast_sma: float,
        slow_sma: float,
        atr_pct: float,
        breakout_level: float,
    ) -> float:
        trend_strength = min(max((fast_sma - slow_sma) / latest_close, 0.0), 0.05) / 0.05
        breakout_strength = min(max((latest_close - breakout_level) / latest_close, 0.0), 0.03) / 0.03
        volatility_strength = min(max(atr_pct / 4.0, 0.0), 1.0)

        raw_score = (
            0.45 * trend_strength
            + 0.35 * breakout_strength
            + 0.20 * volatility_strength
        )

        confidence = 0.55 + (raw_score * 0.35)
        return round(min(max(confidence, 0.55), 0.90), 2)