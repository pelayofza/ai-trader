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
# El ATR vive en `shared/indicators.py` porque este modulo y `momentum_crypto.py` tenian
# la MISMA funcion duplicada. Se importa con el nombre local de siempre para que ninguna
# llamada de aqui abajo cambie (y porque `atr` a secas chocaria con la variable local).
from ai_trader.shared.indicators import atr as _atr
from ai_trader.shared.schemas import Side, Signal

logger = logging.getLogger(__name__)


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    # Desviacion muestral (ddof=1), coherente con volatility_pct del motor de metricas.
    return series.rolling(window=window, min_periods=window).std(ddof=1)


@dataclass(slots=True)
class MeanReversionConfig:
    """
    Primitiva de reversion a la media, regimen opuesto al momentum.

    Compra cuando el precio esta estirado por debajo de su media movil mas de
    `entry_z` desviaciones (banda inferior de Bollinger / z-score sobrevendido) y
    apunta a la reversion hacia la media. Puramente parametrica y sin IA: el CEM la
    optimiza sobre los backtests sinteticos.

    Las salidas siguen siendo mecanicas (las fija el riesgo a partir del SL/TP que
    sugerimos + max_holding_days del runner); aqui solo se decide la ENTRADA.
    """

    timeframe: str = "1d"
    lookback: int = 20
    # Umbral de entrada en desviaciones tipicas por DEBAJO de la media.
    entry_z: float = 2.0
    # Objetivo de salida en desviaciones respecto a la media (0.0 = reversion exacta a
    # la media). El take-profit resultante = media + exit_z * sigma.
    exit_z: float = 0.0
    # Stop en multiplos de ATR por debajo del precio de entrada.
    stop_atr_mult: float = 2.5
    atr_window: int = 14
    # Suelo de volatilidad: si sigma/precio es menor que esto (en %), no hay reversion
    # que capturar y se descarta. Tambien evita dividir ruido en el z-score.
    min_std_pct: float = 0.1
    min_bars: int = 30
    # Filtros de regimen cross-sectional (opcionales, solo con MarketRegimeProvider
    # inyectado). Defaults permisivos = comportamiento identico. El CEM los optimiza.
    min_breadth: float = 0.0  # 0 = sin filtro; evita comprar en selloff amplio (cuchillos)
    max_relative_strength: float = 1.0  # 1 = sin filtro; solo compra rezagados del mercado
    # Filtros de SENALES EXTERNAS (opcionales, solo con SignalRadarProvider inyectado).
    # Defaults en el borde exacto del recorte de las z: inertes por construccion, no por
    # convenio (ver observation/signal_radar.py::INERT_MIN_TONE).
    #
    # POLARIDAD, Y AQUI ESTA EL MATIZ QUE PARECE UN ERROR Y NO LO ES. La intensidad si es
    # el eje opuesto al de momentum: esta primitiva compra caidas y lo que la mata es
    # comprarlas mientras algo grande esta ocurriendo, asi que su umbral es un TECHO.
    # El tono, en cambio, es un PISO en las dos, igual que en momentum. No por simetria:
    # el modo de fallo caracteristico de la reversion a la media es comprar una caida de
    # -3 sigma que resulta ser el primer dia de un reprecio permanente —el hack, el
    # desbloqueo, la sancion—, y en esa situacion el tono esta muy negativo. Un techo de
    # tono compraria exactamente esas.
    min_signal_tone: float = INERT_MIN_TONE
    max_signal_intensity: float = INERT_MAX_INTENSITY

    def __post_init__(self) -> None:
        if self.lookback <= 1:
            raise ValueError("lookback must be > 1")
        if self.entry_z <= 0:
            raise ValueError("entry_z must be > 0")
        if self.exit_z < 0:
            raise ValueError("exit_z must be >= 0")
        if self.exit_z >= self.entry_z:
            raise ValueError("exit_z must be < entry_z (target must sit above the entry level)")
        if self.stop_atr_mult <= 0:
            raise ValueError("stop_atr_mult must be > 0")
        if self.atr_window <= 0:
            raise ValueError("atr_window must be > 0")
        if self.min_std_pct <= 0:
            raise ValueError("min_std_pct must be > 0")
        if self.min_bars < self.lookback:
            raise ValueError("min_bars must be >= lookback")
        if not 0.0 <= self.min_breadth <= 1.0:
            raise ValueError("min_breadth must be between 0 and 1")
        if not INERT_MIN_TONE <= self.min_signal_tone <= -INERT_MIN_TONE:
            raise ValueError(
                f"min_signal_tone must be between {INERT_MIN_TONE} and {-INERT_MIN_TONE}"
            )
        if not INERT_MIN_INTENSITY <= self.max_signal_intensity <= INERT_MAX_INTENSITY:
            raise ValueError(
                f"max_signal_intensity must be between {INERT_MIN_INTENSITY} "
                f"and {INERT_MAX_INTENSITY}"
            )


class MeanReversionStrategy:
    strategy_id = "mean_reversion_v1"

    def __init__(self, config: MeanReversionConfig | None = None) -> None:
        self.config = config or MeanReversionConfig()
        # Colaboradores opcionales; los inyecta quien construye (backtest o proceso vivo).
        self._regime = None
        self._signals = None

    def attach_regime_provider(self, provider) -> None:
        self._regime = provider

    def attach_signal_provider(self, provider) -> None:
        self._signals = provider

    def _regime_active(self) -> bool:
        return self.config.min_breadth > 0.0 or self.config.max_relative_strength < 1.0

    def _signals_active(self) -> bool:
        return (
            self.config.min_signal_tone > INERT_MIN_TONE
            or self.config.max_signal_intensity < INERT_MAX_INTENSITY
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

        mean = _sma(close, self.config.lookback)
        std = _rolling_std(close, self.config.lookback)
        atr = _atr(bars, self.config.atr_window)

        latest_close = float(close.iloc[-1])
        latest_mean = mean.iloc[-1]
        latest_std = std.iloc[-1]
        latest_atr = atr.iloc[-1]

        if pd.isna(latest_mean) or pd.isna(latest_std) or pd.isna(latest_atr):
            return None

        if latest_close <= 0 or latest_std <= 0:
            return None

        latest_mean = float(latest_mean)
        latest_std = float(latest_std)
        latest_atr = float(latest_atr)

        std_pct = latest_std / latest_close * 100.0
        z_score = (latest_close - latest_mean) / latest_std

        # Puertas de entrada: precio estirado por debajo de la media Y con volatilidad
        # suficiente para que la reversion valga el coste de operar.
        oversold_ok = bool(z_score <= -self.config.entry_z)
        volatility_ok = bool(std_pct >= self.config.min_std_pct)

        logger.info(
            (
                "MeanRev check | symbol=%s | close=%.6f | mean=%.6f | std=%.6f | "
                "z=%.2f | std_pct=%.2f | oversold_ok=%s | volatility_ok=%s"
            ),
            symbol,
            latest_close,
            latest_mean,
            latest_std,
            z_score,
            std_pct,
            oversold_ok,
            volatility_ok,
        )

        if not (oversold_ok and volatility_ok):
            logger.info("Strategy produced no signal for symbol=%s", symbol)
            return None

        # Puerta de regimen: comprar rezagados, pero no en un selloff amplio (cuchillos).
        if self._regime is not None and self._regime_active():
            regime = self._regime.features(symbol)
            if regime["breadth"] < self.config.min_breadth:
                logger.info("Regime breadth gate blocked symbol=%s", symbol)
                return None
            if regime["relative_strength"] > self.config.max_relative_strength:
                logger.info("Regime relative-strength gate blocked symbol=%s", symbol)
                return None

        # Puerta de senales, DESPUES de la de regimen y con la misma regla de fallo
        # abierto: sin cobertura suficiente no se evalua nada.
        if self._signals is not None and self._signals_active():
            reason = signal_gate_reason(
                self._signals.features(symbol),
                min_tone=self.config.min_signal_tone,
                max_intensity=self.config.max_signal_intensity,
            )
            if reason is not None:
                logger.info("Signal gate blocked symbol=%s (%s)", symbol, reason)
                return None

        stop_loss = latest_close - (latest_atr * self.config.stop_atr_mult)
        take_profit = latest_mean + (self.config.exit_z * latest_std)

        # El objetivo de reversion queda por encima de la entrada por construccion
        # (entrada = media - entry_z*sigma, objetivo = media + exit_z*sigma, exit_z <
        # entry_z). El stop, en cambio, podria caer <= 0 en activos muy volatiles.
        if stop_loss <= 0 or take_profit <= latest_close:
            logger.info("Degenerate SL/TP for symbol=%s; skipping", symbol)
            return None

        confidence = self._build_confidence(z_score=z_score, std_pct=std_pct)

        logger.info("Strategy produced BUY signal for symbol=%s", symbol)

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            timeframe=self.config.timeframe,
            timestamp=utc_now(),
            side=Side.BUY,
            confidence=confidence,
            entry_price=latest_close,
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            reason=(
                f"Mean reversion: z={z_score:.2f} <= -{self.config.entry_z:.2f}, "
                f"std%={std_pct:.2f}"
            ),
            features={
                "close": latest_close,
                "mean": latest_mean,
                "std": latest_std,
                "z_score": z_score,
                "std_pct": std_pct,
                "atr": latest_atr,
            },
            signal_id=f"{self.strategy_id}:{symbol}:{uuid4().hex[:12]}",
        )

    def _build_confidence(self, *, z_score: float, std_pct: float) -> float:
        # Cuanto mas estirado por debajo del umbral, mayor la conviccion; se satura al
        # llegar a 2*entry_z de exceso para no premiar colas absurdas.
        excess = max(-z_score - self.config.entry_z, 0.0)
        stretch_strength = min(excess / self.config.entry_z, 1.0)
        # Algo de volatilidad ayuda (hay reversion que capturar), pero se satura pronto.
        volatility_strength = min(std_pct / 4.0, 1.0)

        raw_score = 0.7 * stretch_strength + 0.3 * volatility_strength
        confidence = 0.55 + (raw_score * 0.35)
        return round(min(max(confidence, 0.55), 0.90), 2)
