from __future__ import annotations

from typing import Any, Callable

from ai_trader.strategies.attention_ignition import (
    AttentionIgnitionConfig,
    AttentionIgnitionStrategy,
)
from ai_trader.strategies.event_calendar_drift import (
    EventCalendarDriftConfig,
    EventCalendarDriftStrategy,
)
from ai_trader.strategies.flow_persistence import FlowPersistenceConfig, FlowPersistenceStrategy
from ai_trader.strategies.liquidation_cascade import (
    LiquidationCascadeConfig,
    LiquidationCascadeStrategy,
)
from ai_trader.strategies.mean_reversion import MeanReversionConfig, MeanReversionStrategy
from ai_trader.strategies.momentum_crypto import CryptoMomentumConfig, CryptoMomentumStrategy
from ai_trader.strategies.polymarket_threshold import (
    PolymarketThresholdConfig,
    PolymarketThresholdStrategy,
)
from ai_trader.strategies.signal_composite import SignalCompositeConfig, SignalCompositeStrategy
from ai_trader.strategies.vol_term_structure import (
    VolTermStructureConfig,
    VolTermStructureStrategy,
)

# Registro de estrategias parametrizadas: nombre -> constructor(params) -> Strategy.
#
# Es el punto de entrada del generador de estrategias con IA: generar una estrategia
# se reduce a emitir {"type": <clave>, "params": {...}}, sin escribir codigo nuevo.
StrategyFactory = Callable[[dict[str, Any]], Any]


def _build_crypto_momentum(params: dict[str, Any]):
    return CryptoMomentumStrategy(CryptoMomentumConfig(**params))


def _build_mean_reversion(params: dict[str, Any]):
    return MeanReversionStrategy(MeanReversionConfig(**params))


def _build_polymarket_threshold(params: dict[str, Any]):
    return PolymarketThresholdStrategy(PolymarketThresholdConfig(**params))


# --- las seis primitivas TEMATICAS ---------------------------------------------------
#
# Cada una lee UN tema del radar (`observation/signal_themes.py`) salvo la ultima, que los
# lee los cinco. Todas tienen la misma forma: nucleo de precio que corre solo + capa de senal
# INERTE por defecto, de modo que sin cobertura degradan a su variante ciega en vez de dejar
# de operar. Ninguna puede bloquear por falta de datos.


def _build_liquidation_cascade(params: dict[str, Any]):
    return LiquidationCascadeStrategy(LiquidationCascadeConfig(**params))


def _build_vol_term_structure(params: dict[str, Any]):
    return VolTermStructureStrategy(VolTermStructureConfig(**params))


def _build_event_calendar_drift(params: dict[str, Any]):
    return EventCalendarDriftStrategy(EventCalendarDriftConfig(**params))


def _build_attention_ignition(params: dict[str, Any]):
    return AttentionIgnitionStrategy(AttentionIgnitionConfig(**params))


def _build_flow_persistence(params: dict[str, Any]):
    return FlowPersistenceStrategy(FlowPersistenceConfig(**params))


def _build_signal_composite(params: dict[str, Any]):
    return SignalCompositeStrategy(SignalCompositeConfig(**params))


STRATEGY_REGISTRY: dict[str, StrategyFactory] = {
    "crypto_momentum": _build_crypto_momentum,
    "mean_reversion": _build_mean_reversion,
    "polymarket_threshold": _build_polymarket_threshold,
    "liquidation_cascade": _build_liquidation_cascade,
    "vol_term_structure": _build_vol_term_structure,
    "event_calendar_drift": _build_event_calendar_drift,
    "attention_ignition": _build_attention_ignition,
    "flow_persistence": _build_flow_persistence,
    "signal_composite": _build_signal_composite,
}


def build_strategy(
    strategy_type: str,
    params: dict[str, Any] | None = None,
    strategy_id: str | None = None,
) -> Any:
    """Instancia una estrategia del registro por su clave.

    Devuelve `Any` y no un tipo comun a proposito: el registro es el punto de entrada del
    generador con IA, que emite `{"type": ..., "params": {...}}` sin escribir codigo, y las
    estrategias solo comparten forma por duck typing (ver `StrategyFactory`)."""
    factory = STRATEGY_REGISTRY.get(strategy_type)

    if factory is None:
        known = ", ".join(sorted(STRATEGY_REGISTRY))
        raise ValueError(f"Unknown strategy type '{strategy_type}'. Known types: {known}")

    try:
        strategy = factory(dict(params or {}))
    except TypeError as exc:
        raise ValueError(f"Invalid params for strategy '{strategy_type}': {exc}") from exc

    # El id es responsabilidad del registro, no de cada config: asi dos instancias de
    # la misma estrategia con parametros distintos pueden coexistir e identificarse.
    if strategy_id:
        strategy.strategy_id = strategy_id

    return strategy
