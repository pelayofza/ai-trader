from __future__ import annotations

from typing import Any, Callable

from ai_trader.strategies.momentum_crypto import CryptoMomentumConfig, CryptoMomentumStrategy
from ai_trader.strategies.polymarket_threshold import (
    PolymarketThresholdConfig,
    PolymarketThresholdStrategy,
)

# Registro de estrategias parametrizadas: nombre -> constructor(params) -> Strategy.
#
# Es el punto de entrada del generador de estrategias con IA: generar una estrategia
# se reduce a emitir {"type": <clave>, "params": {...}}, sin escribir codigo nuevo.
StrategyFactory = Callable[[dict[str, Any]], Any]


def _build_crypto_momentum(params: dict[str, Any]):
    return CryptoMomentumStrategy(CryptoMomentumConfig(**params))


def _build_polymarket_threshold(params: dict[str, Any]):
    return PolymarketThresholdStrategy(PolymarketThresholdConfig(**params))


STRATEGY_REGISTRY: dict[str, StrategyFactory] = {
    "crypto_momentum": _build_crypto_momentum,
    "polymarket_threshold": _build_polymarket_threshold,
}


def build_strategy(
    strategy_type: str,
    params: dict[str, Any] | None = None,
    strategy_id: str | None = None,
):
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
