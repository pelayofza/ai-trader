from __future__ import annotations

from typing import Protocol

from ai_trader.execution.paper import PaperExecutionEngine
from ai_trader.execution.polymarket_paper import PolymarketPaperExecutionEngine
from ai_trader.shared.instruments import AssetClass
from ai_trader.shared.schemas import ExecutionResult, OrderRequest


class ExecutionEngine(Protocol):
    def execute(
        self,
        order_request: OrderRequest,
        reference_price: float | None = None,
    ) -> ExecutionResult:
        ...


class SpotPaperEngine:
    """Adapta PaperExecutionEngine al contrato del router (necesita precio de referencia)."""

    def __init__(self, engine: PaperExecutionEngine) -> None:
        self.engine = engine

    def execute(
        self,
        order_request: OrderRequest,
        reference_price: float | None = None,
    ) -> ExecutionResult:
        if reference_price is None:
            raise ValueError(f"reference_price is required for {order_request.symbol}")
        return self.engine.execute(order_request=order_request, market_price=reference_price)


class PredictionPaperEngine:
    """Adapta PolymarketPaperExecutionEngine: el precio lo resuelve el propio motor."""

    def __init__(self, engine: PolymarketPaperExecutionEngine) -> None:
        self.engine = engine

    def execute(
        self,
        order_request: OrderRequest,
        reference_price: float | None = None,
    ) -> ExecutionResult:
        return self.engine.execute(order_request, reference_price=reference_price)


class ExecutionRouter:
    """
    Enruta cada orden al motor de su clase de activo.

    Sustituye el cableado anterior, en el que el orquestador recibia un motor por
    cada venue como argumento con nombre. Anadir renta variable o ejecucion real
    es registrar un motor mas, no tocar el orquestador.
    """

    def __init__(self, engines: dict[AssetClass, ExecutionEngine]) -> None:
        if not engines:
            raise ValueError("ExecutionRouter requires at least one engine")
        self.engines = engines

    def execute(
        self,
        order_request: OrderRequest,
        reference_price: float | None = None,
    ) -> ExecutionResult:
        asset_class = order_request.asset_class
        engine = self.engines.get(asset_class) if asset_class else None

        if engine is None:
            raise ValueError(
                f"No execution engine registered for asset class "
                f"{asset_class.value if asset_class else 'unknown'} "
                f"(symbol={order_request.symbol})"
            )

        return engine.execute(order_request, reference_price=reference_price)

    @classmethod
    def paper(
        cls,
        spot_engine: PaperExecutionEngine,
        prediction_engine: PolymarketPaperExecutionEngine,
    ) -> ExecutionRouter:
        spot = SpotPaperEngine(spot_engine)
        return cls(
            {
                AssetClass.CRYPTO: spot,
                AssetClass.STOCK: spot,
                AssetClass.PREDICTION: PredictionPaperEngine(prediction_engine),
            }
        )
