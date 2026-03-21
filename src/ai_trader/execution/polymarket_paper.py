from __future__ import annotations

from ai_trader.data.providers.polymarket_clob import PolymarketClobProvider
from ai_trader.execution.paper import PaperExecutionEngine
from ai_trader.shared.instruments import AssetClass, Venue
from ai_trader.shared.schemas import ExecutionResult, OrderRequest


class PolymarketPaperExecutionEngine:
    def __init__(
        self,
        clob_provider: PolymarketClobProvider | None = None,
        paper_engine: PaperExecutionEngine | None = None,
    ) -> None:
        self.clob_provider = clob_provider or PolymarketClobProvider()
        self.paper_engine = paper_engine or PaperExecutionEngine()

    def execute(self, order_request: OrderRequest) -> ExecutionResult:
        if order_request.asset_class != AssetClass.PREDICTION:
            raise ValueError("PolymarketPaperExecutionEngine only accepts prediction orders")
        if order_request.venue != Venue.POLYMARKET:
            raise ValueError("PolymarketPaperExecutionEngine only accepts venue=POLYMARKET")
        if not order_request.instrument_id:
            raise ValueError("prediction orders require instrument_id token id")

        midpoint = self.clob_provider.get_midpoint(order_request.instrument_id)
        if midpoint is None:
            raise ValueError("could not resolve polymarket midpoint")

        return self.paper_engine.execute(order_request=order_request, market_price=midpoint)

    def list_fills(self):
        return self.paper_engine.list_fills()

    def get_fill(self, order_id: str):
        return self.paper_engine.get_fill(order_id)

    def clear_fills(self) -> None:
        self.paper_engine.clear_fills()