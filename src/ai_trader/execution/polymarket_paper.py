from __future__ import annotations

from ai_trader.data.providers.polymarket_clob import PolymarketClobProvider
from ai_trader.execution.paper import PaperExecutionEngine
from ai_trader.shared.instruments import AssetClass, Venue
from ai_trader.shared.schemas import ExecutionResult, OrderRequest


class PolymarketPaperExecutionEngine:
    """
    Ejecucion en papel de ordenes de mercados de prediccion.

    No duplica la simulacion de fills: resuelve el precio (el midpoint del CLOB) y
    delega en PaperExecutionEngine. Se le inyectan tanto el proveedor como el motor
    para que comparta instancia con el resto del sistema; antes construia los suyos
    propios, lo que significaba una segunda conexion HTTP y comisiones por defecto
    distintas de las configuradas.
    """

    def __init__(
        self,
        paper_engine: PaperExecutionEngine,
        clob_provider: PolymarketClobProvider | None = None,
    ) -> None:
        self.paper_engine = paper_engine
        self.clob_provider = clob_provider or PolymarketClobProvider()

    def execute(
        self,
        order_request: OrderRequest,
        reference_price: float | None = None,
    ) -> ExecutionResult:
        if order_request.asset_class != AssetClass.PREDICTION:
            raise ValueError("PolymarketPaperExecutionEngine only accepts prediction orders")
        if order_request.venue != Venue.POLYMARKET:
            raise ValueError("PolymarketPaperExecutionEngine only accepts venue=POLYMARKET")
        if not order_request.instrument_id:
            raise ValueError("prediction orders require instrument_id token id")

        # El orquestador ya ha consultado el midpoint para decidir; reutilizarlo evita
        # una segunda llamada HTTP y que el fill imprima a un precio distinto del que
        # la estrategia uso para decidir.
        price = reference_price
        if price is None:
            price = self.clob_provider.get_midpoint(order_request.instrument_id)

        if price is None:
            raise ValueError(
                f"Could not resolve Polymarket midpoint for {order_request.instrument_id}"
            )

        return self.paper_engine.execute(order_request=order_request, market_price=price)
