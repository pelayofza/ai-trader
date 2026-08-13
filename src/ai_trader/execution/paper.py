from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from ai_trader.execution.microstructure import (
    EMPTY_SNAPSHOT,
    LiquidityProvider,
    LiquiditySnapshot,
    SlippageModel,
)
from ai_trader.shared.clock import utc_now
from ai_trader.shared.instruments import AssetClass
from ai_trader.shared.schemas import (
    ExecutionResult,
    OrderRequest,
    OrderStatus,
    OrderType,
)


def fill_price(reference_price: float, side: str, slippage_bps: float) -> float:
    """
    El precio al que se llena: la referencia, empeorada por el deslizamiento.

    Vive fuera del motor porque hay una segunda pregunta legitima que necesita la misma
    aritmetica y no puede construir un motor para hacerla: el estudio de divergencia
    (`backtest/divergence_study.py`) re-tasa una orden YA ARCHIVADA para saber a que
    precio la habria llenado el modelo con la liquidez que vio el backtest. Que las dos
    respuestas salgan de la misma linea es lo que hace que la comparacion mida la
    diferencia de CONTEXTO y no una diferencia de formula.
    """
    slippage_multiplier = slippage_bps / 10_000

    if side == "buy":
        return round(reference_price * (1 + slippage_multiplier), 8)

    return round(reference_price * (1 - slippage_multiplier), 8)


@dataclass(slots=True)
class PaperExecutionConfig:
    fee_rate: float = 0.001
    # Coste plano de REFERENCIA, en bps. El motor ya NO lo cobra: el fill se valora con
    # el modelo de microestructura (`slippage`). Se conserva porque los baselines pasivos
    # (scoring/baselines.py) y la auditoria de costes de la calibracion necesitan un
    # coste unico y comparable; que el baseline pague un plano barato y la estrategia el
    # modelo completo endurece el listado, que es la direccion prudente del error.
    slippage_bps: float = 5.0
    reject_zero_or_negative_price: bool = True
    # Fills parciales activos por defecto: sin techo de capacidad, cualquier tamano se
    # llenaba entero y el backtest podia "operar" mas de lo que el mercado ofrece.
    allow_partial_fills: bool = True
    # Techo de capacidad por barra: fraccion del volumen de la barra que una sola orden
    # puede consumir. Lo que exceda no se llena (fill parcial).
    max_participation: float = 0.10
    enforce_prediction_price_bounds: bool = True
    slippage: SlippageModel = field(default_factory=SlippageModel)

    def __post_init__(self) -> None:
        if self.fee_rate < 0:
            raise ValueError("fee_rate cannot be negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps cannot be negative")
        if not 0 < self.max_participation <= 1:
            raise ValueError("max_participation must be between 0 and 1")
        # El config vive en TOML: `[execution.slippage]` llega como dict.
        if isinstance(self.slippage, dict):
            self.slippage = SlippageModel(**self.slippage)


class PaperExecutionEngine:
    """
    Simulacion de fills en papel.

    Dos decisiones por orden, en este orden:

    1. **Cuanto se llena**: el volumen de la barra pone un techo de capacidad
       (`max_participation`); lo que exceda no se llena. Las ordenes de reduccion
       (`reduce_only`: cierres de posicion) se exceptuan del techo, porque salir es
       obligatorio: se llenan enteras y pagan el impacto de todo el tamano.
    2. **A que precio**: el modelo de microestructura cobra medio spread del simbolo,
       mas volatilidad reciente, mas impacto por el tamano REALMENTE llenado.

    La liquidez se resuelve por una costura inyectable (`liquidity_provider`). Sin ella
    -o para instrumentos sin barras, como los mercados de prediccion- el modelo degrada
    a spread + volatilidad de referencia, sin impacto ni techo.
    """

    def __init__(
        self,
        config: PaperExecutionConfig | None = None,
        liquidity_provider: LiquidityProvider | None = None,
    ) -> None:
        self.config = config or PaperExecutionConfig()
        self.liquidity_provider = liquidity_provider

    def execute(
        self,
        order_request: OrderRequest,
        market_price: float,
    ) -> ExecutionResult:
        self._validate_order_request(order_request)
        self._validate_market_price(order_request, market_price)

        order_id = self._next_order_id()
        snapshot = self._liquidity(order_request)

        filled_size, status = self._resolve_fill_size(order_request, snapshot)
        if filled_size <= 0:
            return self._rejected(order_request, order_id)

        # El deslizamiento se cobra sobre el tamano que DE VERDAD se llena: si la
        # capacidad recorta la orden, se paga el impacto de cruzar esa capacidad, no el
        # de un tamano que nunca llego al libro.
        slippage_bps = self.config.slippage.slippage_bps(
            symbol=order_request.symbol,
            size=filled_size,
            snapshot=snapshot,
            asset_class=order_request.asset_class,
        )

        reference_price = self._resolve_reference_price(order_request, market_price)
        filled_price = self._apply_slippage(reference_price, order_request.side.value, slippage_bps)
        self._validate_filled_price(order_request, filled_price)

        fees = round(filled_price * filled_size * self.config.fee_rate, 8)

        return ExecutionResult(
            success=True,
            status=status,
            message=self._build_message(order_request, filled_size, filled_price, fees, status),
            order_id=order_id,
            filled_price=filled_price,
            filled_size=filled_size,
            fees=fees,
            slippage_bps=slippage_bps,
            venue=order_request.venue,
            asset_class=order_request.asset_class,
            symbol=order_request.symbol,
            instrument_id=order_request.instrument_id,
            outcome=order_request.outcome,
            metadata=dict(order_request.metadata),
        )

    def _validate_order_request(self, order_request: OrderRequest) -> None:
        if order_request.size <= 0:
            raise ValueError("order_request.size must be greater than 0")

        if (
            order_request.order_type == OrderType.LIMIT
            and order_request.limit_price is None
        ):
            raise ValueError("limit orders require limit_price")

    def _validate_market_price(self, order_request: OrderRequest, market_price: float) -> None:
        if self.config.reject_zero_or_negative_price and market_price <= 0:
            raise ValueError("market_price must be greater than 0")

        if (
            self.config.enforce_prediction_price_bounds
            and order_request.asset_class == AssetClass.PREDICTION
            and not 0.0 < market_price < 1.0
        ):
            raise ValueError("prediction market price must be between 0 and 1")

    def _validate_filled_price(self, order_request: OrderRequest, filled_price: float) -> None:
        if (
            self.config.enforce_prediction_price_bounds
            and order_request.asset_class == AssetClass.PREDICTION
            and not 0.0 < filled_price < 1.0
        ):
            raise ValueError("prediction market filled_price must be between 0 and 1")

    def _liquidity(self, order_request: OrderRequest) -> LiquiditySnapshot:
        """Estado de liquidez del simbolo. Los mercados de prediccion no tienen OHLCV:
        no se le pregunta al proveedor por ellos, se cobra solo el spread de su clase."""
        if self.liquidity_provider is None:
            return EMPTY_SNAPSHOT
        if order_request.asset_class == AssetClass.PREDICTION:
            return EMPTY_SNAPSHOT
        return self.liquidity_provider.snapshot(order_request.symbol)

    def _resolve_reference_price(
        self,
        order_request: OrderRequest,
        market_price: float,
    ) -> float:
        if order_request.order_type == OrderType.MARKET:
            return market_price

        assert order_request.limit_price is not None

        if order_request.side.value == "buy":
            if market_price <= order_request.limit_price:
                return market_price
            return order_request.limit_price

        if market_price >= order_request.limit_price:
            return market_price
        return order_request.limit_price

    @staticmethod
    def _apply_slippage(reference_price: float, side: str, slippage_bps: float) -> float:
        return fill_price(reference_price, side, slippage_bps)

    def _resolve_fill_size(
        self,
        order_request: OrderRequest,
        snapshot: LiquiditySnapshot,
    ) -> tuple[float, OrderStatus]:
        """Tamano llenado y estado resultante, segun el techo de capacidad de la barra."""
        requested = order_request.size

        if not self.config.allow_partial_fills or order_request.reduce_only:
            return requested, OrderStatus.FILLED

        volume = snapshot.bar_volume
        if volume is None or volume <= 0:
            # Sin dato de liquidez no se puede afirmar que haya escasez: se llena entero.
            return requested, OrderStatus.FILLED

        capacity = round(volume * self.config.max_participation, 8)
        if capacity <= 0:
            return 0.0, OrderStatus.REJECTED
        if requested <= capacity:
            return requested, OrderStatus.FILLED

        return capacity, OrderStatus.PARTIALLY_FILLED

    def _rejected(self, order_request: OrderRequest, order_id: str) -> ExecutionResult:
        return ExecutionResult(
            success=False,
            status=OrderStatus.REJECTED,
            message=(
                f"Paper order rejected: no capacity for {order_request.symbol} "
                f"(bar volume too thin for any fill)"
            ),
            order_id=order_id,
            filled_price=None,
            filled_size=0.0,
            venue=order_request.venue,
            asset_class=order_request.asset_class,
            symbol=order_request.symbol,
            instrument_id=order_request.instrument_id,
            outcome=order_request.outcome,
            metadata=dict(order_request.metadata),
        )

    def _next_order_id(self) -> str:
        return f"paper-{utc_now():%Y%m%d%H%M%S}-{uuid4().hex[:8]}"

    @staticmethod
    def _build_message(
        order_request: OrderRequest,
        filled_size: float,
        filled_price: float,
        fees: float,
        status: OrderStatus,
    ) -> str:
        instrument_suffix = ""
        if order_request.instrument_id:
            instrument_suffix = f" | instrument_id={order_request.instrument_id}"
        if order_request.outcome:
            instrument_suffix += f" | outcome={order_request.outcome}"
        if status == OrderStatus.PARTIALLY_FILLED:
            instrument_suffix += f" | requested={order_request.size:.8f}"

        return (
            f"Paper order {status.value}: "
            f"{order_request.side.value.upper()} {order_request.symbol} | "
            f"size={filled_size:.8f} | "
            f"price={filled_price:.8f} | "
            f"fees={fees:.8f}"
            f"{instrument_suffix}"
        )
