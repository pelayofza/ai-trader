from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from ai_trader.shared.clock import utc_now
from ai_trader.shared.instruments import AssetClass
from ai_trader.shared.schemas import (
    ExecutionResult,
    OrderRequest,
    OrderStatus,
    OrderType,
)


@dataclass(slots=True)
class PaperExecutionConfig:
    fee_rate: float = 0.001
    slippage_bps: float = 5.0
    reject_zero_or_negative_price: bool = True
    allow_partial_fills: bool = False
    partial_fill_ratio: float = 0.5
    enforce_prediction_price_bounds: bool = True

    def __post_init__(self) -> None:
        if self.fee_rate < 0:
            raise ValueError("fee_rate cannot be negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps cannot be negative")
        if not 0 < self.partial_fill_ratio <= 1:
            raise ValueError("partial_fill_ratio must be between 0 and 1")


class PaperExecutionEngine:
    def __init__(self, config: PaperExecutionConfig | None = None) -> None:
        self.config = config or PaperExecutionConfig()

    def execute(
        self,
        order_request: OrderRequest,
        market_price: float,
    ) -> ExecutionResult:
        self._validate_order_request(order_request)
        self._validate_market_price(order_request, market_price)

        order_id = self._next_order_id()
        reference_price = self._resolve_reference_price(order_request, market_price)
        filled_price = self._apply_slippage(reference_price, order_request.side.value)
        self._validate_filled_price(order_request, filled_price)

        filled_size, status = self._resolve_fill_size(order_request.size)
        fees = round(filled_price * filled_size * self.config.fee_rate, 8)

        return ExecutionResult(
            success=True,
            status=status,
            message=self._build_message(order_request, filled_size, filled_price, fees, status),
            order_id=order_id,
            filled_price=filled_price,
            filled_size=filled_size,
            fees=fees,
            slippage_bps=self.config.slippage_bps,
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

    def _apply_slippage(self, reference_price: float, side: str) -> float:
        slippage_multiplier = self.config.slippage_bps / 10_000

        if side == "buy":
            return round(reference_price * (1 + slippage_multiplier), 8)

        return round(reference_price * (1 - slippage_multiplier), 8)

    def _resolve_fill_size(self, requested_size: float) -> tuple[float, OrderStatus]:
        if not self.config.allow_partial_fills:
            return requested_size, OrderStatus.FILLED

        filled_size = round(requested_size * self.config.partial_fill_ratio, 8)

        if filled_size >= requested_size:
            return requested_size, OrderStatus.FILLED

        return filled_size, OrderStatus.PARTIALLY_FILLED

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

        return (
            f"Paper order {status.value}: "
            f"{order_request.side.value.upper()} {order_request.symbol} | "
            f"size={filled_size:.8f} | "
            f"price={filled_price:.8f} | "
            f"fees={fees:.8f}"
            f"{instrument_suffix}"
        )