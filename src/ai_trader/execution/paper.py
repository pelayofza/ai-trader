from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import count

from ai_trader.shared.schemas import (
    ExecutionResult,
    OrderRequest,
    OrderStatus,
    OrderType,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class PaperExecutionConfig:
    fee_rate: float = 0.001
    slippage_bps: float = 5.0
    reject_zero_or_negative_price: bool = True
    allow_partial_fills: bool = False
    partial_fill_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.fee_rate < 0:
            raise ValueError("fee_rate cannot be negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps cannot be negative")
        if not 0 < self.partial_fill_ratio <= 1:
            raise ValueError("partial_fill_ratio must be between 0 and 1")


@dataclass(slots=True)
class PaperFill:
    order_id: str
    symbol: str
    side: str
    requested_size: float
    filled_size: float
    requested_price: float
    filled_price: float
    fees: float
    slippage_bps: float
    order_type: str
    strategy_id: str
    signal_id: str | None
    status: str
    executed_at: datetime = field(default_factory=utc_now)


class PaperExecutionEngine:
    def __init__(self, config: PaperExecutionConfig | None = None) -> None:
        self.config = config or PaperExecutionConfig()
        self._fills: list[PaperFill] = []
        self._order_sequence = count(1)

    def execute(
        self,
        order_request: OrderRequest,
        market_price: float,
    ) -> ExecutionResult:
        self._validate_order_request(order_request)
        self._validate_market_price(market_price)

        order_id = self._next_order_id()

        reference_price = self._resolve_reference_price(order_request, market_price)
        filled_price = self._apply_slippage(reference_price, order_request.side.value)
        filled_size, status = self._resolve_fill_size(order_request.size)

        fees = round(filled_price * filled_size * self.config.fee_rate, 8)

        fill = PaperFill(
            order_id=order_id,
            symbol=order_request.symbol,
            side=order_request.side.value,
            requested_size=order_request.size,
            filled_size=filled_size,
            requested_price=reference_price,
            filled_price=filled_price,
            fees=fees,
            slippage_bps=self.config.slippage_bps,
            order_type=order_request.order_type.value,
            strategy_id=order_request.strategy_id,
            signal_id=order_request.signal_id,
            status=status.value,
        )
        self._fills.append(fill)

        return ExecutionResult(
            success=True,
            status=status,
            message=self._build_message(order_request, filled_size, filled_price, fees, status),
            order_id=order_id,
            filled_price=filled_price,
            filled_size=filled_size,
            fees=fees,
            slippage_bps=self.config.slippage_bps,
        )

    def list_fills(self) -> list[PaperFill]:
        return list(self._fills)

    def get_fill(self, order_id: str) -> PaperFill | None:
        for fill in self._fills:
            if fill.order_id == order_id:
                return fill
        return None

    def clear_fills(self) -> None:
        self._fills.clear()

    def _validate_order_request(self, order_request: OrderRequest) -> None:
        if order_request.size <= 0:
            raise ValueError("order_request.size must be greater than 0")

        if (
            order_request.order_type == OrderType.LIMIT
            and order_request.limit_price is None
        ):
            raise ValueError("limit orders require limit_price")

    def _validate_market_price(self, market_price: float) -> None:
        if self.config.reject_zero_or_negative_price and market_price <= 0:
            raise ValueError("market_price must be greater than 0")

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
        return f"paper-{next(self._order_sequence):06d}"

    @staticmethod
    def _build_message(
        order_request: OrderRequest,
        filled_size: float,
        filled_price: float,
        fees: float,
        status: OrderStatus,
    ) -> str:
        return (
            f"Paper order {status.value}: "
            f"{order_request.side.value.upper()} {order_request.symbol} | "
            f"size={filled_size:.8f} | "
            f"price={filled_price:.8f} | "
            f"fees={fees:.8f}"
        )