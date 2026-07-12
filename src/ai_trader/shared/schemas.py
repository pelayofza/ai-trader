from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ai_trader.shared.clock import utc_now
from ai_trader.shared.instruments import AssetClass, Venue


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass(slots=True)
class Signal:
    strategy_id: str
    symbol: str
    timeframe: str
    timestamp: datetime
    side: Side
    confidence: float
    entry_price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    reason: str = ""
    features: dict[str, Any] = field(default_factory=dict)
    signal_id: str | None = None
    # Identidad del instrumento. Permiten que una senal de mercado de prediccion
    # viaje por el mismo camino que una de cripto: estrategia -> riesgo -> ejecucion.
    venue: Venue | None = None
    asset_class: AssetClass | None = None
    instrument_id: str | None = None
    outcome: str | None = None

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ValueError("strategy_id cannot be empty")
        if not self.symbol:
            raise ValueError("symbol cannot be empty")
        if not self.timeframe:
            raise ValueError("timeframe cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.entry_price <= 0:
            raise ValueError("entry_price must be greater than 0")
        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError("stop_loss must be greater than 0")
        if self.take_profit is not None and self.take_profit <= 0:
            raise ValueError("take_profit must be greater than 0")


@dataclass(slots=True)
class RiskDecision:
    approved: bool
    size_usd: float
    reason: str
    stop_loss: float | None = None
    take_profit: float | None = None
    warnings: list[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.size_usd < 0:
            raise ValueError("size_usd cannot be negative")
        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError("stop_loss must be greater than 0")
        if self.take_profit is not None and self.take_profit <= 0:
            raise ValueError("take_profit must be greater than 0")


@dataclass(slots=True)
class OrderRequest:
    symbol: str
    side: Side
    size: float
    order_type: OrderType
    strategy_id: str
    limit_price: float | None = None
    signal_id: str | None = None
    submitted_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)
    venue: Venue | None = None
    asset_class: AssetClass | None = None
    instrument_id: str | None = None
    outcome: str | None = None

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol cannot be empty")
        if not self.strategy_id:
            raise ValueError("strategy_id cannot be empty")
        if self.size <= 0:
            raise ValueError("size must be greater than 0")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit_price is required for limit orders")
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("limit_price must be greater than 0")
        if self.instrument_id is not None and not self.instrument_id.strip():
            raise ValueError("instrument_id cannot be blank")
        if self.outcome is not None and not self.outcome.strip():
            raise ValueError("outcome cannot be blank")


@dataclass(slots=True)
class ExecutionResult:
    success: bool
    status: OrderStatus
    message: str
    order_id: str | None = None
    filled_price: float | None = None
    filled_size: float | None = None
    executed_at: datetime = field(default_factory=utc_now)
    fees: float = 0.0
    slippage_bps: float = 0.0
    venue: Venue | None = None
    asset_class: AssetClass | None = None
    symbol: str | None = None
    instrument_id: str | None = None
    outcome: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.filled_price is not None and self.filled_price <= 0:
            raise ValueError("filled_price must be greater than 0")
        if self.filled_size is not None and self.filled_size < 0:
            raise ValueError("filled_size cannot be negative")
        if self.fees < 0:
            raise ValueError("fees cannot be negative")
        if self.instrument_id is not None and not self.instrument_id.strip():
            raise ValueError("instrument_id cannot be blank")
        if self.outcome is not None and not self.outcome.strip():
            raise ValueError("outcome cannot be blank")


@dataclass(slots=True)
class Position:
    symbol: str
    side: Side
    size: float
    entry_price: float
    opened_at: datetime
    strategy_id: str
    status: PositionStatus = PositionStatus.OPEN
    position_id: str | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    closed_at: datetime | None = None
    exit_price: float | None = None
    realized_pnl: float | None = None
    close_reason: str | None = None
    # Comisiones pagadas. Se computaban en la ejecucion pero nunca se restaban del
    # PnL, asi que todo resultado publicado era optimista.
    entry_fees_usd: float = 0.0
    exit_fees_usd: float = 0.0
    venue: Venue | None = None
    asset_class: AssetClass | None = None
    instrument_id: str | None = None
    outcome: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol cannot be empty")
        if not self.strategy_id:
            raise ValueError("strategy_id cannot be empty")
        if self.size <= 0:
            raise ValueError("size must be greater than 0")
        if self.entry_price <= 0:
            raise ValueError("entry_price must be greater than 0")
        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError("stop_loss must be greater than 0")
        if self.take_profit is not None and self.take_profit <= 0:
            raise ValueError("take_profit must be greater than 0")
        if self.instrument_id is not None and not self.instrument_id.strip():
            raise ValueError("instrument_id cannot be blank")
        if self.outcome is not None and not self.outcome.strip():
            raise ValueError("outcome cannot be blank")

    @property
    def is_open(self) -> bool:
        return self.status == PositionStatus.OPEN

    @property
    def notional_value(self) -> float:
        return self.size * self.entry_price

    @property
    def total_fees_usd(self) -> float:
        return self.entry_fees_usd + self.exit_fees_usd

    def gross_pnl_at(self, price: float) -> float:
        direction = 1.0 if self.side == Side.BUY else -1.0
        return direction * (price - self.entry_price) * self.size

    def net_pnl_at(self, price: float) -> float:
        """PnL neto de comisiones. Es el unico numero que se debe publicar."""
        return self.gross_pnl_at(price) - self.total_fees_usd