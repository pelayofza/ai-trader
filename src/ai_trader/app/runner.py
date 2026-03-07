from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol

from ai_trader.execution.paper import PaperExecutionEngine
from ai_trader.risk.engine import PortfolioState, RiskEngine
from ai_trader.shared.schemas import (
    ExecutionResult,
    OrderRequest,
    OrderType,
    Position,
    Side,
    Signal,
)


logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MarketDataReader(Protocol):
    def get_daily_bars(self, symbol: str, start: datetime, end: datetime): ...


class Strategy(Protocol):
    strategy_id: str

    def generate_signal(self, symbol: str, bars) -> Signal | None: ...


class Supervisor(Protocol):
    def send_info(self, message: str) -> None: ...
    def send_warning(self, message: str) -> None: ...
    def send_error(self, message: str) -> None: ...


@dataclass(slots=True)
class RunnerConfig:
    symbols: list[str]
    lookback_days: int = 180
    order_type: OrderType = OrderType.MARKET
    timeframe_label: str = "1d"
    cycle_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("symbols cannot be empty")
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be greater than 0")


@dataclass(slots=True)
class RunnerState:
    open_positions: list[Position] = field(default_factory=list)
    execution_results: list[ExecutionResult] = field(default_factory=list)
    daily_realized_pnl_usd: float = 0.0
    is_paused: bool = False

    def build_portfolio_state(self) -> PortfolioState:
        return PortfolioState(
            open_positions=self.open_positions,
            daily_realized_pnl_usd=self.daily_realized_pnl_usd,
        )


class NullSupervisor:
    def send_info(self, message: str) -> None:
        logger.info(message)

    def send_warning(self, message: str) -> None:
        logger.warning(message)

    def send_error(self, message: str) -> None:
        logger.error(message)


class TradingRunner:
    def __init__(
        self,
        config: RunnerConfig,
        market_data_reader: MarketDataReader,
        strategies: list[Strategy],
        risk_engine: RiskEngine,
        execution_engine: PaperExecutionEngine,
        supervisor: Supervisor | None = None,
    ) -> None:
        if not strategies:
            raise ValueError("strategies cannot be empty")

        self.config = config
        self.market_data_reader = market_data_reader
        self.strategies = strategies
        self.risk_engine = risk_engine
        self.execution_engine = execution_engine
        self.supervisor = supervisor or NullSupervisor()
        self.state = RunnerState()

    def run_cycle(self) -> list[ExecutionResult]:
        if not self.config.cycle_enabled:
            self.supervisor.send_warning("Runner cycle is disabled in config.")
            return []

        if self.state.is_paused:
            self.supervisor.send_warning("Runner is paused. Skipping cycle.")
            return []

        cycle_results: list[ExecutionResult] = []

        for symbol in self.config.symbols:
            try:
                symbol_results = self._process_symbol(symbol)
                cycle_results.extend(symbol_results)
            except Exception as exc:
                logger.exception("Error while processing symbol %s", symbol)
                self.supervisor.send_error(
                    f"Error processing {symbol}: {exc}"
                )

        self.state.execution_results.extend(cycle_results)

        self.supervisor.send_info(self._build_cycle_summary(cycle_results))
        return cycle_results

    def pause(self) -> None:
        self.state.is_paused = True
        self.supervisor.send_warning("Trading runner paused.")

    def resume(self) -> None:
        self.state.is_paused = False
        self.supervisor.send_info("Trading runner resumed.")

    def get_status(self) -> str:
        return (
            f"runner_paused={self.state.is_paused} | "
            f"open_positions={len([p for p in self.state.open_positions if p.is_open])} | "
            f"daily_realized_pnl_usd={self.state.daily_realized_pnl_usd:.2f} | "
            f"stored_results={len(self.state.execution_results)}"
        )

    def _process_symbol(self, symbol: str) -> list[ExecutionResult]:
        bars = self._load_bars(symbol)
        if bars is None or len(bars) == 0:
            self.supervisor.send_warning(f"No bars available for {symbol}.")
            return []

        results: list[ExecutionResult] = []

        for strategy in self.strategies:
            signal = strategy.generate_signal(symbol, bars)
            if signal is None:
                logger.info("No signal for %s from strategy %s", symbol, strategy.strategy_id)
                continue

            self.supervisor.send_info(self._format_signal_message(signal))

            risk_decision = self.risk_engine.evaluate(
                signal=signal,
                portfolio_state=self.state.build_portfolio_state(),
            )

            if not risk_decision.approved:
                self.supervisor.send_warning(
                    f"Risk rejected {signal.symbol} ({signal.strategy_id}): {risk_decision.reason}"
                )
                continue

            order_request = self._build_order_request(signal, risk_decision.size_usd)
            execution_result = self.execution_engine.execute(
                order_request=order_request,
                market_price=signal.entry_price,
            )

            results.append(execution_result)

            if execution_result.success and execution_result.filled_price is not None:
                self._register_position(
                    signal=signal,
                    order_request=order_request,
                    execution_result=execution_result,
                )
                self.supervisor.send_info(
                    self._format_execution_message(signal, order_request, execution_result)
                )

        return results

    def _load_bars(self, symbol: str):
        end = utc_now()
        start = end - timedelta(days=self.config.lookback_days)
        return self.market_data_reader.get_daily_bars(symbol, start, end)

    def _build_order_request(self, signal: Signal, size_usd: float) -> OrderRequest:
        size_units = round(size_usd / signal.entry_price, 8)

        limit_price = None
        if self.config.order_type == OrderType.LIMIT:
            limit_price = signal.entry_price

        return OrderRequest(
            symbol=signal.symbol,
            side=signal.side,
            size=size_units,
            order_type=self.config.order_type,
            limit_price=limit_price,
            strategy_id=signal.strategy_id,
            signal_id=signal.signal_id,
            metadata={
                "timeframe": signal.timeframe,
                "confidence": signal.confidence,
                "reason": signal.reason,
            },
        )

    def _register_position(
        self,
        signal: Signal,
        order_request: OrderRequest,
        execution_result: ExecutionResult,
    ) -> None:
        assert execution_result.filled_price is not None
        assert execution_result.filled_size is not None

        position = Position(
            symbol=signal.symbol,
            side=signal.side,
            size=execution_result.filled_size,
            entry_price=execution_result.filled_price,
            opened_at=utc_now(),
            strategy_id=signal.strategy_id,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            position_id=execution_result.order_id,
        )
        self.state.open_positions.append(position)

    @staticmethod
    def _format_signal_message(signal: Signal) -> str:
        return (
            f"Signal detected | "
            f"strategy={signal.strategy_id} | "
            f"symbol={signal.symbol} | "
            f"side={signal.side.value.upper()} | "
            f"confidence={signal.confidence:.2f} | "
            f"entry={signal.entry_price:.2f}"
        )

    @staticmethod
    def _format_execution_message(
        signal: Signal,
        order_request: OrderRequest,
        execution_result: ExecutionResult,
    ) -> str:
        return (
            f"Paper execution | "
            f"strategy={signal.strategy_id} | "
            f"symbol={signal.symbol} | "
            f"side={order_request.side.value.upper()} | "
            f"status={execution_result.status.value} | "
            f"size={execution_result.filled_size or 0:.8f} | "
            f"price={execution_result.filled_price or 0:.2f}"
        )

    def _build_cycle_summary(self, cycle_results: list[ExecutionResult]) -> str:
        filled = sum(1 for result in cycle_results if result.success)
        return (
            f"Cycle complete | "
            f"symbols={len(self.config.symbols)} | "
            f"executions={len(cycle_results)} | "
            f"successful={filled} | "
            f"open_positions={len([p for p in self.state.open_positions if p.is_open])}"
        )