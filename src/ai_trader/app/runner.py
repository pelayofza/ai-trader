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
    Signal,
)

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MarketDataReader(Protocol):
    def get_daily_bars(self, symbol: str, start: datetime, end: datetime):
        ...


class Strategy(Protocol):
    strategy_id: str

    def generate_signal(self, symbol: str, bars) -> Signal | None:
        ...


class Supervisor(Protocol):
    def send_info(self, message: str) -> None:
        ...

    def send_warning(self, message: str) -> None:
        ...

    def send_error(self, message: str) -> None:
        ...


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


@dataclass(slots=True)
class SymbolCycleDiagnostics:
    symbol: str
    bars_loaded: int = 0
    strategies_run: int = 0
    signals_generated: int = 0
    risk_approved: int = 0
    risk_rejected: int = 0
    executions_attempted: int = 0
    executions_successful: int = 0
    notes: list[str] = field(default_factory=list)


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

        logger.info(
            "Starting run_cycle | symbols=%s | lookback_days=%s | strategies=%s",
            self.config.symbols,
            self.config.lookback_days,
            [strategy.strategy_id for strategy in self.strategies],
        )

        cycle_results: list[ExecutionResult] = []
        diagnostics: list[SymbolCycleDiagnostics] = []

        for symbol in self.config.symbols:
            diag = SymbolCycleDiagnostics(symbol=symbol)
            diagnostics.append(diag)

            try:
                logger.info("Processing symbol=%s", symbol)
                symbol_results = self._process_symbol(symbol, diag)
                cycle_results.extend(symbol_results)

            except Exception as exc:
                logger.exception("Error while processing symbol=%s", symbol)
                diag.notes.append(f"exception={exc}")
                self.supervisor.send_error(f"Error processing {symbol}: {exc}")

        self.state.execution_results.extend(cycle_results)

        summary = self._build_cycle_summary(cycle_results, diagnostics)
        logger.info(summary)
        self.supervisor.send_info(summary)

        for diag in diagnostics:
            logger.info(
                (
                    "Cycle diagnostics | symbol=%s | bars=%s | strategies_run=%s | "
                    "signals=%s | risk_approved=%s | risk_rejected=%s | "
                    "executions_attempted=%s | executions_successful=%s | notes=%s"
                ),
                diag.symbol,
                diag.bars_loaded,
                diag.strategies_run,
                diag.signals_generated,
                diag.risk_approved,
                diag.risk_rejected,
                diag.executions_attempted,
                diag.executions_successful,
                diag.notes,
            )

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

    def _process_symbol(
        self,
        symbol: str,
        diag: SymbolCycleDiagnostics,
    ) -> list[ExecutionResult]:
        bars = self._load_bars(symbol)

        if bars is None or len(bars) == 0:
            diag.notes.append("no_bars")
            self.supervisor.send_warning(f"No bars available for {symbol}.")
            return []

        diag.bars_loaded = len(bars)

        logger.info(
            "Bars loaded | symbol=%s | rows=%s | columns=%s",
            symbol,
            len(bars),
            list(getattr(bars, "columns", [])),
        )

        results: list[ExecutionResult] = []

        for strategy in self.strategies:
            diag.strategies_run += 1

            logger.info(
                "Running strategy | symbol=%s | strategy=%s",
                symbol,
                strategy.strategy_id,
            )

            signal = strategy.generate_signal(symbol, bars)

            if signal is None:
                logger.info(
                    "No signal | symbol=%s | strategy=%s",
                    symbol,
                    strategy.strategy_id,
                )
                diag.notes.append(f"{strategy.strategy_id}:no_signal")
                continue

            diag.signals_generated += 1

            logger.info(
                (
                    "Signal generated | symbol=%s | strategy=%s | side=%s | "
                    "confidence=%.2f | entry=%.6f | stop_loss=%s | take_profit=%s"
                ),
                signal.symbol,
                signal.strategy_id,
                signal.side.value,
                signal.confidence,
                signal.entry_price,
                signal.stop_loss,
                signal.take_profit,
            )

            self.supervisor.send_info(self._format_signal_message(signal))

            risk_decision = self.risk_engine.evaluate(
                signal=signal,
                portfolio_state=self.state.build_portfolio_state(),
            )

            logger.info(
                "Risk decision | symbol=%s | strategy=%s | approved=%s | reason=%s | size_usd=%s",
                signal.symbol,
                signal.strategy_id,
                risk_decision.approved,
                risk_decision.reason,
                risk_decision.size_usd,
            )

            if not risk_decision.approved:
                diag.risk_rejected += 1
                diag.notes.append(
                    f"{strategy.strategy_id}:risk_rejected:{risk_decision.reason}"
                )
                self.supervisor.send_warning(
                    f"Risk rejected {signal.symbol} ({signal.strategy_id}): {risk_decision.reason}"
                )
                continue

            diag.risk_approved += 1

            order_request = self._build_order_request(signal, risk_decision.size_usd)

            logger.info(
                (
                    "Submitting paper order | symbol=%s | strategy=%s | "
                    "size_usd=%.2f | size_units=%.8f | order_type=%s"
                ),
                signal.symbol,
                signal.strategy_id,
                risk_decision.size_usd,
                order_request.size,
                order_request.order_type.value,
            )

            diag.executions_attempted += 1

            execution_result = self.execution_engine.execute(
                order_request=order_request,
                market_price=signal.entry_price,
            )

            logger.info(
                (
                    "Execution result | symbol=%s | strategy=%s | success=%s | "
                    "status=%s | filled_size=%s | filled_price=%s | fees=%s | reason=%s"
                ),
                signal.symbol,
                signal.strategy_id,
                execution_result.success,
                execution_result.status.value,
                execution_result.filled_size,
                execution_result.filled_price,
                execution_result.fees,
                execution_result.message,
            )

            results.append(execution_result)

            if execution_result.success and execution_result.filled_price is not None:
                diag.executions_successful += 1
                self._register_position(
                    signal=signal,
                    order_request=order_request,
                    execution_result=execution_result,
                )
                self.supervisor.send_info(
                    self._format_execution_message(signal, order_request, execution_result)
                )
            else:
                diag.notes.append(
                    f"{strategy.strategy_id}:execution_failed:{execution_result.reason}"
                )

        return results

    def _load_bars(self, symbol: str):
        end = utc_now()
        start = end - timedelta(days=self.config.lookback_days)

        logger.info(
            "Loading bars | symbol=%s | start=%s | end=%s",
            symbol,
            start.isoformat(),
            end.isoformat(),
        )

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

        logger.info(
            "Position registered | symbol=%s | strategy=%s | position_id=%s | size=%s | entry_price=%s",
            signal.symbol,
            signal.strategy_id,
            position.position_id,
            position.size,
            position.entry_price,
        )

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

    def _build_cycle_summary(
        self,
        cycle_results: list[ExecutionResult],
        diagnostics: list[SymbolCycleDiagnostics],
    ) -> str:
        filled = sum(1 for result in cycle_results if result.success)
        total_signals = sum(diag.signals_generated for diag in diagnostics)
        total_risk_rejected = sum(diag.risk_rejected for diag in diagnostics)
        total_no_signal = sum(
            1
            for diag in diagnostics
            for note in diag.notes
            if note.endswith(":no_signal")
        )

        return (
            f"Cycle complete | "
            f"symbols={len(self.config.symbols)} | "
            f"signals={total_signals} | "
            f"risk_rejected={total_risk_rejected} | "
            f"no_signal_events={total_no_signal} | "
            f"executions={len(cycle_results)} | "
            f"successful={filled} | "
            f"open_positions={len([p for p in self.state.open_positions if p.is_open])}"
        )