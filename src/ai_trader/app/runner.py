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
from ai_trader.app.state_store import JsonStateStore
from ai_trader.shared.schemas import Side

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
        state_store: JsonStateStore | None = None,
    ) -> None:
        if not strategies:
            raise ValueError("strategies cannot be empty")

        self.config = config
        self.market_data_reader = market_data_reader
        self.strategies = strategies
        self.risk_engine = risk_engine
        self.execution_engine = execution_engine
        self.supervisor = supervisor or NullSupervisor()
        self.state_store = state_store or JsonStateStore()
        payload = self.state_store.load()

        if not payload:
            self.state = RunnerState()
        else:
            self.state = RunnerState(
                open_positions=payload.get("open_positions", []),
                execution_results=payload.get("execution_results", []),
                daily_realized_pnl_usd=payload.get("daily_realized_pnl_usd", 0.0),
                is_paused=payload.get("is_paused", False),
            )

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
        self._persist_state()

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
        self._persist_state()
        self.supervisor.send_warning("Trading runner paused.")

    def resume(self) -> None:
        self.state.is_paused = False
        self._persist_state()
        self.supervisor.send_info("Trading runner resumed.")

    def get_status(self) -> str:
        return (
            f"runner_paused={self.state.is_paused} | "
            f"open_positions={len([p for p in self.state.open_positions if p.is_open])} | "
            f"daily_realized_pnl_usd={self.state.daily_realized_pnl_usd:.2f} | "
            f"stored_results={len(self.state.execution_results)}"
        )

    def get_positions(self) -> list[Position]:
        return [position for position in self.state.open_positions if position.is_open]

    def get_positions_report(self) -> str:
        positions = self.get_positions()

        if not positions:
            return "No open positions."

        lines: list[str] = [f"Open positions: {len(positions)}"]
        total_unrealized = 0.0

        for position in positions:
            last_price = self._get_last_price(position.symbol)
            pnl_usd = None
            pnl_pct = None

            if last_price is not None:
                pnl_usd = self._compute_unrealized_pnl_usd(position, last_price)
                pnl_pct = self._compute_unrealized_pnl_pct(position, last_price)
                total_unrealized += pnl_usd

            lines.append("")
            lines.append(f"Symbol: {position.symbol}")
            lines.append(f"Side: {position.side.value.upper()}")
            lines.append(f"Size: {position.size:.8f}")
            lines.append(f"Entry: {position.entry_price:,.2f}")
            lines.append(
                f"Notional at entry: {position.notional_value:,.2f} USD"
            )

            if last_price is None:
                lines.append("Last price: unavailable")
                lines.append("Unrealized PnL: unavailable")
            else:
                lines.append(f"Last price: {last_price:,.2f}")
                lines.append(f"Unrealized PnL: {pnl_usd:,.2f} USD")
                lines.append(f"Unrealized PnL %: {pnl_pct:,.2f}%")

            lines.append(f"Strategy: {position.strategy_id}")

            if position.stop_loss is not None:
                lines.append(f"Stop loss: {position.stop_loss:,.2f}")

            if position.take_profit is not None:
                lines.append(f"Take profit: {position.take_profit:,.2f}")

            lines.append(f"Opened at: {position.opened_at.isoformat()}")

        lines.append("")
        lines.append(f"Total unrealized PnL: {total_unrealized:,.2f} USD")

        return "\n".join(lines)

    def get_risk_report(self) -> str:
        portfolio_state = self.state.build_portfolio_state()
        limits = self.risk_engine.limits
        positions = self.get_positions()

        lines: list[str] = [
            f"Runner paused: {self.state.is_paused}",
            f"Open positions: {portfolio_state.open_positions_count()} / {limits.max_open_positions}",
            (
                f"Daily realized PnL: "
                f"{portfolio_state.daily_realized_pnl_usd:,.2f} USD / "
                f"-{limits.max_daily_loss_usd:,.2f} USD limit"
            ),
            (
                f"Total exposure: "
                f"{portfolio_state.total_exposure_usd:,.2f} USD / "
                f"{limits.max_total_exposure_usd:,.2f} USD"
            ),
            (
                f"Max position size: "
                f"{limits.max_position_size_usd:,.2f} USD"
            ),
            (
                f"Confidence range: "
                f"{limits.min_confidence_per_trade:.2f} - "
                f"{limits.max_confidence_per_trade:.2f}"
            ),
            "",
            "Exposure by symbol:",
        ]

        if not positions:
            lines.append("None")
        else:
            for symbol in sorted({position.symbol for position in positions}):
                symbol_exposure = portfolio_state.symbol_exposure_usd(symbol)
                lines.append(
                    f"{symbol}: {symbol_exposure:,.2f} USD / {limits.max_symbol_exposure_usd:,.2f} USD"
                )

        return "\n".join(lines)

    def _get_last_price(self, symbol: str) -> float | None:
        end = utc_now()
        start = end - timedelta(days=30)
        bars = self.market_data_reader.get_daily_bars(symbol, start, end)

        if bars is None or bars.empty:
            return None

        last = bars.iloc[-1]

        for candidate in ("close", "Close"):
            if candidate in bars.columns:
                return float(last[candidate])

        return None

    def _persist_state(self) -> None:
        self.state_store.save({
            "open_positions": self.state.open_positions,
            "execution_results": self.state.execution_results,
            "daily_realized_pnl_usd": self.state.daily_realized_pnl_usd,
            "is_paused": self.state.is_paused,
        })
    
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
        self._persist_state()

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

    @staticmethod
    def _compute_unrealized_pnl_usd(position: Position, last_price: float) -> float:
        if position.side == Side.BUY:
            return round((last_price - position.entry_price) * position.size, 2)

        return round((position.entry_price - last_price) * position.size, 2)

    @staticmethod
    def _compute_unrealized_pnl_pct(position: Position, last_price: float) -> float:
        if position.side == Side.BUY:
            return round(((last_price / position.entry_price) - 1.0) * 100.0, 2)

        return round(((position.entry_price / last_price) - 1.0) * 100.0, 2)