from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol

from ai_trader.execution.paper import PaperExecutionEngine
from ai_trader.execution.polymarket_paper import PolymarketPaperExecutionEngine
from ai_trader.risk.engine import PortfolioState, RiskEngine
from ai_trader.shared.instruments import AssetClass, Venue
from ai_trader.shared.schemas import (
    ExecutionResult,
    OrderRequest,
    OrderType,
    Position,
    PositionStatus,
    Signal,
    Side,
)

from ai_trader.app.state_store import JsonStateStore


logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MarketDataReader(Protocol):
    def get_daily_bars(self, symbol: str, start: datetime, end: datetime):
        ...

    def get_prediction_midpoint(self, token_id: str) -> float | None:
        ...

    def get_prediction_market(self, slug: str):
        ...

class Strategy(Protocol):
    strategy_id: str

    def generate_signal(self, symbol: str, bars) -> Signal | None:
        ...

class PredictionStrategy(Protocol):
    strategy_id: str

    def supports_symbol(self, symbol: str) -> bool:
        ...

    def build_order_request(self, market, midpoint: float) -> OrderRequest | None:
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
    max_holding_days: int = 10
    symbol_cooldown_hours: int = 12
    max_trades_per_cycle: int = 5

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("symbols cannot be empty")
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be greater than 0")
        if self.max_holding_days <= 0:
            raise ValueError("max_holding_days must be greater than 0")


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
        polymarket_execution_engine: PolymarketPaperExecutionEngine | None = None,
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
        self.polymarket_execution_engine = (
            polymarket_execution_engine or PolymarketPaperExecutionEngine()
        )
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
                
        closed_positions = self._process_open_positions()

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

            if len(cycle_results) >= self.config.max_trades_per_cycle:
                break

            try:
                logger.info("Processing symbol=%s", symbol)

                if symbol.startswith("PM::"):
                    symbol_results = self._process_prediction_symbol(symbol, diag)
                else:
                    symbol_results = self._process_symbol(symbol, diag)

                cycle_results.extend(symbol_results)

            except Exception as exc:
                logger.exception("Error while processing symbol=%s", symbol)
                diag.notes.append(f"exception={exc}")
                self.supervisor.send_error(f"Error processing {symbol}: {exc}")

        self.state.execution_results.extend(cycle_results)
        self._persist_state()

        summary = self._build_cycle_summary(cycle_results, diagnostics, closed_positions)
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

    def _has_open_prediction_position(self, instrument_id: str) -> bool:
        for position in self.state.open_positions:
            if not position.is_open:
                continue
            if position.instrument_id == instrument_id:
                return True
        return False

    def _get_position_mark_price(self, position: Position) -> float | None:
        if (
            position.asset_class == AssetClass.PREDICTION
            and position.venue == Venue.POLYMARKET
            and position.instrument_id
        ):
            return self.market_data_reader.get_prediction_midpoint(position.instrument_id)

        return self._get_last_price(position.symbol)

    def get_positions_report(self) -> str:
        positions = self.get_positions()

        if not positions:
            return "No open positions."

        lines: list[str] = [f"Open positions: {len(positions)}"]
        total_unrealized = 0.0

        for position in positions:
            last_price = self._get_position_mark_price(position)
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
            lines.append(f"Entry: {position.entry_price:,.4f}")
            lines.append(
                f"Notional at entry: {position.notional_value:,.2f} USD"
            )

            if last_price is None:
                lines.append("Last price: unavailable")
                lines.append("Unrealized PnL: unavailable")
            else:
                lines.append(f"Last price: {last_price:,.4f}")
                lines.append(f"Unrealized PnL: {pnl_usd:,.2f} USD")
                lines.append(f"Unrealized PnL %: {pnl_pct:,.2f}%")

            lines.append(f"Strategy: {position.strategy_id}")

            if position.venue is not None:
                lines.append(f"Venue: {position.venue.value}")

            if position.asset_class is not None:
                lines.append(f"Asset class: {position.asset_class.value}")

            if position.instrument_id:
                lines.append(f"Instrument ID: {position.instrument_id}")

            if position.outcome:
                lines.append(f"Outcome: {position.outcome}")

            if position.stop_loss is not None:
                lines.append(f"Stop loss: {position.stop_loss:,.2f}")

            if position.take_profit is not None:
                lines.append(f"Take profit: {position.take_profit:,.2f}")

            lines.append(f"Opened at: {position.opened_at.isoformat()}")

        lines.append("")
        lines.append(f"Total unrealized PnL: {total_unrealized:,.2f} USD")

        return "\n".join(lines)

    def get_risk_report(self) -> str:
        limits = self.risk_engine.limits
        open_positions = self.get_positions()

        total_exposure = sum(position.size * self._get_last_price(position.symbol) for position in open_positions)

        exposure_by_symbol: dict[str, float] = {}
        positions_by_symbol: dict[str, int] = {}

        for position in open_positions:
            exposure_by_symbol[position.symbol] = (
                exposure_by_symbol.get(position.symbol, 0.0) + position.size * self._get_last_price(position.symbol)
            )

        lines: list[str] = [
            f"Runner paused: {self.state.is_paused}",
            f"Open positions: {len(open_positions)} / {limits.max_open_positions}",
            (
                f"Daily realized PnL: "
                f"{self.state.daily_realized_pnl_usd:,.2f} USD / "
                f"-{limits.max_daily_loss_usd:,.2f} USD limit"
            ),
            (
                f"Total current exposure: "
                f"{total_exposure:,.2f} USD / "
                f"{limits.max_total_exposure_usd:,.2f} USD"
            ),
            f"Max position size: {limits.max_position_size_usd:,.2f} USD",
        ]

        if limits.max_confidence_per_trade >= 1.0:
            lines.append(
                f"Minimum confidence required: {limits.min_confidence_per_trade:.2f}"
            )
        else:
            lines.append(
                "Accepted confidence range: "
                f"{limits.min_confidence_per_trade:.2f} - "
                f"{limits.max_confidence_per_trade:.2f}"
            )

        lines.append("")
        lines.append("Current exposure by symbol:")

        if not open_positions:
            lines.append("None")
        else:
            for symbol in sorted(exposure_by_symbol):
                lines.append(
                    f"{symbol}: "
                    f"{exposure_by_symbol[symbol]:,.2f} USD / "
                    f"{limits.max_symbol_exposure_usd:,.2f} USD"
                )

        return "\n".join(lines)

    def get_history_report(self) -> str:
        closed_positions = [
            position
            for position in self.state.open_positions
            if position.status == PositionStatus.CLOSED
        ]

        if not closed_positions:
            return "No closed positions yet."

        closed_positions = sorted(
            closed_positions,
            key=lambda position: position.closed_at or position.opened_at,
            reverse=True,
        )

        total_realized = sum(position.realized_pnl or 0.0 for position in closed_positions)
        wins = sum(1 for position in closed_positions if (position.realized_pnl or 0.0) > 0)
        losses = sum(1 for position in closed_positions if (position.realized_pnl or 0.0) < 0)
        total = len(closed_positions)
        win_rate = (wins / total * 100.0) if total > 0 else 0.0

        lines = [
            f"Closed positions: {total}",
            f"Realized PnL: {total_realized:,.2f} USD",
            f"Wins: {wins}",
            f"Losses: {losses}",
            f"Win rate: {win_rate:.2f}%",
            "",
            "Last closed positions:",
        ]

        for position in closed_positions[:10]:
            lines.append("")
            lines.append(f"Symbol: {position.symbol}")
            lines.append(f"Side: {position.side.value.upper()}")
            lines.append(f"Strategy: {position.strategy_id}")
            lines.append(f"Entry: {position.entry_price:,.2f}")
            lines.append(f"Exit: {(position.exit_price or 0.0):,.2f}")
            lines.append(f"Size: {position.size:.8f}")
            lines.append(f"PnL: {(position.realized_pnl or 0.0):,.2f} USD")
            lines.append(f"Reason: {position.close_reason or 'unknown'}")
            lines.append(f"Opened at: {position.opened_at.isoformat()}")
            lines.append(
                f"Closed at: {position.closed_at.isoformat() if position.closed_at else 'n/a'}"
            )

        return "\n".join(lines)

    def get_performance_report(self) -> str:
        closed_positions = [
            position
            for position in self.state.open_positions
            if position.status == PositionStatus.CLOSED
        ]
        open_positions = self.get_positions()

        realized_pnl = sum(position.realized_pnl or 0.0 for position in closed_positions)

        unrealized_pnl = 0.0
        for position in open_positions:
            last_price = self._get_position_mark_price(position)
            if last_price is None:
                continue
            unrealized_pnl += self._compute_unrealized_pnl_usd(position, last_price)

        winners = [p for p in closed_positions if (p.realized_pnl or 0.0) > 0]
        losers = [p for p in closed_positions if (p.realized_pnl or 0.0) < 0]

        gross_profit = sum(p.realized_pnl or 0.0 for p in winners)
        gross_loss = abs(sum(p.realized_pnl or 0.0 for p in losers))

        win_rate = (len(winners) / len(closed_positions) * 100.0) if closed_positions else 0.0
        avg_win = (gross_profit / len(winners)) if winners else 0.0
        avg_loss = (sum((p.realized_pnl or 0.0) for p in losers) / len(losers)) if losers else 0.0

        if gross_loss == 0:
            profit_factor = "n/a"
        else:
            profit_factor = f"{(gross_profit / gross_loss):.2f}"

        holding_days_values = []
        for position in closed_positions:
            if position.closed_at is not None:
                holding_days_values.append((position.closed_at - position.opened_at).days)

        avg_holding_days = (
            sum(holding_days_values) / len(holding_days_values)
            if holding_days_values
            else 0.0
        )

        net_pnl = realized_pnl + unrealized_pnl

        return "\n".join([
            f"Closed trades: {len(closed_positions)}",
            f"Open positions: {len(open_positions)}",
            f"Realized PnL: {realized_pnl:,.2f} USD",
            f"Unrealized PnL: {unrealized_pnl:,.2f} USD",
            f"Net PnL: {net_pnl:,.2f} USD",
            f"Winners: {len(winners)}",
            f"Losers: {len(losers)}",
            f"Win rate: {win_rate:.2f}%",
            f"Average win: {avg_win:,.2f} USD",
            f"Average loss: {avg_loss:,.2f} USD",
            f"Profit factor: {profit_factor}",
            f"Average holding days: {avg_holding_days:.2f}",
        ])

    def get_symbols_report(self) -> str:
        lines = ["Symbols report:"]

        for symbol in self.config.symbols:
            open_positions = [
                position
                for position in self.get_positions()
                if position.symbol == symbol
            ]
            closed_positions = [
                position
                for position in self.state.open_positions
                if position.symbol == symbol and position.status == PositionStatus.CLOSED
            ]

            last_price = self._get_last_price(symbol)
            realized_pnl = sum(position.realized_pnl or 0.0 for position in closed_positions)

            unrealized_pnl = 0.0
            current_exposure = 0.0

            for position in open_positions:
                current_exposure += position.notional_value
                if last_price is not None:
                    unrealized_pnl += self._compute_unrealized_pnl_usd(position, last_price)

            lines.append("")
            lines.append(f"Symbol: {symbol}")
            lines.append(f"Open positions: {len(open_positions)}")
            lines.append(f"Closed positions: {len(closed_positions)}")
            lines.append(f"Current exposure: {current_exposure:,.2f} USD")
            lines.append(
                f"Last price: {last_price:,.2f}" if last_price is not None else "Last price: unavailable"
            )
            lines.append(f"Realized PnL: {realized_pnl:,.2f} USD")
            lines.append(f"Unrealized PnL: {unrealized_pnl:,.2f} USD")

        return "\n".join(lines)

    def submit_order(
        self,
        order_request: OrderRequest,
        *,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> ExecutionResult:
        execution_result = self._execute_order(order_request=order_request)

        self.state.execution_results.append(execution_result)

        if execution_result.success and execution_result.filled_price is not None:
            assert execution_result.filled_size is not None

            position = Position(
                symbol=order_request.symbol,
                side=order_request.side,
                size=execution_result.filled_size,
                entry_price=execution_result.filled_price,
                opened_at=utc_now(),
                strategy_id=order_request.strategy_id,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_id=execution_result.order_id,
                venue=execution_result.venue,
                asset_class=execution_result.asset_class,
                instrument_id=execution_result.instrument_id,
                outcome=execution_result.outcome,
                metadata=dict(order_request.metadata),
            )
            self.state.open_positions.append(position)

        self._persist_state()
        return execution_result

    def _get_last_price(self, symbol: str) -> float | None:
        end = utc_now()
        start = end - timedelta(days=7)
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
    
    def _process_open_positions(self) -> list[Position]:
        closed_positions: list[Position] = []

        for position in self.get_positions():
            last_price = self._get_position_mark_price(position)

            if last_price is None:
                logger.warning(
                    "Could not load last price for open position symbol=%s",
                    position.symbol,
                )
                continue

            close_reason = self._get_close_reason(position, last_price)

            if close_reason is None:
                continue

            logger.info(
                "Closing position | symbol=%s | position_id=%s | reason=%s | last_price=%s",
                position.symbol,
                position.position_id,
                close_reason,
                last_price,
            )

            self._close_position(
                position=position,
                exit_price=last_price,
                reason=close_reason,
            )
            closed_positions.append(position)

        if closed_positions:
            self._persist_state()

        return closed_positions


    def _get_close_reason(self, position: Position, last_price: float) -> str | None:
        if position.side == Side.BUY:
            if position.stop_loss is not None and last_price <= position.stop_loss:
                return "stop_loss"

            if position.take_profit is not None and last_price >= position.take_profit:
                return "take_profit"

        holding_days = (utc_now() - position.opened_at).days
        if holding_days >= self.config.max_holding_days:
            return "max_holding_days"

        return None


    def _symbol_in_cooldown(self, symbol: str) -> bool:
        if self.config.symbol_cooldown_hours <= 0:
            return False

        last_event_at: datetime | None = None

        for position in self.state.open_positions:
            if position.symbol != symbol:
                continue

            candidate = position.closed_at or position.opened_at

            if last_event_at is None or candidate > last_event_at:
                last_event_at = candidate

        if last_event_at is None:
            return False

        cooldown_seconds = self.config.symbol_cooldown_hours * 3600
        elapsed_seconds = (utc_now() - last_event_at).total_seconds()

        return elapsed_seconds < cooldown_seconds

    def _close_position(
        self,
        position: Position,
        exit_price: float,
        reason: str,
    ) -> None:
        position.status = PositionStatus.CLOSED
        position.closed_at = utc_now()
        position.exit_price = exit_price
        position.close_reason = reason

        realized_pnl = self._compute_realized_pnl(position, exit_price)
        position.realized_pnl = realized_pnl
        self.state.daily_realized_pnl_usd += realized_pnl

        logger.info(
            (
                "Position closed | symbol=%s | position_id=%s | reason=%s | "
                "entry_price=%s | exit_price=%s | size=%s | realized_pnl=%s"
            ),
            position.symbol,
            position.position_id,
            reason,
            position.entry_price,
            exit_price,
            position.size,
            realized_pnl,
        )

        self.supervisor.send_info(
            self._format_position_closed_message(position, reason)
        )


    @staticmethod
    def _compute_realized_pnl(position: Position, exit_price: float) -> float:
        if position.side == Side.BUY:
            return round((exit_price - position.entry_price) * position.size, 2)

        return round((position.entry_price - exit_price) * position.size, 2)


    @staticmethod
    def _format_position_closed_message(position: Position, reason: str) -> str:
        return (
            f"Position closed | "
            f"symbol={position.symbol} | "
            f"side={position.side.value.upper()} | "
            f"reason={reason} | "
            f"entry={position.entry_price:.2f} | "
            f"exit={position.exit_price or 0:.2f} | "
            f"pnl={position.realized_pnl or 0:.2f} USD"
        )

    def _process_prediction_symbol(
        self,
        symbol: str,
        diag: SymbolCycleDiagnostics,
    ) -> list[ExecutionResult]:
        slug = symbol.removeprefix("PM::").strip().lower()
        if not slug:
            diag.notes.append("invalid_prediction_symbol")
            return []

        market = self.market_data_reader.get_prediction_market(slug)
        if market is None:
            diag.notes.append("prediction_market_not_found")
            self.supervisor.send_warning(f"Prediction market not found for symbol {symbol}")
            return []

        results: list[ExecutionResult] = []

        for strategy in self.strategies:
            if len(results) >= self.config.max_trades_per_cycle:
                break

            if not hasattr(strategy, "supports_symbol"):
                continue
            if not hasattr(strategy, "build_order_request"):
                continue

            if not strategy.supports_symbol(symbol):
                continue

            diag.strategies_run += 1

            outcome_name = getattr(strategy, "config", None)
            if outcome_name is not None:
                outcome_name = getattr(strategy.config, "outcome", "yes")
            else:
                outcome_name = "yes"

            token = market.yes_token if str(outcome_name).lower() == "yes" else market.no_token
            if token is None:
                diag.notes.append("missing_outcome_token")
                continue

            midpoint = self.market_data_reader.get_prediction_midpoint(token.token_id)
            if midpoint is None:
                diag.notes.append("missing_prediction_midpoint")
                self.supervisor.send_warning(
                    f"Could not resolve midpoint for prediction market {symbol}"
                )
                continue

            if self._has_open_prediction_position(token.token_id):
                diag.notes.append("prediction_position_already_open")
                continue

            order_request = strategy.build_order_request(market, midpoint)
            if order_request is None:
                diag.notes.append("prediction_strategy_no_signal")
                continue

            diag.signals_generated += 1
            diag.risk_approved += 1
            diag.executions_attempted += 1

            execution_result = self.submit_order(order_request)
            results.append(execution_result)

            if execution_result.success:
                diag.executions_successful += 1

            if len(results) >= self.config.max_trades_per_cycle:
                break

        return results

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

        if any(position.symbol == symbol for position in self.get_positions()):
            logger.info("Skipping symbol=%s because there is already an open position", symbol)
            diag.notes.append("open_position_exists")
            return []

        if self._symbol_in_cooldown(symbol):
            logger.info("Skipping symbol=%s because it is in cooldown", symbol)
            diag.notes.append("symbol_cooldown")
            return []

        for strategy in self.strategies:
            if not hasattr(strategy, "generate_signal"):
                continue
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

            execution_result = self._execute_order(
                order_request=order_request,
                reference_price=signal.entry_price,
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
                    f"{strategy.strategy_id}:execution_failed:{execution_result.message}"
                )

        return results

    def _execute_order(
        self,
        order_request: OrderRequest,
        reference_price: float | None = None,
    ) -> ExecutionResult:
        if (
            order_request.asset_class == AssetClass.PREDICTION
            and order_request.venue == Venue.POLYMARKET
        ):
            return self.polymarket_execution_engine.execute(order_request)

        if reference_price is None:
            raise ValueError("reference_price is required for non-prediction paper orders")

        return self.execution_engine.execute(
            order_request=order_request,
            market_price=reference_price,
        )

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
            venue=execution_result.venue,
            asset_class=execution_result.asset_class,
            instrument_id=execution_result.instrument_id,
            outcome=execution_result.outcome,
            metadata=dict(order_request.metadata),
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
        closed_positions: list[Position],
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
            f"closed_positions={len(closed_positions)} | "
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