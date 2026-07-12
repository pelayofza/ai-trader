from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Protocol

import pandas as pd

from ai_trader.app.reports import ReportBuilder
from ai_trader.app.state_store import JsonStateStore
from ai_trader.execution.router import ExecutionRouter
from ai_trader.notifications.base import NullNotifier, Notifier
from ai_trader.risk.engine import PortfolioState, RiskEngine
from ai_trader.shared import bars as bar_schema
from ai_trader.shared.clock import Clock, LiveClock
from ai_trader.shared.instruments import AssetClass, PredictionMarket, Venue
from ai_trader.shared.schemas import (
    ExecutionResult,
    OrderRequest,
    OrderType,
    Position,
    PositionStatus,
    RiskDecision,
    Side,
    Signal,
)

logger = logging.getLogger(__name__)

PREDICTION_PREFIX = "PM::"


class MarketDataReader(Protocol):
    def get_daily_bars(self, symbol: str, start, end) -> pd.DataFrame | None:
        ...

    def get_prediction_midpoint(self, token_id: str) -> float | None:
        ...

    def get_prediction_market(self, slug: str) -> PredictionMarket | None:
        ...

    def detect_asset_class(self, symbol: str) -> AssetClass:
        ...


# Venue por defecto de cada clase de activo, para las estrategias que no declaran
# una identidad de instrumento propia (las de prediccion si lo hacen).
DEFAULT_VENUE: dict[AssetClass, Venue] = {
    AssetClass.CRYPTO: Venue.CCXT,
    AssetClass.STOCK: Venue.ALPACA,
    AssetClass.PREDICTION: Venue.POLYMARKET,
}


class Strategy(Protocol):
    """
    Contrato unico de estrategia.

    `generate_signal` recibe el simbolo y su contexto de mercado y devuelve una Signal
    o None. Toda Signal, venga de donde venga, pasa por el motor de riesgo antes de
    convertirse en orden.
    """

    strategy_id: str

    def supports_symbol(self, symbol: str) -> bool:
        ...

    def generate_signal(self, symbol: str, context) -> Signal | None:
        ...


@dataclass(slots=True)
class RunnerConfig:
    symbols: list[str]
    lookback_days: int = 180
    order_type: OrderType = OrderType.MARKET
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
    # Contiene posiciones abiertas Y cerradas. Antes se llamaba `open_positions`,
    # lo que hacia creer que estaba filtrado.
    positions: list[Position] = field(default_factory=list)
    execution_results: list[ExecutionResult] = field(default_factory=list)
    daily_realized_pnl_usd: float = 0.0
    is_paused: bool = False

    def open_positions(self) -> list[Position]:
        return [p for p in self.positions if p.is_open]

    def build_portfolio_state(self) -> PortfolioState:
        return PortfolioState(
            open_positions=self.open_positions(),
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


class TradingRunner:
    """
    Orquestador del sistema.

    Un ciclo: cerrar lo que toque cerrar, y para cada simbolo del universo resolver
    su contexto de mercado, pedir senal a las estrategias que lo soportan, pasarla
    por el riesgo y, si aprueba, enrutarla al motor de ejecucion de su venue.
    """

    def __init__(
        self,
        config: RunnerConfig,
        market_data_reader: MarketDataReader,
        strategies: list[Strategy],
        risk_engine: RiskEngine,
        execution_router: ExecutionRouter,
        notifier: Notifier | None = None,
        state_store: JsonStateStore | None = None,
        clock: Clock | None = None,
    ) -> None:
        if not strategies:
            raise ValueError("strategies cannot be empty")

        self.config = config
        self.market_data_reader = market_data_reader
        self.strategies = strategies
        self.risk_engine = risk_engine
        self.execution_router = execution_router
        self.notifier = notifier or NullNotifier()
        self.state_store = state_store or JsonStateStore()
        self.clock = clock or LiveClock()

        payload = self.state_store.load()
        self.state = RunnerState(
            positions=payload.get("positions", []),
            execution_results=payload.get("execution_results", []),
            daily_realized_pnl_usd=payload.get("daily_realized_pnl_usd", 0.0),
            is_paused=payload.get("is_paused", False),
        )

    # --- ciclo principal ---

    def run_cycle(self) -> list[ExecutionResult]:
        if not self.config.cycle_enabled:
            self.notifier.warning("Runner cycle is disabled in config.")
            return []

        if self.state.is_paused:
            self.notifier.warning("Runner is paused. Skipping cycle.")
            return []

        closed_positions = self._process_open_positions()

        logger.info(
            "Starting run_cycle | symbols=%s | strategies=%s",
            len(self.config.symbols),
            [s.strategy_id for s in self.strategies],
        )

        cycle_results: list[ExecutionResult] = []
        diagnostics: list[SymbolCycleDiagnostics] = []

        for symbol in self.config.symbols:
            if len(cycle_results) >= self.config.max_trades_per_cycle:
                break

            diag = SymbolCycleDiagnostics(symbol=symbol)
            diagnostics.append(diag)

            try:
                cycle_results.extend(self._process_symbol(symbol, diag))
            except Exception as exc:
                logger.exception("Error while processing symbol=%s", symbol)
                diag.notes.append(f"exception={exc}")
                self.notifier.error(f"Error processing {symbol}: {exc}")

        self.state.execution_results.extend(cycle_results)
        self._persist_state()

        summary = self._build_cycle_summary(cycle_results, diagnostics, closed_positions)
        logger.info(summary)
        self.notifier.info(summary)

        for diag in diagnostics:
            logger.info("Cycle diagnostics | %s", diag)

        return cycle_results

    def _process_symbol(
        self,
        symbol: str,
        diag: SymbolCycleDiagnostics,
    ) -> list[ExecutionResult]:
        if any(p.symbol == symbol for p in self.state.open_positions()):
            diag.notes.append("open_position_exists")
            return []

        if self._symbol_in_cooldown(symbol):
            diag.notes.append("symbol_cooldown")
            return []

        context = self._build_context(symbol, diag)
        if context is None:
            return []

        results: list[ExecutionResult] = []

        for strategy in self.strategies:
            if len(results) >= self.config.max_trades_per_cycle:
                break

            if not strategy.supports_symbol(symbol):
                continue

            diag.strategies_run += 1

            signal = strategy.generate_signal(symbol, context)
            if signal is None:
                diag.notes.append(f"{strategy.strategy_id}:no_signal")
                continue

            diag.signals_generated += 1
            self.notifier.info(self._format_signal(signal))

            # Puerta unica de riesgo. Antes las ordenes de prediccion la esquivaban
            # por completo: se contaban como aprobadas sin evaluarlas siquiera.
            decision = self.risk_engine.evaluate(
                signal=signal,
                portfolio_state=self.state.build_portfolio_state(),
            )

            logger.info(
                "Risk decision | symbol=%s | strategy=%s | approved=%s | reason=%s | size_usd=%s",
                signal.symbol,
                signal.strategy_id,
                decision.approved,
                decision.reason,
                decision.size_usd,
            )

            if not decision.approved:
                diag.risk_rejected += 1
                diag.notes.append(f"{strategy.strategy_id}:risk_rejected:{decision.reason}")
                self.notifier.warning(
                    f"Risk rejected {signal.symbol} ({signal.strategy_id}): {decision.reason}"
                )
                continue

            for warning in decision.warnings:
                self.notifier.warning(f"{signal.symbol}: {warning}")

            diag.risk_approved += 1
            diag.executions_attempted += 1

            result = self._open_position(signal, decision, diag)
            if result is not None:
                results.append(result)

        return results

    def _build_context(self, symbol: str, diag: SymbolCycleDiagnostics):
        """Resuelve el contexto de mercado del simbolo segun su clase de activo."""
        if symbol.upper().startswith(PREDICTION_PREFIX):
            slug = symbol[len(PREDICTION_PREFIX) :].strip().lower()
            if not slug:
                diag.notes.append("invalid_prediction_symbol")
                return None

            market = self.market_data_reader.get_prediction_market(slug)
            if market is None:
                diag.notes.append("prediction_market_not_found")
                self.notifier.warning(f"Prediction market not found: {symbol}")
                return None

            # Gamma devuelve precios que pueden estar rancios. El precio con el que
            # se decide y con el que se ejecuta debe ser el mismo: el midpoint vivo
            # del CLOB. Se refresca aqui, una sola vez por ciclo y por token.
            for token in market.outcomes:
                midpoint = self.market_data_reader.get_prediction_midpoint(token.token_id)
                if midpoint is not None:
                    token.price = midpoint

            if all(token.price is None for token in market.outcomes):
                diag.notes.append("missing_prediction_midpoint")
                return None

            return market

        bars = self._load_bars(symbol)
        if bars is None or bars.empty:
            diag.notes.append("no_bars")
            return None

        diag.bars_loaded = len(bars)
        return bars

    def _open_position(
        self,
        signal: Signal,
        decision: RiskDecision,
        diag: SymbolCycleDiagnostics,
    ) -> ExecutionResult | None:
        order_request = self._build_order_request(signal, decision.size_usd)

        result = self.execution_router.execute(
            order_request,
            reference_price=signal.entry_price,
        )

        logger.info(
            "Execution result | symbol=%s | success=%s | status=%s | price=%s | fees=%s",
            signal.symbol,
            result.success,
            result.status.value,
            result.filled_price,
            result.fees,
        )

        if not (result.success and result.filled_price is not None):
            diag.notes.append(f"{signal.strategy_id}:execution_failed:{result.message}")
            return result

        diag.executions_successful += 1

        position = Position(
            symbol=signal.symbol,
            side=signal.side,
            size=result.filled_size or order_request.size,
            entry_price=result.filled_price,
            opened_at=self.clock.now(),
            strategy_id=signal.strategy_id,
            # El stop y el take profit los fija el riesgo, no la estrategia.
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            entry_fees_usd=result.fees,
            position_id=result.order_id,
            venue=result.venue,
            asset_class=result.asset_class,
            instrument_id=result.instrument_id,
            outcome=result.outcome,
            metadata=dict(order_request.metadata),
        )
        self.state.positions.append(position)
        self._persist_state()

        self.notifier.info(self._format_execution(position, result))
        return result

    # --- gestion de posiciones abiertas ---

    def _process_open_positions(self) -> list[Position]:
        closed: list[Position] = []

        for position in self.state.open_positions():
            mark_price = self._get_mark_price(position)
            if mark_price is None:
                logger.warning("No mark price for open position symbol=%s", position.symbol)
                continue

            reason = self._get_close_reason(position, mark_price)
            if reason is None:
                continue

            self._close_position(position, mark_price, reason)
            closed.append(position)

        if closed:
            self._persist_state()

        return closed

    def _get_close_reason(self, position: Position, mark_price: float) -> str | None:
        # Antes solo se evaluaba el stop en posiciones BUY, asi que un corto nunca
        # habria saltado por stop loss.
        if position.side == Side.BUY:
            if position.stop_loss is not None and mark_price <= position.stop_loss:
                return "stop_loss"
            if position.take_profit is not None and mark_price >= position.take_profit:
                return "take_profit"
        else:
            if position.stop_loss is not None and mark_price >= position.stop_loss:
                return "stop_loss"
            if position.take_profit is not None and mark_price <= position.take_profit:
                return "take_profit"

        holding_days = (self.clock.now() - position.opened_at).days
        if holding_days >= self.config.max_holding_days:
            return "max_holding_days"

        return None

    def _close_position(self, position: Position, exit_price: float, reason: str) -> None:
        # El cierre pasa por el motor de ejecucion, igual que la apertura: si no,
        # la salida no pagaria ni slippage ni comisiones y el PnL seria ficticio.
        closing_order = OrderRequest(
            symbol=position.symbol,
            side=Side.SELL if position.side == Side.BUY else Side.BUY,
            size=position.size,
            order_type=OrderType.MARKET,
            strategy_id=position.strategy_id,
            venue=position.venue,
            asset_class=position.asset_class,
            instrument_id=position.instrument_id,
            outcome=position.outcome,
            metadata={"close_reason": reason, "position_id": position.position_id or ""},
        )

        try:
            result = self.execution_router.execute(closing_order, reference_price=exit_price)
            fill_price = result.filled_price or exit_price
            exit_fees = result.fees
            self.state.execution_results.append(result)
        except Exception as exc:
            logger.exception("Closing order failed | symbol=%s", position.symbol)
            self.notifier.error(f"Closing order failed for {position.symbol}: {exc}")
            fill_price = exit_price
            exit_fees = 0.0

        position.status = PositionStatus.CLOSED
        position.closed_at = self.clock.now()
        position.exit_price = fill_price
        position.exit_fees_usd = exit_fees
        position.close_reason = reason
        position.realized_pnl = round(position.net_pnl_at(fill_price), 2)

        self.state.daily_realized_pnl_usd += position.realized_pnl

        logger.info(
            "Position closed | symbol=%s | reason=%s | entry=%s | exit=%s | net_pnl=%s",
            position.symbol,
            reason,
            position.entry_price,
            fill_price,
            position.realized_pnl,
        )

        self.notifier.info(
            f"Position closed | symbol={position.symbol} | "
            f"side={position.side.value.upper()} | reason={reason} | "
            f"entry={position.entry_price:.4f} | exit={fill_price:.4f} | "
            f"net_pnl={position.realized_pnl:.2f} USD (fees {position.total_fees_usd:.4f})"
        )

    def _symbol_in_cooldown(self, symbol: str) -> bool:
        if self.config.symbol_cooldown_hours <= 0:
            return False

        events = [
            p.closed_at or p.opened_at
            for p in self.state.positions
            if p.symbol == symbol
        ]
        if not events:
            return False

        elapsed = (self.clock.now() - max(events)).total_seconds()
        return elapsed < self.config.symbol_cooldown_hours * 3600

    # --- precios ---

    def _get_mark_price(self, position: Position) -> float | None:
        if (
            position.asset_class == AssetClass.PREDICTION
            and position.venue == Venue.POLYMARKET
            and position.instrument_id
        ):
            return self.market_data_reader.get_prediction_midpoint(position.instrument_id)

        return self._get_last_price(position.symbol)

    def _get_last_price(self, symbol: str) -> float | None:
        end = self.clock.now()
        bars = self.market_data_reader.get_daily_bars(symbol, end - timedelta(days=7), end)
        return bar_schema.last_close(bars) if bars is not None else None

    def _load_bars(self, symbol: str) -> pd.DataFrame | None:
        end = self.clock.now()
        start = end - timedelta(days=self.config.lookback_days)
        return self.market_data_reader.get_daily_bars(symbol, start, end)

    def _resolve_asset_class(self, symbol: str) -> AssetClass:
        if symbol.upper().startswith(PREDICTION_PREFIX):
            return AssetClass.PREDICTION
        return self.market_data_reader.detect_asset_class(symbol)

    def _build_order_request(self, signal: Signal, size_usd: float) -> OrderRequest:
        size_units = round(size_usd / signal.entry_price, 8)

        # Una estrategia puede declarar la identidad del instrumento (las de prediccion
        # lo hacen, porque necesitan el token id). Si no lo hace, la resuelve el
        # orquestador: sin clase de activo el router no sabe a que motor enrutar.
        asset_class = signal.asset_class or self._resolve_asset_class(signal.symbol)
        venue = signal.venue or DEFAULT_VENUE.get(asset_class)

        return OrderRequest(
            symbol=signal.symbol,
            side=signal.side,
            size=size_units,
            order_type=self.config.order_type,
            limit_price=signal.entry_price if self.config.order_type == OrderType.LIMIT else None,
            strategy_id=signal.strategy_id,
            signal_id=signal.signal_id,
            venue=venue,
            asset_class=asset_class,
            instrument_id=signal.instrument_id,
            outcome=signal.outcome,
            metadata={
                "timeframe": signal.timeframe,
                "confidence": signal.confidence,
                "reason": signal.reason,
            },
        )

    # --- control y estado ---

    def pause(self) -> None:
        self.state.is_paused = True
        self._persist_state()
        self.notifier.warning("Trading runner paused.")

    def resume(self) -> None:
        self.state.is_paused = False
        self._persist_state()
        self.notifier.info("Trading runner resumed.")

    def get_positions(self) -> list[Position]:
        return self.state.open_positions()

    def _persist_state(self) -> None:
        self.state_store.save(
            {
                "positions": self.state.positions,
                "execution_results": self.state.execution_results,
                "daily_realized_pnl_usd": self.state.daily_realized_pnl_usd,
                "is_paused": self.state.is_paused,
            }
        )

    # --- informes (delegados a ReportBuilder) ---

    def _reports(self) -> ReportBuilder:
        return ReportBuilder(
            positions=self.state.positions,
            daily_realized_pnl_usd=self.state.daily_realized_pnl_usd,
            is_paused=self.state.is_paused,
            limits=self.risk_engine.limits,
            price_lookup=self._get_mark_price,
        )

    def get_status(self) -> str:
        return self._reports().status()

    def get_positions_report(self) -> str:
        return self._reports().positions_report()

    def get_risk_report(self) -> str:
        return self._reports().risk_report()

    def get_history_report(self) -> str:
        return self._reports().history_report()

    def get_performance_report(self) -> str:
        return self._reports().performance_report()

    def get_symbols_report(self) -> str:
        return self._reports().symbols_report(self.config.symbols)

    # --- formateo de eventos ---

    def _build_cycle_summary(
        self,
        results: list[ExecutionResult],
        diagnostics: list[SymbolCycleDiagnostics],
        closed: list[Position],
    ) -> str:
        return (
            f"Cycle complete | "
            f"symbols={len(self.config.symbols)} | "
            f"closed_positions={len(closed)} | "
            f"signals={sum(d.signals_generated for d in diagnostics)} | "
            f"risk_rejected={sum(d.risk_rejected for d in diagnostics)} | "
            f"executions={len(results)} | "
            f"successful={sum(1 for r in results if r.success)} | "
            f"open_positions={len(self.state.open_positions())}"
        )

    @staticmethod
    def _format_signal(signal: Signal) -> str:
        return (
            f"Signal | strategy={signal.strategy_id} | symbol={signal.symbol} | "
            f"side={signal.side.value.upper()} | confidence={signal.confidence:.2f} | "
            f"entry={signal.entry_price:.4f}"
        )

    @staticmethod
    def _format_execution(position: Position, result: ExecutionResult) -> str:
        return (
            f"Paper execution | strategy={position.strategy_id} | "
            f"symbol={position.symbol} | side={position.side.value.upper()} | "
            f"status={result.status.value} | size={position.size:.8f} | "
            f"price={position.entry_price:.4f} | fees={position.entry_fees_usd:.4f} | "
            f"SL={position.stop_loss:.4f} | TP={position.take_profit:.4f}"
        )
