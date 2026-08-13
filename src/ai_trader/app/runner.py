from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Protocol

import pandas as pd

from ai_trader.app.accounting import realized_pnl_usd, unrealized_pnl_usd
from ai_trader.app.journal import (
    STATUS_DISABLED,
    STATUS_PAUSED,
    STATUS_RAN,
    Journal,
    NullJournal,
    closed_record,
    cycle_record,
    exit_record,
    fill_record,
    open_record,
    order_record,
    risk_record,
    signal_record,
)
from ai_trader.app.reports import ReportBuilder
from ai_trader.app.state_store import JsonStateStore
from ai_trader.execution.market_model import CloseMarketModel, MarketModel
from ai_trader.execution.router import ExecutionRouter
from ai_trader.notifications.base import NullNotifier, Notifier
from ai_trader.risk.engine import PortfolioState, RiskEngine
from ai_trader.shared.clock import Clock, LiveClock
from ai_trader.shared.instruments import (
    PREDICTION_PREFIX,
    AssetClass,
    PredictionMarket,
    Venue,
)
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
    # Mercados de prediccion que se OBSERVAN sin operarlos: por cada ciclo se anota su
    # midpoint en el diario. VACIA por defecto, y eso importa: con la lista vacia no se
    # hace ni una llamada de red y el ciclo es identico al de antes. No es universo -no
    # genera senales, no pasa por riesgo, no abre posicion-; es la unica forma de que
    # dentro de unos meses exista un historico de Polymarket, que hoy no se puede
    # comprar ni descargar.
    prediction_watchlist: list[str] = field(default_factory=list)

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
    """
    Lo que le paso a un simbolo en un ciclo: los contadores (el resumen) y los eventos
    (el detalle que va al diario).

    Los contadores ya existian y son lo que se registra en el log. Los tres eventos son
    lo que hace auditable el ciclo: una senal sin su confianza, un rechazo sin su motivo
    o un fill sin su `slippage_bps` no sirven para medir la divergencia live-vs-backtest.
    Van aqui y no en una estructura nueva porque `diag` ya viaja a los dos sitios donde
    se generan (`_process_symbol` y `_open_position`), asi que no hay que enhebrar nada.
    """

    symbol: str
    bars_loaded: int = 0
    strategies_run: int = 0
    signals_generated: int = 0
    risk_approved: int = 0
    risk_rejected: int = 0
    executions_attempted: int = 0
    executions_successful: int = 0
    notes: list[str] = field(default_factory=list)
    # Eventos, ya serializados por `app/journal.py` (el esquema del diario vive alli).
    signals: list[dict] = field(default_factory=list)
    risk: list[dict] = field(default_factory=list)
    orders: list[dict] = field(default_factory=list)
    fills: list[dict] = field(default_factory=list)

    def as_record(self) -> dict:
        """La forma del simbolo dentro de la linea del diario."""
        return {
            "symbol": self.symbol,
            "bars_loaded": self.bars_loaded,
            "strategies_run": self.strategies_run,
            "notes": list(self.notes),
            "signals": list(self.signals),
            "risk": list(self.risk),
            "orders": list(self.orders),
            "fills": list(self.fills),
        }


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
        market_model: MarketModel | None = None,
        starting_equity: float | None = None,
        journal: Journal | None = None,
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
        # Costura de precios. Por defecto, cierre-contra-cierre (comportamiento en vivo);
        # el backtest inyecta el modelo intrabar realista.
        self.market_model = market_model or CloseMarketModel(market_data_reader, self.clock)
        # Capital inicial de la cuenta. Si es None, el riesgo dimensiona en absoluto.
        self.starting_equity = starting_equity
        # Diario de ciclos. Por defecto NO escribe: el backtest conduce este mismo runner
        # miles de veces por ventana y ahi el diario seria ruido. Quien opera en vivo
        # (`main.build_runner`) inyecta un `CycleJournal`.
        self.journal: Journal = journal or NullJournal()
        # Fills de SALIDA del ciclo en curso. Se acumulan aqui porque el cierre ocurre en
        # `_close_position`, que no devuelve el resultado de la ejecucion a nadie, y se
        # vacian al escribir la linea. Solo se llenan si hay diario detras.
        self._exit_fills: list[dict] = []

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
            self._write_journal(STATUS_DISABLED, [], [])
            return []

        if self.state.is_paused:
            self.notifier.warning("Runner is paused. Skipping cycle.")
            # Un ciclo pausado TAMBIEN deja linea. Es lo que distingue "el proceso sigue
            # vivo y no opera" de "el proceso se cayo", que desde fuera son iguales.
            self._write_journal(STATUS_PAUSED, [], [])
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

        # El diario, DESPUES de persistir el estado y antes de notificar: si el proceso
        # muere aqui, la foto y la pelicula cuentan lo mismo.
        self._write_journal(STATUS_RAN, diagnostics, closed_positions)

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
            diag.signals.append(signal_record(signal))
            self.notifier.info(self._format_signal(signal))

            # Puerta unica de riesgo. Antes las ordenes de prediccion la esquivaban
            # por completo: se contaban como aprobadas sin evaluarlas siquiera.
            decision = self.risk_engine.evaluate(
                signal=signal,
                portfolio_state=self._build_portfolio_state(),
            )
            diag.risk.append(risk_record(signal, decision))

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
        reference_price = self.market_model.entry_reference_price(signal.symbol, signal)
        if reference_price is None:
            diag.notes.append(f"{signal.strategy_id}:no_entry_price")
            return None

        order_request = self._build_order_request(signal, decision.size_usd, reference_price)
        # El instante EXACTO en que la decision queda tomada: despues de que el riesgo
        # haya aprobado y el precio de referencia este fijado, justo antes de enrutar.
        # Restarselo a `executed_at` da la latencia real de esta orden.
        decided_at = self.clock.now()
        diag.orders.append(
            order_record(
                signal.strategy_id,
                order_request.side.value,
                order_request.size,
                reference_price,
                decided_at=decided_at,
            )
        )

        result = self.execution_router.execute(
            order_request,
            reference_price=reference_price,
        )
        diag.fills.append(fill_record(result))

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
            # Primero, salida por precio (stop-loss / take-profit). El modelo de mercado
            # decide si se ha disparado y a que precio se sale.
            price_exit = self.market_model.price_exit(position)
            if price_exit is not None:
                reason, exit_price = price_exit
                self._close_position(position, exit_price, reason)
                closed.append(position)
                continue

            # Salida por tiempo. Es la misma para vivo y backtest, asi que la lleva
            # el runner, no el modelo de mercado.
            holding_days = (self.clock.now() - position.opened_at).days
            if holding_days >= self.config.max_holding_days:
                mark = self.market_model.mark_price(position)
                if mark is None:
                    logger.warning("No mark price for open position symbol=%s", position.symbol)
                    continue
                self._close_position(position, mark, "max_holding_days")
                closed.append(position)

        if closed:
            self._persist_state()

        return closed

    def force_liquidate(self, reason: str = "backtest_end") -> list[Position]:
        """Cierra todo lo abierto al precio de marca actual. Lo usa el backtest al
        cerrar cada ventana para que el equity final sea limpio y comparable."""
        closed: list[Position] = []
        for position in self.state.open_positions():
            mark = self.market_model.mark_price(position)
            if mark is None:
                continue
            self._close_position(position, mark, reason)
            closed.append(position)
        if closed:
            self._persist_state()
        return closed

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
            # Salida: exenta del techo de capacidad. Si se llenara parcialmente, la
            # posicion quedaria cerrada en el estado pero solo pagada a medias.
            reduce_only=True,
            metadata={"close_reason": reason, "position_id": position.position_id or ""},
        )

        decided_at = self.clock.now()
        try:
            result = self.execution_router.execute(closing_order, reference_price=exit_price)
            fill_price = result.filled_price or exit_price
            exit_fees = result.fees
            self.state.execution_results.append(result)
            if self._journal_active():
                self._exit_fills.append(
                    exit_record(
                        position,
                        result,
                        reason,
                        reference_price=exit_price,
                        decided_at=decided_at,
                    )
                )
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

    # --- equity y estado de cartera ---

    def _build_portfolio_state(self) -> PortfolioState:
        base = self.state.build_portfolio_state()

        # Sin capital inicial (modo vivo), el riesgo dimensiona en absoluto y no hace
        # falta calcular equity.
        if self.starting_equity is None:
            return base

        open_positions = self.state.open_positions()
        # Las dos sumas viven en `app/accounting.py`: `reports.py` las tenia repetidas
        # con el cuerpo identico y el diario habria sido la tercera copia.
        realized_cum = realized_pnl_usd(self.state.positions)
        deployed_cost = sum(p.size * p.entry_price + p.entry_fees_usd for p in open_positions)
        unrealized = unrealized_pnl_usd(self.state.positions, self.market_model.mark_price)

        base.equity_usd = self.starting_equity + realized_cum + unrealized
        base.available_cash_usd = max(self.starting_equity + realized_cum - deployed_cost, 0.0)
        return base

    def current_equity(self) -> float | None:
        """Equity actual marcado a mercado. None en modo vivo (sin cuenta)."""
        return self._build_portfolio_state().equity_usd

    def reset_daily_pnl(self) -> None:
        """Reinicia el PnL realizado del dia. El backtest lo llama al inicio de cada
        dia simulado para que el limite de perdida diaria tenga semantica diaria real."""
        self.state.daily_realized_pnl_usd = 0.0

    def _load_bars(self, symbol: str) -> pd.DataFrame | None:
        end = self.clock.now()
        start = end - timedelta(days=self.config.lookback_days)
        return self.market_data_reader.get_daily_bars(symbol, start, end)

    def _resolve_asset_class(self, symbol: str) -> AssetClass:
        if symbol.upper().startswith(PREDICTION_PREFIX):
            return AssetClass.PREDICTION
        return self.market_data_reader.detect_asset_class(symbol)

    def _build_order_request(
        self,
        signal: Signal,
        size_usd: float,
        reference_price: float,
    ) -> OrderRequest:
        # El tamano en unidades se calcula sobre el precio de entrada real (que en
        # backtest es el open de hoy, no el cierre de la barra de decision).
        size_units = round(size_usd / reference_price, 8)

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
            limit_price=reference_price if self.config.order_type == OrderType.LIMIT else None,
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

    # --- diario de ciclos ---

    def _journal_active(self) -> bool:
        """Si no hay diario detras no se recoge nada ni se construye la linea. El motor
        de backtest conduce ESTE mismo runner miles de veces por ventana: pagar el marcado
        a mercado de cada posicion abierta para tirar el resultado seria un coste real en
        el camino caliente."""
        return not getattr(self.journal, "is_null", False)

    def _write_journal(
        self,
        status: str,
        diagnostics: list[SymbolCycleDiagnostics],
        closed: list[Position],
    ) -> None:
        """
        Deja la huella del ciclo. NUNCA tumba el ciclo: el diario es auditoria, no
        operativa, y un disco lleno no puede impedir que se cierre una posicion.

        Un ciclo que NO se ejecuto (pausado o deshabilitado) deja una linea mas ligera:
        se registra que el proceso esta vivo, pero no se marca a mercado ni se consulta
        la watchlist. Pausar tiene que seguir significando que el sistema no toca la red.
        """
        exits, self._exit_fills = self._exit_fills, []
        if not self._journal_active():
            return

        try:
            marked = status == STATUS_RAN
            open_positions = self.state.open_positions()
            portfolio = (
                self._build_portfolio_state()
                if marked
                else self.state.build_portfolio_state()
            )
            record = cycle_record(
                timestamp=self.clock.now(),
                status=status,
                symbols=[d.as_record() for d in diagnostics],
                opened=(
                    [open_record(p, self.market_model.mark_price(p)) for p in open_positions]
                    if marked
                    else []
                ),
                closed=[closed_record(p) for p in closed],
                exits=exits,
                equity_usd=portfolio.equity_usd,
                realized_pnl_usd=realized_pnl_usd(self.state.positions),
                unrealized_pnl_usd=(
                    unrealized_pnl_usd(self.state.positions, self.market_model.mark_price)
                    if marked
                    else None
                ),
                # La exposicion que el motor de riesgo compara contra
                # `max_total_exposure_usd`, no una tercera definicion de exposicion.
                exposure_usd=portfolio.total_exposure_usd,
                open_positions=len(open_positions),
                max_open_positions=self.risk_engine.limits.max_open_positions,
                daily_realized_pnl_usd=self.state.daily_realized_pnl_usd,
                watchlist=self._observe_watchlist() if marked else [],
            )
            self.journal.append(record)
        except Exception:
            logger.exception("No se pudo escribir el diario de ciclos")

    def _observe_watchlist(self) -> list[dict]:
        """
        Midpoints de los mercados de prediccion OBSERVADOS, sin operar ninguno.

        Con la lista vacia (el defecto) devuelve `[]` sin tocar la red. Un fallo por
        mercado se anota y no corta el resto: es lectura opcional, no operativa.
        """
        out: list[dict] = []
        for slug in self.config.prediction_watchlist:
            try:
                market = self.market_data_reader.get_prediction_market(slug)
                if market is None:
                    out.append({"slug": slug, "error": "market_not_found"})
                    continue
                for token in market.outcomes:
                    midpoint = self.market_data_reader.get_prediction_midpoint(token.token_id)
                    out.append(
                        {
                            "slug": slug,
                            "question": market.question,
                            "outcome": token.outcome,
                            "token_id": token.token_id,
                            # El midpoint VIVO del CLOB; `token.price` viene de Gamma y
                            # puede estar rancio (mismo motivo que en `_build_context`).
                            "midpoint": midpoint,
                            "gamma_price": token.price,
                            "closed": market.closed,
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - la watchlist nunca corta el ciclo
                logger.warning("Watchlist: fallo al leer %s (%s)", slug, exc)
                out.append({"slug": slug, "error": str(exc)})
        return out

    # --- informes (delegados a ReportBuilder) ---

    def _reports(self) -> ReportBuilder:
        return ReportBuilder(
            positions=self.state.positions,
            daily_realized_pnl_usd=self.state.daily_realized_pnl_usd,
            is_paused=self.state.is_paused,
            limits=self.risk_engine.limits,
            price_lookup=self.market_model.mark_price,
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
