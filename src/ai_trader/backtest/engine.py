from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from ai_trader.app.runner import TradingRunner
from ai_trader.app.state_store import InMemoryStateStore
from ai_trader.backtest.metrics import EquityPoint, PerformanceMetrics, compute_metrics
from ai_trader.config import AppConfig
from ai_trader.data.backtest_source import HistoricalDataSource
from ai_trader.execution.market_model import IntrabarMarketModel
from ai_trader.execution.paper import PaperExecutionEngine
from ai_trader.execution.polymarket_paper import PolymarketPaperExecutionEngine
from ai_trader.execution.router import ExecutionRouter
from ai_trader.notifications.base import NullNotifier
from ai_trader.risk.engine import RiskEngine
from ai_trader.shared.clock import HistoricalClock
from ai_trader.shared.schemas import Position
from ai_trader.strategies.registry import build_strategy

logger = logging.getLogger(__name__)

DEFAULT_STARTING_EQUITY = 10_000.0
DEFAULT_SPLIT_RATIO = 0.7


@dataclass(slots=True)
class WindowResult:
    label: str
    start: datetime
    end: datetime
    metrics: PerformanceMetrics
    equity_curve: list[EquityPoint]
    trades: list[Position]

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "start": self.start.date().isoformat(),
            "end": self.end.date().isoformat(),
            "metrics": self.metrics.as_dict(),
        }


@dataclass(slots=True)
class BacktestResult:
    train: WindowResult
    test: WindowResult
    starting_equity: float

    @property
    def headline_score(self) -> float:
        """Metrica con la que se puntua la estrategia: Calmar OUT-OF-SAMPLE (test).
        Es lo que el RL debe optimizar; el train solo sirve de referencia de ajuste."""
        return self.test.metrics.calmar

    def as_dict(self) -> dict:
        return {
            "starting_equity": self.starting_equity,
            "headline_score_calmar_test": round(self.headline_score, 3),
            "train": self.train.as_dict(),
            "test": self.test.as_dict(),
        }


class BacktestEngine:
    """
    Conduce el TradingRunner real sobre historico.

    No reimplementa nada del sistema: inyecta reloj simulado, datos con anti
    look-ahead, modelo de mercado intrabar y estado en memoria, y llama a run_cycle()
    un dia por vez. Lo que testeas es exactamente lo que opera en vivo.
    """

    def __init__(
        self,
        config: AppConfig,
        market_data_service,
        starting_equity: float = DEFAULT_STARTING_EQUITY,
    ) -> None:
        self.config = config
        self.market_data_service = market_data_service
        self.starting_equity = starting_equity

    def run(
        self,
        start: datetime,
        end: datetime,
        *,
        split_ratio: float = DEFAULT_SPLIT_RATIO,
        split_date: datetime | None = None,
    ) -> BacktestResult:
        if end <= start:
            raise ValueError("end must be after start")

        cutoff = self._resolve_cutoff(start, end, split_ratio, split_date)
        logger.info(
            "Backtest | train=[%s, %s) | test=[%s, %s]",
            start.date(), cutoff.date(), cutoff.date(), end.date(),
        )

        # Se precarga una vez, con calentamiento suficiente para que el primer dia de
        # train ya tenga lookback completo. Cada ventana usa una fuente con su reloj.
        warmup = timedelta(days=self.config.runner.lookback_days + 30)
        bars = HistoricalDataSource.fetch_bars(
            self.market_data_service,
            self.config.runner.symbols,
            start - warmup,
            end,
        )
        if not bars:
            raise ValueError("No historical bars available for the requested universe/range")

        train = self._run_window("train", bars, start, cutoff)
        test = self._run_window("test", bars, cutoff, end)

        return BacktestResult(train=train, test=test, starting_equity=self.starting_equity)

    def _resolve_cutoff(
        self,
        start: datetime,
        end: datetime,
        split_ratio: float,
        split_date: datetime | None,
    ) -> datetime:
        if split_date is not None:
            if not start < split_date < end:
                raise ValueError("split_date must fall strictly between start and end")
            return split_date

        if not 0.0 < split_ratio < 1.0:
            raise ValueError("split_ratio must be between 0 and 1")

        span = end - start
        return start + timedelta(days=int(span.days * split_ratio))

    def _run_window(
        self,
        label: str,
        bars: dict,
        start: datetime,
        end: datetime,
    ) -> WindowResult:
        clock = HistoricalClock(start)
        source = HistoricalDataSource(bars, clock)

        days = [d for d in source.trading_days(start, end)]
        if len(days) < 2:
            raise ValueError(
                f"Window '{label}' has {len(days)} trading day(s); need at least 2. "
                "Widen the date range or check data availability."
            )

        runner = self._build_runner(source, clock)

        curve: list[EquityPoint] = []
        for i, day in enumerate(days):
            clock.set(day)
            runner.reset_daily_pnl()
            runner.run_cycle()

            # Al ultimo dia se liquida todo lo abierto para que el equity final sea
            # limpio y comparable entre estrategias.
            if i == len(days) - 1:
                runner.force_liquidate("window_end")

            equity = runner.current_equity()
            curve.append(EquityPoint(day=day.to_pydatetime(), equity=float(equity)))

        closed = [p for p in runner.state.positions if not p.is_open]
        metrics = compute_metrics(curve, closed)

        logger.info(
            "Window '%s' done | trades=%s | return=%.2f%% | maxDD=%.2f%% | calmar=%.3f",
            label, metrics.num_trades, metrics.total_return_pct,
            metrics.max_drawdown_pct, metrics.calmar,
        )

        return WindowResult(
            label=label,
            start=days[0].to_pydatetime(),
            end=days[-1].to_pydatetime(),
            metrics=metrics,
            equity_curve=curve,
            trades=closed,
        )

    def _build_runner(self, source: HistoricalDataSource, clock: HistoricalClock) -> TradingRunner:
        strategies = [
            build_strategy(spec.type, spec.params, strategy_id=spec.id)
            for spec in self.config.strategies
        ]
        paper_engine = PaperExecutionEngine(self.config.execution)
        router = ExecutionRouter.paper(
            spot_engine=paper_engine,
            prediction_engine=PolymarketPaperExecutionEngine(paper_engine=paper_engine),
        )

        return TradingRunner(
            config=self.config.runner,
            market_data_reader=source,
            strategies=strategies,
            risk_engine=RiskEngine(self.config.risk),
            execution_router=router,
            notifier=NullNotifier(),
            state_store=InMemoryStateStore(),
            clock=clock,
            market_model=IntrabarMarketModel(source, clock),
            starting_equity=self.starting_equity,
        )


def parse_date(value: str) -> datetime:
    """Parsea YYYY-MM-DD a datetime UTC medianoche."""
    ts = pd.Timestamp(value).to_pydatetime()
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)
