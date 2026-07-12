from __future__ import annotations

import pandas as pd
import pytest

from ai_trader.app.runner import RunnerConfig, TradingRunner
from ai_trader.app.state_store import JsonStateStore
from ai_trader.execution.paper import PaperExecutionConfig, PaperExecutionEngine
from ai_trader.execution.polymarket_paper import PolymarketPaperExecutionEngine
from ai_trader.execution.router import ExecutionRouter
from ai_trader.risk.engine import RiskEngine
from ai_trader.shared.instruments import AssetClass, OutcomeToken, PredictionMarket, Venue
from ai_trader.strategies.polymarket_threshold import (
    PolymarketThresholdConfig,
    PolymarketThresholdStrategy,
)
from tests.conftest import build_bars


class FakeMarketData:
    """Datos de mercado en memoria. Ningun test toca la red."""

    def __init__(
        self,
        bars: dict[str, pd.DataFrame] | None = None,
        markets: dict[str, PredictionMarket] | None = None,
        midpoints: dict[str, float] | None = None,
    ) -> None:
        self.bars = bars or {}
        self.markets = markets or {}
        self.midpoints = midpoints or {}

    def get_daily_bars(self, symbol, start, end):
        return self.bars.get(symbol)

    def get_prediction_market(self, slug):
        return self.markets.get(slug)

    def get_prediction_midpoint(self, token_id):
        return self.midpoints.get(token_id)

    def detect_asset_class(self, symbol):
        if symbol.upper().startswith("PM::"):
            return AssetClass.PREDICTION
        return AssetClass.CRYPTO if "/" in symbol else AssetClass.STOCK


class SpyRiskEngine(RiskEngine):
    """Envuelve el motor real para contar cuantas senales lo atraviesan."""

    def __init__(self, limits):
        super().__init__(limits)
        self.evaluated = []

    def evaluate(self, signal, portfolio_state):
        self.evaluated.append(signal)
        return super().evaluate(signal, portfolio_state)


def build_runner(tmp_path, limits, market_data, strategies, **config_overrides):
    paper_engine = PaperExecutionEngine(PaperExecutionConfig(fee_rate=0.001, slippage_bps=5.0))
    risk_engine = SpyRiskEngine(limits)

    config_kwargs = {"symbols": ["BTC/USDT"], "symbol_cooldown_hours": 0, **config_overrides}

    runner = TradingRunner(
        config=RunnerConfig(**config_kwargs),
        market_data_reader=market_data,
        strategies=strategies,
        risk_engine=risk_engine,
        execution_router=ExecutionRouter.paper(
            spot_engine=paper_engine,
            prediction_engine=PolymarketPaperExecutionEngine(paper_engine=paper_engine),
        ),
        state_store=JsonStateStore(tmp_path / "state.json"),
    )
    return runner, risk_engine


def prediction_market(yes_price: float) -> PredictionMarket:
    return PredictionMarket(
        market_id="1",
        question="Will it happen?",
        slug="test-market",
        active=True,
        closed=False,
        archived=False,
        enable_order_book=True,
        outcomes=[
            OutcomeToken(outcome="Yes", token_id="tok-yes", price=yes_price),
            OutcomeToken(outcome="No", token_id="tok-no", price=1.0 - yes_price),
        ],
    )


class TestPredictionOrdersGoThroughRisk:
    """
    El agujero mas grave del sistema anterior: las ordenes de Polymarket se contaban
    como aprobadas por riesgo sin llamar nunca al motor. No tenian limite de tamano,
    ni de exposicion, ni stop loss.
    """

    @pytest.fixture
    def strategy(self):
        return PolymarketThresholdStrategy(
            PolymarketThresholdConfig(
                slug="test-market",
                outcome="yes",
                buy_below_price=0.40,
                confidence=0.80,
            )
        )

    def test_prediction_signal_is_evaluated_by_the_risk_engine(
        self, tmp_path, limits, strategy
    ):
        market_data = FakeMarketData(
            markets={"test-market": prediction_market(0.30)},
            midpoints={"tok-yes": 0.30, "tok-no": 0.70},
        )
        runner, risk = build_runner(
            tmp_path, limits, market_data, [strategy], symbols=["PM::test-market"]
        )

        runner.run_cycle()

        assert len(risk.evaluated) == 1
        assert risk.evaluated[0].asset_class == AssetClass.PREDICTION

    def test_prediction_position_gets_a_stop_loss_from_risk(self, tmp_path, limits, strategy):
        market_data = FakeMarketData(
            markets={"test-market": prediction_market(0.30)},
            midpoints={"tok-yes": 0.30, "tok-no": 0.70},
        )
        runner, _ = build_runner(
            tmp_path, limits, market_data, [strategy], symbols=["PM::test-market"]
        )

        runner.run_cycle()
        positions = runner.get_positions()

        assert len(positions) == 1
        assert positions[0].venue == Venue.POLYMARKET
        # Antes esto era None: las posiciones de prediccion no tenian salida definida.
        assert positions[0].stop_loss is not None
        assert positions[0].take_profit is not None

    def test_risk_can_reject_a_prediction_signal(self, tmp_path, limits, strategy):
        limits.min_confidence_per_trade = 0.95  # por encima de la confianza de la estrategia
        market_data = FakeMarketData(
            markets={"test-market": prediction_market(0.30)},
            midpoints={"tok-yes": 0.30, "tok-no": 0.70},
        )
        runner, _ = build_runner(
            tmp_path, limits, market_data, [strategy], symbols=["PM::test-market"]
        )

        results = runner.run_cycle()

        assert results == []
        assert runner.get_positions() == []


class TestPositionLifecycle:
    def test_opening_a_position_records_the_entry_fees(self, tmp_path, limits):
        from ai_trader.strategies.momentum_crypto import CryptoMomentumStrategy

        closes = [float(100 + i * 2) for i in range(60)]
        market_data = FakeMarketData(bars={"BTC/USDT": build_bars(closes)})
        runner, _ = build_runner(tmp_path, limits, market_data, [CryptoMomentumStrategy()])

        runner.run_cycle()
        positions = runner.get_positions()

        assert len(positions) == 1
        # Antes las comisiones se calculaban y se tiraban a la basura.
        assert positions[0].entry_fees_usd > 0


class TestStatePersistence:
    def test_state_survives_a_restart(self, tmp_path, limits):
        from ai_trader.strategies.momentum_crypto import CryptoMomentumStrategy

        closes = [float(100 + i * 2) for i in range(60)]
        market_data = FakeMarketData(bars={"BTC/USDT": build_bars(closes)})

        runner, _ = build_runner(tmp_path, limits, market_data, [CryptoMomentumStrategy()])
        runner.run_cycle()
        assert len(runner.get_positions()) == 1

        # Un runner nuevo apuntando al mismo estado debe reencontrar la posicion.
        reloaded, _ = build_runner(tmp_path, limits, market_data, [CryptoMomentumStrategy()])

        assert len(reloaded.get_positions()) == 1
        assert reloaded.get_positions()[0].symbol == "BTC/USDT"

    def test_paused_runner_does_not_trade(self, tmp_path, limits):
        from ai_trader.strategies.momentum_crypto import CryptoMomentumStrategy

        closes = [float(100 + i * 2) for i in range(60)]
        market_data = FakeMarketData(bars={"BTC/USDT": build_bars(closes)})
        runner, _ = build_runner(tmp_path, limits, market_data, [CryptoMomentumStrategy()])

        runner.pause()
        results = runner.run_cycle()

        assert results == []
        assert runner.get_positions() == []


class TestReportsDoNotCrash:
    def test_risk_report_survives_a_missing_price(self, tmp_path, limits, make_position):
        """
        get_risk_report hacia `size * last_price` sin comprobar None, asi que reventaba
        con TypeError en cuanto un simbolo no devolvia precio.
        """
        market_data = FakeMarketData()  # ningun simbolo devuelve barras
        runner, _ = build_runner(tmp_path, limits, market_data, [_NoopStrategy()])

        runner.state.positions.append(make_position(symbol="BTC/USDT"))

        report = runner.get_risk_report()

        assert "Open positions: 1" in report
        assert "BTC/USDT" in report

    def test_reports_are_empty_but_valid_with_no_positions(self, tmp_path, limits):
        runner, _ = build_runner(tmp_path, limits, FakeMarketData(), [_NoopStrategy()])

        assert "No open positions." in runner.get_positions_report()
        assert "No closed positions yet." in runner.get_history_report()
        assert "Closed trades: 0" in runner.get_performance_report()


class _NoopStrategy:
    strategy_id = "noop"

    def supports_symbol(self, symbol: str) -> bool:
        return True

    def generate_signal(self, symbol, context):
        return None
