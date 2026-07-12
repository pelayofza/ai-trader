from __future__ import annotations

import pytest

from ai_trader.risk.engine import PortfolioState, RiskEngine
from ai_trader.shared.schemas import Side


@pytest.fixture
def engine(limits) -> RiskEngine:
    return RiskEngine(limits)


def test_approves_a_clean_signal(engine, make_signal):
    decision = engine.evaluate(make_signal(confidence=0.80), PortfolioState())

    assert decision.approved
    assert decision.size_usd == pytest.approx(800.0)  # 1000 * 0.80


def test_rejects_below_min_confidence(engine, make_signal):
    decision = engine.evaluate(make_signal(confidence=0.50), PortfolioState())

    assert not decision.approved
    assert decision.size_usd == 0.0
    assert "confidence" in decision.reason.lower()


def test_rejects_when_daily_loss_limit_reached(engine, make_signal):
    portfolio = PortfolioState(daily_realized_pnl_usd=-250.0)  # limite: -200

    decision = engine.evaluate(make_signal(), portfolio)

    assert not decision.approved
    assert "daily loss" in decision.reason.lower()


def test_rejects_duplicate_symbol(engine, make_signal, make_position):
    portfolio = PortfolioState(open_positions=[make_position(symbol="BTC/USDT")])

    decision = engine.evaluate(make_signal(symbol="BTC/USDT"), portfolio)

    assert not decision.approved
    assert "already exists" in decision.reason.lower()


def test_rejects_when_max_open_positions_reached(engine, make_signal, make_position):
    portfolio = PortfolioState(
        open_positions=[make_position(symbol=f"SYM{i}/USDT") for i in range(5)]
    )

    decision = engine.evaluate(make_signal(symbol="NEW/USDT"), portfolio)

    assert not decision.approved
    assert "open positions" in decision.reason.lower()


def test_rejects_when_total_exposure_would_be_exceeded(engine, make_signal, make_position):
    # 4 posiciones x 1200 USD = 4800; el limite total es 5000.
    portfolio = PortfolioState(
        open_positions=[
            make_position(symbol=f"SYM{i}/USDT", size=12.0, entry_price=100.0)
            for i in range(4)
        ]
    )

    decision = engine.evaluate(make_signal(symbol="NEW/USDT", confidence=0.80), portfolio)

    assert not decision.approved
    assert "total portfolio exposure" in decision.reason.lower()


class TestExitPolicy:
    """El riesgo es el dueno del stop-loss y del take-profit."""

    def test_applies_defaults_when_strategy_proposes_none(self, engine, make_signal):
        decision = engine.evaluate(make_signal(entry_price=100.0), PortfolioState())

        assert decision.approved
        assert decision.stop_loss == pytest.approx(95.0)  # -5%
        assert decision.take_profit == pytest.approx(110.0)  # +10%

    def test_respects_a_reasonable_strategy_proposal(self, engine, make_signal):
        signal = make_signal(entry_price=100.0, stop_loss=93.0, take_profit=120.0)

        decision = engine.evaluate(signal, PortfolioState())

        assert decision.stop_loss == pytest.approx(93.0)
        assert decision.take_profit == pytest.approx(120.0)

    def test_tightens_a_stop_that_is_too_far_away(self, engine, make_signal):
        # La estrategia pide un stop a -30%; el maximo tolerado es 15%.
        signal = make_signal(entry_price=100.0, stop_loss=70.0)

        decision = engine.evaluate(signal, PortfolioState())

        assert decision.stop_loss == pytest.approx(85.0)
        assert decision.warnings
        assert "tightened" in decision.warnings[0].lower()

    def test_exits_are_mirrored_for_short_positions(self, engine, make_signal):
        decision = engine.evaluate(
            make_signal(side=Side.SELL, entry_price=100.0), PortfolioState()
        )

        # En un corto el stop esta por encima de la entrada y el objetivo por debajo.
        assert decision.stop_loss == pytest.approx(105.0)
        assert decision.take_profit == pytest.approx(90.0)
