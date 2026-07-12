from __future__ import annotations

import pytest

from ai_trader.shared.instruments import AssetClass, OutcomeToken, PredictionMarket, Venue
from ai_trader.shared.schemas import Side
from ai_trader.strategies import build_strategy
from ai_trader.strategies.momentum_crypto import CryptoMomentumConfig, CryptoMomentumStrategy
from ai_trader.strategies.polymarket_threshold import (
    PolymarketThresholdConfig,
    PolymarketThresholdStrategy,
)
from tests.conftest import build_bars


class TestCryptoMomentum:
    def test_no_signal_without_enough_bars(self):
        strategy = CryptoMomentumStrategy()

        assert strategy.generate_signal("BTC/USDT", build_bars([100.0] * 10)) is None

    def test_no_signal_in_a_downtrend(self):
        strategy = CryptoMomentumStrategy()
        closes = [float(200 - i) for i in range(60)]  # bajando

        assert strategy.generate_signal("BTC/USDT", build_bars(closes)) is None

    def test_signal_on_an_uptrend_breakout(self):
        strategy = CryptoMomentumStrategy()
        closes = [float(100 + i * 2) for i in range(60)]  # subiendo y rompiendo maximos

        signal = strategy.generate_signal("BTC/USDT", build_bars(closes))

        assert signal is not None
        assert signal.side == Side.BUY
        assert signal.entry_price == pytest.approx(closes[-1])
        assert 0.0 <= signal.confidence <= 1.0

    def test_breakout_gate_is_enforced_when_required(self):
        """
        El filtro de ruptura estaba anulado a mano (`breakout_ok = True`), asi que la
        estrategia operaba sin el pese a llamarse "breakout".
        """
        # Tendencia alcista, pero el ultimo cierre no supera el maximo reciente.
        closes = [float(100 + i) for i in range(59)] + [130.0]
        bars = build_bars(closes)

        strict = CryptoMomentumStrategy(CryptoMomentumConfig(require_breakout=True))
        lenient = CryptoMomentumStrategy(CryptoMomentumConfig(require_breakout=False))

        assert strict.generate_signal("BTC/USDT", bars) is None
        assert lenient.generate_signal("BTC/USDT", bars) is not None

    def test_does_not_support_prediction_symbols(self):
        strategy = CryptoMomentumStrategy()

        assert strategy.supports_symbol("BTC/USDT")
        assert not strategy.supports_symbol("PM::some-market")


def make_market(yes_price: float, slug: str = "test-market") -> PredictionMarket:
    return PredictionMarket(
        market_id="1",
        question="Will it happen?",
        slug=slug,
        active=True,
        closed=False,
        archived=False,
        enable_order_book=True,
        outcomes=[
            OutcomeToken(outcome="Yes", token_id="tok-yes", price=yes_price),
            OutcomeToken(outcome="No", token_id="tok-no", price=1.0 - yes_price),
        ],
    )


class TestPolymarketThreshold:
    @pytest.fixture
    def strategy(self) -> PolymarketThresholdStrategy:
        return PolymarketThresholdStrategy(
            PolymarketThresholdConfig(slug="test-market", outcome="yes", buy_below_price=0.40)
        )

    def test_no_signal_above_the_threshold(self, strategy):
        assert strategy.generate_signal("PM::test-market", make_market(0.55)) is None

    def test_signal_below_the_threshold(self, strategy):
        signal = strategy.generate_signal("PM::test-market", make_market(0.30))

        assert signal is not None
        assert signal.side == Side.BUY
        assert signal.entry_price == pytest.approx(0.30)

    def test_signal_carries_the_instrument_identity(self, strategy):
        """
        Emitir una Signal (y no una OrderRequest) es lo que hace que estas operaciones
        pasen por el motor de riesgo, del que antes se escapaban por completo.
        """
        signal = strategy.generate_signal("PM::test-market", make_market(0.30))

        assert signal.venue == Venue.POLYMARKET
        assert signal.asset_class == AssetClass.PREDICTION
        assert signal.instrument_id == "tok-yes"
        assert signal.outcome == "Yes"

    def test_supports_only_its_own_market(self, strategy):
        assert strategy.supports_symbol("PM::test-market")
        assert not strategy.supports_symbol("PM::another-market")
        assert not strategy.supports_symbol("BTC/USDT")


class TestRegistry:
    def test_builds_a_strategy_from_type_and_params(self):
        strategy = build_strategy("crypto_momentum", {"fast_sma_window": 5, "slow_sma_window": 15})

        assert isinstance(strategy, CryptoMomentumStrategy)
        assert strategy.config.fast_sma_window == 5

    def test_unknown_type_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown strategy type"):
            build_strategy("does_not_exist")

    def test_invalid_params_are_rejected(self):
        with pytest.raises(ValueError, match="Invalid params"):
            build_strategy("crypto_momentum", {"not_a_real_param": 1})
