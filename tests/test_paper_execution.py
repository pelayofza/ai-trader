from __future__ import annotations

import pytest

from ai_trader.execution.paper import PaperExecutionConfig, PaperExecutionEngine
from ai_trader.shared.instruments import AssetClass, Venue
from ai_trader.shared.schemas import OrderRequest, OrderStatus, OrderType, Side


@pytest.fixture
def engine() -> PaperExecutionEngine:
    return PaperExecutionEngine(PaperExecutionConfig(fee_rate=0.001, slippage_bps=10.0))


def order(**overrides) -> OrderRequest:
    defaults = dict(
        symbol="BTC/USDT",
        side=Side.BUY,
        size=2.0,
        order_type=OrderType.MARKET,
        strategy_id="test_strategy",
        asset_class=AssetClass.CRYPTO,
    )
    return OrderRequest(**{**defaults, **overrides})


def test_buy_fills_above_market_because_of_slippage(engine):
    result = engine.execute(order(side=Side.BUY), market_price=100.0)

    assert result.success
    assert result.status == OrderStatus.FILLED
    assert result.filled_price == pytest.approx(100.1)  # +10 bps


def test_sell_fills_below_market_because_of_slippage(engine):
    result = engine.execute(order(side=Side.SELL), market_price=100.0)

    assert result.filled_price == pytest.approx(99.9)  # -10 bps


def test_fees_are_charged_on_notional(engine):
    result = engine.execute(order(size=2.0), market_price=100.0)

    # 100.1 (con slippage) * 2 unidades * 0.1%
    assert result.fees == pytest.approx(0.2002)


def test_prediction_prices_must_stay_within_zero_and_one(engine):
    prediction_order = order(
        symbol="PM::some-market",
        asset_class=AssetClass.PREDICTION,
        venue=Venue.POLYMARKET,
        instrument_id="token-123",
        outcome="yes",
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        engine.execute(prediction_order, market_price=1.5)


def test_rejects_non_positive_market_price(engine):
    with pytest.raises(ValueError, match="market_price"):
        engine.execute(order(), market_price=0.0)


def test_limit_buy_does_not_fill_worse_than_the_limit(engine):
    result = engine.execute(
        order(order_type=OrderType.LIMIT, limit_price=95.0),
        market_price=100.0,
    )

    # El mercado esta por encima del limite: se referencia el limite, no el mercado.
    assert result.filled_price == pytest.approx(95.095)  # 95 + 10 bps


def test_partial_fills_when_enabled():
    engine = PaperExecutionEngine(
        PaperExecutionConfig(allow_partial_fills=True, partial_fill_ratio=0.5)
    )

    result = engine.execute(order(size=10.0), market_price=100.0)

    assert result.status == OrderStatus.PARTIALLY_FILLED
    assert result.filled_size == pytest.approx(5.0)
