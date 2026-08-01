from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ai_trader.execution.market_model import IntrabarMarketModel
from ai_trader.shared.bars import Bar
from ai_trader.shared.schemas import Side


class FakeClock:
    def __init__(self, moment):
        self._moment = moment

    def now(self):
        return self._moment


class FakeBarSource:
    def __init__(self, bar: Bar | None):
        self._bar = bar

    def bar_on(self, symbol, day):
        return self._bar


def bar(open_, high, low, close):
    return Bar(
        timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
        open=open_,
        high=high,
        low=low,
        close=close,
    )


@pytest.fixture
def clock():
    return FakeClock(datetime(2026, 1, 2, tzinfo=timezone.utc))


def model(bar_obj, clock):
    return IntrabarMarketModel(FakeBarSource(bar_obj), clock)


class TestEntryAndMark:
    def test_entry_fills_at_the_open_of_the_current_bar(self, clock, make_signal):
        m = model(bar(open_=99.0, high=101.0, low=98.0, close=100.0), clock)

        signal = make_signal(entry_price=100.0)  # la senal se decidio con el cierre de ayer

        # No entra al cierre de decision (100): entra al open de hoy (99).
        assert m.entry_reference_price("BTC/USDT", signal) == pytest.approx(99.0)

    def test_mark_price_is_todays_close(self, clock, make_position):
        m = model(bar(open_=99.0, high=101.0, low=98.0, close=100.5), clock)

        assert m.mark_price(make_position()) == pytest.approx(100.5)

    def test_returns_none_when_no_bar(self, clock, make_signal, make_position):
        m = model(None, clock)

        assert m.entry_reference_price("BTC/USDT", make_signal()) is None
        assert m.mark_price(make_position()) is None
        assert m.price_exit(make_position()) is None


class TestIntrabarStops:
    def _pos(self, make_position, *, side=Side.BUY, stop=None, tp=None):
        p = make_position(side=side, entry_price=100.0)
        p.stop_loss = stop
        p.take_profit = tp
        return p

    def test_long_stops_out_at_the_stop_level(self, clock, make_position):
        # El low toca el stop; el open estaba por encima -> se sale en el stop.
        m = model(bar(open_=99.0, high=100.0, low=94.0, close=96.0), clock)
        exit = m.price_exit(self._pos(make_position, stop=95.0, tp=110.0))

        assert exit == ("stop_loss", pytest.approx(95.0))

    def test_long_gap_down_fills_at_the_open_not_the_stop(self, clock, make_position):
        # Abre ya por debajo del stop -> se llena al open (peor que el stop).
        m = model(bar(open_=90.0, high=91.0, low=88.0, close=89.0), clock)
        exit = m.price_exit(self._pos(make_position, stop=95.0, tp=110.0))

        assert exit == ("stop_loss", pytest.approx(90.0))

    def test_long_takes_profit_at_the_target(self, clock, make_position):
        m = model(bar(open_=101.0, high=112.0, low=100.0, close=108.0), clock)
        exit = m.price_exit(self._pos(make_position, stop=95.0, tp=110.0))

        assert exit == ("take_profit", pytest.approx(110.0))

    def test_long_gap_up_fills_at_the_open(self, clock, make_position):
        m = model(bar(open_=115.0, high=118.0, low=114.0, close=116.0), clock)
        exit = m.price_exit(self._pos(make_position, stop=95.0, tp=110.0))

        assert exit == ("take_profit", pytest.approx(115.0))

    def test_stop_wins_when_both_hit_in_the_same_bar(self, clock, make_position):
        # Convencion pesimista: si la barra toca stop y objetivo, se asume el stop.
        m = model(bar(open_=100.0, high=112.0, low=94.0, close=105.0), clock)
        exit = m.price_exit(self._pos(make_position, stop=95.0, tp=110.0))

        assert exit[0] == "stop_loss"

    def test_no_exit_when_price_stays_between_levels(self, clock, make_position):
        m = model(bar(open_=100.0, high=104.0, low=97.0, close=101.0), clock)

        assert m.price_exit(self._pos(make_position, stop=95.0, tp=110.0)) is None

    def test_short_stop_is_mirrored(self, clock, make_position):
        # En un corto el stop esta por encima de la entrada.
        m = model(bar(open_=101.0, high=106.0, low=100.0, close=104.0), clock)
        exit = m.price_exit(self._pos(make_position, side=Side.SELL, stop=105.0, tp=90.0))

        assert exit == ("stop_loss", pytest.approx(105.0))

    def test_short_takes_profit_downward(self, clock, make_position):
        m = model(bar(open_=99.0, high=100.0, low=88.0, close=91.0), clock)
        exit = m.price_exit(self._pos(make_position, side=Side.SELL, stop=105.0, tp=90.0))

        assert exit == ("take_profit", pytest.approx(90.0))
