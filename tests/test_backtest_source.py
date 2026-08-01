from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from ai_trader.data.backtest_source import HistoricalDataSource
from ai_trader.shared.clock import HistoricalClock
from ai_trader.shared.instruments import AssetClass


def daily_df(start_day: int, n: int) -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [datetime(2026, 1, start_day + i, tzinfo=timezone.utc) for i in range(n)],
        name="timestamp",
    )
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1000.0] * n,
        },
        index=index,
    )


@pytest.fixture
def source():
    clock = HistoricalClock(datetime(2026, 1, 5, tzinfo=timezone.utc))
    return HistoricalDataSource({"BTC/USDT": daily_df(1, 10)}, clock), clock


class TestNoLookAhead:
    def test_only_returns_bars_strictly_before_today(self, source):
        src, clock = source
        clock.set(datetime(2026, 1, 5, tzinfo=timezone.utc))

        bars = src.get_daily_bars(
            "BTC/USDT",
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 5, tzinfo=timezone.utc),
        )

        # Hoy es el 5; la estrategia solo debe ver hasta el 4 (barra cerrada).
        assert bars.index.max() < pd.Timestamp("2026-01-05", tz="UTC")
        assert bars.index.max() == pd.Timestamp("2026-01-04", tz="UTC")

    def test_window_grows_as_the_clock_advances(self, source):
        src, clock = source
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)

        clock.set(datetime(2026, 1, 3, tzinfo=timezone.utc))
        early = src.get_daily_bars("BTC/USDT", start, clock.now())

        clock.set(datetime(2026, 1, 8, tzinfo=timezone.utc))
        later = src.get_daily_bars("BTC/USDT", start, clock.now())

        assert len(later) > len(early)


class TestBarAccess:
    def test_bar_on_returns_the_exact_day(self, source):
        src, _ = source

        bar = src.bar_on("BTC/USDT", datetime(2026, 1, 5, tzinfo=timezone.utc))

        assert bar is not None
        # Es el quinto dia: open = 100 + 4.
        assert bar.open == pytest.approx(104.0)
        assert bar.close == pytest.approx(104.5)

    def test_bar_on_returns_none_outside_range(self, source):
        src, _ = source

        assert src.bar_on("BTC/USDT", datetime(2026, 2, 1, tzinfo=timezone.utc)) is None

    def test_unknown_symbol_is_none(self, source):
        src, _ = source

        assert src.get_daily_bars("DOGE/USDT", datetime(2026, 1, 1), datetime(2026, 1, 9)) is None
        assert src.bar_on("DOGE/USDT", datetime(2026, 1, 5)) is None


class TestCalendarAndRouting:
    def test_trading_days_spans_the_range(self, source):
        src, _ = source

        days = src.trading_days(
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 10, tzinfo=timezone.utc),
        )

        assert len(days) == 10
        assert days[0] == pd.Timestamp("2026-01-01", tz="UTC")

    def test_prediction_symbols_have_no_ohlcv(self, source):
        src, _ = source

        assert src.get_prediction_market("some-slug") is None
        assert src.get_prediction_midpoint("token") is None
        assert src.detect_asset_class("PM::x") == AssetClass.PREDICTION
        assert src.detect_asset_class("BTC/USDT") == AssetClass.CRYPTO
