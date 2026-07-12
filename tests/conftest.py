from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ai_trader.risk.engine import RiskLimits
from ai_trader.shared.schemas import Position, Side, Signal


@pytest.fixture
def limits() -> RiskLimits:
    return RiskLimits(
        max_position_size_usd=1_000.0,
        max_open_positions=5,
        max_symbol_exposure_usd=2_000.0,
        max_total_exposure_usd=5_000.0,
        max_daily_loss_usd=200.0,
        min_confidence_per_trade=0.65,
        default_stop_loss_pct=5.0,
        default_take_profit_pct=10.0,
        max_stop_distance_pct=15.0,
    )


@pytest.fixture
def make_signal():
    def _make(
        *,
        symbol: str = "BTC/USDT",
        side: Side = Side.BUY,
        confidence: float = 0.80,
        entry_price: float = 100.0,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Signal:
        return Signal(
            strategy_id="test_strategy",
            symbol=symbol,
            timeframe="1d",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            side=side,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    return _make


@pytest.fixture
def make_position():
    def _make(
        *,
        symbol: str = "BTC/USDT",
        side: Side = Side.BUY,
        size: float = 1.0,
        entry_price: float = 100.0,
        entry_fees_usd: float = 0.0,
        exit_fees_usd: float = 0.0,
    ) -> Position:
        return Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            opened_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            strategy_id="test_strategy",
            entry_fees_usd=entry_fees_usd,
            exit_fees_usd=exit_fees_usd,
        )

    return _make


def build_bars(closes: list[float], *, high_offset: float = 1.0) -> pd.DataFrame:
    """Barras diarias sinteticas en la forma canonica (columnas en minuscula, indice UTC)."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    index = pd.DatetimeIndex(
        [start + timedelta(days=i) for i in range(len(closes))], name="timestamp"
    )
    closes_arr = np.array(closes, dtype=float)

    return pd.DataFrame(
        {
            "open": closes_arr,
            "high": closes_arr + high_offset,
            "low": closes_arr - high_offset,
            "close": closes_arr,
            "volume": np.full(len(closes), 1_000.0),
        },
        index=index,
    )
