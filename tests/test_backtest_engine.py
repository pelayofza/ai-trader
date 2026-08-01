from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ai_trader.backtest.engine import BacktestEngine
from ai_trader.config import AppConfig, StrategySpec
from ai_trader.execution.paper import PaperExecutionConfig
from ai_trader.risk.engine import RiskLimits
from ai_trader.app.runner import RunnerConfig


def trending_df(n_days: int, start_price: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    """
    Rampa alcista lineal y determinista. El paso diario (step) supera el rango
    intradia, de modo que cada cierre marca un maximo nuevo y el filtro de breakout
    del momentum deja entrar. El open del dia es el cierre del dia anterior.
    """
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    index = pd.DatetimeIndex([start + pd.Timedelta(days=i) for i in range(n_days)], name="timestamp")

    closes = np.array([start_price + i * step for i in range(n_days)])
    opens = np.array([start_price] + [closes[i - 1] for i in range(1, n_days)])
    highs = np.maximum(opens, closes) + 0.1
    lows = np.minimum(opens, closes) - 0.1

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1000.0},
        index=index,
    )


class FakeService:
    """Sirve el historico completo desde memoria; el clamping lo hace la fuente."""

    def __init__(self, bars: dict[str, pd.DataFrame]):
        self._bars = bars

    def get_daily_bars(self, symbol, start, end):
        return self._bars.get(symbol.strip().upper())


def make_config() -> AppConfig:
    return AppConfig(
        runner=RunnerConfig(
            symbols=["BTC/USDT"],
            lookback_days=120,
            max_holding_days=10,
            symbol_cooldown_hours=0,  # re-entrada libre para generar varios trades
            max_trades_per_cycle=5,
        ),
        risk=RiskLimits(
            min_confidence_per_trade=0.50,
            risk_fraction_per_trade=0.10,
            max_symbol_exposure_usd=1_000_000.0,  # generoso: en modo equity no debe estorbar
            max_total_exposure_usd=1_000_000.0,
            max_daily_loss_usd=1_000_000.0,
        ),
        execution=PaperExecutionConfig(fee_rate=0.001, slippage_bps=5.0),
        strategies=[
            StrategySpec(
                type="crypto_momentum",
                id="mom",
                params={"min_bars": 30, "require_breakout": True, "min_atr_pct": 0.1},
            )
        ],
    )


@pytest.fixture
def engine():
    bars = {"BTC/USDT": trending_df(400)}
    return BacktestEngine(make_config(), FakeService(bars), starting_equity=10_000.0)


class TestSplit:
    def test_produces_train_and_test_windows(self, engine):
        result = engine.run(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
            split_ratio=0.7,
        )

        assert result.train.label == "train"
        assert result.test.label == "test"
        # El train termina antes de que empiece el test.
        assert result.train.end <= result.test.start

    def test_explicit_cutoff_overrides_ratio(self, engine):
        cutoff = datetime(2024, 10, 1, tzinfo=timezone.utc)
        result = engine.run(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
            split_date=cutoff,
        )

        assert result.test.start >= cutoff

    def test_rejects_cutoff_outside_range(self, engine):
        with pytest.raises(ValueError, match="split_date"):
            engine.run(
                start=datetime(2024, 6, 1, tzinfo=timezone.utc),
                end=datetime(2025, 1, 1, tzinfo=timezone.utc),
                split_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
            )


class TestExecution:
    def test_an_uptrend_produces_trades_and_grows_equity(self, engine):
        result = engine.run(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        assert result.train.metrics.num_trades > 0
        # Serie claramente alcista: el equity de entrenamiento deberia crecer.
        assert result.train.metrics.ending_equity > result.train.metrics.starting_equity

    def test_equity_curve_has_one_point_per_trading_day(self, engine):
        start = datetime(2024, 6, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = engine.run(start=start, end=end, split_ratio=0.7)

        assert len(result.test.equity_curve) >= 2
        # Monotonia temporal de la curva.
        days = [p.day for p in result.test.equity_curve]
        assert days == sorted(days)

    def test_headline_score_is_out_of_sample_calmar(self, engine):
        result = engine.run(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        assert result.headline_score == result.test.metrics.calmar


class TestDeterminism:
    def test_two_identical_runs_give_the_same_equity_curve(self):
        bars = {"BTC/USDT": trending_df(400)}
        args = dict(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
            split_ratio=0.7,
        )

        r1 = BacktestEngine(make_config(), FakeService(bars), 10_000.0).run(**args)
        r2 = BacktestEngine(make_config(), FakeService(bars), 10_000.0).run(**args)

        c1 = [round(p.equity, 6) for p in r1.test.equity_curve]
        c2 = [round(p.equity, 6) for p in r2.test.equity_curve]
        assert c1 == c2
        assert r1.headline_score == r2.headline_score


class TestCompounding:
    def test_larger_starting_capital_scales_position_sizes(self):
        bars = {"BTC/USDT": trending_df(400)}
        args = dict(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        small = BacktestEngine(make_config(), FakeService(bars), 10_000.0).run(**args)
        big = BacktestEngine(make_config(), FakeService(bars), 100_000.0).run(**args)

        # Con sizing por fraccion de equity, 10x capital -> ~10x notional por trade,
        # asi que el retorno PORCENTUAL debe ser parecido (no el absoluto).
        assert big.train.metrics.ending_equity > 5 * small.train.metrics.ending_equity
        assert math.isclose(
            big.train.metrics.total_return_pct,
            small.train.metrics.total_return_pct,
            rel_tol=0.05,
        )
