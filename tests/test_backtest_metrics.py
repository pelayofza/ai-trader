from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ai_trader.backtest.metrics import (
    EquityPoint,
    compute_metrics,
    max_drawdown_pct,
    sharpe_ratio,
)


def curve(values: list[float]) -> list[EquityPoint]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [EquityPoint(day=start + timedelta(days=i), equity=v) for i, v in enumerate(values)]


class TestMaxDrawdown:
    def test_no_drawdown_on_monotonic_growth(self):
        assert max_drawdown_pct(curve([100, 110, 120, 130])) == pytest.approx(0.0)

    def test_measures_peak_to_trough(self):
        # Pico 200, valle 150 -> -25%.
        assert max_drawdown_pct(curve([100, 200, 150, 180])) == pytest.approx(25.0)

    def test_takes_the_worst_of_several_drawdowns(self):
        assert max_drawdown_pct(curve([100, 90, 100, 200, 100])) == pytest.approx(50.0)


class TestSharpe:
    def test_positive_for_steady_gains(self):
        assert sharpe_ratio([0.01, 0.012, 0.009, 0.011]) > 0

    def test_negative_for_steady_losses(self):
        assert sharpe_ratio([-0.01, -0.012, -0.009, -0.011]) < 0

    def test_zero_when_no_variance(self):
        assert sharpe_ratio([0.0, 0.0, 0.0]) == 0.0


class TestComputeMetrics:
    def test_headline_calmar_rewards_return_and_penalizes_drawdown(self, make_position):
        # Curva que sube a 130 con un valle intermedio.
        c = curve([100, 120, 90, 130])
        m = compute_metrics(c, [])

        assert m.total_return_pct == pytest.approx(30.0)
        assert m.max_drawdown_pct == pytest.approx(25.0)  # 120 -> 90
        # calmar = cagr / maxdd, ambos positivos.
        assert m.calmar > 0

    def test_trade_stats_come_from_closed_positions(self, make_position):
        winner = make_position(symbol="BTC/USDT")
        winner.realized_pnl = 50.0
        winner.closed_at = winner.opened_at + timedelta(days=3)
        loser = make_position(symbol="ETH/USDT")
        loser.realized_pnl = -20.0
        loser.closed_at = loser.opened_at + timedelta(days=1)

        m = compute_metrics(curve([100, 130]), [winner, loser])

        assert m.num_trades == 2
        assert m.win_rate_pct == pytest.approx(50.0)
        assert m.profit_factor == pytest.approx(2.5)  # 50 / 20
        assert m.avg_win_usd == pytest.approx(50.0)
        assert m.avg_loss_usd == pytest.approx(-20.0)
        assert m.avg_holding_days == pytest.approx(2.0)

    def test_profit_factor_is_none_without_losses(self):
        m = compute_metrics(curve([100, 110]), [])
        assert m.profit_factor is None

    def test_empty_curve_is_rejected(self):
        with pytest.raises(ValueError):
            compute_metrics([], [])
