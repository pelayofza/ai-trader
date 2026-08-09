from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ai_trader.backtest.metrics import (
    DEFAULT_HEADLINE_WEIGHTS,
    EquityPoint,
    HeadlineWeights,
    compute_metrics,
    headline_score,
    kurtosis,
    max_drawdown_pct,
    sharpe_ratio,
    skewness,
    turnover_ratio,
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


def closed(make_position, *, size, entry, exit_price, days=1):
    p = make_position(size=size, entry_price=entry)
    p.exit_price = exit_price
    p.closed_at = p.opened_at + timedelta(days=days)
    p.realized_pnl = size * (exit_price - entry)
    return p


class TestTurnover:
    def test_counts_both_legs_of_every_trade_per_day(self, make_position):
        # 10 dias de ventana, equity inicial 1000. Un round trip de 100 de entrada
        # y 100 de salida = 200 rotados -> 200 / (1000 * 10) = 0.02 al dia.
        c = curve([1000.0] * 11)
        trade = closed(make_position, size=1.0, entry=100.0, exit_price=100.0)

        assert turnover_ratio(c, [trade]) == pytest.approx(0.02)

    def test_is_zero_without_trades(self):
        assert turnover_ratio(curve([100, 110, 120]), []) == 0.0

    def test_scales_with_size_not_just_trade_count(self, make_position):
        c = curve([1000.0] * 11)
        small = closed(make_position, size=1.0, entry=100.0, exit_price=100.0)
        big = closed(make_position, size=4.0, entry=100.0, exit_price=100.0)

        assert turnover_ratio(c, [big]) == pytest.approx(4 * turnover_ratio(c, [small]))

    def test_appears_in_compute_metrics(self, make_position):
        c = curve([1000.0] * 11)
        trade = closed(make_position, size=1.0, entry=100.0, exit_price=100.0)

        assert compute_metrics(c, [trade]).turnover == pytest.approx(0.02)


class TestHeadlineScore:
    def test_is_sharpe_minus_turnover_and_drawdown_penalties(self, make_position):
        m = compute_metrics(curve([100, 120, 90, 130]), [])
        w = HeadlineWeights(lambda_turnover=0.5, kappa_maxdd=1.0)

        # maxDD 25% -> 0.25 de penalizacion; sin trades el turnover no resta.
        assert headline_score(m, w) == pytest.approx(m.sharpe - 0.25)

    def test_drawdown_is_a_soft_penalty_not_a_denominator(self):
        """El Calmar explota cuando el maxDD tiende a 0; el headline no: es aditivo."""
        flat = compute_metrics(curve([100, 101, 102, 103]), [])

        assert flat.max_drawdown_pct == pytest.approx(0.0)
        assert flat.calmar == 0.0  # el Calmar degenera justo cuando mejor va todo
        assert headline_score(flat) == pytest.approx(flat.sharpe)  # el headline no

    def test_does_not_reward_inactivity(self):
        """El optimo degenerado del Calmar (casi no operar -> maxDD minusculo ->
        Calmar altisimo) desaparece: una curva plana puntua 0, no infinito."""
        idle = compute_metrics(curve([100.0] * 30), [])
        working = compute_metrics(curve([100 + i for i in range(30)]), [])

        assert headline_score(idle) == pytest.approx(0.0)
        assert headline_score(working) > headline_score(idle)

    def test_churn_costs_score(self, make_position):
        c = curve([1000.0] * 11)
        quiet = compute_metrics(c, [])
        churner = compute_metrics(
            c, [closed(make_position, size=5.0, entry=100.0, exit_price=100.0) for _ in range(6)]
        )

        # Misma curva de equity: la unica diferencia es la rotacion, y resta.
        assert churner.sharpe == pytest.approx(quiet.sharpe)
        assert headline_score(churner) < headline_score(quiet)

    def test_default_weights_are_the_documented_ones(self):
        assert DEFAULT_HEADLINE_WEIGHTS.lambda_turnover == 0.5
        assert DEFAULT_HEADLINE_WEIGHTS.kappa_maxdd == 1.0


class TestReturnMoments:
    def test_symmetric_returns_have_no_skew(self):
        assert skewness([-0.02, -0.01, 0.0, 0.01, 0.02]) == pytest.approx(0.0, abs=1e-12)

    def test_a_fat_left_tail_is_negatively_skewed(self):
        assert skewness([0.01, 0.01, 0.01, 0.01, -0.20]) < 0

    def test_kurtosis_is_not_in_excess(self):
        # Fallback declarado para series demasiado cortas: normal (3.0), no 0.
        assert kurtosis([0.01]) == pytest.approx(3.0)
        assert kurtosis([0.01, 0.01, 0.01, 0.01, -0.20]) > 3.0


class TestComputeMetrics:
    def test_calmar_is_still_reported_even_though_it_is_not_the_headline(self):
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
