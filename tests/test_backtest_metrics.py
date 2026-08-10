from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

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
from ai_trader.scoring.weight_calibration import (
    CALIBRATION_REPORT,
    grid_point,
    load_calibration_report,
)

ROOT = Path(__file__).resolve().parents[1]


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


class TestDefaultWeightsAreCalibrated:
    """
    Congela los pesos por defecto Y los ata a la evidencia que los justifica.

    Un test que solo comprobara las dos constantes seria un espejo del codigo: pasaria
    igual si alguien las cambiara a ojo. Estos ademas exigen que el informe publicado del
    estudio (data/calibration) cubra exactamente esos pesos y siga respaldandolos, asi
    que mover los pesos obliga a re-correr el estudio y republicar la evidencia, que es
    el punto entero de haberlo medido.
    """

    LAMBDA_TURNOVER = 0.25
    KAPPA_MAXDD = 0.0

    @pytest.fixture(scope="class")
    def report(self) -> dict:
        published = load_calibration_report(ROOT / CALIBRATION_REPORT)
        assert published is not None, (
            f"Falta la evidencia de calibracion en {CALIBRATION_REPORT}. "
            "Regenerala con: python -m ai_trader.scoring.weight_study"
        )
        return published

    def test_weights_are_frozen(self):
        assert DEFAULT_HEADLINE_WEIGHTS.lambda_turnover == self.LAMBDA_TURNOVER
        assert DEFAULT_HEADLINE_WEIGHTS.kappa_maxdd == self.KAPPA_MAXDD

    def test_the_published_grid_actually_swept_these_weights(self, report):
        """No basta con documentar un numero: tiene que estar EN la rejilla medida."""
        point = grid_point(report, self.LAMBDA_TURNOVER, self.KAPPA_MAXDD)

        assert point is not None, "Los pesos por defecto no estan en la rejilla barrida"
        assert report["plan"]["library_id"] == "ai_v2"
        assert len(report["configs"]["kept"]) >= 8
        assert report["configs"]["n_samples"] >= 20

    def test_penalising_costs_at_most_a_tenth_of_the_ranking_signal(self, report):
        """
        El estudio midio que penalizar NO estabiliza el ranking: lo degrada un poco, de
        forma monotona en ambos pesos. Asumido eso, lo que hay que acotar es el precio.

        La cifra es la perdida PAREADA de rank IC frente a no penalizar (misma muestra,
        asi que la varianza comun se cancela), y el liston es el 10% del nivel de la
        senal. Los pesos anteriores (0.5, 1.0) costaban un 17% y este test los habria
        rechazado: no es un espejo de las constantes, es un criterio que sabe fallar.
        """
        point = grid_point(report, self.LAMBDA_TURNOVER, self.KAPPA_MAXDD)
        best = max(p["rank_ic_mean"] for p in report["grid"]["points"])

        assert point["rank_ic_gain"] <= 0.0  # el maximo esta en no penalizar
        assert abs(point["rank_ic_gain"]) <= 0.10 * best

    def test_the_weights_do_not_change_which_config_wins(self, report):
        """
        El hallazgo que convierte a lambda y kappa en un regularizador barato y no en el
        motor del ranking: en TODA la rejilla gana la misma configuracion. Si algun dia
        deja de ser cierto, la eleccion de pesos vuelve a ser una decision de producto y
        este test tiene que obligar a revisarla.
        """
        winners = {p["selected_config"] for p in report["grid"]["points"]}

        assert len(winners) == 1

    def test_a_zero_turnover_price_would_break_a_committed_property(self):
        """
        Por que lambda no es 0 aunque la evidencia lo prefiera: el headline se comprometio
        a que rotar mas sobre la MISMA curva de equity puntue peor. Con lambda = 0 esa
        propiedad desaparece, asi que el minimo de la rejilla no es una opcion.
        """
        assert DEFAULT_HEADLINE_WEIGHTS.lambda_turnover > 0.0

    def test_turnover_penalty_does_not_double_charge_the_costs_already_paid(self, report):
        """
        Punto (2) del encargo. La curva de equity ya paga fee_rate + slippage al rotar, y
        ese coste vive DENTRO del Sharpe. lambda es un regularizador anadido, asi que
        tiene que ser una fraccion pequena del coste implicito: si se acercara a el,
        estariamos cobrando dos veces la misma rotacion.
        """
        implied = report["cost_audit_active"]["implied_lambda_median"]

        assert implied > 0
        assert self.LAMBDA_TURNOVER < 0.25 * implied

    def test_the_cost_chain_was_validated_against_fees_actually_charged(self, report):
        """El lambda implicito se apoya en reconstruir el notional desde el turnover; el
        informe debe demostrar que esa reconstruccion reproduce el fee_rate real."""
        audit = report["cost_audit_active"]

        assert audit["measured_fee_rate"] == pytest.approx(audit["fee_rate"], rel=0.02)


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
