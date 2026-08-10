from __future__ import annotations

import math

import numpy as np
import pytest

from ai_trader.backtest.metrics import (
    TRADING_DAYS_PER_YEAR,
    HeadlineWeights,
    PerformanceMetrics,
    headline_score,
)
from ai_trader.scoring.aggregate import aggregate_reward
from ai_trader.scoring.scenario_split import ScenarioSplit
from ai_trader.scoring.weight_calibration import (
    CalibrationMatrix,
    CalibrationSample,
    WindowComponents,
    candidate_specs,
    evaluate_weight_point,
    filter_active_configs,
    spearman,
    sweep_weights,
    turnover_cost_audit,
)
from ai_trader.strategies import build_strategy

STARTING_EQUITY = 10_000.0
DAYS = 100


def metrics(
    *,
    sharpe: float = 0.0,
    turnover: float = 0.0,
    max_dd: float = 0.0,
    volatility_pct: float = 20.0,
    num_trades: int = 10,
    total_fees_usd: float = 0.0,
) -> PerformanceMetrics:
    return PerformanceMetrics(
        starting_equity=STARTING_EQUITY,
        ending_equity=STARTING_EQUITY,
        total_return_pct=0.0,
        cagr_pct=0.0,
        max_drawdown_pct=max_dd,
        sharpe=sharpe,
        sortino=0.0,
        calmar=0.0,
        volatility_pct=volatility_pct,
        turnover=turnover,
        num_trades=num_trades,
        win_rate_pct=0.0,
        profit_factor=None,
        avg_win_usd=0.0,
        avg_loss_usd=0.0,
        avg_holding_days=0.0,
        total_fees_usd=total_fees_usd,
    )


def components(**kwargs) -> WindowComponents:
    return WindowComponents(
        metrics=metrics(**kwargs), days=DAYS, starting_equity=STARTING_EQUITY
    )


def sample(
    config_id: str,
    scenario_id: str,
    *,
    is_kwargs: dict,
    oos_kwargs: dict,
    path_index: int = 0,
    failed: bool = False,
) -> CalibrationSample:
    return CalibrationSample(
        config_id=config_id,
        strategy_type="crypto_momentum",
        scenario_id=scenario_id,
        path_index=path_index,
        in_sample=components(**is_kwargs),
        out_of_sample=components(**oos_kwargs),
        failed=failed,
    )


class TestSpearman:
    def test_perfect_agreement_and_perfect_inversion(self):
        assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
        assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_only_ranks_matter_not_magnitudes(self):
        assert spearman([1, 2, 3], [1, 2, 1000]) == pytest.approx(1.0)

    def test_ties_are_averaged(self):
        # Empate en la segunda serie: rangos 0, 1.5, 1.5. No es ni 1 ni -1.
        rho = spearman([1, 2, 3], [1, 5, 5])
        assert 0.0 < rho < 1.0

    def test_constant_series_has_no_ranking_to_correlate(self):
        assert math.isnan(spearman([1, 2, 3], [7, 7, 7]))

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            spearman([1, 2], [1, 2, 3])


class TestCandidateSpecs:
    def test_is_deterministic(self):
        a = candidate_specs("crypto_momentum", 8, seed=3)
        b = candidate_specs("crypto_momentum", 8, seed=3)

        assert [s.params for s in a] == [s.params for s in b]
        assert [s.id for s in a] == [s.id for s in b]

    def test_every_candidate_builds_a_valid_strategy(self):
        for strategy_type in ("crypto_momentum", "mean_reversion"):
            for spec in candidate_specs(strategy_type, 6, seed=1):
                assert build_strategy(spec.type, spec.params, strategy_id=spec.id) is not None

    def test_latin_hypercube_covers_each_dimension(self):
        """Si el conjunto no contuviera configuraciones lentas Y rapidas, barrer lambda
        no informaria de nada: la cobertura por dimension es el requisito del estudio."""
        specs = candidate_specs("crypto_momentum", 12, seed=1)
        lookbacks = [s.params["breakout_lookback"] for s in specs]

        assert min(lookbacks) <= 5  # el espacio va de 2 a 20
        assert max(lookbacks) >= 17

    def test_needs_at_least_two_configs_to_rank(self):
        with pytest.raises(ValueError):
            candidate_specs("crypto_momentum", 1)


class TestWindowComponents:
    def test_score_delegates_to_the_single_headline_formula(self):
        window = components(sharpe=1.5, turnover=0.2, max_dd=10.0)
        weights = HeadlineWeights(lambda_turnover=0.5, kappa_maxdd=1.0)

        assert window.score(weights) == pytest.approx(headline_score(window.metrics, weights))
        assert window.score(weights) == pytest.approx(1.5 - 0.1 - 0.1)

    def test_traded_notional_reconstructs_from_turnover(self):
        window = components(turnover=0.2)

        assert window.traded_notional_usd == pytest.approx(0.2 * STARTING_EQUITY * DAYS)

    def test_survives_a_json_round_trip(self):
        window = components(sharpe=1.25, turnover=0.33, max_dd=7.5)

        assert WindowComponents.from_dict(window.as_dict()) == window


class TestCalibrationMatrix:
    def _samples(self) -> list[CalibrationSample]:
        out = []
        for config_id, sharpe in (("a", 1.0), ("b", 2.0)):
            for scenario_id in ("s1", "s2"):
                out.append(
                    sample(
                        config_id,
                        scenario_id,
                        is_kwargs={"sharpe": sharpe},
                        oos_kwargs={"sharpe": sharpe},
                    )
                )
        return out

    def test_builds_a_config_by_sample_matrix(self):
        matrix = CalibrationMatrix(self._samples())

        assert matrix.config_ids == ["a", "b"]
        assert matrix.n_samples == 2
        assert matrix.scores(HeadlineWeights(), window="out_of_sample").shape == (2, 2)

    def test_drops_configs_with_failed_backtests_and_declares_them(self):
        samples = self._samples()
        samples[0] = sample(
            "a", "s1", is_kwargs={}, oos_kwargs={}, failed=True
        )

        matrix = CalibrationMatrix(samples)

        assert matrix.config_ids == ["b"]
        assert matrix.dropped == ["a"]

    def test_drops_configs_missing_a_sample(self):
        samples = [s for s in self._samples() if not (s.config_id == "a" and s.scenario_id == "s2")]

        matrix = CalibrationMatrix(samples)

        assert matrix.config_ids == ["b"]
        assert matrix.dropped == ["a"]

    def test_sample_mask_selects_by_scenario(self):
        matrix = CalibrationMatrix(self._samples())

        assert list(matrix.sample_mask(["s2"])) == [False, True]


class TestFilterActiveConfigs:
    def test_keeps_only_configs_that_actually_trade(self):
        samples = [
            sample("busy", "s1", is_kwargs={}, oos_kwargs={"num_trades": 40}),
            sample("busy", "s2", is_kwargs={}, oos_kwargs={"num_trades": 60}),
            sample("idle", "s1", is_kwargs={}, oos_kwargs={"num_trades": 2}),
            sample("idle", "s2", is_kwargs={}, oos_kwargs={"num_trades": 3}),
        ]

        kept = filter_active_configs(samples, min_median_trades=20)

        assert {s.config_id for s in kept} == {"busy"}


class TestWeightGrid:
    """
    Matriz amanada donde la respuesta correcta se conoce de antemano.

    'churner' compra su ventaja con rotacion: brilla dentro de muestra y en los
    escenarios de train, y se desploma tanto fuera de muestra como en el arquetipo
    reservado. 'steady' es la eleccion buena. Subir lambda tiene que corregir la
    eleccion en LOS DOS EJES que mide el estudio (temporal y de escenarios); si no lo
    hiciera, la maquinaria no mide lo que dice medir.
    """

    TRAIN = ("s1", "s2", "s3")
    VALIDATION = ("s4",)

    def _matrix(self) -> tuple[CalibrationMatrix, ScenarioSplit]:
        samples = []
        for i, scenario_id in enumerate(self.TRAIN + self.VALIDATION):
            held_out = scenario_id in self.VALIDATION
            samples.append(
                sample(
                    "churner",
                    scenario_id,
                    is_kwargs={"sharpe": 3.0 + 0.1 * i, "turnover": 1.0},
                    oos_kwargs={"sharpe": 0.0 if held_out else 2.0 + 0.1 * i, "turnover": 1.0},
                )
            )
            samples.append(
                sample(
                    "steady",
                    scenario_id,
                    is_kwargs={"sharpe": 2.0 + 0.1 * i, "turnover": 0.05},
                    oos_kwargs={"sharpe": 1.5 + 0.1 * i, "turnover": 0.05},
                )
            )
            samples.append(
                sample(
                    "middling",
                    scenario_id,
                    is_kwargs={"sharpe": 1.0 + 0.1 * i, "turnover": 0.3},
                    oos_kwargs={"sharpe": 0.8 + 0.1 * i, "turnover": 0.3},
                )
            )
        split = ScenarioSplit(train=self.TRAIN, validation=self.VALIDATION, seed=0)
        return CalibrationMatrix(samples), split

    def test_without_penalty_the_rotator_wins_and_the_choice_does_not_survive(self):
        matrix, split = self._matrix()

        point = evaluate_weight_point(matrix, HeadlineWeights(0.0, 0.0), split)

        # Eje de escenarios: se elige al rotador y se hunde en el arquetipo reservado.
        assert point.selected_config == "churner"
        assert point.selection_gap == pytest.approx(2.0)
        # Eje temporal: el que gana dentro de muestra es el peor fuera.
        assert point.top1_oos_pct == pytest.approx(100.0 / 6)

    def test_turnover_penalty_fixes_the_selection_on_both_axes(self):
        matrix, split = self._matrix()

        point = evaluate_weight_point(matrix, HeadlineWeights(2.0, 0.0), split)

        assert point.selected_config == "steady"
        assert point.val_turnover == pytest.approx(0.05)
        assert point.selection_gap < 0.0  # ya no se desinfla al salir del train
        assert point.top1_oos_pct > 80.0

    def test_gap_is_reported_against_the_scenario_holdout(self):
        matrix, split = self._matrix()

        point = evaluate_weight_point(matrix, HeadlineWeights(0.0, 0.0), split)

        assert point.selection_gap == pytest.approx(point.train_reward - point.validation_reward)
        assert point.val_sharpe == pytest.approx(0.0)  # el churner en el hold-out
        assert point.val_max_drawdown_pct == pytest.approx(0.0)

    def test_normalized_gap_divides_by_the_spread_between_configs(self):
        matrix, split = self._matrix()

        point = evaluate_weight_point(matrix, HeadlineWeights(0.0, 0.0), split)
        rewards = [
            row.min() for row in matrix.scores(HeadlineWeights(0.0, 0.0), window="out_of_sample")
        ]
        spread = float(np.std([r for r in rewards], ddof=1))

        # CVaR@25% sobre 3 muestras de train = la peor; la dispersion entre configs es
        # la unidad en la que el gap deja de depender de la escala de los pesos.
        assert not math.isnan(point.selection_gap_norm)
        assert spread > 0

    def test_sweep_covers_the_whole_grid(self):
        matrix, split = self._matrix()

        grid = sweep_weights(matrix, split, lambdas=(0.0, 1.0), kappas=(0.0, 0.5, 1.0))

        assert len(grid) == 6
        assert {(p.lambda_turnover, p.kappa_maxdd) for p in grid} == {
            (0.0, 0.0), (0.0, 0.5), (0.0, 1.0), (1.0, 0.0), (1.0, 0.5), (1.0, 1.0)
        }

    def test_paired_gain_is_measured_against_not_penalising_at_all(self):
        matrix, split = self._matrix()

        grid = sweep_weights(matrix, split, lambdas=(0.0, 2.0), kappas=(0.0,))
        neutral = next(p for p in grid if p.lambda_turnover == 0.0)
        penalised = next(p for p in grid if p.lambda_turnover == 2.0)

        # El punto de referencia no puede ganarse a si mismo.
        assert neutral.rank_ic_gain == pytest.approx(0.0)
        # Y la ganancia es exactamente la diferencia de medias sobre las mismas muestras.
        assert penalised.rank_ic_gain == pytest.approx(
            penalised.rank_ic_mean - neutral.rank_ic_mean
        )

    def test_paired_error_is_tighter_than_the_unpaired_one(self):
        """Es la razon de ser del estadistico pareado: la varianza comun entre muestras
        se cancela, asi que detecta diferencias que restar dos medias sueltas no ve."""
        matrix, split = self._matrix()

        grid = sweep_weights(matrix, split, lambdas=(0.0, 1.0), kappas=(0.0,))
        penalised = next(p for p in grid if p.lambda_turnover == 1.0)

        assert penalised.rank_ic_gain_se <= penalised.rank_ic_se

    def test_pooled_rank_ic_uses_the_same_aggregation_as_the_optimizer(self):
        matrix, split = self._matrix()
        weights = HeadlineWeights(1.0, 0.0)

        point = evaluate_weight_point(matrix, weights, split)

        expected = spearman(
            [aggregate_reward(row).reward for row in matrix.scores(weights, window="in_sample")],
            [
                aggregate_reward(row).reward
                for row in matrix.scores(weights, window="out_of_sample")
            ],
        )
        assert point.rank_ic_pooled == pytest.approx(expected)
        assert -1.0 <= point.rank_ic_mean <= 1.0


class TestPenaltyRescuesTheTemporalRanking:
    """
    El eje temporal, aislado: aqui el rotador brilla DENTRO de muestra y se hunde FUERA
    en todos los escenarios. Es el caso que la penalizacion de rotacion existe para
    detectar, y el rank IC tiene que reflejarlo pasando de negativo a positivo.
    """

    def _matrix(self) -> tuple[CalibrationMatrix, ScenarioSplit]:
        rows = (
            ("churner", 3.0, 0.1, 1.0),
            ("steady", 2.0, 1.5, 0.05),
            ("middling", 1.0, 0.8, 0.3),
        )
        samples = [
            sample(
                config_id,
                scenario_id,
                is_kwargs={"sharpe": is_sharpe, "turnover": turnover},
                oos_kwargs={"sharpe": oos_sharpe, "turnover": turnover},
            )
            for config_id, is_sharpe, oos_sharpe, turnover in rows
            for scenario_id in ("s1", "s2")
        ]
        return CalibrationMatrix(samples), ScenarioSplit(("s1",), ("s2",), 0)

    def test_penalising_turnover_flips_the_ranking_from_misleading_to_useful(self):
        matrix, split = self._matrix()

        grid = sweep_weights(matrix, split, lambdas=(0.0, 2.0), kappas=(0.0,))
        neutral = next(p for p in grid if p.lambda_turnover == 0.0)
        penalised = next(p for p in grid if p.lambda_turnover == 2.0)

        assert neutral.rank_ic_mean == pytest.approx(-0.5)
        assert penalised.rank_ic_mean == pytest.approx(0.5)
        assert penalised.rank_ic_gain == pytest.approx(1.0)


class TestTurnoverCostAudit:
    """
    Punto (2) del encargo: comprobar que la penalizacion explicita de rotacion es del
    mismo orden que los costes que la curva de equity YA paga, para no cobrar dos veces
    sin saberlo.
    """

    def _samples(self, *, turnover: float, volatility_pct: float, fee_rate: float):
        notional = turnover * STARTING_EQUITY * DAYS
        return [
            sample(
                "c",
                f"s{i}",
                is_kwargs={},
                oos_kwargs={
                    "turnover": turnover,
                    "volatility_pct": volatility_pct,
                    "num_trades": 50,
                    "total_fees_usd": fee_rate * notional,
                },
            )
            for i in range(4)
        ]

    def test_measured_fee_rate_reconstructs_the_configured_one(self):
        """Control de que la cadena turnover -> notional -> coste es correcta: si las
        comisiones realmente cobradas no reproducen el fee_rate, el resto sobra."""
        samples = self._samples(turnover=0.2, volatility_pct=20.0, fee_rate=0.001)

        audit = turnover_cost_audit(samples, fee_rate=0.001, slippage_bps=5.0)

        assert audit.measured_fee_rate == pytest.approx(0.001)
        assert audit.cost_rate == pytest.approx(0.0015)

    def test_implied_lambda_is_cost_rate_annualized_over_volatility(self):
        samples = self._samples(turnover=0.2, volatility_pct=25.0, fee_rate=0.001)

        audit = turnover_cost_audit(samples, fee_rate=0.001, slippage_bps=5.0)

        expected = 0.0015 * TRADING_DAYS_PER_YEAR / 0.25
        assert audit.implied_lambda_median == pytest.approx(expected)
        # Y el arrastre total es ese lambda por el turnover realmente rotado.
        assert audit.median_sharpe_drag == pytest.approx(expected * 0.2)

    def test_lower_volatility_makes_the_same_costs_bite_harder(self):
        cheap = turnover_cost_audit(
            self._samples(turnover=0.2, volatility_pct=40.0, fee_rate=0.001),
            fee_rate=0.001, slippage_bps=5.0,
        )
        dear = turnover_cost_audit(
            self._samples(turnover=0.2, volatility_pct=10.0, fee_rate=0.001),
            fee_rate=0.001, slippage_bps=5.0,
        )

        assert dear.implied_lambda_median == pytest.approx(4 * cheap.implied_lambda_median)

    def test_ignores_windows_below_the_activity_threshold(self):
        samples = self._samples(turnover=0.2, volatility_pct=20.0, fee_rate=0.001)
        samples.append(
            sample(
                "idle", "s9", is_kwargs={},
                oos_kwargs={"turnover": 0.001, "volatility_pct": 1.0, "num_trades": 2},
            )
        )

        audit = turnover_cost_audit(samples, fee_rate=0.001, slippage_bps=5.0, min_trades=20)

        assert audit.n_windows == 4  # la ventana casi inactiva no distorsiona la mediana

    def test_refuses_to_report_without_usable_windows(self):
        idle = [
            sample("idle", "s1", is_kwargs={}, oos_kwargs={"num_trades": 0}),
        ]

        with pytest.raises(ValueError):
            turnover_cost_audit(idle, fee_rate=0.001, slippage_bps=5.0)


class TestNoScoreFormulaDuplication:
    def test_components_and_metrics_agree_on_random_inputs(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            weights = HeadlineWeights(
                lambda_turnover=float(rng.uniform(0, 4)),
                kappa_maxdd=float(rng.uniform(0, 4)),
            )
            window = components(
                sharpe=float(rng.normal()),
                turnover=float(rng.uniform(0, 1)),
                max_dd=float(rng.uniform(0, 60)),
            )
            assert window.score(weights) == headline_score(window.metrics, weights)
