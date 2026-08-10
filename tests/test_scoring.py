from __future__ import annotations

import inspect
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ai_trader.app.runner import RunnerConfig
from ai_trader.config import AppConfig
from ai_trader.execution.paper import PaperExecutionConfig
from ai_trader.risk.engine import RiskLimits
from ai_trader.backtest.metrics import CRYPTO_PERIODS_PER_YEAR, STOCK_PERIODS_PER_YEAR
from ai_trader.scoring.aggregate import aggregate_reward
from ai_trader.scoring.baselines import (
    BASELINE_BTC,
    BASELINE_EQUAL_WEIGHT,
    BASELINE_SPY,
    compute_baselines,
    gate,
)
from ai_trader.scoring.cem import CEMConfig, maximize
from ai_trader.scoring.optimize import DEFAULT_LIBRARY_ID, run_optimization
from ai_trader.scoring.overfit import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from ai_trader.scoring.scenario_split import split_scenarios
from ai_trader.scoring.search_space import SPACES, get_space
from ai_trader.strategies import build_strategy


class TestScenarioSplit:
    def test_is_deterministic(self):
        ids = [f"s{i}" for i in range(30)]

        a = split_scenarios(ids, seed=7)
        b = split_scenarios(ids, seed=7)

        assert a.train == b.train
        assert a.validation == b.validation

    def test_partition_is_disjoint_and_complete(self):
        ids = [f"s{i}" for i in range(30)]

        split = split_scenarios(ids, validation_fraction=0.27)

        assert set(split.train).isdisjoint(split.validation)
        assert set(split.train) | set(split.validation) == set(ids)
        assert split.n_validation == round(30 * 0.27)  # 8
        assert split.n_train == 22

    def test_different_seeds_give_different_holdouts(self):
        ids = [f"s{i}" for i in range(30)]

        assert split_scenarios(ids, seed=1).validation != split_scenarios(ids, seed=2).validation

    def test_always_leaves_one_on_each_side(self):
        split = split_scenarios(["a", "b"], validation_fraction=0.99)

        assert split.n_train >= 1
        assert split.n_validation >= 1


class TestSearchSpace:
    @pytest.mark.parametrize("strategy_type", list(SPACES))
    def test_midpoint_builds_a_valid_strategy(self, strategy_type):
        space = get_space(strategy_type)
        params = space.to_params(space.midpoint())

        # No debe lanzar: los params proyectados son siempre validos.
        strategy = build_strategy(strategy_type, params)
        assert strategy is not None

    @pytest.mark.parametrize("strategy_type", list(SPACES))
    def test_random_in_bounds_vectors_are_always_valid(self, strategy_type):
        space = get_space(strategy_type)
        rng = np.random.default_rng(0)

        for _ in range(50):
            vector = rng.uniform(space.lows, space.highs)
            params = build_strategy(strategy_type, space.to_params(vector))
            assert params is not None

    def test_momentum_enforces_fast_below_slow_and_min_bars(self):
        space = get_space("crypto_momentum")
        # Vector con fast >= slow para forzar la correccion de coherencia.
        vector = np.array([30, 20, 5, 0.5, 2.0, 3.0, 0.0, -1.0], dtype=float)
        params = space.to_params(vector)

        assert params["fast_sma_window"] < params["slow_sma_window"]
        assert params["min_bars"] >= params["slow_sma_window"]

    def test_mean_reversion_keeps_exit_below_entry(self):
        space = get_space("mean_reversion")
        # exit_z alto y entry_z bajo: la finalizacion debe recolocar exit_z por debajo.
        vector = np.array([20, 1.0, 1.0, 2.0, 0.2, 0.0, 1.0], dtype=float)
        params = space.to_params(vector)

        assert params["exit_z"] < params["entry_z"]

    def test_unknown_type_is_rejected(self):
        with pytest.raises(ValueError, match="No search space"):
            get_space("nope")


class TestAggregateReward:
    def test_reward_is_the_cvar_of_the_worst_quartile(self):
        stats = aggregate_reward([0.0, 1.0, 2.0, 3.0, 4.0])

        # CVaR@25% de 5 muestras = media del peor cuartil, ceil(0.25*5)=2 -> (0+1)/2.
        assert stats.cvar25 == pytest.approx(0.5)
        assert stats.reward == pytest.approx(stats.cvar25)

    def test_ranks_by_the_bad_tail_not_by_the_centre(self):
        """La distribucion con mejor media pierde si su cola mala es peor: es
        exactamente lo que 'media - lambda*std' no garantizaba."""
        steady = aggregate_reward([1.0, 1.0, 1.0, 1.0])
        lottery = aggregate_reward([-3.0, 1.0, 2.0, 4.0])  # media 1.0, cola horrible

        assert lottery.mean >= steady.mean
        assert steady.reward > lottery.reward

    def test_upside_variance_is_not_punished_like_downside(self):
        """Una politica que a veces explota AL ALZA no paga por ello (a diferencia de
        media - lambda*std), porque el CVaR solo mira el peor cuartil."""
        flat = aggregate_reward([1.0, 1.0, 1.0, 1.0])
        upside = aggregate_reward([1.0, 1.0, 1.0, 9.0])  # misma cola, mucha mas std

        assert upside.std > flat.std
        assert upside.reward == pytest.approx(flat.reward)

    def test_reports_mean_std_and_p25_without_using_them_to_rank(self):
        stats = aggregate_reward([0.0, 1.0, 2.0, 3.0, 4.0])

        assert stats.mean == pytest.approx(2.0)
        assert stats.std == pytest.approx(np.std([0, 1, 2, 3, 4], ddof=1))
        assert stats.p25 == pytest.approx(1.0)
        assert stats.worst == 0.0
        assert stats.best == 4.0
        assert stats.n == 5

    def test_alpha_widens_the_tail(self):
        scores = [0.0, 1.0, 2.0, 3.0]

        assert aggregate_reward(scores, alpha=0.25).reward == pytest.approx(0.0)
        assert aggregate_reward(scores, alpha=0.5).reward == pytest.approx(0.5)
        assert aggregate_reward(scores, alpha=1.0).reward == pytest.approx(1.5)

    def test_empty_is_neutral(self):
        stats = aggregate_reward([])
        assert stats.reward == 0.0
        assert stats.n == 0

    def test_invalid_alpha_is_rejected(self):
        with pytest.raises(ValueError, match="alpha"):
            aggregate_reward([1.0], alpha=0.0)


class TestBaselines:
    """Los baselines pasivos: lo que consigue cualquiera sin hacer nada."""

    def _bars(self) -> dict:
        anchor = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return {
            "BTC/USDT": _trending_df(60, 1.0, anchor),   # sube
            "SPY": _trending_df(60, 0.2, anchor),        # sube poco
            "GLD": _trending_df(60, -0.5, anchor),       # baja
        }

    def _compute(self, bars, **kwargs):
        return compute_baselines(
            bars,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 2, 29, tzinfo=timezone.utc),
            starting_equity=10_000.0,
            **kwargs,
        )

    def test_builds_the_three_passive_alternatives(self):
        out = self._compute(self._bars())

        assert set(out) == {BASELINE_BTC, BASELINE_SPY, BASELINE_EQUAL_WEIGHT}
        assert out[BASELINE_BTC].symbols == ("BTC/USDT",)
        assert out[BASELINE_EQUAL_WEIGHT].symbols == ("BTC/USDT", "GLD", "SPY")

    def test_a_missing_symbol_is_omitted_not_faked(self):
        bars = self._bars()
        del bars["SPY"]

        out = self._compute(bars)

        assert BASELINE_SPY not in out
        assert BASELINE_BTC in out  # los demas siguen

    def test_buy_and_hold_tracks_its_symbol(self):
        out = self._compute(self._bars())

        btc = out[BASELINE_BTC].metrics
        # Rampa de +1/dia sobre 100 durante ~59 dias: la cartera casi se multiplica.
        assert btc.total_return_pct > 40.0
        assert btc.num_trades == 1  # una compra y una liquidacion

    def test_the_equal_weight_portfolio_sits_between_its_components(self):
        out = self._compute(self._bars())

        ew = out[BASELINE_EQUAL_WEIGHT].metrics.total_return_pct
        btc = out[BASELINE_BTC].metrics.total_return_pct
        assert btc > ew  # GLD, que cae, arrastra a la equiponderada hacia abajo

    def test_every_baseline_is_annualised_on_the_same_scale(self):
        """
        El gate compara el Sharpe de la estrategia con el del mejor baseline. Si
        `spy_hold` -puramente bursatil- se anualizara por 252 y la estrategia por 365,
        la comparacion seria entre dos escalas y el baseline saldria artificialmente
        peor. El factor lo fija el universo de la muestra, no los simbolos de cada
        baseline: los tres recorren el mismo calendario union.
        """
        out = self._compute(self._bars())  # universo mixto: cripto + SPY + GLD

        factors = {name: b.metrics.periods_per_year for name, b in out.items()}
        assert set(factors.values()) == {CRYPTO_PERIODS_PER_YEAR}

    def test_a_stock_only_sample_annualises_by_sessions(self):
        bars = self._bars()
        del bars["BTC/USDT"]

        out = self._compute(bars)

        assert out[BASELINE_SPY].metrics.periods_per_year == STOCK_PERIODS_PER_YEAR

    def test_the_caller_can_impose_the_strategy_factor(self):
        """`evaluate_baselines` lo pasa desde el universo del config para que baselines
        y estrategia usen exactamente el mismo, aunque las barras de la muestra no
        traigan todos los simbolos del universo."""
        out = self._compute(self._bars(), periods_per_year=STOCK_PERIODS_PER_YEAR)

        assert out[BASELINE_BTC].metrics.periods_per_year == STOCK_PERIODS_PER_YEAR

    def test_baselines_pay_the_same_frictions_as_the_strategy(self):
        free = self._compute(self._bars())[BASELINE_BTC].metrics
        costly = self._compute(
            self._bars(), fee_rate=0.01, slippage_bps=50.0
        )[BASELINE_BTC].metrics

        assert costly.total_fees_usd > 0
        assert costly.ending_equity < free.ending_equity

    def test_is_deterministic(self):
        a = self._compute(self._bars())
        b = self._compute(self._bars())

        assert {k: v.score for k, v in a.items()} == {k: v.score for k, v in b.items()}


class TestBaselineGate:
    """El gate: una estrategia solo aprueba si bate al MEJOR baseline."""

    def test_approves_a_strategy_that_beats_every_baseline(self):
        verdict = gate(
            [2.0, 2.0, 2.0, 2.0],
            {BASELINE_BTC: [1.0, 1.0, 1.0, 1.0], BASELINE_SPY: [0.5, 0.5, 0.5, 0.5]},
        )

        assert verdict.approved
        assert verdict.best_name == BASELINE_BTC
        assert verdict.margin == pytest.approx(1.0)
        assert verdict.win_rate_pct == pytest.approx(100.0)

    def test_rejects_a_strategy_that_loses_to_the_best_baseline(self):
        verdict = gate(
            [0.9, 0.9, 0.9, 0.9],
            {BASELINE_BTC: [1.0, 1.0, 1.0, 1.0]},
        )

        assert not verdict.approved
        assert verdict.margin < 0

    def test_the_gate_is_played_on_the_tail_not_on_the_average(self):
        """Una estrategia con mejor media pero peor cuartil malo NO aprueba: se compara
        por la recompensa (CVaR@25%), igual que se optimiza."""
        verdict = gate(
            [-2.0, 1.5, 1.5, 4.0],  # media 1.25
            {BASELINE_BTC: [1.0, 1.0, 1.0, 1.0]},  # media 1.0
        )

        assert not verdict.approved

    def test_without_baselines_nothing_is_approved(self):
        verdict = gate([5.0, 5.0], {}, missing=[BASELINE_BTC, BASELINE_SPY])

        assert not verdict.approved
        assert verdict.best_name is None
        assert verdict.missing == (BASELINE_BTC, BASELINE_SPY)

    def test_win_rate_compares_sample_by_sample_against_the_best_rival(self):
        verdict = gate(
            [2.0, 0.0],
            {BASELINE_BTC: [1.0, 1.0], BASELINE_SPY: [3.0, -1.0]},
        )

        # Muestra 0: mejor rival 3.0 (pierde). Muestra 1: mejor rival 1.0 (pierde).
        assert verdict.win_rate_pct == pytest.approx(0.0)

    def test_best_baseline_is_deterministic_under_ties(self):
        tied = {BASELINE_SPY: [1.0, 1.0], BASELINE_BTC: [1.0, 1.0]}

        assert gate([0.0, 0.0], tied).best_name == BASELINE_BTC  # desempate por nombre


class TestDeflatedSharpe:
    """DSR: el Sharpe del ganador descontado por cuantas configuraciones se probaron."""

    def test_more_trials_raise_the_bar(self):
        trials_few = [0.5, 1.0, 1.5]
        trials_many = trials_few * 40  # misma dispersion, 120 intentos

        few = deflated_sharpe_ratio(1.5, trials_few, 500)
        many = deflated_sharpe_ratio(1.5, trials_many, 500)

        assert many.expected_max_sharpe > few.expected_max_sharpe
        assert many.dsr < few.dsr

    def test_a_single_trial_deflates_nothing(self):
        single = deflated_sharpe_ratio(1.5, [1.5], 500)

        assert single.expected_max_sharpe == 0.0
        assert single.dsr > 0.5  # sigue siendo un Sharpe positivo creible
        assert single.computable

    def test_a_lucky_winner_among_noise_is_discounted(self):
        """El mismo Sharpe observado deja de ser creible si sale de un barrido ancho
        con mucha dispersion entre intentos."""
        noise = [(-1.0) ** i * 1.2 for i in range(200)]

        assert deflated_sharpe_ratio(1.5, noise, 500).dsr < 0.5

    def test_fat_left_tails_lower_the_confidence(self):
        symmetric = deflated_sharpe_ratio(1.5, [1.0, 1.5], 500, skew=0.0, kurtosis=3.0)
        ugly = deflated_sharpe_ratio(1.5, [1.0, 1.5], 500, skew=-1.5, kurtosis=12.0)

        assert ugly.dsr < symmetric.dsr

    def test_declares_itself_uncomputable_without_a_track_record(self):
        assert not deflated_sharpe_ratio(1.5, [1.0, 1.5], 1).computable

    def test_is_deterministic(self):
        args = (1.4, [0.5, 1.0, 1.5, 2.0], 400)
        assert deflated_sharpe_ratio(*args).dsr == deflated_sharpe_ratio(*args).dsr


class TestPBO:
    """PBO por CSCV: cuando elijo la mejor configuracion por backtest, ¿acierto?"""

    def test_genuine_skill_gives_a_low_pbo(self):
        # La config 0 es mejor en TODAS las muestras: elegirla nunca decepciona.
        matrix = [[1.0, 0.0, -1.0] for _ in range(8)]

        result = probability_of_backtest_overfitting(matrix, n_blocks=4)

        assert result.computable
        assert result.pbo == pytest.approx(0.0)
        assert result.median_logit > 0

    def test_pure_overfitting_gives_a_high_pbo(self):
        # Cada config gana en la mitad de los bloques y pierde en la otra: la ganadora
        # in-sample es sistematicamente la perdedora out-of-sample.
        matrix = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]

        result = probability_of_backtest_overfitting(matrix, n_blocks=4)

        assert result.pbo == pytest.approx(1.0)

    def test_reports_the_splits_and_samples_actually_used(self):
        matrix = [[1.0, 0.0] for _ in range(9)]  # 9 filas, 4 bloques -> sobra 1

        result = probability_of_backtest_overfitting(matrix, n_blocks=4)

        assert result.n_splits == 6  # C(4,2)
        assert result.n_samples_used == 8
        assert result.n_blocks == 4

    def test_a_single_configuration_is_not_computable(self):
        result = probability_of_backtest_overfitting([[1.0], [2.0], [3.0], [4.0]])

        assert not result.computable
        assert result.n_splits == 0

    def test_blocks_are_clamped_to_an_even_count_that_fits(self):
        matrix = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]  # 3 muestras, se piden 10 bloques

        result = probability_of_backtest_overfitting(matrix, n_blocks=10)

        assert result.n_blocks == 2
        assert result.computable

    def test_is_deterministic(self):
        matrix = [[0.3, 0.7], [0.9, 0.1], [0.5, 0.5], [0.2, 0.8]]

        a = probability_of_backtest_overfitting(matrix, n_blocks=4)
        b = probability_of_backtest_overfitting(matrix, n_blocks=4)
        assert a.pbo == b.pbo and a.median_logit == b.median_logit


class TestCEM:
    def test_converges_towards_the_optimum(self):
        target = np.array([7.0, 3.0])

        def objective(x):
            return -float(np.sum((x - target) ** 2))

        result = maximize(
            objective,
            lows=np.array([0.0, 0.0]),
            highs=np.array([10.0, 10.0]),
            config=CEMConfig(population=40, iterations=25, seed=0),
        )

        assert np.allclose(result.best_vector, target, atol=0.5)
        assert result.best_score > -0.5

    def test_is_deterministic(self):
        def objective(x):
            return -float(np.sum(x**2))

        args = dict(lows=np.array([-5.0, -5.0]), highs=np.array([5.0, 5.0]))
        a = maximize(objective, config=CEMConfig(seed=3, iterations=5), **args)
        b = maximize(objective, config=CEMConfig(seed=3, iterations=5), **args)

        assert np.array_equal(a.best_vector, b.best_vector)
        assert a.best_score == b.best_score


# --- end-to-end sobre un store falso (sin depender de ninguna libreria en disco) ---


def _trending_df(n_days: int, slope: float, anchor: datetime) -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [anchor + pd.Timedelta(days=i) for i in range(n_days)], name="timestamp"
    )
    closes = np.array([100.0 + slope * i for i in range(n_days)])
    opens = np.array([100.0] + [closes[i - 1] for i in range(1, n_days)])
    highs = np.maximum(opens, closes) + 0.1
    lows = np.minimum(opens, closes) - 0.1
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1000.0},
        index=index,
    )


class _FakeManifest:
    anchor = "2024-01-01"
    horizon_days = 200
    n_paths = 3

    def __init__(self, scenario_ids):
        self.scenarios = [{"id": s} for s in scenario_ids]


class _FakeStore:
    """Sirve barras sinteticas en memoria con el mismo contrato que SyntheticStore
    (load_manifest + load_bars), sin tocar disco."""

    def __init__(self, scenario_slopes: dict[str, float]):
        self._slopes = scenario_slopes
        self._anchor = datetime(2024, 1, 1, tzinfo=timezone.utc)
        self.requested_libraries: list[str] = []  # que sustrato pidio el optimizador

    def load_manifest(self, library_id):
        self.requested_libraries.append(library_id)
        return _FakeManifest(list(self._slopes))

    def load_bars(self, library_id, scenario_id, path_index):
        self.requested_libraries.append(library_id)
        # Cada path perturba levemente la pendiente para que la distribucion no sea trivial.
        slope = self._slopes[scenario_id] + path_index * 0.05
        return {"BTC/USDT": _trending_df(200, slope, self._anchor)}


def _fake_base_config() -> AppConfig:
    return AppConfig(
        runner=RunnerConfig(
            symbols=["BTC/USDT"],
            lookback_days=30,
            max_holding_days=10,
            symbol_cooldown_hours=0,
            max_trades_per_cycle=5,
        ),
        risk=RiskLimits(
            min_confidence_per_trade=0.50,
            risk_fraction_per_trade=0.10,
            max_symbol_exposure_usd=1_000_000.0,
            max_total_exposure_usd=1_000_000.0,
            max_daily_loss_usd=1_000_000.0,
        ),
        execution=PaperExecutionConfig(fee_rate=0.001, slippage_bps=5.0),
        strategies=[],  # lo rellena el optimizador con la candidata
    )


class TestDefaultLibrary:
    """El sustrato por defecto del scoring debe ser la libreria realista (ai_v2), no el
    ruido iid de ai_v1: optimizar sobre iid premia sesgos optimistas."""

    def test_constant_points_to_the_realistic_library(self):
        assert DEFAULT_LIBRARY_ID == "ai_v2"

    def test_signature_default_is_the_constant(self):
        default = inspect.signature(run_optimization).parameters["library_id"].default
        assert default == DEFAULT_LIBRARY_ID

    def test_optimizer_reads_ai_v2_when_no_library_is_given(self):
        store = _FakeStore({"up": 1.0, "flat": 0.05, "down": -0.5})

        run_optimization(
            "crypto_momentum",
            store=store,
            base_config=_fake_base_config(),
            cem_config=CEMConfig(population=2, iterations=1, seed=0),
            n_paths=1,
        )

        assert store.requested_libraries  # se toco el store
        assert set(store.requested_libraries) == {"ai_v2"}

    def test_explicit_library_still_wins(self):
        store = _FakeStore({"up": 1.0, "flat": 0.05, "down": -0.5})

        run_optimization(
            "crypto_momentum",
            library_id="ai_v1",
            store=store,
            base_config=_fake_base_config(),
            cem_config=CEMConfig(population=2, iterations=1, seed=0),
            n_paths=1,
        )

        assert set(store.requested_libraries) == {"ai_v1"}


class TestRunOptimizationEndToEnd:
    def _run(self):
        store = _FakeStore({"up": 1.0, "flat": 0.05, "down": -0.5})
        return run_optimization(
            "crypto_momentum",
            store=store,
            base_config=_fake_base_config(),
            cem_config=CEMConfig(population=4, iterations=2, seed=0),
            n_paths=2,
        )

    def test_produces_valid_best_params_and_holdout_stats(self):
        result = self._run()

        # Los mejores params construyen una estrategia valida.
        assert build_strategy("crypto_momentum", result.best_params) is not None
        # Train = 2 escenarios x 2 paths; validation = 1 escenario x 2 paths.
        assert result.train.n == result.split.n_train * 2
        assert result.validation.n == result.split.n_validation * 2
        assert result.n_paths_per_scenario == 2
        assert result.total_paths_available == 3
        assert len(result.history) == 2
        assert isinstance(result.overfit_gap, float)

    def test_the_gate_runs_on_the_holdout_and_declares_missing_baselines(self):
        result = self._run()

        # El universo falso es solo BTC/USDT: no hay SPY que comprar, y se dice.
        assert set(result.gate.baselines) == {BASELINE_BTC, BASELINE_EQUAL_WEIGHT}
        assert BASELINE_SPY in result.gate.missing
        # Un baseline por muestra de VALIDATION, no de train.
        assert result.gate.baselines[BASELINE_BTC].n == result.validation.n
        assert result.approved == result.gate.approved
        assert isinstance(result.approved, bool)

    def test_reports_multiple_testing_discounts(self):
        result = self._run()

        # 4 candidatos x 2 iteraciones = 8 configuraciones probadas.
        assert result.pbo.n_trials == 8
        assert result.dsr.n_trials == 8
        assert 0.0 <= result.pbo.pbo <= 1.0
        assert 0.0 <= result.dsr.dsr <= 1.0

    def test_serializes_the_whole_verdict(self):
        payload = self._run().as_dict()

        assert set(payload) >= {"approved", "gate", "pbo", "dsr", "headline_weights"}
        assert payload["gate"]["best_baseline"] in (BASELINE_BTC, BASELINE_EQUAL_WEIGHT)

    def test_is_deterministic(self):
        a = self._run()
        b = self._run()

        assert a.best_params == b.best_params
        assert a.train.reward == b.train.reward
        assert a.validation.reward == b.validation.reward
        assert a.gate.approved == b.gate.approved
        assert a.gate.best_reward == b.gate.best_reward
        assert a.pbo.pbo == b.pbo.pbo
        assert a.dsr.dsr == b.dsr.dsr
