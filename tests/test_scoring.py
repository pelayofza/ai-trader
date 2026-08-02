from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ai_trader.app.runner import RunnerConfig
from ai_trader.config import AppConfig
from ai_trader.execution.paper import PaperExecutionConfig
from ai_trader.risk.engine import RiskLimits
from ai_trader.scoring.aggregate import aggregate_reward
from ai_trader.scoring.cem import CEMConfig, maximize
from ai_trader.scoring.optimize import run_optimization
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
    def test_reward_penalizes_dispersion(self):
        tight = aggregate_reward([1.0, 1.0, 1.0], lam=0.5)
        spread = aggregate_reward([-1.0, 1.0, 3.0], lam=0.5)  # misma media 1.0

        assert tight.mean == pytest.approx(spread.mean)
        assert tight.reward > spread.reward  # menos dispersion -> mas recompensa

    def test_reports_tail_statistics(self):
        stats = aggregate_reward([0.0, 1.0, 2.0, 3.0, 4.0], lam=0.0)

        assert stats.worst == 0.0
        assert stats.best == 4.0
        # CVaR@25% de 5 muestras = media del peor cuartil, ceil(0.25*5)=2 -> (0+1)/2.
        assert stats.cvar25 == pytest.approx(0.5)
        assert stats.n == 5

    def test_empty_is_neutral(self):
        stats = aggregate_reward([], lam=0.5)
        assert stats.reward == 0.0
        assert stats.n == 0


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


# --- end-to-end sobre un store falso (sin depender de la libreria ai_v1 en disco) ---


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

    def load_manifest(self, library_id):
        return _FakeManifest(list(self._slopes))

    def load_bars(self, library_id, scenario_id, path_index):
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

    def test_is_deterministic(self):
        a = self._run()
        b = self._run()

        assert a.best_params == b.best_params
        assert a.train.reward == b.train.reward
        assert a.validation.reward == b.validation.reward
