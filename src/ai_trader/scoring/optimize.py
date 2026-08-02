from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from ai_trader.backtest.engine import DEFAULT_STARTING_EQUITY
from ai_trader.config import AppConfig, StrategySpec, load_config
from ai_trader.scoring.aggregate import DEFAULT_LAMBDA, RewardStats, aggregate_reward
from ai_trader.scoring.cem import CEMConfig, maximize
from ai_trader.scoring.sample_eval import evaluate_sample
from ai_trader.scoring.scenario_split import (
    DEFAULT_VALIDATION_FRACTION,
    ScenarioSplit,
    split_scenarios,
)
from ai_trader.scoring.search_space import ParamSpace, get_space
from ai_trader.synthetic.service import sample_window
from ai_trader.synthetic.store import SyntheticStore

logger = logging.getLogger(__name__)

DEFAULT_SYNTHETIC_CONFIG = Path("config") / "synthetic.toml"


@dataclass(slots=True)
class OptimizationResult:
    """
    Resultado de optimizar los params de una primitiva por CEM sobre la libreria
    sintetica. Reporta train Y validation (hold-out de escenarios enteros) para hacer
    visible el gap de overfitting: la unidad de evaluacion es la distribucion, y la
    validacion son arquetipos que el CEM nunca vio.
    """

    strategy_type: str
    best_params: dict
    train: RewardStats
    validation: RewardStats
    split: ScenarioSplit
    n_paths_per_scenario: int
    total_paths_available: int
    history: list[dict] = field(default_factory=list)

    @property
    def overfit_gap(self) -> float:
        """Cuanto peor rinde en validacion que en train. Positivo = sobreajuste."""
        return self.train.reward - self.validation.reward

    def as_dict(self) -> dict:
        return {
            "strategy_type": self.strategy_type,
            "best_params": self.best_params,
            "train": self.train.as_dict(),
            "validation": self.validation.as_dict(),
            "overfit_gap": round(self.overfit_gap, 4),
            "split": {
                "n_train": self.split.n_train,
                "n_validation": self.split.n_validation,
                "seed": self.split.seed,
            },
            "n_paths_per_scenario": self.n_paths_per_scenario,
            "total_paths_available": self.total_paths_available,
        }


class _SampleEvaluator:
    """Evalua specs sobre muestras (escenario, path), cacheando las barras cargadas
    para no releer parquet en cada iteracion del CEM."""

    def __init__(
        self,
        store,
        library_id: str,
        base_config: AppConfig,
        start: datetime,
        end: datetime,
        n_paths: int,
        *,
        split_ratio: float,
        starting_equity: float,
    ) -> None:
        self._store = store
        self._library_id = library_id
        self._base_config = base_config
        self._start = start
        self._end = end
        self._n_paths = n_paths
        self._split_ratio = split_ratio
        self._starting_equity = starting_equity
        self._cache: dict[tuple[str, int], dict[str, pd.DataFrame]] = {}

    def _bars(self, scenario_id: str, path_index: int) -> dict[str, pd.DataFrame]:
        key = (scenario_id, path_index)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._store.load_bars(self._library_id, scenario_id, path_index)
            self._cache[key] = cached
        return cached

    def scores(self, spec: StrategySpec, scenario_ids: tuple[str, ...]) -> list[float]:
        out: list[float] = []
        for scenario_id in scenario_ids:
            for path_index in range(self._n_paths):
                bars = self._bars(scenario_id, path_index)
                out.append(
                    evaluate_sample(
                        self._base_config,
                        spec,
                        bars,
                        self._start,
                        self._end,
                        split_ratio=self._split_ratio,
                        starting_equity=self._starting_equity,
                    )
                )
        return out


def run_optimization(
    strategy_type: str,
    *,
    library_id: str = "ai_v1",
    store: SyntheticStore | None = None,
    base_config: AppConfig | None = None,
    cem_config: CEMConfig | None = None,
    lam: float = DEFAULT_LAMBDA,
    warmup_days: int | None = None,
    split_ratio: float = 0.7,
    starting_equity: float = DEFAULT_STARTING_EQUITY,
    n_paths: int | None = None,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    split_seed: int = 0,
) -> OptimizationResult:
    """
    Optimiza por CEM los parametros de una primitiva sobre la libreria sintetica.

    - Recompensa: media - lam*std del Calmar OOS sobre las muestras de TRAIN.
    - Hold-out: escenarios enteros reservados como validation (nunca vistos por el CEM).
    - Subsampling: `n_paths` limita paths por escenario para acotar el coste; si es
      menor que el total disponible, se AVISA por log (nada de recortes silenciosos).
    - Determinista de punta a punta (split_seed + cem_config.seed + backtest).
    """
    store = store or SyntheticStore()
    base_config = base_config or load_config(DEFAULT_SYNTHETIC_CONFIG)
    space: ParamSpace = get_space(strategy_type)

    manifest = store.load_manifest(library_id)
    scenario_ids = [s["id"] for s in manifest.scenarios]
    split = split_scenarios(
        scenario_ids, validation_fraction=validation_fraction, seed=split_seed
    )

    warmup = warmup_days if warmup_days is not None else base_config.runner.lookback_days + 5
    start, end = sample_window(manifest, warmup)

    total_paths = manifest.n_paths
    used_paths = total_paths if n_paths is None else min(n_paths, total_paths)
    if used_paths < total_paths:
        logger.warning(
            "Subsampling %d/%d paths per scenario (speed vs variance tradeoff)",
            used_paths, total_paths,
        )

    evaluator = _SampleEvaluator(
        store, library_id, base_config, start, end, used_paths,
        split_ratio=split_ratio, starting_equity=starting_equity,
    )

    def make_spec(vector) -> StrategySpec:
        params = space.to_params(vector)
        return StrategySpec(type=strategy_type, id=f"{strategy_type}_cem", params=params)

    def objective(vector) -> float:
        spec = make_spec(vector)
        return aggregate_reward(evaluator.scores(spec, split.train), lam=lam).reward

    logger.info(
        "Optimizing '%s' | train=%d scenarios x %d paths | validation=%d scenarios",
        strategy_type, split.n_train, used_paths, split.n_validation,
    )
    cem_result = maximize(objective, space.lows, space.highs, cem_config or CEMConfig())

    best_params = space.to_params(cem_result.best_vector)
    best_spec = StrategySpec(type=strategy_type, id=f"{strategy_type}_cem", params=best_params)

    train_stats = aggregate_reward(evaluator.scores(best_spec, split.train), lam=lam)
    validation_stats = aggregate_reward(evaluator.scores(best_spec, split.validation), lam=lam)

    logger.info(
        "Done '%s' | train reward=%.4f | validation reward=%.4f | overfit gap=%.4f",
        strategy_type, train_stats.reward, validation_stats.reward,
        train_stats.reward - validation_stats.reward,
    )

    return OptimizationResult(
        strategy_type=strategy_type,
        best_params=best_params,
        train=train_stats,
        validation=validation_stats,
        split=split,
        n_paths_per_scenario=used_paths,
        total_paths_available=total_paths,
        history=cem_result.history,
    )
