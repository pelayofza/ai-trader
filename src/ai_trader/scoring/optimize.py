from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from ai_trader.backtest.engine import DEFAULT_STARTING_EQUITY
from ai_trader.backtest.metrics import DEFAULT_HEADLINE_WEIGHTS, HeadlineWeights
from ai_trader.config import AppConfig, StrategySpec, load_config
from ai_trader.scoring.aggregate import DEFAULT_CVAR_ALPHA, RewardStats, aggregate_reward
from ai_trader.scoring.baselines import BASELINE_LABELS, BaselineGate, gate
from ai_trader.scoring.cem import CEMConfig, maximize
from ai_trader.scoring.overfit import (
    DeflatedSharpe,
    PBOResult,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from ai_trader.scoring.sample_eval import (
    SampleEvaluation,
    evaluate_baselines,
    evaluate_sample_detailed,
)
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

# Sustrato por defecto del scoring. 'ai_v2' es la libreria realista (colas t-Student,
# clustering de volatilidad y autocorrelacion serial); 'ai_v1' es ruido iid y se conserva
# solo como referencia comparativa: optimizar sobre ella premia sesgos optimistas.
DEFAULT_LIBRARY_ID = "ai_v2"


@dataclass(slots=True)
class OptimizationResult:
    """
    Resultado de optimizar los params de una primitiva por CEM sobre la libreria
    sintetica. Reporta train Y validation (hold-out de escenarios enteros) para hacer
    visible el gap de overfitting: la unidad de evaluacion es la distribucion, y la
    validacion son arquetipos que el CEM nunca vio.

    Tres capas de honestidad ademas del hold-out:
    - `gate`: la estrategia solo APRUEBA si bate al mejor baseline pasivo en validation.
    - `pbo`: probabilidad de que elegir por backtest sea elegir ruido.
    - `dsr`: Sharpe del ganador descontado por el nº de configuraciones probadas.
    """

    strategy_type: str
    best_params: dict
    train: RewardStats
    validation: RewardStats
    split: ScenarioSplit
    n_paths_per_scenario: int
    total_paths_available: int
    gate: BaselineGate
    pbo: PBOResult
    dsr: DeflatedSharpe
    headline_weights: HeadlineWeights = DEFAULT_HEADLINE_WEIGHTS
    history: list[dict] = field(default_factory=list)

    @property
    def overfit_gap(self) -> float:
        """Cuanto peor rinde en validacion que en train. Positivo = sobreajuste."""
        return self.train.reward - self.validation.reward

    @property
    def approved(self) -> bool:
        """Veredicto unico: bate al mejor baseline fuera de muestra."""
        return self.gate.approved

    def as_dict(self) -> dict:
        return {
            "strategy_type": self.strategy_type,
            "best_params": self.best_params,
            "approved": self.approved,
            "train": self.train.as_dict(),
            "validation": self.validation.as_dict(),
            "overfit_gap": round(self.overfit_gap, 4),
            "gate": self.gate.as_dict(),
            "pbo": self.pbo.as_dict(),
            "dsr": self.dsr.as_dict(),
            "headline_weights": self.headline_weights.as_dict(),
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
    para no releer parquet en cada iteracion del CEM. Los baselines de cada muestra no
    dependen de la estrategia, asi que se calculan una vez y se cachean tambien."""

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
        headline_weights: HeadlineWeights,
    ) -> None:
        self._store = store
        self._library_id = library_id
        self._base_config = base_config
        self._start = start
        self._end = end
        self._n_paths = n_paths
        self._split_ratio = split_ratio
        self._starting_equity = starting_equity
        self._headline_weights = headline_weights
        self._cache: dict[tuple[str, int], dict[str, pd.DataFrame]] = {}
        self._baseline_cache: dict[tuple[str, int], dict[str, float]] = {}

    def _bars(self, scenario_id: str, path_index: int) -> dict[str, pd.DataFrame]:
        key = (scenario_id, path_index)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._store.load_bars(self._library_id, scenario_id, path_index)
            self._cache[key] = cached
        return cached

    def _samples(self, scenario_ids: tuple[str, ...]):
        for scenario_id in scenario_ids:
            for path_index in range(self._n_paths):
                yield scenario_id, path_index

    def evaluations(
        self, spec: StrategySpec, scenario_ids: tuple[str, ...]
    ) -> list[SampleEvaluation]:
        return [
            evaluate_sample_detailed(
                self._base_config,
                spec,
                self._bars(scenario_id, path_index),
                self._start,
                self._end,
                split_ratio=self._split_ratio,
                starting_equity=self._starting_equity,
                headline_weights=self._headline_weights,
            )
            for scenario_id, path_index in self._samples(scenario_ids)
        ]

    def scores(self, spec: StrategySpec, scenario_ids: tuple[str, ...]) -> list[float]:
        return [e.score for e in self.evaluations(spec, scenario_ids)]

    def baseline_scores(self, scenario_ids: tuple[str, ...]) -> dict[str, list[float]]:
        """Scores de los baselines por muestra, en el mismo orden que `scores`. Un
        baseline solo entra si esta disponible en TODAS las muestras: comparar contra
        una serie con huecos seria comparar contra otra cosa."""
        per_sample: list[dict[str, float]] = []
        for scenario_id, path_index in self._samples(scenario_ids):
            key = (scenario_id, path_index)
            cached = self._baseline_cache.get(key)
            if cached is None:
                cached = {
                    name: b.score
                    for name, b in evaluate_baselines(
                        self._base_config,
                        self._bars(scenario_id, path_index),
                        self._start,
                        self._end,
                        split_ratio=self._split_ratio,
                        starting_equity=self._starting_equity,
                        headline_weights=self._headline_weights,
                    ).items()
                }
                self._baseline_cache[key] = cached
            per_sample.append(cached)

        if not per_sample:
            return {}

        complete = set.intersection(*(set(d) for d in per_sample))
        return {name: [d[name] for d in per_sample] for name in sorted(complete)}


def run_optimization(
    strategy_type: str,
    *,
    library_id: str = DEFAULT_LIBRARY_ID,
    store: SyntheticStore | None = None,
    base_config: AppConfig | None = None,
    cem_config: CEMConfig | None = None,
    cvar_alpha: float = DEFAULT_CVAR_ALPHA,
    headline_weights: HeadlineWeights = DEFAULT_HEADLINE_WEIGHTS,
    warmup_days: int | None = None,
    split_ratio: float = 0.7,
    starting_equity: float = DEFAULT_STARTING_EQUITY,
    n_paths: int | None = None,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    split_seed: int = 0,
) -> OptimizationResult:
    """
    Optimiza por CEM los parametros de una primitiva sobre la libreria sintetica.

    - Sustrato: `library_id`, por defecto DEFAULT_LIBRARY_ID ('ai_v2', la libreria
      realista). Pasa el id explicitamente para optimizar sobre otra libreria.
    - Score por muestra: `Sharpe - lambda*turnover - kappa*maxDD` OUT-OF-SAMPLE.
    - Recompensa: CVaR@`cvar_alpha` de esos scores sobre las muestras de TRAIN, o sea
      la media del peor cuartil: se optimiza la cola mala, no el centro.
    - Hold-out: escenarios enteros reservados como validation (nunca vistos por el CEM).
    - Gate: en validation, la estrategia debe batir al MEJOR baseline pasivo para
      aprobar. Sin baseline disponible no hay aprobado.
    - Sobreajuste por multiples pruebas: se reportan PBO (sobre la matriz muestras x
      configuraciones que el CEM genero) y DSR (Sharpe del ganador deflactado).
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
        headline_weights=headline_weights,
    )

    def make_spec(vector) -> StrategySpec:
        params = space.to_params(vector)
        return StrategySpec(type=strategy_type, id=f"{strategy_type}_cem", params=params)

    # Cada configuracion que el CEM prueba se guarda entera (score por muestra + Sharpe
    # medio): es la evidencia con la que despues se mide el sobreajuste por multiples
    # pruebas. Sin ella, PBO y DSR tendrian que inventarse el nº de intentos.
    trial_scores: list[list[float]] = []
    trial_sharpes: list[float] = []

    def objective(vector) -> float:
        evaluations = evaluator.evaluations(make_spec(vector), split.train)
        scores = [e.score for e in evaluations]
        trial_scores.append(scores)
        trial_sharpes.append(_mean([e.sharpe for e in evaluations]))
        return aggregate_reward(scores, alpha=cvar_alpha).reward

    logger.info(
        "Optimizing '%s' | train=%d scenarios x %d paths | validation=%d scenarios",
        strategy_type, split.n_train, used_paths, split.n_validation,
    )
    cem_result = maximize(objective, space.lows, space.highs, cem_config or CEMConfig())

    best_params = space.to_params(cem_result.best_vector)
    best_spec = StrategySpec(type=strategy_type, id=f"{strategy_type}_cem", params=best_params)

    train_evals = evaluator.evaluations(best_spec, split.train)
    validation_evals = evaluator.evaluations(best_spec, split.validation)
    train_stats = aggregate_reward([e.score for e in train_evals], alpha=cvar_alpha)
    validation_stats = aggregate_reward([e.score for e in validation_evals], alpha=cvar_alpha)

    # El gate se juega en VALIDATION: batir baselines en los escenarios que el CEM ya vio
    # no probaria nada.
    baseline_scores = evaluator.baseline_scores(split.validation)
    baseline_gate = gate(
        [e.score for e in validation_evals],
        baseline_scores,
        alpha=cvar_alpha,
        missing=_missing_baselines(baseline_scores),
    )

    # Matriz muestras x configuraciones (train) para el CSCV.
    pbo = probability_of_backtest_overfitting(_transpose(trial_scores))
    dsr = deflated_sharpe_ratio(
        _mean([e.sharpe for e in validation_evals]),
        trial_sharpes,
        _median_int([e.oos_observations for e in validation_evals]),
        skew=_mean([e.returns_skew for e in validation_evals]),
        kurtosis=_mean([e.returns_kurtosis for e in validation_evals], default=3.0),
    )

    logger.info(
        "Done '%s' | train reward=%.4f | validation reward=%.4f | overfit gap=%.4f "
        "| gate=%s (mejor baseline: %s %.4f) | PBO=%.3f | DSR=%.3f",
        strategy_type, train_stats.reward, validation_stats.reward,
        train_stats.reward - validation_stats.reward,
        "APROBADO" if baseline_gate.approved else "RECHAZADO",
        baseline_gate.best_name, baseline_gate.best_reward,
        pbo.pbo, dsr.dsr,
    )

    return OptimizationResult(
        strategy_type=strategy_type,
        best_params=best_params,
        train=train_stats,
        validation=validation_stats,
        split=split,
        n_paths_per_scenario=used_paths,
        total_paths_available=total_paths,
        gate=baseline_gate,
        pbo=pbo,
        dsr=dsr,
        headline_weights=headline_weights,
        history=cem_result.history,
    )


def _mean(values: list[float], *, default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _median_int(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _transpose(rows: list[list[float]]) -> list[list[float]]:
    """De [config][muestra] a [muestra][config], que es la forma que espera el CSCV.
    Las configuraciones con distinto nº de muestras se descartan (no deberia pasar)."""
    if not rows:
        return []
    width = len(rows[0])
    usable = [r for r in rows if len(r) == width]
    return [list(col) for col in zip(*usable)]


def _missing_baselines(available: dict[str, list[float]]) -> tuple[str, ...]:
    """Baselines que no se pudieron construir en esta libreria (p.ej. SPY en un universo
    solo cripto). Se declaran, no se silencian."""
    return tuple(name for name in sorted(BASELINE_LABELS) if name not in available)
