from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

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
from ai_trader.scoring.real_source import RealWindowSource, TemporalSplit
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
    Resultado de optimizar los params de una primitiva por CEM sobre un sustrato.

    Reporta train Y validation (hold-out de UNIDADES enteras) para hacer visible el gap de
    overfitting: la unidad de evaluacion es la distribucion, y la validacion son unidades
    que el CEM nunca vio. `substrate` dice sobre que se optimizo -- sin eso las cifras no
    son auditables, porque el mismo numero significa cosas distintas segun el sustrato.

    Tres capas de honestidad ademas del hold-out:
    - `gate`: la estrategia solo APRUEBA si bate al mejor baseline pasivo en validation Y
      supera el suelo de actividad (`scoring.activity`): no hacer nada bate a los pasivos
      en un mercado que cae, y eso no es batirlos. `train.activity` y
      `validation.activity` publican las operaciones por muestra al lado de la recompensa.
    - `pbo`: probabilidad de que elegir por backtest sea elegir ruido.
    - `dsr`: Sharpe del ganador descontado por el nº de configuraciones probadas.
    """

    strategy_type: str
    best_params: dict
    train: RewardStats
    validation: RewardStats
    split: ScenarioSplit | TemporalSplit
    substrate: dict
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
            "rankable": self.gate.eligible,
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
            "substrate": self.substrate,
        }


class SampleSource(Protocol):
    """
    De donde salen las muestras sobre las que se puntua una configuracion.

    El optimizador no sabe -y no debe saber- si detras hay caminos de una libreria
    sintetica o folds de una sub-ventana del historico real. Solo pide dos cosas: que
    UNIDADES hay a cada lado del hold-out, y como se evalua una spec sobre unas unidades.

    Una unidad es la unidad de HOLD-OUT (un escenario entero, una sub-ventana entera), no
    una muestra suelta: si una muestra de validacion pudiera pertenecer a una unidad ya
    vista en train, el gap de sobreajuste quedaria enmascarado.
    """

    @property
    def train_units(self) -> tuple[str, ...]:
        ...

    @property
    def validation_units(self) -> tuple[str, ...]:
        ...

    def evaluations(
        self, spec: StrategySpec, units: tuple[str, ...]
    ) -> list[SampleEvaluation]:
        ...

    def baseline_scores(self, units: tuple[str, ...]) -> dict[str, list[float]]:
        ...

    def describe(self) -> dict:
        """El sustrato, serializado junto al resultado: sin esto las cifras no son
        auditables, porque no se sabria sobre QUE se optimizo."""
        ...


class SyntheticSampleSource:
    """Muestras = (escenario, path) de una libreria sintetica almacenada.

    Cachea las barras cargadas para no releer parquet en cada iteracion del CEM. Los
    baselines de cada muestra no dependen de la estrategia, asi que se calculan una vez
    y se cachean tambien."""

    def __init__(
        self,
        store,
        library_id: str,
        base_config: AppConfig,
        start: datetime,
        end: datetime,
        n_paths: int,
        split: ScenarioSplit,
        total_paths: int,
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
        self.split = split
        self.total_paths = total_paths
        self._split_ratio = split_ratio
        self._starting_equity = starting_equity
        self._headline_weights = headline_weights
        self._cache: dict[tuple[str, int], dict[str, pd.DataFrame]] = {}
        self._baseline_cache: dict[tuple[str, int], dict[str, float]] = {}

    @classmethod
    def build(
        cls,
        *,
        library_id: str = DEFAULT_LIBRARY_ID,
        store: SyntheticStore | None = None,
        base_config: AppConfig | None = None,
        warmup_days: int | None = None,
        n_paths: int | None = None,
        validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
        split_seed: int = 0,
        split_ratio: float = 0.7,
        starting_equity: float = DEFAULT_STARTING_EQUITY,
        headline_weights: HeadlineWeights = DEFAULT_HEADLINE_WEIGHTS,
    ) -> SyntheticSampleSource:
        """Resuelve libreria, particion de escenarios y ventana desde el manifiesto.

        El hold-out es de ESCENARIOS ENTEROS: los arquetipos macro reservados aqui son los
        que el CEM no vera nunca. `n_paths` acota el coste subsampleando caminos por
        escenario y AVISA por log si recorta (nada de recortes silenciosos)."""
        store = store or SyntheticStore()
        base_config = base_config or load_config(DEFAULT_SYNTHETIC_CONFIG)
        manifest = store.load_manifest(library_id)
        scenario_ids = [s["id"] for s in manifest.scenarios]
        split = split_scenarios(
            scenario_ids, validation_fraction=validation_fraction, seed=split_seed
        )
        warmup = (
            warmup_days if warmup_days is not None else base_config.runner.lookback_days + 5
        )
        start, end = sample_window(manifest, warmup)

        total_paths = manifest.n_paths
        used_paths = total_paths if n_paths is None else min(n_paths, total_paths)
        if used_paths < total_paths:
            logger.warning(
                "Subsampling %d/%d paths per scenario (speed vs variance tradeoff)",
                used_paths, total_paths,
            )
        return cls(
            store, library_id, base_config, start, end, used_paths, split, total_paths,
            split_ratio=split_ratio, starting_equity=starting_equity,
            headline_weights=headline_weights,
        )

    @property
    def train_units(self) -> tuple[str, ...]:
        return self.split.train

    @property
    def validation_units(self) -> tuple[str, ...]:
        return self.split.validation

    def describe(self) -> dict:
        return {
            "substrate": "sintetico",
            "library_id": self._library_id,
            "n_paths_per_unit": self._n_paths,
            "total_paths_available": self.total_paths,
            "window": {
                "start": self._start.date().isoformat(),
                "end": self._end.date().isoformat(),
            },
        }

    def _bars(self, scenario_id: str, path_index: int) -> dict[str, pd.DataFrame]:
        key = (scenario_id, path_index)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._store.load_bars(self._library_id, scenario_id, path_index)
            self._cache[key] = cached
        return cached

    def _samples(self, units: tuple[str, ...]):
        for scenario_id in units:
            for path_index in range(self._n_paths):
                yield scenario_id, path_index

    def evaluations(
        self, spec: StrategySpec, units: tuple[str, ...]
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
            for scenario_id, path_index in self._samples(units)
        ]

    def scores(self, spec: StrategySpec, units: tuple[str, ...]) -> list[float]:
        return [e.score for e in self.evaluations(spec, units)]

    def baseline_scores(self, units: tuple[str, ...]) -> dict[str, list[float]]:
        """Scores de los baselines por muestra, en el mismo orden que `scores`. Un
        baseline solo entra si esta disponible en TODAS las muestras: comparar contra
        una serie con huecos seria comparar contra otra cosa."""
        per_sample: list[dict[str, float]] = []
        for scenario_id, path_index in self._samples(units):
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
    source: SampleSource | None = None,
    base_config: AppConfig | None = None,
    cem_config: CEMConfig | None = None,
    cvar_alpha: float = DEFAULT_CVAR_ALPHA,
    headline_weights: HeadlineWeights = DEFAULT_HEADLINE_WEIGHTS,
    starting_equity: float = DEFAULT_STARTING_EQUITY,
    offline: bool = True,
    signals: dict | None = None,
) -> OptimizationResult:
    """
    Optimiza por CEM los parametros de una primitiva sobre el HISTORICO REAL.

    - Sustrato: `source`, y por omision `RealWindowSource` -- sub-ventanas del mercado con
      CPCV purgado y hold-out temporal. Fue sintetico hasta que el estudio de transferencia
      midio que el ranking de los dos mundos no se parece (rho = -0,04, y -0,67 entre las
      que operan de verdad): un juez del que se sabe que no transfiere no puede seguir
      eligiendo. Para volver a puntuar sobre una libreria generada hay que pedirlo
      explicitamente pasando `source=SyntheticSampleSource.build(...)`.
    - Score por muestra: `Sharpe - lambda*turnover - kappa*maxDD` OUT-OF-SAMPLE.
    - Recompensa: CVaR@`cvar_alpha` de esos scores sobre las muestras de TRAIN, o sea
      la media del peor cuartil: se optimiza la cola mala, no el centro.
    - Hold-out: unidades enteras reservadas como validation (nunca vistas por el CEM).
    - Gate: en validation, la estrategia debe batir al MEJOR baseline pasivo para
      aprobar. Sin baseline disponible no hay aprobado.
    - Sobreajuste por multiples pruebas: se reportan PBO (sobre la matriz muestras x
      configuraciones que el CEM genero) y DSR (Sharpe del ganador deflactado). Sobre el
      sustrato real IMPORTAN MAS, no menos: son cuatro unidades de train de un unico camino
      historico, y ese es el problema que el sintetico venia a resolver y no resolvio.
    - Determinista de punta a punta (geometria de folds + cem_config.seed + backtest).
    - COSTE: ~8 min por candidata con la cache caliente (ver `scoring.real_source`). Una
      corrida completa de CEM se mide en horas, no en minutos.
    """
    space: ParamSpace = get_space(strategy_type)

    source = source or RealWindowSource.build(
        base_config=base_config,
        offline=offline,
        starting_equity=starting_equity,
        headline_weights=headline_weights,
        signals=signals,
    )
    split = source.split
    train_units, validation_units = source.train_units, source.validation_units

    def make_spec(vector) -> StrategySpec:
        params = space.to_params(vector)
        return StrategySpec(type=strategy_type, id=f"{strategy_type}_cem", params=params)

    # Cada configuracion que el CEM prueba se guarda entera (score por muestra + Sharpe
    # medio): es la evidencia con la que despues se mide el sobreajuste por multiples
    # pruebas. Sin ella, PBO y DSR tendrian que inventarse el nº de intentos.
    trial_scores: list[list[float]] = []
    trial_sharpes: list[float] = []

    def objective(vector) -> float:
        evaluations = source.evaluations(make_spec(vector), train_units)
        scores = [e.score for e in evaluations]
        trial_scores.append(scores)
        trial_sharpes.append(_mean([e.sharpe for e in evaluations]))
        return aggregate_reward(scores, alpha=cvar_alpha).reward

    logger.info(
        "Optimizing '%s' | train=%d unidades | validation=%d unidades | sustrato=%s",
        strategy_type, len(train_units), len(validation_units),
        source.describe().get("substrate", "?"),
    )
    cem_result = maximize(objective, space.lows, space.highs, cem_config or CEMConfig())

    best_params = space.to_params(cem_result.best_vector)
    best_spec = StrategySpec(type=strategy_type, id=f"{strategy_type}_cem", params=best_params)

    train_evals = source.evaluations(best_spec, train_units)
    validation_evals = source.evaluations(best_spec, validation_units)
    # La actividad acompana a la recompensa desde el primer sitio en el que se calcula: una
    # muestra sin operaciones puntua 0 EXACTO y ese 0 es indistinguible de "no perdio" si
    # no se publica al lado cuantas veces se opero. Ver `scoring.activity`.
    train_stats = aggregate_reward(
        [e.score for e in train_evals], alpha=cvar_alpha,
        trades=[e.num_trades for e in train_evals],
    )
    validation_stats = aggregate_reward(
        [e.score for e in validation_evals], alpha=cvar_alpha,
        trades=[e.num_trades for e in validation_evals],
    )

    # El gate se juega en VALIDATION: batir baselines en los escenarios que el CEM ya vio
    # no probaria nada.
    baseline_scores = source.baseline_scores(validation_units)
    baseline_gate = gate(
        [e.score for e in validation_evals],
        baseline_scores,
        alpha=cvar_alpha,
        missing=_missing_baselines(baseline_scores),
        trades=[e.num_trades for e in validation_evals],
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

    activity = validation_stats.activity
    logger.info(
        "Done '%s' | train reward=%.4f | validation reward=%.4f | overfit gap=%.4f "
        "| ops/muestra=%.1f (%.0f%% vacias)%s | gate=%s (mejor baseline: %s %.4f) "
        "| PBO=%.3f | DSR=%.3f",
        strategy_type, train_stats.reward, validation_stats.reward,
        train_stats.reward - validation_stats.reward,
        activity.trades_per_window if activity else float("nan"),
        activity.zero_window_pct if activity else float("nan"),
        "" if baseline_gate.eligible else " NO RANKEABLE",
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
        substrate=source.describe(),
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
