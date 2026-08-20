"""
El sustrato SINTETICO del optimizador. APARCADO: ver `ai_trader/research/__init__.py`.

Fue el sustrato por defecto de `run_optimization` hasta que el estudio de transferencia
midio que el ranking sintetico no se parece al real (rho = -0,04, y -0,67 entre las
configuraciones que operan de verdad). Sigue aqui, entero y funcionando, porque los
estudios de esta carpeta lo usan y porque un resultado negativo caro no se borra; pero hay
que pedirlo explicitamente:

    from ai_trader.research.synthetic_source import SyntheticSampleSource
    run_optimization("crypto_momentum", source=SyntheticSampleSource.build())
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from ai_trader.backtest.engine import DEFAULT_STARTING_EQUITY
from ai_trader.backtest.metrics import DEFAULT_HEADLINE_WEIGHTS, HeadlineWeights
from ai_trader.config import AppConfig, StrategySpec, load_config
from ai_trader.research.synthetic.service import sample_window
from ai_trader.research.synthetic.store import SyntheticStore
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

logger = logging.getLogger(__name__)

DEFAULT_SYNTHETIC_CONFIG = Path("config") / "synthetic.toml"

# Libreria por defecto de este sustrato. 'ai_v2' es la realista (colas t-Student,
# clustering de volatilidad y autocorrelacion serial); 'ai_v1' es ruido iid y se conserva
# solo como referencia comparativa: optimizar sobre ella premia sesgos optimistas.
DEFAULT_LIBRARY_ID = "ai_v2"


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
