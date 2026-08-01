from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from ai_trader.synthetic.designer import ScenarioDesigner
from ai_trader.synthetic.engine import DEFAULT_ANCHOR, generate_paths
from ai_trader.synthetic.store import LibraryManifest, SyntheticStore
from ai_trader.synthetic.universe import (
    DEFAULT_UNIVERSE,
    SyntheticUniverse,
    universe_summary,
)

logger = logging.getLogger(__name__)

DEFAULT_N_SCENARIOS = 24
DEFAULT_N_PATHS = 30
DEFAULT_HORIZON_DAYS = 730
DEFAULT_SEED_BASE = 1_000


class SyntheticDataService:
    """
    Orquesta el generador: pide escenarios al disenador (IA o plantilla), sintetiza el
    ensemble Monte Carlo de cada uno con el motor determinista y lo persiste.

    Es la fachada de la pieza. No conoce backtest, riesgo ni estrategias: solo produce
    y guarda datos. El puente hacia el backtest es SyntheticStore.load_bars.
    """

    def __init__(
        self,
        designer: ScenarioDesigner,
        universe: SyntheticUniverse = DEFAULT_UNIVERSE,
        store: SyntheticStore | None = None,
        *,
        anchor: datetime = DEFAULT_ANCHOR,
    ) -> None:
        self.designer = designer
        self.universe = universe
        self.store = store or SyntheticStore()
        self.anchor = anchor

    def generate(
        self,
        library_id: str,
        *,
        n_scenarios: int = DEFAULT_N_SCENARIOS,
        n_paths: int = DEFAULT_N_PATHS,
        horizon_days: int = DEFAULT_HORIZON_DAYS,
        seed_base: int = DEFAULT_SEED_BASE,
        created_at: datetime | None = None,
    ) -> LibraryManifest:
        logger.info(
            "Designing %s scenarios (%s paths each, horizon %s) with %s",
            n_scenarios, n_paths, horizon_days, type(self.designer).__name__,
        )
        specs = self.designer.design(self.universe, n_scenarios, horizon_days)

        paths_by_scenario = {}
        scenario_meta = []
        for i, spec in enumerate(specs):
            # Semillas contiguas y sin solape entre escenarios: reproducibilidad total.
            scenario_seed = seed_base + i * n_paths
            paths_by_scenario[spec.id] = generate_paths(
                spec,
                self.universe,
                n_paths=n_paths,
                seed_base=scenario_seed,
                anchor=self.anchor,
            )
            scenario_meta.append(
                {
                    "id": spec.id,
                    "name": spec.name,
                    "narrative": spec.narrative,
                    "horizon_days": spec.horizon_days,
                    "seed_base": scenario_seed,
                }
            )

        stamp = created_at or datetime.now(timezone.utc)
        manifest = LibraryManifest(
            library_id=library_id,
            created_at=stamp.isoformat(),
            horizon_days=horizon_days,
            anchor=self.anchor.isoformat(),
            n_paths=n_paths,
            seed_base=seed_base,
            designer=type(self.designer).__name__,
            factors=list(self.universe.factors),
            universe=universe_summary(self.universe),
            scenarios=scenario_meta,
        )
        self.store.save(manifest, specs, paths_by_scenario)
        return manifest


def sample_window(
    manifest: LibraryManifest, warmup_days: int
) -> tuple[datetime, datetime]:
    """
    Rango (start, end) recomendado para backtestear una muestra de esta libreria.

    Deja `warmup_days` de calentamiento al principio para que la estrategia tenga
    lookback completo el primer dia negociable. `end` es el ultimo dia de la serie.
    """
    anchor = datetime.fromisoformat(manifest.anchor)
    start = anchor + timedelta(days=warmup_days)
    end = anchor + timedelta(days=manifest.horizon_days - 1)
    if start >= end:
        raise ValueError(
            f"warmup_days={warmup_days} leaves no room in a {manifest.horizon_days}-day horizon"
        )
    return start, end
