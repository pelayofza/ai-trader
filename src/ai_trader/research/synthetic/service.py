from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from ai_trader.research.synthetic.designer import ScenarioDesigner
from ai_trader.research.synthetic.engine import DEFAULT_ANCHOR, generate_paths
from ai_trader.research.synthetic.scenarios import ScenarioSpec
from ai_trader.research.synthetic.store import LibraryManifest, SyntheticStore
from ai_trader.research.synthetic.universe import (
    DEFAULT_UNIVERSE,
    SyntheticUniverse,
    universe_from_summary,
    universe_summary,
)

logger = logging.getLogger(__name__)

DEFAULT_N_SCENARIOS = 24
DEFAULT_N_PATHS = 30
DEFAULT_HORIZON_DAYS = 730
DEFAULT_SEED_BASE = 1_000

# Separacion entre los rangos de semillas de escenarios distintos. Ancho y fijo (no
# depende de n_paths) para poder ampliar el numero de paths mas tarde sin que las
# semillas de un escenario invadan las del siguiente: los paths ya generados siguen
# siendo identicos y solo se anaden los nuevos.
SEED_STRIDE = 1_000_000


def _scenario_seed(seed_base: int, index: int) -> int:
    return seed_base + index * SEED_STRIDE


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

        paths_by_scenario, scenario_meta = self._synthesize(
            specs, self.universe, seed_base, n_paths, self.anchor
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

    def resynthesize(self, library_id: str, *, n_paths: int | None = None) -> LibraryManifest:
        """
        Regenera los paths de una libreria existente a partir de sus escenarios ya
        guardados, SIN llamar a la IA. Sirve para ampliar el ensemble (mas paths) o
        para reconstruir los datos si se borraron: solo los spec.json son insustituibles.

        Los paths ya existentes se conservan identicos (mismas semillas); con un n_paths
        mayor solo se anaden los nuevos. El universo se reconstruye del propio manifiesto
        (autocontenido); si es una libreria antigua sin start_price, cae al universo del
        codigo. De paso, actualiza el manifiesto al formato autocontenido.
        """
        manifest = self.store.load_manifest(library_id)
        universe = self._universe_for(manifest)
        total = n_paths or manifest.n_paths
        anchor = datetime.fromisoformat(manifest.anchor)
        specs = self.store.load_specs(library_id)

        logger.info(
            "Resynthesizing '%s' from %s stored scenarios -> %s paths (no API call)",
            library_id, len(specs), total,
        )
        paths_by_scenario, scenario_meta = self._synthesize(
            specs, universe, manifest.seed_base, total, anchor
        )

        new_manifest = LibraryManifest(
            library_id=manifest.library_id,
            created_at=manifest.created_at,
            horizon_days=manifest.horizon_days,
            anchor=manifest.anchor,
            n_paths=total,
            seed_base=manifest.seed_base,
            designer=manifest.designer,
            factors=list(universe.factors),
            universe=universe_summary(universe),
            scenarios=scenario_meta,
        )
        self.store.save(new_manifest, specs, paths_by_scenario)
        return new_manifest

    def derive_library(
        self,
        source_id: str,
        target_id: str,
        *,
        enricher: Callable[[ScenarioSpec], ScenarioSpec],
        n_paths: int | None = None,
    ) -> LibraryManifest:
        """
        Deriva una nueva libreria a partir de otra existente aplicando `enricher` a cada
        spec, SIN llamar a la IA. Es el camino del retrofit determinista: toma ai_v1,
        enriquece cada escenario con microestructura y produce ai_v2 con las mismas
        semillas/universo/anchor. La libreria fuente se conserva intacta.
        """
        source = self.store.load_manifest(source_id)
        universe = self._universe_for(source)
        total = n_paths or source.n_paths
        anchor = datetime.fromisoformat(source.anchor)
        specs = [enricher(spec) for spec in self.store.load_specs(source_id)]

        logger.info(
            "Deriving '%s' from '%s' (%s scenarios, %s paths) via %s",
            target_id, source_id, len(specs), total, getattr(enricher, "__name__", "enricher"),
        )
        paths_by_scenario, scenario_meta = self._synthesize(
            specs, universe, source.seed_base, total, anchor
        )

        manifest = LibraryManifest(
            library_id=target_id,
            created_at=source.created_at,
            horizon_days=source.horizon_days,
            anchor=source.anchor,
            n_paths=total,
            seed_base=source.seed_base,
            designer=f"{source.designer}+{getattr(enricher, '__name__', 'enricher')}",
            factors=list(universe.factors),
            universe=universe_summary(universe),
            scenarios=scenario_meta,
        )
        self.store.save(manifest, specs, paths_by_scenario)
        return manifest

    def _synthesize(
        self,
        specs: list[ScenarioSpec],
        universe: SyntheticUniverse,
        seed_base: int,
        n_paths: int,
        anchor: datetime,
    ) -> tuple[dict, list[dict]]:
        """Sintetiza el ensemble de cada escenario y devuelve (paths, metadatos)."""
        paths_by_scenario: dict = {}
        scenario_meta: list[dict] = []
        for i, spec in enumerate(specs):
            seed = _scenario_seed(seed_base, i)
            paths_by_scenario[spec.id] = generate_paths(
                spec, universe, n_paths=n_paths, seed_base=seed, anchor=anchor
            )
            scenario_meta.append(
                {
                    "id": spec.id,
                    "name": spec.name,
                    "narrative": spec.narrative,
                    "horizon_days": spec.horizon_days,
                    "seed_base": seed,
                }
            )
        return paths_by_scenario, scenario_meta

    def _universe_for(self, manifest: LibraryManifest) -> SyntheticUniverse:
        """Universo con el que regenerar: el del propio manifiesto si es autocontenido;
        si no (libreria antigua sin start_price), el universo del codigo."""
        try:
            return universe_from_summary(manifest.universe, tuple(manifest.factors))
        except (KeyError, ValueError) as exc:
            logger.warning(
                "Manifest universe not self-contained (%s); falling back to code universe",
                exc,
            )
            return DEFAULT_UNIVERSE


# Margen de calentamiento EXTRA, sobre `lookback_days`, con el que los ESTUDIOS abren su
# ventana sintetica. Vive aqui porque tiene que ser el MISMO en todos: dos estudios que
# empiecen en dias distintos evaluan las mismas configuraciones sobre ventanas distintas y
# sus cifras dejan de ser comparables sin que nada avise. Lo estreno el estudio de
# transferencia; el del canal de observacion lo necesita identico para que su celda de
# control reproduzca `data/transfer/units_ai_v3.json` score a score.
STUDY_WARMUP_MARGIN_DAYS = 5


def study_window(
    manifest: LibraryManifest, lookback_days: int
) -> tuple[datetime, datetime]:
    """La ventana con la que los estudios puntuan una muestra de esta libreria."""
    return sample_window(manifest, lookback_days + STUDY_WARMUP_MARGIN_DAYS)


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
