from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Iterable, Sequence
from typing import Iterator

import pandas as pd

from ai_trader.shared.bars import OHLCV_COLUMNS
from ai_trader.synthetic.scenarios import ScenarioSpec

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path("data") / "synthetic"
_PATH_COL = "path"
_SYMBOL_COL = "symbol"
_TS_COL = "timestamp"


@dataclass(slots=True)
class LibraryManifest:
    """Metadatos de una libreria: lo minimo para re-usar los datos sin re-generarlos."""

    library_id: str
    created_at: str
    horizon_days: int
    anchor: str
    n_paths: int
    seed_base: int
    designer: str
    factors: list[str]
    universe: list[dict]
    scenarios: list[dict] = field(default_factory=list)

    @property
    def num_scenarios(self) -> int:
        return len(self.scenarios)

    @property
    def num_samples(self) -> int:
        return self.num_scenarios * self.n_paths

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> LibraryManifest:
        return cls(**data)


class SyntheticStore:
    """
    Persistencia en disco de las librerias de escenarios sinteticos.

    Layout:
      <root>/<library_id>/manifest.json
      <root>/<library_id>/scenarios/<scenario_id>/spec.json
      <root>/<library_id>/scenarios/<scenario_id>/paths.parquet   (formato largo)

    El parquet guarda TODOS los paths y simbolos de un escenario en formato largo
    (columnas path, symbol, timestamp, OHLCV). load_bars pivota a dict[symbol->df],
    que es exactamente lo que consume BacktestEngine.from_bars.
    """

    def __init__(self, root: Path | str = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    # --- escritura ---------------------------------------------------------

    def save(
        self,
        manifest: LibraryManifest,
        specs: list[ScenarioSpec],
        paths_by_scenario: dict[str, list[dict[str, pd.DataFrame]]],
    ) -> Path:
        lib_dir = self.root / manifest.library_id
        (lib_dir / "scenarios").mkdir(parents=True, exist_ok=True)

        for spec in specs:
            sc_dir = lib_dir / "scenarios" / spec.id
            sc_dir.mkdir(parents=True, exist_ok=True)
            (sc_dir / "spec.json").write_text(
                json.dumps(spec.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
            paths = paths_by_scenario.get(spec.id, [])
            self._write_paths(sc_dir / "paths.parquet", paths)

        (lib_dir / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "Saved synthetic library '%s' (%s scenarios x %s paths) to %s",
            manifest.library_id, manifest.num_scenarios, manifest.n_paths, lib_dir,
        )
        return lib_dir

    def _write_paths(self, target: Path, paths: list[dict[str, pd.DataFrame]]) -> None:
        frames = []
        for path_index, bars in enumerate(paths):
            for symbol, df in bars.items():
                block = df.reset_index()
                block.columns = [_TS_COL] + list(df.columns)
                block.insert(0, _SYMBOL_COL, symbol)
                block.insert(0, _PATH_COL, path_index)
                frames.append(block)
        if not frames:
            return
        pd.concat(frames, ignore_index=True).to_parquet(target, index=False)

    # --- lectura -----------------------------------------------------------

    def list_libraries(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / "manifest.json").exists())

    def load_manifest(self, library_id: str) -> LibraryManifest:
        path = self.root / library_id / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"No synthetic library at {path}")
        return LibraryManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load_spec(self, library_id: str, scenario_id: str) -> ScenarioSpec:
        path = self.root / library_id / "scenarios" / scenario_id / "spec.json"
        return ScenarioSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load_specs(self, library_id: str) -> list[ScenarioSpec]:
        """Todos los specs de la libreria, en el orden del manifiesto. Son la salida
        (cara) de la IA; a partir de ellos se regeneran paths sin volver a llamarla."""
        manifest = self.load_manifest(library_id)
        return [self.load_spec(library_id, s["id"]) for s in manifest.scenarios]

    def load_bars(
        self, library_id: str, scenario_id: str, path_index: int
    ) -> dict[str, pd.DataFrame]:
        """Puente hacia el backtest: un (escenario, path) -> dict[symbol -> OHLCV]."""
        df = self._read_paths_frame(library_id, scenario_id)
        subset = df[df[_PATH_COL] == path_index]
        if subset.empty:
            raise ValueError(f"Path {path_index} not found in {scenario_id}")
        return self._pivot_paths(subset, OHLCV_COLUMNS)

    def load_scenario_paths(
        self,
        library_id: str,
        scenario_id: str,
        *,
        path_indices: Iterable[int] | None = None,
        columns: Sequence[str] = OHLCV_COLUMNS,
    ) -> dict[int, dict[str, pd.DataFrame]]:
        """
        Varios paths de un escenario con UNA sola lectura del parquet.

        `load_bars` relee el fichero entero en cada llamada, lo que es correcto cuando se
        quiere una muestra suelta y muy caro cuando se quieren muchas del mismo escenario
        (el estudio de fidelidad recorre la libreria entera). `columns` permite ademas
        leer solo lo que se va a usar: con los cierres basta para medir stylized-facts.
        """
        df = self._read_paths_frame(library_id, scenario_id, columns=columns)
        if path_indices is not None:
            df = df[df[_PATH_COL].isin(list(path_indices))]
        return {
            int(path_index): self._pivot_paths(group, columns)
            for path_index, group in df.groupby(_PATH_COL)
        }

    def _read_paths_frame(
        self, library_id: str, scenario_id: str, *, columns: Sequence[str] | None = None
    ) -> pd.DataFrame:
        parquet = self.root / library_id / "scenarios" / scenario_id / "paths.parquet"
        if not parquet.exists():
            raise FileNotFoundError(f"No paths parquet at {parquet}")
        read_columns = (
            None if columns is None else [_PATH_COL, _SYMBOL_COL, _TS_COL, *columns]
        )
        return pd.read_parquet(parquet, columns=read_columns)

    @staticmethod
    def _pivot_paths(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, pd.DataFrame]:
        bars: dict[str, pd.DataFrame] = {}
        for symbol, group in frame.groupby(_SYMBOL_COL):
            ohlcv = group.set_index(_TS_COL)[list(columns)].sort_index()
            ohlcv.index.name = _TS_COL
            bars[str(symbol)] = ohlcv
        return bars

    def iter_samples(
        self, library_id: str, scenario_id: str | None = None
    ) -> Iterator[tuple[str, int, dict[str, pd.DataFrame]]]:
        """Recorre las muestras (escenario, path, bars) para alimentar backtest/RL."""
        manifest = self.load_manifest(library_id)
        scenario_ids = [scenario_id] if scenario_id else [s["id"] for s in manifest.scenarios]
        for sc_id in scenario_ids:
            for path_index in range(manifest.n_paths):
                yield sc_id, path_index, self.load_bars(library_id, sc_id, path_index)
