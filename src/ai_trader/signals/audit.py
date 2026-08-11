"""
AUDITORIA: el descubrimiento es MEDICION, no curacion.

Dos preguntas, las dos con numero y ninguna con opinion:

1. ¿Que parte del universo tiene ENTIDAD, y de donde sale? (`audit_entities`)
   Cuantos simbolos resuelve la regla derivada, cuantos dependen de un override y cuantos
   no se resuelven en absoluto. Es la cifra que decide si la tabla de overrides
   (`shared/entities.py::ENTITY_OVERRIDES`, hoy vacia) tiene que crecer y por que caso
   concreto. Sin ella, la tabla se rellena "por si acaso", que es como se acaba con
   veinticuatro entradas escritas a mano que nadie sabe si siguen siendo ciertas.

2. ¿Que hay ARCHIVADO, por fuente, entidad y ano? (`audit_archive`)
   Dias con dato, primera y ultima observacion, huecos y el hueco mas largo. Es lo que
   convierte `history_from` de promesa en medicion: la fecha del catalogo se escribe
   cuando esta auditoria dice desde cuando hay dato de verdad, no cuando el proveedor
   anuncia su backfill.

EL DENOMINADOR DE LA COBERTURA, Y POR QUE NO ES EL ANO ENTERO
--------------------------------------------------------------
La cobertura de un ano se mide contra el TRAMO REALMENTE CUBIERTO (de la primera a la
ultima observacion de ese ano), no contra los 365 dias. Una fuente que se conecto en
noviembre no tiene un 8% de cobertura de ese ano: tiene el 100% de lo que lleva capturando,
y esa es la cifra que dice si hay huecos. Cuando lo que se quiera saber sea la profundidad,
la respuesta es `first_day`, que esta al lado y no se puede confundir con un porcentaje.

Para una fuente por evento (`cadence='event'`) la cobertura es None, no 100. No hay
cadencia esperada: un dia sin hack no es un dia sin dato, y llamarlo "falta" o "completo"
seria inventarse el significado.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from ai_trader.shared.clock import utc_now
from ai_trader.shared.entities import (
    SOURCE_OVERRIDE,
    SOURCE_RULE,
    SOURCE_UNMAPPED,
    EntityRef,
    resolve_entities,
)
from ai_trader.shared.signals import DAY, ENTITY, OBSERVED
from ai_trader.signals.catalog import CATALOG, SignalSource
from ai_trader.signals.source import RAW_DAY, RAW_ENTITY, RAW_FETCHED_AT, build_adapter
from ai_trader.signals.store import SignalStore


# --- 1. cobertura del mapeo simbolo -> entidad --------------------------------------


@dataclass(frozen=True, slots=True)
class EntityAudit:
    """Como resuelve el universo actual, con la procedencia de cada mapeo."""

    refs: tuple[EntityRef, ...]

    @property
    def n_symbols(self) -> int:
        return len(self.refs)

    @property
    def by_source(self) -> dict[str, int]:
        counts = {SOURCE_RULE: 0, SOURCE_OVERRIDE: 0, SOURCE_UNMAPPED: 0}
        for ref in self.refs:
            counts[ref.source] = counts.get(ref.source, 0) + 1
        return counts

    @property
    def n_resolved(self) -> int:
        return sum(1 for ref in self.refs if ref.resolved)

    @property
    def coverage_pct(self) -> float:
        return 100.0 * self.n_resolved / self.n_symbols if self.n_symbols else 0.0

    @property
    def unmapped(self) -> tuple[str, ...]:
        """Los simbolos sin entidad. Cada uno es una linea candidata a la tabla de
        overrides, y ninguna se escribe antes de aparecer aqui."""
        return tuple(ref.symbol for ref in self.refs if not ref.resolved)

    @property
    def collisions(self) -> dict[str, tuple[str, ...]]:
        """Entidades a las que apunta mas de un simbolo.

        NO es un error: "BTC/USDT" y "BTC/USD" son el mismo activo y deben compartir
        entidad —de hecho es la propiedad que hace que una fuente se descargue una sola
        vez—. Se publica porque una colision INESPERADA (dos activos distintos cayendo en
        la misma clave) es exactamente el fallo que la regla derivada puede cometer, y sin
        esta lista seria invisible.
        """
        groups: dict[str, list[str]] = {}
        for ref in self.refs:
            if ref.resolved:
                groups.setdefault(ref.key, []).append(ref.symbol)
        return {k: tuple(v) for k, v in sorted(groups.items()) if len(v) > 1}

    @property
    def by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for ref in self.refs:
            counts[ref.kind.value] = counts.get(ref.kind.value, 0) + 1
        return counts

    def as_dict(self) -> dict:
        return {
            "n_symbols": self.n_symbols,
            "n_resolved": self.n_resolved,
            "coverage_pct": round(self.coverage_pct, 2),
            "by_source": self.by_source,
            "by_kind": self.by_kind,
            "unmapped": list(self.unmapped),
            "collisions": {k: list(v) for k, v in self.collisions.items()},
            "rows": [ref.as_dict() for ref in self.refs],
        }


def audit_entities(symbols: Sequence[str]) -> EntityAudit:
    """Resuelve el universo entero y mide de donde sale cada entidad."""
    return EntityAudit(tuple(resolve_entities(symbols).values()))


# --- 2. cobertura del archivo -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class YearCoverage:
    """Lo archivado de una (fuente, entidad) en un ano."""

    entity: str
    year: int
    days: int
    observations: int
    first_day: str
    last_day: str
    expected_days: int | None
    max_gap_days: int

    @property
    def coverage_pct(self) -> float | None:
        """% de los dias esperados DENTRO del tramo cubierto. None en fuentes por evento."""
        if not self.expected_days:
            return None
        return round(min(100.0, 100.0 * self.days / self.expected_days), 2)

    def as_dict(self) -> dict:
        return {
            "entity": self.entity,
            "year": self.year,
            "days": self.days,
            "observations": self.observations,
            "first_day": self.first_day,
            "last_day": self.last_day,
            "expected_days": self.expected_days,
            "coverage_pct": self.coverage_pct,
            "max_gap_days": self.max_gap_days,
        }


@dataclass(frozen=True, slots=True)
class SourceCoverage:
    """Lo archivado de una fuente, con lo que el catalogo declara sobre ella."""

    source_key: str
    tier: str
    pit: str
    cadence: str
    connected: bool
    backtestable: bool
    history_from: str | None
    years: tuple[YearCoverage, ...]

    @property
    def entities(self) -> tuple[str, ...]:
        seen: list[str] = []
        for row in self.years:
            if row.entity not in seen:
                seen.append(row.entity)
        return tuple(seen)

    @property
    def records(self) -> int:
        return sum(row.observations for row in self.years)

    @property
    def measured_history_from(self) -> str | None:
        """La primera fecha con dato ARCHIVADO. Es la candidata a `history_from` del
        catalogo: la que se escribe cuando hay medicion, en vez de la que promete el
        proveedor."""
        days = [row.first_day for row in self.years if row.first_day]
        return min(days) if days else None

    @property
    def declared_matches_measured(self) -> bool | None:
        """¿Coincide lo declarado con lo medido? None si falta uno de los dos.

        Detecta las dos direcciones del desajuste: un `history_from` escrito a mano que el
        archivo no respalda, y un archivo mas profundo que lo declarado.
        """
        if self.history_from is None or self.measured_history_from is None:
            return None
        return self.history_from == self.measured_history_from

    def as_dict(self) -> dict:
        return {
            "source_key": self.source_key,
            "tier": self.tier,
            "pit": self.pit,
            "cadence": self.cadence,
            "connected": self.connected,
            "backtestable": self.backtestable,
            "history_from": self.history_from,
            "measured_history_from": self.measured_history_from,
            "declared_matches_measured": self.declared_matches_measured,
            "n_entities": len(self.entities),
            "records": self.records,
            "years": [row.as_dict() for row in self.years],
        }


@dataclass(frozen=True, slots=True)
class ArchiveAudit:
    """El estado del archivo crudo entero."""

    generated_at: str
    root: str
    sources: tuple[SourceCoverage, ...]

    @property
    def n_with_archive(self) -> int:
        return sum(1 for s in self.sources if s.records > 0)

    @property
    def records(self) -> int:
        return sum(s.records for s in self.sources)

    def as_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "root": self.root,
            "n_sources": len(self.sources),
            "n_with_archive": self.n_with_archive,
            "records": self.records,
            "sources": [s.as_dict() for s in self.sources],
        }


def audit_archive(
    store: SignalStore | None = None,
    sources: Sequence[SignalSource] | None = None,
) -> ArchiveAudit:
    """
    Mide el archivo crudo fuente a fuente, entidad a entidad y ano a ano.

    Cuando la fuente tiene adaptador se deriva con SU `daily_from_raw`, que es la unica
    lectura correcta de su payload. Cuando no lo tiene —hoy, todas— se cae a los metadatos
    de archivo (`day`, y si no `fetched_at`): mide menos, pero mide algo real y no exige
    que exista codigo que aun no se ha escrito.
    """
    archive = store if store is not None else SignalStore()
    catalog = tuple(sources) if sources is not None else CATALOG

    rows: list[SourceCoverage] = []
    for source in catalog:
        adapter = build_adapter(source)
        records = archive.read(source.key)
        frame = _daily_frame(source, adapter, records)
        rows.append(
            SourceCoverage(
                source_key=source.key,
                tier=source.tier,
                pit=source.pit,
                cadence=source.cadence,
                connected=adapter is not None,
                backtestable=source.backtestable,
                history_from=source.history_from.isoformat() if source.history_from else None,
                years=_year_rows(frame, source),
            )
        )

    return ArchiveAudit(
        generated_at=utc_now().isoformat(), root=str(archive.root), sources=tuple(rows)
    )


def audit_report(symbols: Sequence[str], store: SignalStore | None = None) -> dict:
    """Las dos auditorias juntas, en la forma en que las publican dashboard y metodologia."""
    return {
        "entities": audit_entities(symbols).as_dict(),
        "archive": audit_archive(store).as_dict(),
    }


# --- interno ------------------------------------------------------------------------


def _daily_frame(source: SignalSource, adapter, records: Sequence[Mapping]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=[ENTITY, DAY, OBSERVED])

    if adapter is not None:
        derived = adapter.daily_from_raw(records)
        if derived is not None and not derived.empty:
            return derived.reset_index()[[ENTITY, DAY, OBSERVED]]

    return pd.DataFrame(
        [
            {
                ENTITY: str(r.get(RAW_ENTITY) or ""),
                DAY: r.get(RAW_DAY) or r.get(RAW_FETCHED_AT),
                OBSERVED: 1,
            }
            for r in records
        ]
    )


def _year_rows(frame: pd.DataFrame, source: SignalSource) -> tuple[YearCoverage, ...]:
    if frame is None or frame.empty:
        return ()

    work = frame.copy()
    work[DAY] = pd.to_datetime(work[DAY], utc=True, errors="coerce").dt.normalize()
    work = work[work[DAY].notna() & (work[ENTITY].astype(str) != "")]
    if work.empty:
        return ()

    cadence_days = source.expected_days_between_observations
    out: list[YearCoverage] = []
    work["_year"] = work[DAY].dt.year

    for (entity, year), block in work.groupby([ENTITY, "_year"], sort=True):
        days = sorted(set(block[DAY]))
        span_days = (days[-1] - days[0]).days
        expected = (
            int(span_days / cadence_days) + 1 if cadence_days else None
        )
        out.append(
            YearCoverage(
                entity=str(entity),
                year=int(year),
                days=len(days),
                observations=int(pd.to_numeric(block[OBSERVED], errors="coerce").fillna(1).sum()),
                first_day=days[0].date().isoformat(),
                last_day=days[-1].date().isoformat(),
                expected_days=expected,
                max_gap_days=_max_gap(days),
            )
        )
    return tuple(out)


def _max_gap(days: Iterable[pd.Timestamp]) -> int:
    """El hueco mas largo entre dias consecutivos CON dato, en dias naturales.

    1 = sin huecos (dias seguidos). Es la cifra que destapa una caida del proveedor, que
    un porcentaje de cobertura promedia y esconde.
    """
    ordered = list(days)
    if len(ordered) < 2:
        return 0
    return max((b - a).days for a, b in zip(ordered[:-1], ordered[1:], strict=True))


__all__ = [
    "ArchiveAudit",
    "EntityAudit",
    "SourceCoverage",
    "YearCoverage",
    "audit_archive",
    "audit_entities",
    "audit_report",
]
