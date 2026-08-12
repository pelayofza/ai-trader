"""
LA SONDA DE PROFUNDIDAD: convierte `history_from` de promesa en medicion.

EL CAMPO QUE ESTE MODULO EXISTE PARA LLENAR
-------------------------------------------
`SignalSource.history_from` decide si una fuente puede entrar en un backtest. Arranco en
`None` en las diecisiete, y `None` significa "solo hacia adelante". La tentacion, con el
adaptador ya escrito, es rellenarlo con lo que dice el proveedor —"serie completa desde
enero de 2024"— y seguir. Este modulo existe para que eso no pase: la fecha se pide a la
API, se deriva con el `daily_from_raw` de verdad, y lo que sale se escribe en un fichero
—`data/signals/history_depth.json`— con la fecha de la medicion y el metodo.

El catalogo declara despues, a mano, lo que este fichero dice. Y un test compara las dos
cosas: si alguien escribe en el catalogo una fecha que el fichero no respalda, falla. La
honestidad deja de depender de que nadie se despiste.

POR QUE LA MEDICION ARCHIVA
---------------------------
Medir la profundidad de una fuente con backfill significa descargarla entera. Tirar esa
descarga y volver a bajarla en la captura siguiente seria pagar dos veces por lo mismo y,
en las fuentes con limite de peticiones, arriesgar un 429 evitable. Asi que la sonda
archiva lo que baja: la medicion y el primer backfill son la misma operacion.

Con una consecuencia que conviene tener presente: **correr la sonda dos veces re-archiva la
serie entera**, porque el archivo es append-only y no deduplica. No es un desperdicio en una
fuente `archive_revisable` —las dos copias, cada una con su `fetched_at`, son precisamente
como se mide cuanto reescribe un proveedor— pero si es cara en disco. La operacion de todos
los dias no es esta: es `signals capture`, que pide una ventana de treinta dias
(`capture.py::CAPTURE_WINDOW_DAYS`) en vez de doce anos.

LO QUE UNA MEDICION NO PUEDE DECIR
----------------------------------
De una fuente `forward_capture` (P2P, funding, colas de staking) la sonda solo puede medir
lo que YA se haya capturado: su pasado no existe en ningun sitio. Ahi el resultado honesto
es `method='archive'` y una fecha que crece un dia por dia real. Y de una fuente con
credencial que no esta en el entorno, la sonda no mide NADA y lo declara con su error, en
vez de dejar un hueco que se confunda con "no hay historia".
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path

from ai_trader.shared.clock import utc_now
from ai_trader.shared.signals import DAY, ENTITY, OBSERVED
from ai_trader.signals.capture import entities_for
from ai_trader.signals.catalog import CATALOG, SignalSource, get_source
from ai_trader.signals.source import build_adapter
from ai_trader.signals.store import SignalStore

logger = logging.getLogger(__name__)

# El registro de mediciones. Va en data/ y no en .cache/ por el mismo motivo que el archivo
# crudo: es la EVIDENCIA de lo que se comprobo y cuando, y no se puede re-derivar.
DEPTH_LEDGER = Path("data") / "signals" / "history_depth.json"

METHOD_PROVIDER = "provider"  # se pidio a la API y se derivo con el adaptador real
METHOD_ARCHIVE = "archive"  # se midio sobre lo ya archivado (unica opcion en forward_capture)

# Cuanto pasado se pide al sondear. Doce anos cubre a Bitcoin en GitHub (2010) sin pedirle
# a nadie una ventana absurda; las fuentes que ignoran el rango devuelven su serie entera
# igualmente.
PROBE_YEARS = 12

# Ventana medida que se exige para que el catalogo pueda DECLARAR `history_from`. Un ano.
#
# El motivo no es estetico. En una fuente `forward_capture` el primer dia con dato es el dia
# que arranco la captura, asi que declararlo seria literalmente cierto y practicamente
# enganoso: `backtestable` pasaria a True en una serie de un dia. Con el umbral, el registro
# de mediciones guarda desde cuando se captura —que es el dato util— y el catalogo espera a
# que exista una ventana con la que se pueda backtestear algo.
#
# SE MIDE EN DIAS DE CALENDARIO (last_day - first_day), NO EN FILAS. Empezo contando filas,
# que para una serie diaria es lo mismo, y dejo de serlo el dia que entraron las fuentes de
# EVENTO: el calendario del FOMC son ~60 fechas exactas repartidas en una decada, y contar
# filas lo habria dejado fuera del backtest por "no llegar a un ano" teniendo nueve. Lo que
# la regla quiere es una VENTANA sobre la que se pueda backtestear; cuantas observaciones
# hay dentro es otra pregunta, y la responde `observations` en el mismo registro.
MIN_MEASURED_DAYS = 365


@dataclass(frozen=True, slots=True)
class SourceDepth:
    """Lo que se midio de UNA fuente."""

    source_key: str
    measured_at: str
    method: str
    connected: bool
    first_day: str | None
    last_day: str | None
    days: int
    observations: int
    entities: dict[str, str]  # entidad -> su primer dia con dato
    error: str | None = None

    @property
    def measured(self) -> bool:
        return self.first_day is not None

    def as_dict(self) -> dict:
        return {
            "source_key": self.source_key,
            "measured_at": self.measured_at,
            "method": self.method,
            "connected": self.connected,
            "first_day": self.first_day,
            "last_day": self.last_day,
            "days": self.days,
            "observations": self.observations,
            "n_entities": len(self.entities),
            "entities": dict(sorted(self.entities.items())),
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class DepthReport:
    """Una pasada de medicion sobre el catalogo."""

    generated_at: str
    sources: tuple[SourceDepth, ...]

    @property
    def n_measured(self) -> int:
        return sum(1 for s in self.sources if s.measured)

    def by_key(self) -> dict[str, SourceDepth]:
        return {s.source_key: s for s in self.sources}

    def as_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "n_sources": len(self.sources),
            "n_measured": self.n_measured,
            "probe_years": PROBE_YEARS,
            "sources": [s.as_dict() for s in self.sources],
        }


def _connect() -> None:
    """Registra los diecisiete adaptadores. Importar aqui y no arriba mantiene el
    modulo importable sin `ccxt` ni ningun cliente detras."""
    from ai_trader.signals.adapters import register_all

    register_all()


def measure_depth(
    symbols: Sequence[str] = (),
    *,
    sources: Sequence[SignalSource] | None = None,
    store: SignalStore | None = None,
    archive: bool = True,
    from_archive: bool = False,
    ledger_path: Path | str | None = DEPTH_LEDGER,
) -> DepthReport:
    """
    Mide desde cuando hay dato de verdad, fuente a fuente.

    `from_archive=True` no toca la red y mide lo que ya hay en `data/signals_raw/`: es la
    unica lectura posible de una fuente `forward_capture` y la barata para todo lo demas.
    """
    _connect()
    archive_store = store if store is not None else SignalStore()
    catalog = tuple(sources) if sources is not None else CATALOG
    now = utc_now()
    start = now - timedelta(days=365 * PROBE_YEARS)

    results: list[SourceDepth] = []
    for source in catalog:
        adapter = build_adapter(source)
        targets = entities_for(source, symbols)
        method = METHOD_ARCHIVE if (from_archive or adapter is None) else METHOD_PROVIDER
        error: str | None = None
        records: list = []

        if method == METHOD_PROVIDER:
            try:
                records = adapter.fetch_raw(targets, start, now)
                if archive and records:
                    archive_store.append(source.key, records)
            except Exception as exc:  # noqa: BLE001 - medir una fuente no puede tumbar el resto
                error = str(exc)
                logger.warning("· %s: no se pudo medir (%s)", source.key, exc)
                records = archive_store.read(source.key)
                method = METHOD_ARCHIVE
        else:
            records = archive_store.read(source.key)

        results.append(
            _measure(source, adapter, records, method=method, error=error, measured_at=now)
        )

    report = DepthReport(generated_at=now.isoformat(), sources=tuple(results))
    if ledger_path is not None:
        write_ledger(report, ledger_path)
    return report


def _measure(
    source: SignalSource,
    adapter,
    records: Sequence,
    *,
    method: str,
    error: str | None,
    measured_at: datetime,
) -> SourceDepth:
    empty = SourceDepth(
        source_key=source.key,
        measured_at=measured_at.isoformat(),
        method=method,
        connected=adapter is not None,
        first_day=None,
        last_day=None,
        days=0,
        observations=0,
        entities={},
        error=error,
    )
    if adapter is None or not records:
        return empty

    try:
        frame = adapter.daily_from_raw(records)
    except Exception as exc:  # noqa: BLE001 - un mapeo roto se declara, no se traga
        return replace(empty, error=error or f"derivacion: {exc}")

    if frame is None or frame.empty:
        return empty

    days = frame.index.get_level_values(DAY)
    per_entity: dict[str, str] = {}
    for entity, block in frame.groupby(level=ENTITY, sort=True):
        per_entity[str(entity)] = block.index.get_level_values(DAY).min().date().isoformat()

    return SourceDepth(
        source_key=source.key,
        measured_at=measured_at.isoformat(),
        method=method,
        connected=True,
        first_day=days.min().date().isoformat(),
        last_day=days.max().date().isoformat(),
        days=int(len(frame.index.unique())),
        observations=int(frame[OBSERVED].sum()) if OBSERVED in frame else int(len(frame)),
        entities=per_entity,
        error=error,
    )


def write_ledger(
    report: DepthReport, path: Path | str = DEPTH_LEDGER, *, merge: bool = True
) -> Path:
    """
    Escribe el registro de mediciones. Por defecto FUSIONA con lo que ya hubiera.

    Sondear una sola fuente es la operacion normal —las demas cuestan cuota, minutos y, en
    las revisables, un re-archivado entero— y sin fusion esa pasada borraria del registro
    todo lo medido antes. Con ella, cada fila conserva SU `measured_at`: el fichero es un
    registro de mediciones fechadas una a una, no una foto simultanea, y leerlo como si lo
    fuera seria el unico modo de equivocarse aqui.

    `merge=False` para una pasada que quiera dejar el registro con EXACTAMENTE lo que acaba
    de medir (es lo que hacen los tests con su propio fichero temporal).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = report.as_dict()
    if merge:
        previous = load_ledger(target) or {}
        rows = {str(r.get("source_key")): r for r in previous.get("sources") or []}
        rows.update({r["source_key"]: r for r in payload["sources"]})
        merged = [rows[key] for key in sorted(rows)]
        payload["sources"] = merged
        payload["n_sources"] = len(merged)
        payload["n_measured"] = sum(1 for r in merged if r.get("first_day"))
        payload["measured_in_this_pass"] = [s.source_key for s in report.sources]

    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def load_ledger(path: Path | str = DEPTH_LEDGER) -> dict | None:
    """La ultima medicion, o None si nunca se ha corrido la sonda."""
    target = Path(path)
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - fichero corrupto
        return None


def measured_span_days(row: Mapping) -> int:
    """Dias de CALENDARIO cubiertos por una fila del registro. 0 si no hay medicion."""
    first, last = row.get("first_day"), row.get("last_day")
    if not first:
        return 0
    if not last:
        return 1
    return (date.fromisoformat(str(last)) - date.fromisoformat(str(first))).days + 1


def measured_history(
    path: Path | str = DEPTH_LEDGER, *, min_days: int = MIN_MEASURED_DAYS
) -> dict[str, date]:
    """clave de fuente -> primer dia MEDIDO, si la VENTANA medida es suficiente.

    Es literalmente lo que el catalogo tiene derecho a declarar: una fuente cuya ventana
    medida no llega a `min_days` dias de calendario no aparece, por mucho que la sonda le
    haya visto un primer dia (ver `MIN_MEASURED_DAYS`). Con `min_days=0` se obtiene la
    medicion cruda, que es lo que hace falta para AUDITAR el registro, no para declarar."""
    ledger = load_ledger(path) or {}
    out: dict[str, date] = {}
    for row in ledger.get("sources") or []:
        first = row.get("first_day")
        if first and measured_span_days(row) >= min_days:
            out[str(row.get("source_key"))] = date.fromisoformat(first)
    return out


def declared_vs_measured(path: Path | str = DEPTH_LEDGER) -> dict[str, dict]:
    """
    Compara lo que declara el catalogo con lo que dice el registro de mediciones.

    Tres desajustes posibles y los tres importan:
      - declarado sin medir      -> alguien escribio una promesa
      - medido y no declarado    -> hay profundidad que el backtest no esta usando
      - fechas distintas         -> el proveedor cambio, o la declaracion se quedo vieja
    """
    raw = measured_history(path, min_days=0)
    declarable = measured_history(path)
    out: dict[str, dict] = {}
    for key in {*raw, *(s.key for s in CATALOG)}:
        try:
            declared = get_source(key).history_from
        except KeyError:  # pragma: no cover - fuente retirada del catalogo
            declared = None
        first = raw.get(key)
        allowed = declarable.get(key)
        out[key] = {
            "declared": declared.isoformat() if declared else None,
            # Lo MEDIDO y lo DECLARABLE no son lo mismo: una fuente con tres dias tiene
            # medicion y no tiene derecho a una fecha en el catalogo (MIN_MEASURED_DAYS).
            "measured": first.isoformat() if first else None,
            "declarable": allowed.isoformat() if allowed else None,
            "matches": (declared == allowed) if (declared and allowed) else None,
        }
    return dict(sorted(out.items()))


__all__ = [
    "DEPTH_LEDGER",
    "METHOD_ARCHIVE",
    "METHOD_PROVIDER",
    "PROBE_YEARS",
    "MIN_MEASURED_DAYS",
    "DepthReport",
    "SourceDepth",
    "declared_vs_measured",
    "load_ledger",
    "measure_depth",
    "measured_history",
    "measured_span_days",
    "write_ledger",
]
