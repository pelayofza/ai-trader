"""
CAPTURA: recorre el catalogo y archiva. Se arranca YA, antes de que nada este cableado.

POR QUE ANTES QUE NADA
----------------------
Cinco de las diecisiete fuentes del catalogo son `pit='forward_capture'`: su pasado no se
puede descargar porque nadie lo publica. Para esas, la profundidad historica del sistema
no depende de escribir mejor codigo mas adelante, sino de UNA sola variable —desde cuando
se esta capturando— y esa variable solo avanza con el calendario. Cada dia que la captura
no corre es un dia de profundidad que no se recupera nunca, ni pagando.

De ahi que este modulo exista antes que los adaptadores, y que corra igual sin ninguno.

QUE HACE, EXACTAMENTE
---------------------
Registra los adaptadores escritos (`connect_adapters`), recorre el catalogo, resuelve que
entidades le tocan a cada fuente y archiva lo que devuelva. El informe publica las tres
cifras que importan: declaradas, conectadas y con error. Que registrar sea un paso de la
CAPTURA y no del import del catalogo es deliberado —importar una lista de declaraciones no
puede abrir clientes HTTP— pero tiene que ser automatico: una captura que archivara cero
por haber olvidado registrar diria "todo correcto, 0 registros", que es el peor informe
posible.

Y captura una VENTANA, no la historia entera: ver `CAPTURE_WINDOW_DAYS`.

FALLA ABIERTA, FUENTE A FUENTE
------------------------------
Un proveedor caido, un token caducado o un JSON que cambio de forma afectan a SU fuente y
a ninguna mas: el error se captura, se declara en el informe y la captura sigue. Una
captura diaria que se aborta entera porque una de diecisiete fuentes devolvio un 500
perderia las dieciseis restantes, que es precisamente el dato irrecuperable.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from ai_trader.shared.clock import utc_now
from ai_trader.shared.entities import MARKET_ENTITY, resolve_entities
from ai_trader.signals.catalog import CATALOG, SignalSource, catalog_summary
from ai_trader.signals.source import build_adapter
from ai_trader.signals.store import SignalStore

logger = logging.getLogger(__name__)

# Informe de la ultima captura. Va en data/ (no en .cache/) porque es la bitacora de que
# se archivo y que no: es lo que permite, dentro de seis meses, saber si el hueco de una
# serie fue un proveedor caido o una fuente que nunca se conecto.
CAPTURE_REPORT = Path("data") / "signals" / "capture_report.json"

# Dias hacia atras que pide una captura cuando nadie dice otra cosa.
#
# NO es una preferencia: la mitad de estas fuentes devuelven su historia COMPLETA en cada
# llamada (DefiLlama manda 2.800 puntos por serie), y el archivo es append-only y no se
# poda. Sin ventana, la captura de manana volveria a archivar entera la historia de hoy y el
# archivo crecería megas al dia sin un solo dato nuevo dentro.
#
# Treinta dias y no uno porque las fuentes `archive_revisable` corrigen hacia atras: el
# solape es lo que recoge esas revisiones, y comparar las dos lineas del mismo dia —cada una
# con su `fetched_at`— es lo unico que permite medir cuanto revisa un proveedor. El backfill
# completo es otra operacion, explicita: `signals depth`.
CAPTURE_WINDOW_DAYS = 30


@dataclass(frozen=True, slots=True)
class SourceCapture:
    """Lo que paso con UNA fuente en una pasada."""

    source_key: str
    tier: str
    pit: str
    connected: bool  # ¿tiene adaptador escrito?
    entities: tuple[str, ...]
    records: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.connected and self.error is None

    def as_dict(self) -> dict:
        return {
            "source_key": self.source_key,
            "tier": self.tier,
            "pit": self.pit,
            "connected": self.connected,
            "entities": list(self.entities),
            "n_entities": len(self.entities),
            "records": self.records,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class CaptureReport:
    """El resultado de una pasada completa sobre el catalogo."""

    started_at: str
    finished_at: str
    root: str
    sources: tuple[SourceCapture, ...]

    @property
    def n_sources(self) -> int:
        return len(self.sources)

    @property
    def n_connected(self) -> int:
        return sum(1 for s in self.sources if s.connected)

    @property
    def n_pending(self) -> int:
        """Fuentes declaradas SIN adaptador. La cuenta pendiente, publicada."""
        return sum(1 for s in self.sources if not s.connected)

    @property
    def n_failed(self) -> int:
        return sum(1 for s in self.sources if s.connected and s.error is not None)

    @property
    def records(self) -> int:
        return sum(s.records for s in self.sources)

    def as_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "root": self.root,
            "catalog": catalog_summary(),
            "n_sources": self.n_sources,
            "n_connected": self.n_connected,
            "n_pending": self.n_pending,
            "n_failed": self.n_failed,
            "records": self.records,
            "sources": [s.as_dict() for s in self.sources],
        }


def entities_for(source: SignalSource, symbols: Sequence[str] = ()) -> tuple[str, ...]:
    """
    Que entidades le toca capturar a esta fuente.

    Tres casos, y el orden es el que evita preguntas absurdas (la cola de staking de
    "SOL/USDT" o el flujo de ETF del "mercado"):

    1. Alcance de mercado -> UNA entidad, `MARKET_ENTITY`. No depende del universo.
    2. Entidades declaradas en el catalogo (cadenas, series macro) -> esas.
    3. El resto -> las del universo configurado, resueltas por `shared/entities.py` y
       FILTRADAS por el tipo que la fuente declara consumir. Una fuente de tokens no
       recibe una accion aunque este en el universo.
    """
    if source.is_market_scoped:
        return (MARKET_ENTITY,)
    if source.fixed_entities:
        return source.fixed_entities

    out: list[str] = []
    for ref in resolve_entities(symbols).values():
        if ref.resolved and ref.kind is source.entity_kind and ref.key not in out:
            out.append(ref.key)
    return tuple(out)


def connect_adapters() -> tuple[str, ...]:
    """
    Da de alta los adaptadores escritos. Es el punto —el unico— en que el registro deja de
    estar vacio.

    Se llama al arrancar una captura y no al importar el catalogo: importar una lista de
    declaraciones no puede tener como efecto que existan clientes HTTP, y el import de
    `signals/adapters/` arrastra `ccxt` y compania. Es idempotente.
    """
    from ai_trader.signals.adapters import register_all

    return register_all()


def capture(
    symbols: Sequence[str] = (),
    *,
    store: SignalStore | None = None,
    sources: Sequence[SignalSource] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    report_path: Path | str | None = CAPTURE_REPORT,
) -> CaptureReport:
    """
    Una pasada de captura sobre el catalogo. Devuelve el informe y, si se pide, lo escribe.

    No filtra por tier ni por `pit`: se archiva TODO lo que tenga adaptador. Decidir que
    entra en un backtest es cosa de `history_from` y del consumidor, no de la captura —
    tirar hoy un dato porque hoy no sirve es la unica decision aqui que seria irreversible.
    """
    connect_adapters()
    archive = store if store is not None else SignalStore()
    catalog = tuple(sources) if sources is not None else CATALOG
    started = utc_now()
    # La ventana por defecto (ver CAPTURE_WINDOW_DAYS). Quien quiera el backfill entero lo
    # pide a mano, que es justo lo que hace la sonda de profundidad.
    window_start = start if start is not None else started - timedelta(days=CAPTURE_WINDOW_DAYS)

    results: list[SourceCapture] = []
    for source in catalog:
        targets = entities_for(source, symbols)
        adapter = build_adapter(source)

        if adapter is None:
            logger.info("· %s: sin adaptador todavia (%s entidades en espera)",
                        source.key, len(targets))
            results.append(
                SourceCapture(source.key, source.tier, source.pit, False, targets, 0)
            )
            continue

        try:
            records = adapter.fetch_raw(targets, window_start, end)
            written = archive.append(source.key, records)
            logger.info("· %s: %s registros archivados", source.key, written)
            results.append(
                SourceCapture(source.key, source.tier, source.pit, True, targets, written)
            )
        except Exception as exc:  # noqa: BLE001 - una fuente caida no puede tumbar la captura
            logger.warning("· %s: fallo la captura (%s)", source.key, exc)
            results.append(
                SourceCapture(source.key, source.tier, source.pit, True, targets, 0, str(exc))
            )

    report = CaptureReport(
        started_at=started.isoformat(),
        finished_at=utc_now().isoformat(),
        root=str(archive.root),
        sources=tuple(results),
    )

    if report_path is not None:
        write_capture_report(report, report_path)

    return report


def write_capture_report(report: CaptureReport, path: Path | str = CAPTURE_REPORT) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    return target


def load_capture_report(path: Path | str = CAPTURE_REPORT) -> dict | None:
    """El ultimo informe de captura, o None si nunca se ha corrido."""
    target = Path(path)
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - informe corrupto
        return None


__all__ = [
    "CAPTURE_REPORT",
    "CAPTURE_WINDOW_DAYS",
    "CaptureReport",
    "SourceCapture",
    "capture",
    "connect_adapters",
    "entities_for",
    "load_capture_report",
    "write_capture_report",
]
