"""
DEL ARCHIVO A LA DECISION: la pieza que faltaba entre `signals/` y `observation/`.

El archivo crudo lleva meses llenandose y hasta hoy no lo leia nadie salvo la auditoria.
Este modulo es el unico camino por el que ese archivo llega al espacio de observacion, y
hace exactamente tres cosas: registrar los adaptadores, leer el crudo de cada fuente y
derivarlo con su `daily_from_raw`. Ni normaliza (eso es `normalize.py`), ni codifica
eventos (eso es `events.py`), ni sabe que es una estrategia.

FALLA ABIERTA, FUENTE A FUENTE, igual que la captura: una fuente cuyo mapeo reviente al
derivar se queda fuera del radar con su error en el log, y las otras dieciseis siguen. La
alternativa —abortar— convertiria un adaptador roto en un sistema que no arranca.

POR QUE NO VIVE EN `observation/`: para que `signals/` siga sin depender de nada aguas
abajo. `observation/signal_radar.py` importa de `signals/`; si `signals/` importara del
radar, el ciclo estaria hecho. Aqui se devuelven frames, que es vocabulario de la ingesta,
y quien construye el proveedor es quien lo va a inyectar.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from ai_trader.signals.capture import connect_adapters
from ai_trader.signals.catalog import CATALOG, SignalSource
from ai_trader.signals.source import build_adapter
from ai_trader.signals.store import SignalStore

logger = logging.getLogger(__name__)


def load_frames(
    sources: Sequence[SignalSource] = CATALOG,
    *,
    store: SignalStore | None = None,
    raw_root: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Frame diario por fuente, derivado del archivo crudo. Sin red y sin reloj.

    Las fuentes sin adaptador, sin archivo o cuya derivacion falle NO aparecen en el
    resultado. Que falten es informacion —el radar la publica como cobertura— y no un
    error que haya que tapar con un frame vacio.
    """
    connect_adapters()
    archive = store if store is not None else SignalStore(raw_root or None)

    frames: dict[str, pd.DataFrame] = {}
    for source in sources:
        adapter = build_adapter(source)
        if adapter is None:
            continue
        try:
            records = archive.read(source.key)
            if not records:
                continue
            frame = adapter.daily_from_raw(records)
        except Exception as exc:  # noqa: BLE001 - una fuente rota no puede tumbar el radar
            logger.warning("· %s: no se pudo derivar del archivo (%s)", source.key, exc)
            continue
        if frame is not None and not frame.empty:
            frames[source.key] = frame

    logger.info(
        "Senales derivadas del archivo: %s/%s fuentes con dato", len(frames), len(sources)
    )
    return frames


def feed_summary(frames: dict[str, pd.DataFrame]) -> dict:
    """Que hay detras del radar, en cifras publicables."""
    out: dict[str, dict] = {}
    for key, frame in sorted(frames.items()):
        days = frame.index.get_level_values("day")
        out[key] = {
            "rows": int(len(frame)),
            "entities": int(frame.index.get_level_values("entity").nunique()),
            "first_day": days.min().date().isoformat() if len(frame) else None,
            "last_day": days.max().date().isoformat() if len(frame) else None,
        }
    return {"n_sources": len(out), "sources": out}


__all__ = ["feed_summary", "load_frames"]
