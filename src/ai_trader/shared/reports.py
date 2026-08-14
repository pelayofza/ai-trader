"""
Como se lee y como se escribe un informe publicado. Una sola vez.

Este repo publica una decena de informes en `data/` —fidelidad, transferencia,
validacion, actividad, sesiones, calibracion, canal de senales, divergencia— y todos
son lo mismo: un JSON que un estudio escribe y que el dashboard y la documentacion
leen. Hasta la auditoria del 2026-08-12 el CARGADOR estaba escrito seis veces con el
cuerpo identico (ver DEBT_BACKLOG.md, item B1).

Que las seis copias fueran identicas no las hacia inofensivas. La politica que
implementan es una decision de producto y no un detalle: **si el informe no esta, se
devuelve None y quien lo pinta degrada a prosa sin cifras**. Un generador de
documentacion no puede reventar porque un estudio aun no se haya corrido, y tampoco
puede inventarse un cero. Con seis copias, cambiar esa politica en una y no en las
otras deja el repo con dos comportamientos para la misma pregunta, y ningun test lo
detecta porque cada copia tiene los suyos.

Los nombres antiguos (`load_fidelity_report`, `load_sessions_report`, ...) siguen
existiendo y delegan aqui: los importan `dashboard/` y `docs/` por su nombre.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def load_report(path: Path | str) -> dict | None:
    """Lee el informe publicado. Devuelve None si no esta, para que los generadores de
    dashboard y documentacion degraden a prosa sin cifras en vez de romperse."""
    report = Path(path)
    if not report.exists():
        return None
    return json.loads(report.read_text(encoding="utf-8"))


def write_report(
    report: Any,
    path: Path | str,
    *,
    indent: int = 1,
    ensure_ascii: bool = False,
) -> Path:
    """
    Publica el informe y devuelve donde quedo. Crea el directorio si hace falta.

    `indent` y `ensure_ascii` son argumentos y no constantes porque los informes ya
    publicados no coinciden en ellos, y reformatear un JSON que esta bajo golden seria
    un diff enorme sin ningun cambio de contenido. El defecto es el del estudio de
    sesiones (indent=1, con acentos), que es el formato de los informes mas recientes.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=indent, ensure_ascii=ensure_ascii), encoding="utf-8")
    return target


class PublishedGridMismatch(RuntimeError):
    """Se iba a sobrescribir un informe publicado con otra rejilla. Ver `guard_published_grid`."""


def _grid_families(report: Any) -> tuple[str, ...] | None:
    """Las familias que describe un informe, mire donde mire cada estudio."""
    if not isinstance(report, dict):
        return None
    plan = report.get("plan")
    if not isinstance(plan, dict):
        return None
    grid = plan.get("grid")
    families = (grid or {}).get("families") if isinstance(grid, dict) else plan.get("families")
    return tuple(families) if isinstance(families, list) else None


def guard_published_grid(
    path: Path | str, families: Sequence[str], *, overwrite: bool = False
) -> None:
    """
    Se NIEGA a sobrescribir un informe cuya rejilla no es la que se va a escribir.

    Es lo que convierte "los informes son aditivos" de promesa en propiedad. `FAMILIES` es
    una constante de modulo compartida por cuatro estudios: el dia que crece, cualquier
    re-ejecucion despistada contra una libreria antigua produce un informe con OTRAS
    configuraciones y el mismo nombre de fichero, y la evidencia publicada se pierde sin que
    nada avise. Un `--library` mal tecleado basta.

    La disciplina no es nueva, solo faltaba aqui: `signal_study` ya devuelve 1 si su celda de
    control se ensucia y `fidelity_study` si la aceptacion falla. **Un estudio se niega a
    publicar cuando lo que iba a publicar no significa lo que dice.**

    `overwrite=True` es la valvula explicita, para cuando la sustitucion SI es lo que se
    quiere; entonces la decision queda escrita en la linea de comando y no en un descuido.
    """
    existing = load_report(path)
    if existing is None:
        return
    published = _grid_families(existing)
    if published is None or published == tuple(families):
        return
    if overwrite:
        return
    raise PublishedGridMismatch(
        f"{path} ya publica una rejilla distinta.\n"
        f"  publicada: {list(published)}\n"
        f"  se iba a escribir: {list(families)}\n"
        "Escribe en otro --out-dir, usa otra libreria, o pasa --overwrite-published si de "
        "verdad quieres reemplazar la evidencia publicada."
    )


__all__ = [
    "PublishedGridMismatch",
    "guard_published_grid",
    "load_report",
    "write_report",
]
