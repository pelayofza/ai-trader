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


__all__ = ["load_report", "write_report"]
