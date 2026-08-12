"""
La forma de una linea del archivo append-only del repo.

Hay dos archivos append-only y no comparten nada mas: el crudo de senales
(`signals/store.py`, gzip, shard por fuente-entidad-mes) y el diario de ciclos del
paper trading en vivo (`app/journal.py`, texto plano, fsync por linea). Lo que SI
tienen que compartir es como se escribe y como se lee una linea, y por dos motivos
que no son de estilo:

- `default=str` en la serializacion. Es lo que hace que un `datetime` o un `Enum`
  no revienten la escritura. Con dos copias, basta con que una lo pierda para que
  un archivo acepte lo que el otro rechaza, y eso no lo detecta ningun test que
  mire un solo archivo.
- La politica ante una linea ilegible: contarla y seguir. Una escritura
  interrumpida —un corte de luz a mitad de linea— deja una cola rota, y perder esa
  linea no puede invalidar el resto del fichero. Es la unica politica compatible
  con un archivo que se escribe durante meses.
"""
from __future__ import annotations

import json
from collections.abc import Mapping


def to_line(record: Mapping) -> str:
    """Una linea JSONL, con el salto de linea incluido.

    `ensure_ascii=False` porque los motivos de rechazo del riesgo y las preguntas de
    los mercados de prediccion llevan acentos, y `default=str` porque los registros
    traen fechas y enums."""
    return json.dumps(record, ensure_ascii=False, default=str) + "\n"


def read_records(handle) -> tuple[list[dict], int]:
    """
    Registros de un flujo JSONL YA ABIERTO, y cuantas lineas resultaron ilegibles.

    Recibe el manejador y no la ruta a proposito: el crudo de senales se abre con
    `gzip.open` y el diario con `open`, y esa es toda la diferencia entre los dos.
    """
    out: list[dict] = []
    broken = 0
    for line in handle:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            broken += 1
    return out, broken


__all__ = ["read_records", "to_line"]
