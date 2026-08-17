"""
Punto de guardado para barridos largos: unidades terminadas, en disco, segun salen.

POR QUE EXISTE. Un barrido de 2.560 unidades tarda quince horas y hasta ahora solo escribia
su fichero de unidades al TERMINAR. Una interrupcion a mitad —apagar el portatil, quedarse
sin bateria, un `Ctrl-C`— tiraba la corrida entera: en la ultima se perdieron cuatro horas y
media de trabajo ya hecho, que estaba solo en memoria. El coste de esa perdida es mayor que
el de este fichero, y por bastante.

POR QUE ES SEGURO REANUDAR, que es la parte que hay que argumentar y no suponer. Cada unidad
es independiente y determinista: se identifica por una clave completa —configuracion, celda,
escenario, camino— y su resultado no depende de que otras unidades se hayan corrido antes ni
en que orden. El barrido ademas ORDENA las filas al final, asi que el fichero de unidades sale
identico se haya corrido de una vez o en cinco tramos. Reanudar no es una aproximacion: es la
misma corrida con una pausa en medio.

LA HUELLA ES LA GUARDA. Reanudar sobre un plan distinto seria mezclar dos estudios en un
informe que dice ser uno, que es exactamente la clase de mentira que este repositorio se niega
a publicar. Por eso la primera linea del fichero lleva la huella de lo que define el barrido
—el plan y las configuraciones— y un fichero con otra huella NO se reutiliza: se aparta con
sufijo `.stale` y se empieza de cero. Apartarlo y no borrarlo porque si la huella cambio sin
querer, ahi esta lo corrido para mirarlo.

FORMATO. JSONL, una linea por unidad, con `flush` despues de cada una: si el proceso muere a
media escritura la ultima linea queda rota, y al cargar se descarta esa y se conservan todas
las anteriores. Un formato que hubiera que cerrar bien (un JSON con corchetes) no tendria esa
propiedad, y es toda la razon de elegir JSONL.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def fingerprint(*parts: Any) -> str:
    """Huella estable de lo que define un barrido. `default=str` para que un dataclass o una
    fecha no revienten el volcado: aqui no se necesita reconstruir nada, solo distinguir."""
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class UnitCheckpoint:
    """Las unidades ya terminadas de un barrido, en disco y por clave."""

    def __init__(self, path: Path | str, expected: str) -> None:
        self._path = Path(path)
        self._expected = expected
        self._handle = None

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> dict[tuple, dict]:
        """Lo ya corrido, o vacio si no hay nada reutilizable. Nunca lanza: un punto de
        guardado ilegible es 'no hay punto de guardado', no un fallo del estudio."""
        if not self._path.exists():
            return {}

        lines = self._path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return {}

        try:
            header = json.loads(lines[0])
            stored = header["fingerprint"]
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Punto de guardado sin cabecera legible: se ignora (%s)", self._path)
            return self._set_aside()

        if stored != self._expected:
            logger.warning(
                "El punto de guardado es de OTRO barrido (huella %s, se esperaba %s). No se "
                "reutiliza: mezclar dos planes daria un informe que no es de ninguno.",
                stored[:12], self._expected[:12],
            )
            return self._set_aside()

        done: dict[tuple, dict] = {}
        for n, line in enumerate(lines[1:], start=2):
            try:
                entry = json.loads(line)
                done[tuple(entry["key"])] = entry["row"]
            except (json.JSONDecodeError, KeyError, TypeError):
                # Solo la ULTIMA puede estar rota por una muerte a media escritura. Una rota
                # en medio significa otra cosa y hay que decirlo, no tragarsela.
                if n < len(lines) + 1:
                    logger.warning("Linea %d ilegible en el punto de guardado: se descarta", n)
        if done:
            logger.info("Punto de guardado: %d unidades ya corridas en %s",
                        len(done), self._path)
        return done

    def append(self, key: Sequence, row: dict) -> None:
        """Una unidad terminada, en disco YA. El `flush` es el punto entero de esto."""
        if self._handle is None:
            fresh = not self._path.exists() or self._path.stat().st_size == 0
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("a", encoding="utf-8")
            if fresh:
                self._handle.write(
                    json.dumps({"fingerprint": self._expected}, ensure_ascii=False) + "\n"
                )
        self._handle.write(
            json.dumps({"key": list(key), "row": row}, ensure_ascii=False) + "\n"
        )
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def discard(self) -> None:
        """Se llama cuando el barrido termina y publica sus unidades: a partir de ahi el
        fichero bueno es el de unidades, y dejar el punto de guardado solo invita a que
        alguien reanude un barrido que ya acabo."""
        self.close()
        self._path.unlink(missing_ok=True)

    def _set_aside(self) -> dict:
        stale = self._path.with_suffix(self._path.suffix + ".stale")
        stale.unlink(missing_ok=True)
        self._path.rename(stale)
        logger.warning("Apartado en %s; el barrido empieza de cero", stale)
        return {}


__all__ = ["UnitCheckpoint", "fingerprint"]
