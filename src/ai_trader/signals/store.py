"""
EL ARCHIVO CRUDO (append-only) y la cache derivada (desechable).

    data/signals_raw/<fuente>/<entidad>/<YYYY-MM>.jsonl.gz     <- NO se re-deriva
    .cache/signals/<fuente>/<entidad>.parquet                  <- se borra sin perder nada

POR QUE EL CRUDO VA EN `data/` Y NO EN `.cache/`
------------------------------------------------
`.cache/` esta en el .gitignore y se puede borrar entero sin consecuencias: todo lo que
hay dentro se vuelve a derivar. El crudo NO se vuelve a derivar. La mitad del catalogo es
`pit='forward_capture'`: nadie publica el pasado de una cola de staking, de un libro P2P o
de la lista SDN de hace tres meses. Y la otra mitad es `archive_revisable`, que es peor de
lo que suena —el proveedor devuelve algo, pero no necesariamente lo que devolvia aquel
dia—. En los dos casos, lo que se descarto no vuelve.

Es exactamente el principio de `synthetic/designer.py:227-233`: el `spec.json` de un
escenario se GUARDA porque una llamada a la IA no es reproducible, y todo lo determinista
que va detras (caminos, barras, metricas) se puede borrar y rehacer bit a bit. Aqui la
frontera cae en el mismo sitio: el payload del proveedor se guarda, y el frame diario que
sale de el es derivado y desechable.

APPEND-ONLY, Y LA CONSECUENCIA QUE PARECE UN BUG
------------------------------------------------
Nada se sobrescribe y nada se deduplica al escribir. Si una fuente revisa el martes pasado,
la linea vieja SIGUE ahi con su `fetched_at` y la nueva se anade detras. Eso hace que el
archivo tenga duplicados aparentes, y es justo lo que lo hace util: comparar las dos lineas
es la unica forma de medir cuanto revisa una fuente `archive_revisable`, que hoy no es mas
que una etiqueta declarada. La deduplicacion vive aguas abajo, en la derivacion
(`shared/signals.py` agrega por `(entity, day)`), donde es reversible.

SHARDING POR MES DE LA OBSERVACION
----------------------------------
El fichero se elige por el mes al que se REFIERE el dato (`day`), no por el dia en que se
descargo. Asi un backfill de dos anos bajado hoy queda repartido en 24 ficheros y leer una
ventana de backtest no obliga a recorrer el archivo entero. Cuando el registro no dice a
que dia se refiere —una foto del estado actual, como la cola de staking— se usa el mes de
`fetched_at`, que en ese caso es la mejor aproximacion que existe.

EL NOMBRE DEL DIRECTORIO NO ES LA ENTIDAD
-----------------------------------------
Se sanea para que sea un nombre de fichero valido en Windows y en Linux ("BTC/USDT" ->
"BTC_USDT", la entidad de mercado "*" -> "_market"). La entidad de verdad va DENTRO de cada
linea. Nadie debe reconstruirla a partir de la ruta.
"""
from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from ai_trader.shared.entities import MARKET_ENTITY
from ai_trader.shared.signals import normalize_signals
from ai_trader.signals.source import RAW_DAY, RAW_ENTITY, RAW_FETCHED_AT

logger = logging.getLogger(__name__)

DEFAULT_RAW_ROOT = Path("data") / "signals_raw"
DEFAULT_CACHE_ROOT = Path(".cache") / "signals"

SHARD_SUFFIX = ".jsonl.gz"
PARQUET_ENGINE = "pyarrow"

# Nombre de directorio de la entidad de mercado. Un asterisco no es un nombre de fichero
# valido en Windows, asi que la unica entidad que no viene de un simbolo tiene el suyo.
MARKET_DIR = "_market"

# Nivel de compresion: 6 es el de por defecto de gzip. Un JSONL de payloads repetitivos
# comprime bien y estos ficheros se leen mucho mas de lo que se escriben.
GZIP_LEVEL = 6


def safe_name(value: str) -> str:
    """Nombre de directorio seguro en los dos sistemas de ficheros. Es LOSSY a proposito:
    la entidad real va dentro de cada linea, no en la ruta."""
    if value == MARKET_ENTITY:
        return MARKET_DIR
    cleaned = "".join(c if (c.isalnum() or c in "._-") else "_" for c in (value or "").strip())
    return cleaned or "_unknown"


def month_of(record: Mapping) -> str:
    """`YYYY-MM` del registro: mes de la OBSERVACION si lo dice, del `fetched_at` si no."""
    stamp = record.get(RAW_DAY) or record.get(RAW_FETCHED_AT) or ""
    text = str(stamp)
    if len(text) >= 7 and text[4] == "-":
        return text[:7]
    parsed = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Registro sin fecha utilizable para archivar: {dict(record)!r}")
    return parsed.strftime("%Y-%m")


class SignalStore:
    """El archivo crudo. Solo sabe de lineas JSON, nunca de features."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_RAW_ROOT

    # --- rutas ---------------------------------------------------------------------

    def source_dir(self, source_key: str) -> Path:
        return self.root / safe_name(source_key)

    def entity_dir(self, source_key: str, entity: str) -> Path:
        return self.source_dir(source_key) / safe_name(entity)

    def shard_path(self, source_key: str, entity: str, month: str) -> Path:
        return self.entity_dir(source_key, entity) / f"{month}{SHARD_SUFFIX}"

    # --- escritura -----------------------------------------------------------------

    def append(self, source_key: str, records: Iterable[Mapping]) -> int:
        """
        Archiva registros crudos. Devuelve cuantas lineas se escribieron.

        Agrupa por (entidad, mes) para abrir cada fichero UNA vez: una captura de 24
        entidades por 24 meses son 576 ficheros, y abrirlos por registro seria el cuello de
        botella de la captura diaria.
        """
        grouped: dict[tuple[str, str], list[Mapping]] = {}
        for record in records:
            if RAW_FETCHED_AT not in record:
                raise ValueError(
                    "Un registro crudo sin 'fetched_at' no se archiva: sin esa marca no se "
                    "puede distinguir una revision de un dato nuevo. Usa signals.source.raw_record."
                )
            entity = str(record.get(RAW_ENTITY) or MARKET_ENTITY)
            grouped.setdefault((entity, month_of(record)), []).append(record)

        written = 0
        for (entity, month), batch in grouped.items():
            path = self.shard_path(source_key, entity, month)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Modo 'at': gzip admite concatenar miembros y el resultado se lee como un solo
            # flujo. Es lo que permite que el archivo sea append-only de verdad, sin
            # reescribir el fichero del mes en cada captura.
            with gzip.open(path, "at", compresslevel=GZIP_LEVEL, encoding="utf-8") as handle:
                for record in batch:
                    handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    written += 1

        return written

    # --- lectura -------------------------------------------------------------------

    def read(
        self,
        source_key: str,
        entity: str | None = None,
        months: Sequence[str] | None = None,
    ) -> list[dict]:
        """Registros crudos, en orden de fichero (es decir, de archivado)."""
        out: list[dict] = []
        entities = [entity] if entity is not None else list(self.entities(source_key))
        for name in entities:
            for path in self._shards(source_key, name, months):
                out.extend(self._read_shard(path))
        return out

    def sources(self) -> tuple[str, ...]:
        if not self.root.exists():
            return ()
        return tuple(sorted(p.name for p in self.root.iterdir() if p.is_dir()))

    def entities(self, source_key: str) -> tuple[str, ...]:
        """Entidades CON ARCHIVO. Son nombres de directorio (saneados), no claves de
        entidad: para la clave real hay que mirar dentro de las lineas."""
        directory = self.source_dir(source_key)
        if not directory.exists():
            return ()
        return tuple(sorted(p.name for p in directory.iterdir() if p.is_dir()))

    def months(self, source_key: str, entity: str) -> tuple[str, ...]:
        directory = self.entity_dir(source_key, entity)
        if not directory.exists():
            return ()
        return tuple(sorted(p.name[: -len(SHARD_SUFFIX)] for p in directory.glob(f"*{SHARD_SUFFIX}")))

    def stats(self) -> dict:
        """Que hay archivado hoy, por fuente. Lo consume la auditoria y el dashboard."""
        out: dict[str, dict] = {}
        for source_key in self.sources():
            entities = self.entities(source_key)
            months: set[str] = set()
            lines = 0
            for entity in entities:
                entity_months = self.months(source_key, entity)
                months.update(entity_months)
                lines += sum(1 for _ in self.read(source_key, entity))
            out[source_key] = {
                "entities": len(entities),
                "months": sorted(months),
                "records": lines,
            }
        return out

    def _shards(self, source_key: str, entity: str, months: Sequence[str] | None) -> list[Path]:
        available = self.months(source_key, entity)
        wanted = available if months is None else [m for m in available if m in set(months)]
        return [self.shard_path(source_key, entity, m) for m in wanted]

    @staticmethod
    def _read_shard(path: Path) -> list[dict]:
        if not path.exists():
            return []
        out: list[dict] = []
        broken = 0
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # Una escritura interrumpida deja una cola ilegible. Se cuenta y se
                    # sigue: perder una linea no puede invalidar el mes entero.
                    broken += 1
        if broken:
            logger.warning("%s lineas ilegibles en %s (se ignoran)", broken, path)
        return out


class DerivedCache:
    """
    La cache DESECHABLE: frames diarios ya derivados, en `.cache/signals/`.

    Todo lo que hay aqui sale de aplicar `daily_from_raw` al archivo crudo. Borrarla entera
    no pierde nada, solo tiempo. Por eso vive bajo `.cache/` —ignorada por git, como
    `.cache/bars/`— y por eso `clear()` es una operacion normal y no una emergencia: es lo
    que hay que hacer cada vez que se corrige un mapeo.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_CACHE_ROOT

    def path(self, source_key: str, entity: str) -> Path:
        return self.root / safe_name(source_key) / f"{safe_name(entity)}.parquet"

    def load(self, source_key: str, entity: str) -> pd.DataFrame | None:
        target = self.path(source_key, entity)
        if not target.exists():
            return None
        frame = pd.read_parquet(target, engine=PARQUET_ENGINE)
        # Se re-normaliza al leer: es idempotente (un grupo por fila) y garantiza que lo
        # que sale de la cache tiene la misma forma que lo que sale del adaptador.
        return normalize_signals(frame)

    def save(self, source_key: str, entity: str, frame: pd.DataFrame) -> Path:
        target = self.path(source_key, entity)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".parquet.tmp")
        frame.reset_index().to_parquet(tmp, engine=PARQUET_ENGINE, index=False)
        tmp.replace(target)
        return target

    def clear(self, source_key: str | None = None) -> int:
        """Borra la cache (de una fuente o entera). Devuelve cuantos ficheros se fueron."""
        base = self.root if source_key is None else self.root / safe_name(source_key)
        if not base.exists():
            return 0
        removed = 0
        for path in sorted(base.rglob("*.parquet")):
            path.unlink()
            removed += 1
        return removed


__all__ = [
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_RAW_ROOT",
    "MARKET_DIR",
    "SHARD_SUFFIX",
    "DerivedCache",
    "SignalStore",
    "month_of",
    "safe_name",
]
