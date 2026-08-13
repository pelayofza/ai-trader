"""
Lo que TODO adaptador de senal repite, escrito una vez.

Tres cosas, y ninguna es un detalle:

1. `unix_day` / `iso_day`: llevar el sello de tiempo del proveedor a un `YYYY-MM-DD` UTC.
   Cada uno lo manda distinto —segundos, milisegundos, segundos EN CADENA, ISO con hora,
   ISO con `.000` al final— y esa conversion es exactamente donde se pierde un dia entero
   por un factor 1000 sin que nada falle.

2. `chart_records`: partir una serie completa en UN REGISTRO CRUDO POR DIA. Media docena de
   estas fuentes devuelven de golpe la historia entera (DefiLlama manda 2.800 puntos, TFTC
   662). Archivar ese bloque como una sola linea funcionaria y seria un error: el archivo
   se shardea por mes de la OBSERVACION (`signals/store.py`), asi que una respuesta entera
   caeria en el mes en que se descargo y leer una ventana de backtest obligaria a
   descomprimir todo. Partido por dia, el backfill de dos anos cae en sus 24 ficheros.

3. `env_secret`: leer una credencial del entorno y no de un fichero de configuracion. El
   catalogo declara el NOMBRE de la variable (`auth_env`); el valor no se escribe en ningun
   sitio del repositorio.
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timezone
from typing import Any

UTC = timezone.utc

# Frontera para decidir si un entero es segundos o milisegundos. 10^11 segundos son el ano
# 5138: cualquier sello por encima es milisegundos, y cualquiera por debajo, segundos.
MILLIS_THRESHOLD = 10**11


def unix_day(value: Any) -> str | None:
    """`YYYY-MM-DD` UTC de un sello unix en segundos o milisegundos. None si no lo es.

    Acepta la cadena porque DefiLlama manda `"1511913600"` con comillas y Socrata manda
    fechas ISO: quien llama no deberia tener que saber cual de los dos le toca.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return iso_day(value)
    if number != number or number <= 0:  # NaN o sello vacio
        return None
    seconds = number / 1000.0 if number >= MILLIS_THRESHOLD else number
    try:
        return datetime.fromtimestamp(seconds, tz=UTC).date().isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def iso_day(value: Any) -> str | None:
    """`YYYY-MM-DD` de una fecha ISO ('2026-01-05', '2026-01-05T00:00:00.000'). None si no."""
    if isinstance(value, datetime):
        return value.astimezone(UTC).date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value or "").strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return None


def wiki_stamp(value: Any) -> str | None:
    """`YYYY-MM-DD` del sello de Wikimedia, que es `YYYYMMDDHH` sin separadores."""
    text = str(value or "").strip()
    if len(text) >= 8 and text[:8].isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return None


def numeric(value: Any) -> float | None:
    """Float o None. Un '.' de FRED, un '' de Socrata y un None son todos 'no hay dato'."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def env_secret(name: str | None) -> str | None:
    """El valor de la variable de entorno, o None. El repositorio nunca guarda el valor."""
    if not name:
        return None
    value = os.environ.get(name, "").strip()
    return value or None


def certifi_bundle() -> str | None:
    """
    El bundle de certificados de `certifi`, o None si no esta instalado.

    Empezo siendo privado del adaptador de OFAC, cuya cadena no valida el almacen de
    Windows (CERTIFICATE_VERIFY_FAILED, MEDIDO). Con el lote de alta friccion resulto que
    no era un caso raro: MEDIDO 2026-08-13, fallan igual api.hyperliquid.xyz,
    stats-data.hyperliquid.xyz, api.upbit.com, api-manager.upbit.com,
    www.federalregister.gov y www.courtlistener.com, mientras que deribit.com,
    efts.sec.gov, api.llama.fi y rss.applemarketingtools.com validan sin problema.

    Sigue siendo POR FUENTE y no global, por lo mismo de siempre: usar otro almacen de
    confianza es una decision, y una decision que se toma de forma global es una que nadie
    vuelve a mirar. Lo que cambia es que la funcion vive aqui, porque seis modulos que
    repiten el mismo try/import son seis sitios donde arreglar el dia que cambie.
    """
    try:
        import certifi
    except ImportError:  # pragma: no cover - sin certifi se usa el almacen del sistema
        return None
    return certifi.where()


def day_or_none(moment: datetime | None) -> str | None:
    """`YYYY-MM-DD` de un instante, o None. El corte por delante de `chart_records`."""
    return None if moment is None else moment.astimezone(UTC).date().isoformat()


def chart_records(
    points: Iterable[Any],
    *,
    day_of,
    payload_of=None,
    since: str | None = None,
) -> list[tuple[str, Any]]:
    """
    Una serie del proveedor -> `[(dia, payload_del_dia)]`, saltando lo que no tenga dia.

    `day_of` y `payload_of` reciben el punto tal cual viene. Lo que se archiva es el punto
    INTACTO (por defecto, el propio punto): si manana resulta que el campo bueno era otro,
    esta ahi. Lo unico que se descarta es el punto cuyo dia no se puede leer, porque un
    dato sin dia no se puede colocar en ninguna serie.

    `since` recorta la serie por delante, y no es una optimizacion: estos endpoints
    devuelven la HISTORIA ENTERA en cada llamada, asi que sin el recorte una captura diaria
    volveria a archivar los 54.000 registros de ayer. Como el archivo es append-only y no se
    poda, eso son varios megas AL DIA sin un solo dato nuevo dentro. Quien quiere la serie
    completa es la sonda de profundidad, y lo pide explicitamente pidiendo doce anos.
    """
    out: list[tuple[str, Any]] = []
    for point in points or ():
        day = day_of(point)
        if not day or (since is not None and day < since):
            continue
        out.append((day, point if payload_of is None else payload_of(point)))
    return out


def safe_call(fetch, *, what: str, logger: Any = None) -> Any:
    """
    Ejecuta una llamada de la capa 1 y devuelve None si el proveedor la rechaza.

    Existe por una leccion MEDIDA: al sondear DefiLlama, un unico slug que devolvia 400
    aborto la fuente entera y se perdieron las veintitres series que si funcionaban. Es el
    mismo principio de `signals/capture.py` —una fuente caida no puede tumbar a las demas—
    un nivel mas abajo: una LLAMADA caida no puede tumbar a las demas de su fuente.

    Lo que no hace es tragarse el fallo en silencio: queda en el log y, como la llamada no
    aporta registros, en la cobertura de la auditoria.
    """
    try:
        return fetch()
    except Exception as exc:  # noqa: BLE001 - deliberado: ver el docstring
        if logger is not None:
            logger.info("· %s: sin dato (%s)", what, exc)
        return None


def unique_records(
    records: Sequence[Mapping[str, Any]],
    *,
    key_of,
) -> list[Mapping[str, Any]]:
    """
    Los registros con clave DISTINTA, quedandose con el ultimo capturado de cada una.

    Existe por un fallo que solo aparece con el tiempo y solo en las features que se SUMAN.
    El archivo es append-only y la captura pide una ventana de treinta dias, asi que un
    evento del dia 3 se vuelve a archivar los dias 4, 5, ... 33. Con `agg='last'` eso da
    igual —la ultima copia gana— pero con `agg='sum'` el mismo hecho se cuenta treinta
    veces, y el resultado no se parece a un error: se parece a un mes muy activo.

    La deduplicacion vive aqui, en la capa PURA, y no al escribir: el archivo tiene que
    conservar las copias con su `fetched_at`, porque comparar dos copias del mismo hecho es
    la unica forma de medir cuanto revisa un proveedor. Lo que no puede es contarlas dos
    veces al derivar.

    `key_of` devuelve la identidad del hecho segun el proveedor —un numero de accession, un
    id de anuncio, un numero de documento—, nunca una fecha: dos hechos distintos del mismo
    dia son dos hechos.
    """
    out: dict[Any, Mapping[str, Any]] = {}
    for record in sorted(records or [], key=lambda r: str(r.get("fetched_at") or "")):
        try:
            key = key_of(record)
        except (TypeError, ValueError, KeyError, AttributeError):
            key = None
        if key is None:
            continue
        out[key] = record
    return list(out.values())


def rows_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    row_of,
) -> list[dict]:
    """Aplica `row_of` a cada registro archivado y descarta los que devuelvan None.

    Es el bucle de la capa 2 de casi todos los adaptadores: leer una linea del archivo,
    traducirla a una fila `{entity, day, features...}` y seguir. Que un payload viejo con
    otra forma devuelva None y no reviente es deliberado: el archivo es de anos y el mapeo
    cambia, asi que la derivacion tiene que sobrevivir a payloads que ya no se parecen a
    los de hoy.
    """
    out: list[dict] = []
    for record in records or ():
        try:
            row = row_of(record)
        except (TypeError, ValueError, KeyError, AttributeError):
            row = None
        if row:
            out.append(row)
    return out


__all__ = [
    "MILLIS_THRESHOLD",
    "UTC",
    "certifi_bundle",
    "chart_records",
    "day_or_none",
    "env_secret",
    "iso_day",
    "numeric",
    "rows_from_records",
    "safe_call",
    "unique_records",
    "unix_day",
    "wiki_stamp",
]
