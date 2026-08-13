"""
EL PUERTO: lo unico que un adaptador de senal tiene que cumplir, en DOS CAPAS.

    fetch_raw(...)      -> list[RawRecord]   (toca la red; devuelve el payload INTACTO)
    daily_from_raw(...) -> DataFrame         (funcion PURA; testeable sin red)

POR QUE DOS CAPAS Y NO UNA
--------------------------
La tentacion es una sola funcion `get_daily(...)` que descargue y devuelva el frame ya
limpio. Es mas corta y tiene un defecto que solo se paga meses despues: el mapeo
—que campo del JSON es cada feature, en que unidad, con que huso, como se agrega el dia—
es la parte que se EQUIVOCA, y en una sola capa corregirlo obliga a volver a descargar. Con
fuentes cuyo pasado no es re-descargable (`pit='forward_capture'`) eso no es una molestia:
es informacion perdida.

Separadas, corregir un mapeo RE-DERIVA sobre el archivo que ya esta en disco. Cambiar de
proveedor tambien: se escribe un `fetch_raw` nuevo y el `daily_from_raw` viejo sigue
sirviendo para todo lo capturado hasta ese dia. Y la parte que concentra los errores queda
cubierta por tests que no tocan red, no dependen de que el proveedor siga en pie y corren
en milisegundos.

Es la misma logica que ya sostiene el resto del repo: el `spec.json` de un escenario
sintetico se GUARDA porque no se puede re-derivar, y todo lo determinista que va detras se
puede borrar y rehacer (`synthetic/designer.py`). Aqui el crudo es el spec.

EL REGISTRO, QUE ES LA UNICA RESPUESTA HONESTA A "QUE FUENTES EXISTEN"
---------------------------------------------------------------------
`REGISTRY` mapea clave de catalogo -> constructor de adaptador. Arranco VACIO a proposito
—el esqueleto se construyo antes que las fuentes— y la cifra que publicaba entonces la
captura, "17 declaradas, 0 conectadas", era el estado real del sistema y no un defecto del
informe. Hoy son 17 de 17, y llegar ahi no ha exigido tocar ni una linea de este modulo:
cada adaptador se escribe, se registra y ya.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from ai_trader.data.providers.http import JsonHttpClient, JsonHttpConfig
from ai_trader.shared.clock import utc_now
from ai_trader.shared.signals import empty_signals, normalize_signals
from ai_trader.signals.catalog import SignalSource

# Claves de un registro crudo. `payload` es del proveedor y NO se toca; el resto son
# metadatos nuestros, y van al mismo nivel para que una linea del archivo se pueda leer
# sin conocer la fuente.
RAW_SOURCE = "source"
RAW_ENTITY = "entity"
RAW_FETCHED_AT = "fetched_at"
RAW_DAY = "day"
RAW_PAYLOAD = "payload"
RAW_REQUEST = "request"

RawRecord = dict[str, Any]


def raw_record(
    *,
    source: str,
    entity: str,
    payload: Any,
    day: str | None = None,
    request: Mapping[str, Any] | None = None,
    fetched_at: datetime | None = None,
) -> RawRecord:
    """
    Un registro crudo listo para archivar.

    `fetched_at` es obligatorio en el resultado (se rellena con el reloj si no se pasa) y
    es lo que distingue una revision de un dato nuevo: cuando una fuente `archive_revisable`
    reescribe el pasado, la linea vieja sigue en el archivo con su marca y la nueva se
    anade detras. Sin esa marca, la revision es indetectable.

    `day` es opcional y solo sirve para SHARDEAR el archivo por el mes de la observacion
    (ver `signals/store.py`). La verdad sobre el dia la fija `daily_from_raw` leyendo el
    payload, no este campo.
    """
    if not source:
        raise ValueError("Un registro crudo necesita la fuente que lo produjo")
    record: RawRecord = {
        RAW_SOURCE: source,
        RAW_ENTITY: entity or "",
        RAW_FETCHED_AT: (fetched_at or utc_now()).isoformat(),
        RAW_PAYLOAD: payload,
    }
    if day:
        record[RAW_DAY] = day
    if request:
        record[RAW_REQUEST] = dict(request)
    return record


@runtime_checkable
class SignalAdapter(Protocol):
    """
    Lo que hay que escribir para conectar una fuente. Nada mas.

    Un adaptador NO decide donde se guarda (`store`), ni cuando se llama (`capture`), ni
    que entidades le tocan (`catalog` + `shared/entities.py`). Solo sabe hablar con su
    proveedor y traducir su payload.
    """

    source: SignalSource

    def fetch_raw(
        self,
        entities: Sequence[str],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[RawRecord]:
        """Payload INTACTO del proveedor, envuelto con metadatos. Toca red."""
        ...

    def daily_from_raw(self, records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        """Frame canonico `(entity, day) -> features`. PURA: ni red, ni reloj, ni disco."""
        ...


class BaseJsonAdapter:
    """
    Andamio comun para una fuente REST/JSON. Es opcional: el puerto es el Protocol.

    Reusa `JsonHttpClient` (`data/providers/http.py`), que ya trae backoff exponencial y la
    regla que importa —un 4xx que no sea 429 NO se reintenta, porque un error del cliente
    no mejora insistiendo—. Escribir otro cliente HTTP para las senales habria significado
    mantener esa regla en dos sitios.

    El cliente se construye PEREZOSAMENTE: importar el catalogo de fuentes no puede abrir
    conexiones, igual que construir el servicio de datos no dispara `load_markets()` en
    `CCXTCrypto`.
    """

    def __init__(
        self,
        source: SignalSource,
        *,
        base_url: str = "",
        http_config: JsonHttpConfig | None = None,
        client: JsonHttpClient | None = None,
    ) -> None:
        self.source = source
        self._base_url = base_url or source.endpoint
        self._http_config = http_config
        self._client = client

    @property
    def client(self) -> JsonHttpClient:
        if self._client is None:
            if not self._base_url:
                raise RuntimeError(
                    f"La fuente '{self.source.key}' no declara endpoint y no se le paso cliente"
                )
            self._client = JsonHttpClient(self._base_url, self._http_config)
        return self._client

    # --- capa 1: red ---------------------------------------------------------------

    def fetch_raw(
        self,
        entities: Sequence[str],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[RawRecord]:  # pragma: no cover - lo implementa cada fuente
        raise NotImplementedError

    # --- capa 2: pura --------------------------------------------------------------

    def daily_from_raw(
        self, records: Sequence[Mapping[str, Any]]
    ) -> pd.DataFrame:  # pragma: no cover - lo implementa cada fuente
        raise NotImplementedError

    # --- utilidad compartida -------------------------------------------------------

    def record(self, entity: str, payload: Any, **extra) -> RawRecord:
        return raw_record(source=self.source.key, entity=entity, payload=payload, **extra)

    def to_daily(self, rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        """
        Filas planas -> frame canonico, con la politica de agregacion DEL CATALOGO.

        El adaptador solo tiene que producir dicts con `entity`, `day` y sus features; la
        forma, el orden de columnas, el huso y la agregacion a dia los pone
        `shared/signals.py`. Que la politica venga del catalogo y no del adaptador es lo
        que impide que dos fuentes de flujo agreguen distinto por descuido.
        """
        features = self.source.feature_names
        if not rows:
            return empty_signals(features)
        return normalize_signals(
            pd.DataFrame(list(rows)),
            features=features,
            agg=self.source.aggregations(),
        )


# --- registro de adaptadores --------------------------------------------------------

AdapterFactory = Callable[[SignalSource], SignalAdapter]

# clave del catalogo -> constructor. VACIO a proposito: ver el docstring del modulo.
REGISTRY: dict[str, AdapterFactory] = {}


def register_adapter(key: str, factory: AdapterFactory) -> None:
    """Da de alta el adaptador de una fuente del catalogo.

    Falla si la clave ya esta registrada: dos adaptadores para la misma fuente escribirian
    en el mismo directorio del archivo con mapeos distintos, y el crudo dejaria de ser una
    sola verdad.
    """
    from ai_trader.signals.catalog import get_source

    get_source(key)  # una fuente que no esta en el catalogo no se puede registrar
    if key in REGISTRY:
        raise ValueError(f"Ya hay un adaptador registrado para '{key}'")
    REGISTRY[key] = factory


def build_adapter(source: SignalSource) -> SignalAdapter | None:
    """El adaptador de la fuente, o None si aun no hay ninguno escrito."""
    factory = REGISTRY.get(source.key)
    return factory(source) if factory is not None else None


def connected_keys() -> tuple[str, ...]:
    """Fuentes del catalogo que YA tienen adaptador. Hoy, las treinta."""
    return tuple(sorted(REGISTRY))


__all__ = [
    "RAW_DAY",
    "RAW_ENTITY",
    "RAW_FETCHED_AT",
    "RAW_PAYLOAD",
    "RAW_REQUEST",
    "RAW_SOURCE",
    "REGISTRY",
    "AdapterFactory",
    "BaseJsonAdapter",
    "RawRecord",
    "SignalAdapter",
    "build_adapter",
    "connected_keys",
    "raw_record",
    "register_adapter",
]
