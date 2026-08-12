"""
LAS SEIS FUENTES DE EVENTO, conectadas. Lo que aqui se mide es un tamano de muestra.

POR QUE ESTE MODULO EXISTE
--------------------------
Durante una fase, estas seis (tier `A`) iban a entrar al sistema por otra puerta —una
guarda de elegibilidad— y no como features. La defensa de esa bifurcacion era una frase:
"son muestras de DECENAS, un optimizador suelto sobre catorce observaciones construye una
estrategia preciosa y falsa". Nadie habia medido esas decenas. Este modulo escribe los
adaptadores que faltaban para que la sonda (`signals/depth.py`) pueda medirlas, que es lo
unico que convierte esa frase en un dato.

LO QUE DEVOLVIO LA MEDICION (2026-08-12), fuente a fuente:

    btc_difficulty    463 ajustes desde 2009-01-03. La hipotesis de "decenas" es FALSA por
                      un factor de diez: cada 2016 bloques hay un evento, y las cabeceras
                      de bloque no se reescriben.
    defillama_hacks   621 incidentes fechados desde 2016-06-17. Era "el caso duro" y
                      resulta ser la segunda muestra mas grande de las seis.
    macro_calendar    reuniones del FOMC desde 2017 (fechas exactas, publicadas con meses
                      de antelacion) y las publicaciones del IPC que el BLS mantiene en su
                      calendario, que es una ventana rodante de ~13 meses.
    ofac_sdn          la lista de HOY. Nadie publica la de una fecha pasada, asi que su
                      profundidad la compra el calendario, un dia por dia real.
    token_unlocks     402 Payment Required. El endpoint de emisiones de DefiLlama dejo de
                      ser abierto. No hay muestra que medir hasta que haya otra via.
    staking_queue     401 Unauthorized. beaconcha.in ahora pide credencial.

Las dos ultimas no son un fallo de este modulo: son el resultado. Una fuente que no se
puede descargar se declara con su error y se queda en `history_from=None`, exactamente
igual que Guavy o FRED cuando falta la credencial.

QUE ES UNA FILA AQUI, Y EN QUE SE DIFERENCIA DEL TIER B
-------------------------------------------------------
Estos adaptadores emiten EVENTOS, no una serie diaria: una fila el dia que pasa algo, con
su magnitud. La densificacion —dias-al-evento, ventana activa— NO se hace aqui: vive en
`signals/events.py`, se aplica igual a las seis y por eso es testeable en un solo sitio.
El motivo de que no la haga cada adaptador es el mismo por el que la agregacion a dia vive
en `shared/signals.py`: repetida seis veces, seis oportunidades de que una se desvie.

Consecuencia practica: `days_to_unlock`, `days_to_adjustment`, `days_to_fomc` y
`days_to_cpi` han dejado de ser columnas del CATALOGO. No se han perdido —son justo lo que
publica el codificador— pero declararlas como si las produjera el proveedor era falso: lo
que el proveedor da es el CALENDARIO, y los dias que faltan son una cuenta nuestra contra
el 'ahora'.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

import pandas as pd

from ai_trader.data.providers.http import JsonHttpConfig
from ai_trader.shared.entities import MARKET_ENTITY
from ai_trader.shared.signals import DAY, ENTITY
from ai_trader.signals.adapters.common import (
    day_or_none,
    env_secret,
    iso_day,
    numeric,
    rows_from_records,
    safe_call,
    unix_day,
)
from ai_trader.signals.source import BaseJsonAdapter, RawRecord

logger = logging.getLogger(__name__)


# =====================================================================================
# 1. Ajuste de dificultad de Bitcoin (mempool.space)
# =====================================================================================

MEMPOOL_BASE = "https://mempool.space"
# La HISTORIA de ajustes, que es la que convierte esta fuente en backtesteable. El
# endpoint declarado en el catalogo (`/api/v1/difficulty-adjustment`) devuelve solo la foto
# de AHORA; este devuelve las 463 retargets desde el bloque 0 en 21 KB. Se comprobo que
# `all` existe: `1y`, `3y` y `all` responden 200 y `all` empieza en 2009-01-03.
DIFFICULTY_HISTORY_PATH = "/api/v1/mining/difficulty-adjustments/all"
DIFFICULTY_NOW_PATH = "/api/v1/difficulty-adjustment"

BITCOIN_ENTITY = "bitcoin"


class MempoolDifficulty(BaseJsonAdapter):
    """
    Un registro por RETARGET. La serie entera cabe en una llamada y no se revisa nunca.

    El ajuste es el unico evento de oferta del catalogo que es a la vez pasado
    reconstruible y futuro estimable: la fecha del proximo retarget sale de los bloques que
    faltan, no de que nadie la anuncie.

    LIMITE DECLARADO. Para el pasado, el codificador cuenta los dias que faltaban usando la
    fecha en que el retarget OCURRIO. Ese dia, en su momento, era una ESTIMACION (bloques
    restantes x 10 minutos) con un error tipico de horas y maximo de un dia o dos. En una
    proximidad acotada a 30 dias eso es ruido, pero es futuro: queda escrito aqui y en las
    notas del catalogo, no disimulado.
    """

    def __init__(self, source, **kwargs) -> None:
        super().__init__(source, base_url=MEMPOOL_BASE, **kwargs)

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        rows = safe_call(
            lambda: self.client.get_json(DIFFICULTY_HISTORY_PATH),
            what="mempool difficulty-adjustments",
            logger=logger,
        )
        since = day_or_none(start)
        out: list[RawRecord] = []
        for row in rows or []:
            if not isinstance(row, (list, tuple)) or len(row) < 4:
                continue
            day = unix_day(row[0])
            if not day or (since is not None and day < since):
                continue
            out.append(
                self.record(
                    BITCOIN_ENTITY, list(row), day=day, request={"series": "adjustments"}
                )
            )
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        return self.to_daily(rows_from_records(records, row_of=_difficulty_row))


def _difficulty_row(record: Mapping) -> dict | None:
    payload = record.get("payload")
    if not isinstance(payload, (list, tuple)) or len(payload) < 4:
        return None
    day = unix_day(payload[0])
    ratio = numeric(payload[3])
    if not day or ratio is None:
        return None
    # El cuarto campo es el cociente contra el retarget anterior (1.00989 = +0,989%). El
    # primero de todos vale 0 —no habia anterior— y se declara como 0% de cambio.
    change_pct = (ratio - 1.0) * 100.0 if ratio > 0 else 0.0
    return {
        ENTITY: record.get("entity") or BITCOIN_ENTITY,
        DAY: day,
        "difficulty_change_pct": change_pct,
    }


# =====================================================================================
# 2. Hacks y exploits fechados (DefiLlama)
# =====================================================================================

LLAMA_BASE = "https://api.llama.fi"
HACKS_PATH = "/hacks"


class DefiLlamaHacks(BaseJsonAdapter):
    """
    Un registro por incidente. 621 medidos, desde 2016.

    El alcance es de MERCADO y no por activo a proposito: el mecanismo que interesa —la
    contraccion de riesgo despues de un robo grande— no se queda en el token afectado. Que
    la entidad sea una sola es ademas lo que hace que el pooling de (b) funcione aqui: la
    muestra son los 621 eventos, no 621/24 por activo.
    """

    def __init__(self, source, **kwargs) -> None:
        super().__init__(source, base_url=LLAMA_BASE, **kwargs)

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        rows = safe_call(
            lambda: self.client.get_json(HACKS_PATH), what="defillama hacks", logger=logger
        )
        since = day_or_none(start)
        out: list[RawRecord] = []
        for row in rows or []:
            day = unix_day((row or {}).get("date"))
            if not day or (since is not None and day < since):
                continue
            out.append(self.record(MARKET_ENTITY, row, day=day))
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        return self.to_daily(rows_from_records(records, row_of=_hack_row))


def _hack_row(record: Mapping) -> dict | None:
    payload = record.get("payload") or {}
    day = unix_day(payload.get("date"))
    if not day:
        return None
    amount = numeric(payload.get("amount"))
    return {
        ENTITY: record.get("entity") or MARKET_ENTITY,
        DAY: day,
        # DefiLlama publica el importe en MILLONES de dolares. Multiplicarlo aqui y no
        # dejarlo "en unidades del proveedor" es lo que impide que un 1,5 de esta serie se
        # compare con un 1,5 de otra que si viene en dolares.
        "hack_amount_usd": None if amount is None else amount * 1e6,
        "hack_count": 1.0,
    }


# =====================================================================================
# 3. Calendario macro (Reserva Federal + BLS)
# =====================================================================================

FED_BASE = "https://www.federalreserve.gov"
FED_CALENDAR_PATH = "/json/calendar.json"
BLS_BASE = "https://www.bls.gov"
BLS_CPI_SCHEDULE_PATH = "/schedule/news_release/cpi.htm"

# El titulo con el que la Fed marca una reunion (frente a las actas, la rueda de prensa o
# un discurso). Se compara en minusculas y sin espacios porque el fichero trae las tres
# variantes: "FOMC Meeting", "FOMC meeting" y " FOMC Minutes" con espacio delante.
FOMC_MEETING_TITLE = "fomc meeting"

_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_DATE_FORMATS = ("%b. %d, %Y", "%B %d, %Y", "%b %d, %Y")


class MacroCalendar(BaseJsonAdapter):
    """
    Fechas exactas, publicadas con meses de antelacion y no revisables.

    Es la mejor propiedad que puede tener una feature de evento y por eso esta fuente se
    conecta aunque su magnitud sea siempre 1: lo que aporta no es "cuanto", es "cuando", y
    el "cuando" se sabe antes.

    DOS PROFUNDIDADES DISTINTAS EN LA MISMA FUENTE, y hay que decirlo igual que se dijo en
    GitHub: el calendario de la Fed llega a 2017; el del BLS es una ventana RODANTE de unos
    trece meses, asi que la historia larga del IPC solo se acumula captura a captura. Quien
    mire `cpi_release` en 2019 no encontrara nada, y eso no es que no hubiera IPC.
    """

    def __init__(self, source, **kwargs) -> None:
        super().__init__(source, base_url=FED_BASE, **kwargs)
        self._bls = None

    @property
    def bls(self):
        if self._bls is None:
            from ai_trader.data.providers.http import JsonHttpClient

            self._bls = JsonHttpClient(BLS_BASE, self._http_config)
        return self._bls

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        since = day_or_none(start)
        out: list[RawRecord] = []

        calendar = safe_call(
            lambda: self.client.get_json(FED_CALENDAR_PATH), what="fed calendar", logger=logger
        )
        for day in fomc_meeting_days(calendar):
            if since is None or day >= since:
                out.append(self.record(MARKET_ENTITY, {"event": "fomc"}, day=day,
                                       request={"series": "fomc"}))

        schedule = safe_call(
            lambda: self.bls.get_text(BLS_CPI_SCHEDULE_PATH),
            what="bls cpi schedule",
            logger=logger,
        )
        for day in cpi_release_days(schedule):
            if since is None or day >= since:
                out.append(self.record(MARKET_ENTITY, {"event": "cpi"}, day=day,
                                       request={"series": "cpi"}))
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        return self.to_daily(rows_from_records(records, row_of=_calendar_row))


def fomc_meeting_days(calendar) -> list[str]:
    """Dias de REUNION del FOMC, en ISO. De una reunion de dos dias se toma el SEGUNDO,
    que es cuando sale el comunicado: es el dia en que la informacion existe."""
    events = (calendar or {}).get("events") if isinstance(calendar, Mapping) else None
    out: list[str] = []
    for event in events or []:
        title = str(event.get("title") or "").strip().lower()
        month = str(event.get("month") or "").strip()
        if not title.startswith(FOMC_MEETING_TITLE) or not _MONTH_RE.match(month):
            continue
        # `days` es "9" o "8-9" (reunion de dos dias). Alguna entrada del fichero viene sin
        # mes utilizable y se descarta arriba: un evento sin fecha no es un evento.
        last = str(event.get("days") or "").split("-")[-1].strip()
        if not last.isdigit():
            continue
        out.append(f"{month}-{int(last):02d}")
    return sorted(set(out))


def cpi_release_days(html: str | None) -> list[str]:
    """Dias de publicacion del IPC, leidos de la tabla del BLS ('Dec. 18, 2025').

    El BLS no publica esto en JSON. Se archiva el HTML crudo entero, asi que el dia que
    cambien la tabla se puede re-derivar sin haber perdido nada.
    """
    out: list[str] = []
    for cell in _CELL_RE.findall(html or ""):
        text = _TAG_RE.sub("", cell).strip()
        for fmt in _DATE_FORMATS:
            try:
                out.append(datetime.strptime(text, fmt).date().isoformat())
                break
            except ValueError:
                continue
    return sorted(set(out))


def _calendar_row(record: Mapping) -> dict | None:
    day = iso_day(record.get("day"))
    series = str(((record.get("request") or {}).get("series")) or "")
    if not day or series not in ("fomc", "cpi"):
        return None
    return {
        ENTITY: record.get("entity") or MARKET_ENTITY,
        DAY: day,
        "fomc_meeting": 1.0 if series == "fomc" else None,
        "cpi_release": 1.0 if series == "cpi" else None,
    }


# =====================================================================================
# 4. Lista OFAC SDN (direcciones sancionadas)
# =====================================================================================

OFAC_BASE = "https://sanctionslistservice.ofac.treas.gov"
OFAC_SDN_PATH = "/api/PublicationPreview/exports/SDN.XML"

# Prefijo con el que OFAC marca una direccion de criptomoneda dentro de la lista.
CRYPTO_ID_PREFIX = "Digital Currency Address"

_ID_RE = re.compile(
    r"<id>\s*<uid>(\d+)</uid>\s*<idType>([^<]*)</idType>\s*<idNumber>([^<]*)</idNumber>", re.S
)
_PUBLISH_RE = re.compile(r"<Publish_Date>([^<]+)</Publish_Date>")


class OfacSdn(BaseJsonAdapter):
    """
    La foto de HOY de la lista de sanciones, reducida a su parte cripto.

    DOS DECISIONES QUE NO SON DETALLES.

    1. EL PAYLOAD ARCHIVADO ES UN SUBCONJUNTO, y es la unica fuente del catalogo donde se
       rompe la regla de "el payload intacto". El fichero son 28 MB de XML al dia (19.199
       registros, 53.473 identificadores) de los que 977 son direcciones de cripto: guardar
       el resto serian diez gigas al ano de barcos, pasaportes y domicilios que ninguna
       feature va a mirar. Lo que se archiva es cada `<id>` de tipo `Digital Currency
       Address` con su uid, su tipo y su direccion, mas la fecha de publicacion. El
       criterio de recorte esta aqui escrito y es re-aplicable; lo que no se puede es
       recuperar manana el XML de hoy, y por eso la decision se declara en vez de asumirse.
    2. EL CERTIFICADO. El almacen de Windows no valida la cadena de este host (MEDIDO:
       CERTIFICATE_VERIFY_FAILED). Se usa el bundle de `certifi` SOLO para esta fuente.
    """

    def __init__(self, source, **kwargs) -> None:
        config = kwargs.pop("http_config", None) or JsonHttpConfig(
            timeout_seconds=120.0, ca_bundle=_certifi_bundle()
        )
        super().__init__(source, base_url=OFAC_BASE, http_config=config, **kwargs)

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        xml = safe_call(
            lambda: self.client.get_text(OFAC_SDN_PATH), what="ofac sdn", logger=logger
        )
        if not xml:
            return []
        payload = crypto_addresses(xml)
        if not payload["addresses"]:
            return []
        # El dia es el de PUBLICACION que declara el propio fichero. Si no lo trae, el de
        # la captura: es lo unico conocible, y `fetched_at` queda al lado para verlo.
        day = payload["publish_date"] or datetime.now(timezone.utc).date().isoformat()
        return [self.record(MARKET_ENTITY, payload, day=day)]

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        rows: list[dict] = []
        seen_before: set[str] = set()
        for index, record in enumerate(_sorted_by_fetch(records)):
            payload = record.get("payload") or {}
            day = iso_day(record.get("day")) or iso_day(payload.get("publish_date"))
            uids = {str(a.get("uid")) for a in payload.get("addresses") or [] if a.get("uid")}
            if not day or not uids:
                continue
            rows.append(
                {
                    ENTITY: record.get("entity") or MARKET_ENTITY,
                    DAY: day,
                    "sdn_crypto_addresses": float(len(uids)),
                    # La PRIMERA captura no tiene con que comparar: sus altas son NaN, no
                    # cero. Un cero diria "ese dia no entro nadie", que no se ha observado.
                    "sdn_new_entries": None if index == 0 else float(len(uids - seen_before)),
                }
            )
            seen_before |= uids
        return self.to_daily(rows)


def crypto_addresses(xml: str) -> dict:
    """El subconjunto cripto de la lista: `{publish_date, addresses[]}`."""
    published = _PUBLISH_RE.search(xml or "")
    day = None
    if published:
        text = published.group(1).strip()
        for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
            try:
                day = datetime.strptime(text, fmt).date().isoformat()
                break
            except ValueError:
                continue

    addresses = [
        {"uid": uid, "type": id_type.replace(CRYPTO_ID_PREFIX, "").strip(" -"), "address": number}
        for uid, id_type, number in _ID_RE.findall(xml or "")
        if id_type.startswith(CRYPTO_ID_PREFIX)
    ]
    return {"publish_date": day, "addresses": addresses}


def _certifi_bundle() -> str | None:
    try:
        import certifi
    except ImportError:  # pragma: no cover - sin certifi se usa el almacen del sistema
        return None
    return certifi.where()


def _sorted_by_fetch(records: Sequence[Mapping]) -> list[Mapping]:
    """Las capturas en orden cronologico. Las altas del dia se cuentan contra lo ya visto,
    asi que el orden no puede depender de como iterase el archivo."""
    return sorted(records or [], key=lambda r: str(r.get("fetched_at") or ""))


# =====================================================================================
# 5. Desbloqueos y vesting (DefiLlama emissions)
# =====================================================================================

EMISSIONS_PATH = "/emissions"


class DefiLlamaUnlocks(BaseJsonAdapter):
    """
    Calendario de desbloqueos. MEDIDO 2026-08-12: 402 Payment Required.

    El endpoint de emisiones de DefiLlama dejo de ser abierto. El adaptador se escribe
    igualmente y por dos razones concretas: es lo que permite que la sonda MIDA el 402 en
    vez de dejar la fuente como "sin adaptador" —que se leeria como "nadie lo ha
    intentado"—, y el dia que haya credencial o via alternativa, la capa 2 ya esta.

    LO QUE NO SE PUEDE AFIRMAR: el mapeo de la capa 2 esta escrito contra la forma que
    DefiLlama documenta, no contra un payload observado. Los adaptadores del tier B se
    escribieron con la respuesta real delante; este no, y hasta que se pueda descargar una
    respuesta, esa diferencia es real y se declara aqui.
    """

    def __init__(self, source, **kwargs) -> None:
        super().__init__(source, base_url=LLAMA_BASE, **kwargs)

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        # Sin safe_call a proposito: el 402 es EL resultado de esta fuente y la sonda tiene
        # que verlo como error declarado, no como "cero registros".
        rows = self.client.get_json(EMISSIONS_PATH)
        wanted = {str(e).upper() for e in entities}
        since = day_or_none(start)
        out: list[RawRecord] = []
        for row in rows or []:
            symbol = str((row or {}).get("token") or (row or {}).get("symbol") or "").upper()
            if wanted and symbol not in wanted:
                continue
            for event in (row or {}).get("events") or []:
                day = unix_day(event.get("timestamp") or event.get("date"))
                if not day or (since is not None and day < since):
                    continue
                out.append(
                    self.record(symbol, {"token": symbol, **event}, day=day,
                                request={"series": "emissions"})
                )
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        return self.to_daily(rows_from_records(records, row_of=_unlock_row))


def _unlock_row(record: Mapping) -> dict | None:
    payload = record.get("payload") or {}
    day = unix_day(payload.get("timestamp") or payload.get("date"))
    if not day:
        return None
    tokens = numeric(payload.get("noOfTokens") or payload.get("amount"))
    float_supply = numeric(payload.get("circulatingSupply") or payload.get("float"))
    adv = numeric(payload.get("volume24h") or payload.get("adv"))
    if tokens is None:
        return None
    return {
        ENTITY: record.get("entity") or "",
        DAY: day,
        # La magnitud NORMALIZADA es toda la tesis de esta fuente: "hay unlock" esta
        # explotado; un 3% del float sobre un token con 40 dias de volumen en circulacion
        # no lo esta. Sin la pata del denominador no hay feature, y NaN lo dice.
        "unlock_pct_float": None if not float_supply else tokens / float_supply * 100.0,
        "unlock_pct_adv": None if not adv else tokens / adv * 100.0,
    }


# =====================================================================================
# 6. Cola de entrada/salida del staking (beaconcha.in)
# =====================================================================================

BEACONCHAIN_BASE = "https://beaconcha.in"
QUEUE_PATH = "/api/v1/validators/queue"


class BeaconchainStakingQueue(BaseJsonAdapter):
    """
    Cola de validadores. MEDIDO 2026-08-12: 401 Unauthorized.

    El endpoint era abierto y ahora pide credencial, asi que el catalogo pasa a declarar
    `auth_env`. Es forward_capture puro: devuelve la foto de AHORA y no guarda la de ayer,
    de modo que cada dia sin capturar es profundidad que no vuelve.
    """

    def __init__(self, source, **kwargs) -> None:
        token = env_secret(source.auth_env)
        config = kwargs.pop("http_config", None) or JsonHttpConfig(
            headers={"Authorization": f"Bearer {token}"} if token else {}
        )
        super().__init__(source, base_url=BEACONCHAIN_BASE, http_config=config, **kwargs)

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        # Como en unlocks: el 401 es el resultado y tiene que llegar a la sonda.
        payload = self.client.get_json(QUEUE_PATH)
        if not payload:
            return []
        day = datetime.now(timezone.utc).date().isoformat()
        return [self.record("ethereum", payload, day=day)]

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        return self.to_daily(rows_from_records(records, row_of=_queue_row))


def _queue_row(record: Mapping) -> dict | None:
    payload = record.get("payload") or {}
    data = payload.get("data") if isinstance(payload.get("data"), Mapping) else payload
    day = iso_day(record.get("day")) or iso_day(record.get("fetched_at"))
    entering = numeric(data.get("beaconchain_entering"))
    exiting = numeric(data.get("beaconchain_exiting"))
    if not day or (entering is None and exiting is None):
        return None
    # La ESPERA EN DIAS no se publica aqui, y no se estima: convertir la cola en dias exige
    # el limite de churn vigente, que depende del numero de validadores activos y cambia
    # con cada fork. Inventar una constante para eso seria la clase de numero que este
    # repositorio no escribe. El catalogo declara las dos colas y nada mas.
    return {
        ENTITY: record.get("entity") or "ethereum",
        DAY: day,
        "staking_entry_queue": entering,
        "staking_exit_queue": exiting,
    }


# =====================================================================================

ADAPTERS = {
    "btc_difficulty": MempoolDifficulty,
    "defillama_hacks": DefiLlamaHacks,
    "macro_calendar": MacroCalendar,
    "ofac_sdn": OfacSdn,
    "token_unlocks": DefiLlamaUnlocks,
    "staking_queue": BeaconchainStakingQueue,
}


def register() -> None:
    """Da de alta las seis. Tolera el duplicado UNA A UNA y no en bloque: si abortara al
    primer repetido, anadir una fuente nueva a `ADAPTERS` dejaria de registrarse en cuanto
    alguien hubiese llamado antes a `register_all()` en el mismo proceso."""
    from ai_trader.signals.source import register_adapter

    for key, factory in ADAPTERS.items():
        try:
            register_adapter(key, factory)
        except ValueError:
            logger.debug("%s ya estaba registrado", key)


__all__ = [
    "ADAPTERS",
    "BITCOIN_ENTITY",
    "BLS_CPI_SCHEDULE_PATH",
    "CRYPTO_ID_PREFIX",
    "DIFFICULTY_HISTORY_PATH",
    "FOMC_MEETING_TITLE",
    "OFAC_SDN_PATH",
    "BeaconchainStakingQueue",
    "DefiLlamaHacks",
    "DefiLlamaUnlocks",
    "MacroCalendar",
    "MempoolDifficulty",
    "OfacSdn",
    "cpi_release_days",
    "crypto_addresses",
    "fomc_meeting_days",
    "register",
]
