"""
LEGAL E INSTITUCIONAL: lo que llega ANTES que la noticia, porque ES la noticia.

Un 8-K esta publicado en EDGAR horas antes de que nadie lo escriba; una norma esta en el
Federal Register el dia que se publica y no el dia que alguien la resume; una demanda existe
en el docket antes de que exista el titular. Las tres fuentes son gratuitas, las tres estan
fechadas con precision de dia y ninguna se revisa hacia atras salvo por enmienda explicita.

LAS TRES MIDEN LO MISMO: CUANTOS, NO QUE DICE
---------------------------------------------
Ninguna de las tres lee el contenido. Cuentan documentos que casan con una consulta, por
dia. Es deliberado y conviene que se vea: extraer la SEMANTICA de un filing es otro
problema —uno que necesita un modelo y una evaluacion— y mezclarlo con la ingesta haria que
un cambio de prompt cambiara una serie historica. Lo que se publica aqui es una cuenta, que
es re-derivable y no opina.

LO QUE CADA UNA COBRA POR SU PROFUNDIDAD, MEDIDO 2026-08-13
-----------------------------------------------------------
    EDGAR full-text     cubre desde 2001 (comprobado en 2001, 2005, 2011 y 2014) y no
                        acepta agregacion por fecha: hay que pedir UN DIA POR PETICION.
    Federal Register    acepta rango y mil documentos por peticion. El archivo llega a
                        1994 y la consulta declarada tiene 398 documentos desde 2010-01-11.
    CourtListener       la v4 responde SIN token (9.569 dockets) y la v3 da 403, que es al
                        reves de lo que documenta media internet. Pagina de veinte en veinte.

De ahi los dos topes declarados de este modulo (`EDGAR_MAX_DAYS`, `COURT_MAX_PAGES`). No son
limites del proveedor: son la linea donde una pasada de captura deja de ser razonable, y
estan escritos para que la profundidad que compran sea una decision visible y no el
resultado accidental de cuanto aguanto el bucle.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from ai_trader.data.providers.http import JsonHttpConfig
from ai_trader.shared.entities import MARKET_ENTITY
from ai_trader.shared.signals import DAY, ENTITY
from ai_trader.signals.adapters.common import (
    certifi_bundle,
    iso_day,
    numeric,
    rows_from_records,
    safe_call,
    unique_records,
)
from ai_trader.signals.source import BaseJsonAdapter, RawRecord

logger = logging.getLogger(__name__)

# La SEC exige un User-Agent que identifique a quien llama, con contacto. No es una
# recomendacion: sin el, devuelve 403.
SEC_USER_AGENT = "ai-trader research pelayofza@gmail.com"


# =====================================================================================
# 1. SEC EDGAR full-text search (EFTS)
# =====================================================================================

EFTS_BASE = "https://efts.sec.gov"
EFTS_PATH = "/LATEST/search-index"

# La consulta. Una frase entrecomillada y no una lista de terminos: EFTS interpreta varios
# terminos sueltos como una conjuncion, y "bitcoin cryptocurrency" devolveria solo los
# documentos que digan las dos cosas.
EDGAR_QUERY = '"bitcoin"'

# Formularios que cuentan como TENENCIA INSTITUCIONAL. Un 8-K de una empresa que menciona
# bitcoin de pasada y una posicion declarada por un gestor no son el mismo hecho, y la
# magnitud del evento se mide con la segunda.
INSTITUTIONAL_FORMS: frozenset[str] = frozenset({"13F-HR", "SC 13G", "SC 13D", "13F-NT"})

# Dias por pasada. EFTS no agrega por fecha, asi que la cuenta diaria cuesta una peticion
# por dia; cuatrocientos veinte compran algo mas de un ano (que es lo que
# `depth.MIN_MEASURED_DAYS` exige para poder declarar `history_from`) en unos minutos, y el
# backfill entero se hace con varias pasadas. La captura diaria pide treinta y no llega aqui.
EDGAR_MAX_DAYS = 420


class SecEdgarFullText(BaseJsonAdapter):
    """
    Un registro por DIA con el recuento de filings que casan con la consulta.

    Lo que se archiva no es la respuesta entera —serian cientos de kilobytes diarios de
    metadatos de cada documento— sino el total y el desglose por formulario, que es lo que
    la respuesta ya trae calculado en `aggregations.form_filter` y es EXACTO. Usar la
    agregacion en vez de contar los hits devueltos no es una optimizacion: la respuesta trae
    como mucho un centenar de hits, asi que contarlos daria un numero bajo en los dias de
    vencimiento de 13F, que son justo los que interesan.
    """

    def __init__(self, source, *, max_days: int = EDGAR_MAX_DAYS, **kwargs) -> None:
        config = kwargs.pop("http_config", None) or JsonHttpConfig(
            timeout_seconds=30.0,
            user_agent=SEC_USER_AGENT,
            headers={"Accept": "application/json"},
        )
        super().__init__(source, base_url=EFTS_BASE, http_config=config, **kwargs)
        self._max_days = max_days

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        stop = (end or datetime.now(timezone.utc)).date()
        begin = (start.date() if start else stop - timedelta(days=30))
        days = _day_range(begin, stop, self._max_days)
        if not days:
            return []
        logger.info("· edgar: %s dias, de %s a %s", len(days), days[0], days[-1])

        out: list[RawRecord] = []
        for day in days:
            payload = safe_call(
                lambda d=day: self.client.get_json(
                    EFTS_PATH, params={"q": EDGAR_QUERY, "startdt": d, "enddt": d}
                ),
                what=f"edgar fts {day}",
                logger=logger,
            )
            counts = form_counts(payload)
            if counts is None:
                continue
            out.append(self.record(MARKET_ENTITY, {"date": day, **counts}, day=day))
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        # Una linea por dia: la ultima capturada gana. No hace falta sumar nada aqui, y por
        # eso el deduplicado es por (dia) y no por documento.
        latest = unique_records(
            records, key_of=lambda r: iso_day((r.get("payload") or {}).get("date"))
        )
        return self.to_daily(rows_from_records(latest, row_of=_edgar_row))


def _day_range(begin: date, stop: date, max_days: int) -> list[str]:
    """Los dias del rango, RECORTADOS POR EL FINAL para quedarse con los mas recientes.

    El recorte va por delante y no por detras a proposito: si una pasada no llega a cubrir
    doce anos, lo util es tener el ano ultimo completo y no los doce primeros meses de 2014.
    """
    if stop < begin:
        return []
    total = (stop - begin).days + 1
    first = begin if total <= max_days else stop - timedelta(days=max_days - 1)
    return [(first + timedelta(days=i)).isoformat() for i in range((stop - first).days + 1)]


def form_counts(payload) -> dict | None:
    """La respuesta de EFTS -> `{total, institutional, by_form, other}`. None si no hay."""
    hits = (payload or {}).get("hits") if isinstance(payload, Mapping) else None
    if not isinstance(hits, Mapping):
        return None
    total = ((hits.get("total") or {}).get("value")) if isinstance(hits.get("total"), Mapping) else None
    if total is None:
        return None

    buckets = (
        ((payload.get("aggregations") or {}).get("form_filter") or {}).get("buckets") or []
    )
    by_form = {str(b.get("key")): int(b.get("doc_count") or 0) for b in buckets if b.get("key")}
    institutional = sum(n for form, n in by_form.items() if form in INSTITUTIONAL_FORMS)
    return {
        "total": int(total),
        "institutional": institutional,
        "by_form": by_form,
        # Cuantos documentos quedaron fuera del desglose. Si es mayor que cero, el recuento
        # institucional es una COTA INFERIOR y esto es lo que permite saberlo despues.
        "other": int(
            ((payload.get("aggregations") or {}).get("form_filter") or {}).get(
                "sum_other_doc_count"
            )
            or 0
        ),
    }


def _edgar_row(record: Mapping) -> dict | None:
    payload = record.get("payload") or {}
    day = iso_day(payload.get("date")) or iso_day(record.get("day"))
    total = numeric(payload.get("total"))
    if not day or total is None:
        return None
    return {
        ENTITY: MARKET_ENTITY,
        DAY: day,
        "edgar_filings": total,
        "edgar_institutional": numeric(payload.get("institutional")) or 0.0,
    }


# =====================================================================================
# 2. Federal Register
# =====================================================================================

FEDREG_BASE = "https://www.federalregister.gov"
FEDREG_PATH = "/api/v1/documents.json"

FEDREG_QUERY = '"digital asset"'
FEDREG_PAGE_SIZE = 1000
FEDREG_MAX_PAGES = 12

# Que tipos cuentan como actividad NORMATIVA. El resto (avisos, correcciones, actos
# presidenciales) sigue contando en el total pero no en la magnitud del evento: el aviso
# administrativo rutinario es la mayor parte del volumen y no mueve nada.
RULE_TYPES: frozenset[str] = frozenset({"Rule", "Proposed Rule"})


class FederalRegister(BaseJsonAdapter):
    """Un registro por DOCUMENTO, fechado el dia en que se publica."""

    def __init__(self, source, **kwargs) -> None:
        # federalregister.gov no valida con el almacen del sistema (MEDIDO 2026-08-13); la
        # SEC, que esta justo arriba en este mismo modulo, si. Por eso va por fuente.
        config = kwargs.pop("http_config", None) or JsonHttpConfig(
            timeout_seconds=60.0, ca_bundle=certifi_bundle()
        )
        super().__init__(source, base_url=FEDREG_BASE, http_config=config, **kwargs)

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        stop = (end or datetime.now(timezone.utc)).date()
        begin = start.date() if start else stop - timedelta(days=30)

        out: list[RawRecord] = []
        for page in range(1, FEDREG_MAX_PAGES + 1):
            payload = safe_call(
                lambda p=page: self.client.get_json(
                    FEDREG_PATH,
                    params={
                        "per_page": FEDREG_PAGE_SIZE,
                        "page": p,
                        "order": "oldest",
                        "conditions[term]": FEDREG_QUERY,
                        "conditions[publication_date][gte]": begin.isoformat(),
                        "conditions[publication_date][lte]": stop.isoformat(),
                        "fields[]": [
                            "document_number",
                            "publication_date",
                            "type",
                            "title",
                            "effective_on",
                            "comments_close_on",
                        ],
                    },
                ),
                what=f"federal register p{page}",
                logger=logger,
            )
            results = (payload or {}).get("results") or []
            if not results:
                break
            for document in results:
                day = iso_day(document.get("publication_date"))
                if day:
                    out.append(self.record(MARKET_ENTITY, document, day=day))
            if len(results) < FEDREG_PAGE_SIZE:
                break
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        latest = unique_records(
            records, key_of=lambda r: str((r.get("payload") or {}).get("document_number") or "")
        )
        return self.to_daily(rows_from_records(latest, row_of=_fedreg_row))


def _fedreg_row(record: Mapping) -> dict | None:
    payload = record.get("payload") or {}
    day = iso_day(payload.get("publication_date")) or iso_day(record.get("day"))
    if not day or not payload.get("document_number"):
        return None
    return {
        ENTITY: MARKET_ENTITY,
        DAY: day,
        "fedreg_documents": 1.0,
        "fedreg_rules": 1.0 if str(payload.get("type") or "") in RULE_TYPES else 0.0,
    }


# =====================================================================================
# 3. CourtListener / RECAP
# =====================================================================================

COURT_BASE = "https://www.courtlistener.com"
COURT_PATH = "/api/rest/v4/search/"

COURT_QUERY = "cryptocurrency"

# Paginas por pasada, de veinte en veinte. Veinticinco son quinientos dockets: cubre de
# sobra la ventana de captura y acota lo que una sonda de doce anos puede llegar a pedir.
# Lo que queda fuera queda fuera y se dice; publicar un recuento truncado como si fuera
# completo seria el unico error grave posible aqui.
COURT_MAX_PAGES = 25

# Pausa entre paginas. MEDIDO 2026-08-13: la pagina 21 seguida devuelve 429 y el backoff del
# cliente no basta. Sin la pausa, una pasada larga se corta sola y declara menos historia de
# la que hay, que es la unica forma de que un limite de tasa mienta en el registro.
COURT_PAUSE_SECONDS = 1.5


class CourtListenerDockets(BaseJsonAdapter):
    """Un registro por DOCKET, fechado el dia en que se presenta."""

    def __init__(
        self,
        source,
        *,
        max_pages: int = COURT_MAX_PAGES,
        pause_seconds: float = COURT_PAUSE_SECONDS,
        **kwargs,
    ) -> None:
        # courtlistener.com: mismo caso medido que el Federal Register.
        config = kwargs.pop("http_config", None) or JsonHttpConfig(
            timeout_seconds=60.0, ca_bundle=certifi_bundle()
        )
        super().__init__(source, base_url=COURT_BASE, http_config=config, **kwargs)
        self._max_pages = max_pages
        self._pause = pause_seconds

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        begin = (start or datetime.now(timezone.utc) - timedelta(days=30)).date()
        out: list[RawRecord] = []
        cursor: str | None = None
        for page in range(self._max_pages):
            if page > 0 and self._pause > 0:
                time.sleep(self._pause)
            params = {
                "q": COURT_QUERY,
                "type": "r",  # RECAP: dockets federales
                "order_by": "dateFiled desc",
                "filed_after": begin.strftime("%m/%d/%Y"),
            }
            if cursor:
                params["cursor"] = cursor
            payload = safe_call(
                lambda p=params: self.client.get_json(COURT_PATH, params=p),
                what=f"courtlistener p{page + 1}",
                logger=logger,
            )
            results = (payload or {}).get("results") or []
            if not results:
                break
            for docket in results:
                day = iso_day(docket.get("dateFiled"))
                if not day:
                    continue
                out.append(
                    self.record(
                        MARKET_ENTITY,
                        {
                            "docket_id": docket.get("docket_id"),
                            "dateFiled": docket.get("dateFiled"),
                            "court_id": docket.get("court_id"),
                            "caseName": docket.get("caseName"),
                            "docketNumber": docket.get("docketNumber"),
                        },
                        day=day,
                    )
                )
            cursor = _cursor_of(payload.get("next"))
            if not cursor:
                break
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        latest = unique_records(
            records, key_of=lambda r: (r.get("payload") or {}).get("docket_id")
        )
        return self.to_daily(rows_from_records(latest, row_of=_court_row))


def _cursor_of(url) -> str | None:
    """El cursor de la url `next` que devuelve la API. None si no hay mas paginas."""
    from urllib.parse import parse_qs, urlparse

    if not url:
        return None
    values = parse_qs(urlparse(str(url)).query).get("cursor")
    return values[0] if values else None


def _court_row(record: Mapping) -> dict | None:
    payload = record.get("payload") or {}
    day = iso_day(payload.get("dateFiled")) or iso_day(record.get("day"))
    if not day or payload.get("docket_id") is None:
        return None
    return {ENTITY: MARKET_ENTITY, DAY: day, "court_dockets": 1.0}


# =====================================================================================

ADAPTERS = {
    "sec_edgar_fts": SecEdgarFullText,
    "federal_register": FederalRegister,
    "courtlistener_dockets": CourtListenerDockets,
}


def register() -> None:
    from ai_trader.signals.source import register_adapter

    for key, factory in ADAPTERS.items():
        try:
            register_adapter(key, factory)
        except ValueError:
            logger.debug("%s ya estaba registrado", key)


__all__ = [
    "ADAPTERS",
    "COURT_MAX_PAGES",
    "COURT_QUERY",
    "EDGAR_MAX_DAYS",
    "EDGAR_QUERY",
    "FEDREG_QUERY",
    "INSTITUTIONAL_FORMS",
    "RULE_TYPES",
    "SEC_USER_AGENT",
    "CourtListenerDockets",
    "FederalRegister",
    "SecEdgarFullText",
    "form_counts",
    "register",
]
