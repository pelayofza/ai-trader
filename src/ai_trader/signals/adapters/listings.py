"""
LISTADOS Y DESLISTADOS: el evento mas limpio que hay en cripto, y el unico que ademas
pide una guarda operativa.

POR QUE UPBIT
-------------
El "efecto Upbit" es de los pocos eventos de cripto que casi no necesita defensa: un token
que se lista en el mercado KRW pasa de no ser comprable por el retail coreano a serlo, y
Corea pesa de forma desproporcionada en las altcoins. La direccion es conocida, el instante
es publico y la reaccion es medible. Y el otro lado —un deslistado— es oferta forzada mas
riesgo de liquidez, que no es lo mismo con el signo cambiado: la oferta forzada es mecanica
y la perdida de liquidez es estructural.

MEDIDO 2026-08-13: el endpoint publico de anuncios devuelve 38 paginas de categoria 'trade'
y llega hasta 2017-10-27. Es, con diferencia, el evento con mas historia gratuita del
catalogo entero despues del ajuste de dificultad.

Y LA CIFRA QUE HAY QUE LEER AL LADO
-----------------------------------
Los 283 mercados KRW tienen una mediana de volumen de 253.000 dolares al dia (p10: 40.193;
maximo: 102,6 M$). El evento es limpio y el tamano que admite, no. Por eso el catalogo
declara `typical_adv_usd` en esta fuente: no es un adorno, es la diferencia entre una senal
util y una senal que solo se puede operar con dinero que no mueva el precio.

TRES CLASES DE ANUNCIO Y UN PARSEO QUE ES UNA HEURISTICA DECLARADA
------------------------------------------------------------------
El endpoint de listado da el TITULO y nada mas. Los titulos estan en coreano y siguen tres
patrones estables, MEDIDOS sobre los anuncios reales:

    "프롬(PROM) KRW, USDT 마켓 디지털 자산 추가"          -> ALTA
    "댑오에스(DOS) 신규 거래지원 안내 (KRW, BTC, USDT 마켓)"  -> ALTA
    "이더리움클래식(ETC/KRW) 상장"                        -> ALTA (formato de 2017)
    "... 거래지원 종료 ..."                              -> BAJA
    "레이븐코인(RVN) 거래 유의 종목 지정 안내"             -> VIGILANCIA (previa a la baja)

El ticker sale de los parentesis. Eso es una HEURISTICA y esta escrita como tal: un titulo
con formato nuevo dara `None` y se descartara, que es preferible a inventarse una entidad y
cruzarla con los precios de otra. El titulo crudo se archiva entero, asi que el dia que
cambie el formato el pasado se vuelve a derivar sin haber perdido nada.

SE FECHA EL ANUNCIO, NO LA EJECUCION
------------------------------------
Es la misma decision que en el COT: el dia con el que se archiva es el de PUBLICACION,
porque es cuando la informacion existe. Para una baja hay ademas una fecha de ejecucion
posterior que vive en el CUERPO del anuncio y no en el listado; leerla costaria una peticion
por anuncio y queda declarado como limite, no resuelto a medias.
"""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone

import pandas as pd

from ai_trader.data.providers.http import JsonHttpConfig
from ai_trader.shared.signals import DAY, ENTITY
from ai_trader.signals.adapters.common import (
    certifi_bundle,
    iso_day,
    safe_call,
    unique_records,
)
from ai_trader.signals.source import BaseJsonAdapter, RawRecord

logger = logging.getLogger(__name__)

UPBIT_BASE = "https://api-manager.upbit.com"
ANNOUNCEMENTS_PATH = "/api/v1/announcements"

# La categoria de anuncios de negociacion. Las otras (deposito, retirada, mantenimiento) no
# cambian el conjunto de lo negociable.
CATEGORY = "trade"
PAGE_SIZE = 20

# Paginas por pasada. Cuarenta cubren el archivo entero MEDIDO (38 paginas hasta 2017), asi
# que el backfill completo cabe en una sola pasada y la captura diaria se para sola en
# cuanto la pagina se sale de la ventana pedida.
MAX_PAGES = 40

# Pausa entre paginas, en segundos. Es el numero que decide cuanta historia compra una
# pasada, y esta calibrado contra el 429 del proveedor MEDIDO tres veces el 2026-08-13:
#
#   sin pausa   la pagina 5 devuelve 429 -> el backfill se para en 2026-04 (4 meses)
#   1,0 s       la pagina 22 devuelve 429 -> se para en 2021-12 (4 anos)
#   2,5 s       llega al fondo del archivo, 38 paginas hasta 2017-10-27 (9 anos)
#
# Lo que hace visible esa tabla es que sin pausa el limite del proveedor no se nota: no hay
# error, hay MENOS HISTORIA, y el registro de profundidad la habria declarado como si fuera
# toda la que existe. En la captura diaria no cuesta nada —la ventana de treinta dias se
# agota en dos paginas— y solo se paga en el backfill, que se hace una vez.
PAGE_PAUSE_SECONDS = 2.5

# --- clasificacion, en el orden en que hay que probarla -------------------------------
#
# EL ORDEN NO ES ALFABETICO Y NO SE PUEDE CAMBIAR: "거래지원 종료" (fin de soporte) contiene
# "거래지원" (soporte), que es la marca de un alta. Probando el alta primero, todas las bajas
# se clasificarian como altas y la feature tendria el signo cambiado justo en los eventos
# que mas importan.
DELISTING_MARKERS: tuple[str, ...] = ("거래지원 종료", "거래 지원 종료", "상장폐지", "거래 종료")
WARNING_MARKERS: tuple[str, ...] = ("유의 종목 지정", "유의종목 지정", "투자유의")
LISTING_MARKERS: tuple[str, ...] = (
    "디지털 자산 추가",
    "신규 거래지원",
    "거래지원 개시",
    "마켓 추가",
    "상장",
)

LISTING_ADDED = "added"
LISTING_REMOVED = "removed"
LISTING_WARNED = "warned"

# Divisas de cotizacion de Upbit. Dentro de un parentesis son el MERCADO y no el token, y
# distinguirlo es lo que impide que "(KRW, BTC, USDT 마켓)" produzca tres listados falsos.
QUOTES: frozenset[str] = frozenset({"KRW", "BTC", "USDT", "USD"})

# La marca coreana de "mercado". Si aparece dentro del parentesis, lo que hay dentro es una
# lista de mercados y no de tokens.
MARKET_WORD = "마켓"

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")
_PAREN_RE = re.compile(r"\(([^()]*)\)")


class UpbitListings(BaseJsonAdapter):
    """Un registro por ANUNCIO, con el titulo intacto."""

    def __init__(
        self,
        source,
        *,
        max_pages: int = MAX_PAGES,
        pause_seconds: float = PAGE_PAUSE_SECONDS,
        **kwargs,
    ) -> None:
        # api-manager.upbit.com no valida con el almacen del sistema (MEDIDO 2026-08-13).
        config = kwargs.pop("http_config", None) or JsonHttpConfig(
            timeout_seconds=30.0, ca_bundle=certifi_bundle()
        )
        super().__init__(source, base_url=UPBIT_BASE, http_config=config, **kwargs)
        self._max_pages = max_pages
        self._pause = pause_seconds

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        floor = start or (datetime.now(timezone.utc) - timedelta(days=30))
        out: list[RawRecord] = []
        for page in range(1, self._max_pages + 1):
            if page > 1 and self._pause > 0:
                time.sleep(self._pause)
            payload = safe_call(
                lambda p=page: self.client.get_json(
                    ANNOUNCEMENTS_PATH,
                    params={"os": "web", "page": p, "per_page": PAGE_SIZE, "category": CATEGORY},
                ),
                what=f"upbit announcements p{page}",
                logger=logger,
            )
            notices = ((payload or {}).get("data") or {}).get("notices") or []
            if not notices:
                break

            oldest = None
            for notice in notices:
                moment = announcement_moment(notice)
                if moment is None:
                    continue
                oldest = moment if oldest is None else min(oldest, moment)
                if moment < floor:
                    continue
                day = iso_day(moment)
                out.append(
                    self.record(
                        # La entidad del CRUDO es el propio anuncio: un anuncio puede
                        # nombrar seis tokens y el archivo no puede duplicarlo seis veces.
                        # El reparto por token se hace al derivar.
                        "upbit",
                        notice,
                        day=day,
                        request={"page": page},
                    )
                )
            # La pagina viene de mas nuevo a mas viejo: en cuanto una entera cae por debajo
            # de la ventana pedida, las siguientes tambien.
            if oldest is not None and oldest < floor:
                break
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        latest = unique_records(
            records, key_of=lambda r: (r.get("payload") or {}).get("id")
        )
        rows: list[dict] = []
        for record in latest:
            rows.extend(_listing_rows(record))
        return self.to_daily(rows)


def announcement_moment(notice: Mapping) -> datetime | None:
    """El instante del anuncio, en UTC. Upbit lo publica en hora de Corea (+09:00)."""
    text = str((notice or {}).get("listed_at") or (notice or {}).get("first_listed_at") or "")
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    return moment.astimezone(timezone.utc) if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def classify(title: str) -> str | None:
    """Que clase de anuncio es. None si el titulo no encaja en ninguno de los tres.

    Devolver None es una respuesta legitima y frecuente: la categoria 'trade' incluye
    cambios de horario, avisos de mantenimiento y correcciones. Forzarlos a una clase
    inventaria eventos.
    """
    text = str(title or "")
    if any(marker in text for marker in DELISTING_MARKERS):
        return LISTING_REMOVED
    if any(marker in text for marker in WARNING_MARKERS):
        return LISTING_WARNED
    if any(marker in text for marker in LISTING_MARKERS):
        return LISTING_ADDED
    return None


def tickers(title: str) -> tuple[str, ...]:
    """Los tickers que nombra el titulo. Heuristica declarada: ver el docstring del modulo."""
    out: list[str] = []
    for group in _PAREN_RE.findall(str(title or "")):
        if MARKET_WORD in group:
            continue  # "(KRW, BTC, USDT 마켓)" es la lista de mercados, no de tokens
        for chunk in group.split(","):
            candidate = chunk.strip().split("/")[0].strip().upper()
            if not _TICKER_RE.match(candidate) or candidate in QUOTES:
                continue
            if candidate not in out:
                out.append(candidate)
    return tuple(out)


def _listing_rows(record: Mapping) -> list[dict]:
    payload = record.get("payload") or {}
    moment = announcement_moment(payload)
    day = iso_day(moment) if moment else iso_day(record.get("day"))
    kind = classify(payload.get("title"))
    if not day or kind is None:
        return []

    symbols = tickers(payload.get("title"))
    if not symbols:
        return []

    change = {LISTING_ADDED: 1.0, LISTING_REMOVED: -1.0}.get(kind)
    return [
        {
            ENTITY: symbol,
            DAY: day,
            # Un anuncio de vigilancia NO es media baja: no cambia lo negociable, avisa de
            # que puede cambiar. Se cuenta en su propia columna y deja `listing_change` a
            # cero para que la magnitud del evento no lo confunda con una baja de verdad.
            "listing_change": change if change is not None else 0.0,
            "listing_warning": 1.0 if kind == LISTING_WARNED else 0.0,
        }
        for symbol in symbols
    ]


ADAPTERS = {"cex_listings": UpbitListings}


def register() -> None:
    from ai_trader.signals.source import register_adapter

    for key, factory in ADAPTERS.items():
        try:
            register_adapter(key, factory)
        except ValueError:
            logger.debug("%s ya estaba registrado", key)


__all__ = [
    "ADAPTERS",
    "DELISTING_MARKERS",
    "LISTING_ADDED",
    "LISTING_MARKERS",
    "LISTING_REMOVED",
    "LISTING_WARNED",
    "MAX_PAGES",
    "QUOTES",
    "WARNING_MARKERS",
    "UpbitListings",
    "announcement_moment",
    "classify",
    "register",
    "tickers",
]
