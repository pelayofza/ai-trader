"""
Wikipedia Pageviews: atencion, DESGLOSADA POR IDIOMA.

QUE ANADE SOBRE UN INDICE DE BUSQUEDAS
--------------------------------------
Google Trends devuelve un indice normalizado, relativo a una ventana y a una region, que
cambia de escala segun lo que se le pida: no es una serie que se pueda archivar y comparar
consigo misma dentro de un ano. Pageviews devuelve un CONTEO, absoluto y estable, con dos
propiedades que aqui importan: es horario y esta partido por proyecto de idioma.

Esa particion es una descomposicion GEOGRAFICA barata de la atencion. Que las visitas a
"Bitcoin" suban un 40% no dice lo mismo si suben en todos los idiomas —ciclo global— que si
suben solo en `tr` o `ru` —moneda local rompiendose, que es el mismo fenomeno que mide la
prima P2P y llega por otra puerta—. `pageviews_lang_concentration` es el HHI sobre el
reparto por idioma: 1/n_idiomas = atencion repartida, cerca de 1 = un solo idioma.

DOS COSAS MEDIDAS QUE CAMBIAN EL DISENO
---------------------------------------
1. El limite de peticiones es AGRESIVO. Ocho peticiones en paralelo devuelven 429, y en
   serie con 0,7 s tambien acaban en 429 (medido). Por eso este adaptador es
   deliberadamente lento y secuencial: `PAUSE_SECONDS` entre llamadas y ningun paralelismo.
   Con 6 idiomas y los 15 articulos de la tabla son 90 peticiones, unos ocho minutos. Una
   fuente diaria puede permitirselo; el paralelismo, no.
2. El User-Agent generico no vale: la politica de Wikimedia exige uno descriptivo y el que
   trae el cliente por defecto se come un 403 (medido). Va en `USER_AGENT` y se puede
   sobreescribir por entorno.

LOS TITULOS SON UNA TABLA, Y ES UN HECHO NO DERIVABLE
-----------------------------------------------------
El articulo de SOL no se llama "SOL" sino "Solana (blockchain platform)", y en japones no
se llama asi en absoluto. No hay regla que derive eso de un ticker: es exactamente el caso
que justifica una tabla explicita, como `ENTITY_OVERRIDES` en `shared/entities.py`. Los
titulos en ingles de la tabla estan COMPROBADOS uno a uno contra la API; los que devolvian
404 no estan, y su hueco aparece como cobertura cero en la auditoria en vez de como una
peticion que falla en silencio cada dia.
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from urllib.parse import quote

import pandas as pd

from ai_trader.data.providers.http import JsonHttpConfig
from ai_trader.shared.signals import DAY, ENTITY, OBSERVED
from ai_trader.signals.adapters.common import UTC, numeric, wiki_stamp
from ai_trader.signals.source import RAW_REQUEST, BaseJsonAdapter, RawRecord

logger = logging.getLogger(__name__)

WIKIMEDIA_BASE = "https://wikimedia.org"
PAGEVIEWS_PATH = "/api/rest_v1/metrics/pageviews/per-article"

# Politica de Wikimedia: un UA descriptivo. El generico se come un 403 (medido).
USER_AGENT = os.environ.get(
    "WIKIMEDIA_USER_AGENT", "ai-trader/0.1.0 (senales de mercado; uso de investigacion)"
)

# Segundos entre peticiones. MEDIDO a base de comerse el 429: con 0,7 s salta a la decima
# peticion y con 2 s tambien acaba saltando en una pasada larga. Cinco segundos aguantan un
# barrido completo. Parece exagerado y no lo es: el limite se cuenta por IP y a lo largo del
# tiempo, asi que la unica forma de terminar el barrido es no correr.
PAUSE_SECONDS = float(os.environ.get("WIKIMEDIA_PAUSE_SECONDS", "5.0"))

# Espera del reintento. Larga a proposito: un 429 no se cura en medio segundo, y el backoff
# corto del cliente por defecto solo sirve para gastar los tres intentos sin conseguir nada.
RETRY_BACKOFF_SECONDS = 15.0

# La API no tiene datos anteriores a esta fecha: pedir 2014 gasta una peticion del limite
# para recibir un rango vacio.
PAGEVIEWS_START = datetime(2015, 7, 1, tzinfo=UTC)

# Idiomas. Seis y no veinte: cada idioma multiplica las peticiones por el numero de
# activos, y estos seis ya separan lo global de lo local, que es para lo que sirve el
# desglose.
LANGUAGES: tuple[str, ...] = ("en", "es", "de", "ja", "ru", "zh")

# entidad -> titulo en ingles. COMPROBADO contra la API; lo que devolvia 404 no esta.
ARTICLES: dict[str, str] = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "SOL": "Solana (blockchain platform)",
    "XRP": "XRP Ledger",
    "ADA": "Cardano (blockchain platform)",
    "DOGE": "Dogecoin",
    "AVAX": "Avalanche (blockchain platform)",
    "DOT": "Polkadot (cryptocurrency)",
    "LINK": "Chainlink (blockchain oracle)",
    "LTC": "Litecoin",
    "UNI": "Uniswap",
    "NEAR": "NEAR Protocol",
    "OP": "Optimism (blockchain)",
    "FIL": "Filecoin",
    "ETC": "Ethereum Classic",
}

# Titulo por idioma cuando NO es el mismo que en ingles. Sin esto, `ja`, `ru` y `zh`
# devolverian 404 para todo y el desglose geografico —el motivo de usar esta fuente— seria
# tres columnas vacias.
ARTICLES_BY_LANGUAGE: dict[str, dict[str, str]] = {
    "BTC": {"ja": "ビットコイン", "ru": "Биткойн", "zh": "比特幣"},
    "ETH": {"ja": "イーサリアム", "zh": "以太坊"},
    "DOGE": {"ja": "ドージコイン", "zh": "狗狗幣"},
    "LTC": {"ja": "ライトコイン", "zh": "萊特幣"},
}

DEFAULT_DAYS = 365


class WikipediaPageviews(BaseJsonAdapter):
    """Un registro crudo por (entidad, idioma, dia). El reparto se calcula en la capa 2."""

    def __init__(self, source, *, pause_seconds: float = PAUSE_SECONDS, **kwargs) -> None:
        kwargs.setdefault(
            "http_config",
            JsonHttpConfig(
                user_agent=USER_AGENT,
                timeout_seconds=30.0,
                backoff_seconds=RETRY_BACKOFF_SECONDS,
            ),
        )
        super().__init__(source, base_url=WIKIMEDIA_BASE, **kwargs)
        self.pause_seconds = pause_seconds

    def article_for(self, entity: str, language: str) -> str | None:
        override = ARTICLES_BY_LANGUAGE.get(entity, {}).get(language)
        return override or ARTICLES.get(entity)

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        finish = end or datetime.now(UTC)
        begin = max(start or finish - timedelta(days=DEFAULT_DAYS), PAGEVIEWS_START)
        span = f"{begin.strftime('%Y%m%d')}/{finish.strftime('%Y%m%d')}"

        out: list[RawRecord] = []
        for entity in entities:
            for language in LANGUAGES:
                article = self.article_for(entity, language)
                if article is None:
                    continue
                path = (
                    f"{PAGEVIEWS_PATH}/{language}.wikipedia/all-access/all-agents/"
                    f"{quote(article, safe='')}/daily/{span}"
                )
                try:
                    payload = self.client.get_json(path)
                except Exception as exc:  # noqa: BLE001 - un articulo sin version en ese idioma
                    logger.info("· wikipedia: %s/%s sin serie (%s)", language, article, exc)
                    payload = None
                finally:
                    # La pausa va tambien tras un fallo: el 429 se cura esperando, y volver
                    # a golpear inmediatamente es lo que lo perpetua.
                    time.sleep(self.pause_seconds)

                for item in (payload or {}).get("items") or []:
                    day = wiki_stamp(item.get("timestamp"))
                    if day:
                        out.append(
                            self.record(
                                entity,
                                item,
                                day=day,
                                request={"language": language, "article": article},
                            )
                        )
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        by_day: dict[tuple[str, str], dict[str, float]] = {}
        for record in records or ():
            payload = record.get("payload") or {}
            day = wiki_stamp(payload.get("timestamp"))
            views = numeric(payload.get("views"))
            language = (record.get(RAW_REQUEST) or {}).get("language") or payload.get("project", "")
            entity = str(record.get("entity") or "")
            if not day or views is None or not entity:
                continue
            # Un mismo (entidad, idioma, dia) puede estar archivado varias veces —cada
            # captura re-descarga la ventana—. Se queda el ULTIMO, que es la revision mas
            # reciente de Wikimedia, en vez de sumarse: sumarlo multiplicaria las visitas
            # por el numero de capturas.
            by_day.setdefault((entity, day), {})[str(language)] = views

        rows = []
        for (entity, day), per_language in by_day.items():
            views = list(per_language.values())
            rows.append(
                {
                    ENTITY: entity,
                    DAY: day,
                    "pageviews": float(sum(views)),
                    "pageviews_lang_concentration": _hhi(views),
                    # `observed` cuenta IDIOMAS, que es lo que hay detras de la celda. Un dia
                    # con seis proyectos y otro con uno no son la misma observacion, y sin
                    # esta cuenta la concentracion de un solo idioma valdria 1.0 —"toda la
                    # atencion es local"— cuando lo que pasa es que faltan cinco series.
                    OBSERVED: len(per_language),
                }
            )
        return self.to_daily(rows)


def _hhi(values: Sequence[float]) -> float | None:
    """Herfindahl del reparto por idioma. 1 = un solo idioma; 1/n = repartido a partes."""
    total = sum(values)
    if total <= 0:
        return None
    return float(sum((v / total) ** 2 for v in values))


def register() -> None:
    from ai_trader.signals.source import register_adapter

    register_adapter("wikipedia_pageviews", WikipediaPageviews)


__all__ = [
    "ARTICLES",
    "ARTICLES_BY_LANGUAGE",
    "LANGUAGES",
    "PAUSE_SECONDS",
    "USER_AGENT",
    "WikipediaPageviews",
    "register",
]
