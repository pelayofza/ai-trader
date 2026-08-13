"""
ATENCION GEOGRAFICA: donde esta entrando el retail, no cuanto se habla.

POR QUE EL DIFERENCIAL Y NO EL NIVEL
------------------------------------
"Coinbase esta en el puesto 12 de EE.UU." no dice casi nada: el nivel absoluto de una lista
de aplicaciones lo domina la epoca del ano, la campana de marketing del banco de turno y
que la lista es general y no de finanzas. Lo que si dice algo es el DIFERENCIAL entre Corea
y Estados Unidos, porque el retail coreano entra ANTES y con mas fuerza, y porque Corea pesa
de forma desproporcionada en las altcoins. Ese diferencial es una comparacion entre dos
listas construidas igual el mismo dia, asi que casi todo lo que contamina el nivel se
cancela.

Naver y Yandex son la misma idea por otra via: los buscadores de dos paises donde el
buscador dominante NO es Google, y cuyos datos son gratuitos y practicamente nadie mira
desde fuera.

LA MEDICION QUE CAMBIO EL DISENO
--------------------------------
La intencion era una serie continua de rankings. MEDIDO 2026-08-13:

  - la lista de Apple tiene CIEN puestos y no mas: pedir 200 devuelve 500 Internal Server
    Error, y el parametro de genero se IGNORA (pedir finanzas devuelve la lista general);
  - ese dia NINGUNA de las cuatro apps (Upbit, Coinbase, Binance, Bitget) estaba en el top
    100 de Corea ni en el de EE.UU.

Una serie continua que es cero casi siempre no la puede normalizar `normalize.py`: mediana
0, rango intercuartilico 0, z indefinida o gigantesca. Asi que la fuente se declara de
EVENTO y el evento es "una app cripto entra en la lista general", que resulta ser
exactamente el hecho que interesa —la entrada de retail masivo— y no un apano. Los dias en
que no entra ninguna no producen fila, que es lo que distingue "hoy no pasa nada" de "no se
nada de esto".

VISIBILIDAD Y NO PUESTO
-----------------------
Publicar el puesto (1..100) tiene dos problemas: esta CENSURADO —fuera de la lista no hay
numero, y un NaN diario no es lo mismo que "no aparece"— y va al reves (mejor es menor). Se
publica la VISIBILIDAD, `(CHART_SIZE + 1 - puesto) / CHART_SIZE`: 1,0 en el primer puesto,
0,01 en el ultimo y 0,0 fuera de la lista. El cero significa algo que se ha observado —"no
esta entre las cien"— y no un hueco.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone

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
    unique_records,
)
from ai_trader.signals.source import BaseJsonAdapter, RawRecord

logger = logging.getLogger(__name__)


# =====================================================================================
# 1. Ranking en App Store (Corea frente a EE.UU.)
# =====================================================================================

APPLE_RSS_BASE = "https://rss.applemarketingtools.com"

# Cuantos puestos tiene la lista. MEDIDO: cien es el maximo que acepta el endpoint.
CHART_SIZE = 100

# Los dos paises que forman el diferencial, y en este orden: primero el que se adelanta.
COUNTRIES: tuple[str, ...] = ("kr", "us")

# Las apps que se buscan en cada lista. Se comparan en minusculas y por subcadena porque el
# nombre publicado cambia con la campana ("Coinbase: Buy Bitcoin & Ether") y porque la
# version coreana de Upbit se publica en hangul. Cada entrada esta MEDIDA contra el nombre
# real que devuelve la lista de su pais.
APPS: dict[str, tuple[str, ...]] = {
    "kr": ("upbit", "업비트", "bithumb", "빗썸", "coinone", "코인원", "binance", "bitget"),
    "us": ("coinbase", "binance", "bitget", "crypto.com", "kraken"),
}


class AppStoreRank(BaseJsonAdapter):
    """Una linea por pais y dia con la lista ENTERA. Ver el docstring sobre el recorte."""

    def __init__(self, source, *, countries: Sequence[str] = COUNTRIES, **kwargs) -> None:
        super().__init__(source, base_url=APPLE_RSS_BASE, **kwargs)
        self._countries = tuple(countries)

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        day = datetime.now(timezone.utc).date().isoformat()
        out: list[RawRecord] = []
        for country in self._countries:
            payload = safe_call(
                lambda c=country: self.client.get_json(
                    f"/api/v2/{c}/apps/top-free/{CHART_SIZE}/apps.json"
                ),
                what=f"app store {country}",
                logger=logger,
            )
            results = ((payload or {}).get("feed") or {}).get("results") or []
            if not results:
                continue
            # Se archiva la lista ENTERA y no solo las apps buscadas: la tabla `APPS` es una
            # decision nuestra y va a cambiar, y con la lista entera en disco el cambio se
            # re-deriva sobre el pasado en vez de empezar de cero.
            out.append(
                self.record(
                    MARKET_ENTITY,
                    {"country": country, "results": results},
                    day=day,
                    request={"country": country},
                )
            )
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        # Deduplicado por (dia, pais): la captura vuelve a bajar la lista todos los dias y
        # sin esto un dia tendria tantas lecturas como veces se capturo.
        latest = unique_records(
            records,
            key_of=lambda r: (
                iso_day(r.get("day")) or iso_day(r.get("fetched_at")),
                str((r.get("request") or {}).get("country") or ""),
            ),
        )

        by_day: dict[str, dict[str, float]] = {}
        for record in latest:
            day = iso_day(record.get("day")) or iso_day(record.get("fetched_at"))
            payload = record.get("payload") or {}
            country = str(payload.get("country") or "")
            if not day or country not in APPS:
                continue
            by_day.setdefault(day, {})[country] = chart_visibility(
                payload.get("results") or [], APPS[country]
            )

        rows: list[dict] = []
        for day, visibility in sorted(by_day.items()):
            korea = visibility.get("kr")
            usa = visibility.get("us")
            # El evento es que ALGUNA app entre en la lista. Un dia en que las dos listas se
            # descargaron y ninguna app aparecio no es un evento: es el estado normal.
            if not korea and not usa:
                continue
            rows.append(
                {
                    ENTITY: MARKET_ENTITY,
                    DAY: day,
                    "app_visibility_kr": korea,
                    "app_visibility_us": usa,
                    # El diferencial solo existe si se observaron LAS DOS listas: con una
                    # sola, un cero del otro lado seria una suposicion y no una lectura.
                    "app_visibility_gap": (
                        None if korea is None or usa is None else korea - usa
                    ),
                }
            )
        return self.to_daily(rows)


def chart_visibility(results: Sequence[Mapping], needles: Sequence[str]) -> float:
    """Visibilidad de las apps buscadas en una lista. Ver el docstring del modulo.

    Se queda con la MEJOR de las apps y no con la suma: la pregunta es si el retail de ese
    pais esta mirando cripto, y dos apps en el puesto 80 no son una en el puesto 40.
    """
    best = 0.0
    for index, entry in enumerate(results):
        name = str((entry or {}).get("name") or "").lower()
        if any(needle in name for needle in needles):
            best = max(best, (CHART_SIZE + 1 - (index + 1)) / CHART_SIZE)
    return float(best)


# =====================================================================================
# 2. Naver DataLab (Corea)
# =====================================================================================

NAVER_BASE = "https://openapi.naver.com"
NAVER_DATALAB_PATH = "/v1/datalab/search"

# La segunda variable de entorno. El catalogo solo declara el nombre de UNA (`auth_env` es
# un campo, no una lista) y aqui se lee la pareja: sin las dos no hay llamada posible.
NAVER_SECRET_ENV = "NAVER_CLIENT_SECRET"

# Las palabras. Son las genericas de cripto en coreano —"bitcoin", "moneda virtual",
# "coin"— y no las de un token concreto: el indice es RELATIVO dentro del grupo, asi que
# mezclar tokens haria que la serie de uno dependiera de la popularidad del otro.
NAVER_KEYWORDS: tuple[str, ...] = ("비트코인", "가상화폐", "코인")

# Longitud FIJA de la ventana que se pide, en dias. No es una preferencia: DataLab devuelve
# un indice normalizado a 100 DENTRO de la ventana pedida, asi que dos ventanas de distinta
# longitud producen series que no se pueden pegar. Con la ventana fija, el valor de un dia
# significa lo mismo en todas las capturas.
NAVER_WINDOW_DAYS = 90


class NaverDataLab(BaseJsonAdapter):
    """Indice de busqueda en Naver. MEDIDO 2026-08-13: 401 sin credencial."""

    def __init__(self, source, **kwargs) -> None:
        client_id = env_secret(source.auth_env)
        secret = env_secret(NAVER_SECRET_ENV)
        headers = (
            {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": secret}
            if client_id and secret
            else {}
        )
        config = kwargs.pop("http_config", None) or JsonHttpConfig(headers=headers)
        super().__init__(source, base_url=NAVER_BASE, http_config=config, **kwargs)
        self._ready = bool(client_id and secret)

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        if not self._ready:
            raise RuntimeError(
                f"La fuente '{self.source.key}' necesita {self.source.auth_env} y "
                f"{NAVER_SECRET_ENV} en el entorno (401 medido sin las dos)"
            )
        stop = end or datetime.now(timezone.utc)
        # La ventana es SIEMPRE de la misma longitud, se pida lo que se pida: ver
        # NAVER_WINDOW_DAYS. Si `start` pide mas, se ignora y queda dicho en el log.
        begin = stop - timedelta(days=NAVER_WINDOW_DAYS)
        if start is not None and start < begin:
            logger.info(
                "· naver: se pidio desde %s y la ventana es fija de %s dias",
                day_or_none(start),
                NAVER_WINDOW_DAYS,
            )

        payload = self.client.post_json(
            NAVER_DATALAB_PATH,
            body={
                "startDate": begin.date().isoformat(),
                "endDate": stop.date().isoformat(),
                "timeUnit": "date",
                "keywordGroups": [
                    {"groupName": "crypto", "keywords": list(NAVER_KEYWORDS)}
                ],
            },
        )
        out: list[RawRecord] = []
        for group in (payload or {}).get("results") or []:
            for point in group.get("data") or []:
                day = iso_day(point.get("period"))
                if day:
                    out.append(self.record(MARKET_ENTITY, point, day=day))
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        return self.to_daily(rows_from_records(records, row_of=_naver_row))


def _naver_row(record: Mapping) -> dict | None:
    payload = record.get("payload") or {}
    day = iso_day(payload.get("period")) or iso_day(record.get("day"))
    ratio = numeric(payload.get("ratio"))
    if not day or ratio is None:
        return None
    return {ENTITY: MARKET_ENTITY, DAY: day, "naver_search_index": ratio}


# =====================================================================================
# 3. Yandex Wordstat (Rusia)
# =====================================================================================

YANDEX_BASE = "https://api.direct.yandex.ru"
YANDEX_PATH = "/v4/json/"

YANDEX_PHRASES: tuple[str, ...] = ("биткоин", "криптовалюта")


class YandexWordstat(BaseJsonAdapter):
    """
    Impresiones mensuales en Yandex. MEDIDO 2026-08-13: sin token, error propio 501.

    Wordstat es MENSUAL y es una foto: el informe se genera contra el momento en que se
    pide y su pasado no se descarga. De las tres fuentes de atencion es la mas pobre, y esta
    porque el hueco geografico que cubre no lo cubre ninguna otra.
    """

    def __init__(self, source, **kwargs) -> None:
        token = env_secret(source.auth_env)
        config = kwargs.pop("http_config", None) or JsonHttpConfig(
            timeout_seconds=60.0,
            headers={"Authorization": f"Bearer {token}"} if token else {},
        )
        super().__init__(source, base_url=YANDEX_BASE, http_config=config, **kwargs)
        self._token = token

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        if not self._token:
            raise RuntimeError(
                f"La fuente '{self.source.key}' necesita {self.source.auth_env} en el entorno"
            )
        # El informe es ASINCRONO: se crea, se espera y se descarga. Aqui se pide y se lee
        # en la MISMA pasada, de modo que la primera captura del mes suele volver vacia y la
        # siguiente trae el informe. Es feo y es del proveedor: encadenar una espera dentro
        # de la captura bloquearia a las otras veintiocho fuentes.
        created = self.client.post_json(
            YANDEX_PATH,
            body={"method": "CreateNewWordstatReport", "param": {"Phrases": list(YANDEX_PHRASES)}},
        )
        report_id = (created or {}).get("data")
        payload = self.client.post_json(
            YANDEX_PATH, body={"method": "GetWordstatReport", "param": report_id}
        )
        rows = (payload or {}).get("data") or []
        if not rows:
            return []
        day = datetime.now(timezone.utc).date().isoformat()
        return [self.record(MARKET_ENTITY, {"report": rows}, day=day)]

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        return self.to_daily(rows_from_records(records, row_of=_yandex_row))


def _yandex_row(record: Mapping) -> dict | None:
    payload = record.get("payload") or {}
    day = iso_day(record.get("day")) or iso_day(record.get("fetched_at"))
    total = 0.0
    seen = False
    for block in payload.get("report") or []:
        for phrase in (block or {}).get("SearchedWith") or [block]:
            shows = numeric((phrase or {}).get("Shows"))
            if shows is not None:
                total += shows
                seen = True
    if not day or not seen:
        return None
    return {ENTITY: MARKET_ENTITY, DAY: day, "yandex_search_shows": total}


# =====================================================================================

ADAPTERS = {
    "appstore_rank": AppStoreRank,
    "naver_datalab": NaverDataLab,
    "yandex_wordstat": YandexWordstat,
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
    "APPS",
    "CHART_SIZE",
    "COUNTRIES",
    "NAVER_KEYWORDS",
    "NAVER_SECRET_ENV",
    "NAVER_WINDOW_DAYS",
    "YANDEX_PHRASES",
    "AppStoreRank",
    "NaverDataLab",
    "YandexWordstat",
    "chart_visibility",
    "register",
]
