"""
Prima P2P en monedas en crisis: un driver que no es el ciclo especulativo.

QUE MIDE
--------
Lo que cuesta comprar un dolar sintetico (USDT) con lira, peso argentino o naira, contra el
tipo de cambio oficial del mismo momento. Cuando esa prima se abre, no es que el mercado
cripto este alcista: es que alguien esta pagando por salir de su moneda. Es demanda por
huida, y llega por una puerta distinta a la del apetito de riesgo global —aunque a veces
coincida con la que mide el desglose por idioma de Wikipedia, que es justo lo interesante
de tener las dos.

POR QUE EL PAYLOAD LLEVA LAS DOS PATAS JUNTAS
---------------------------------------------
Una prima es un cociente, y el denominador —el tipo oficial— tampoco es re-descargable con
fecha pasada en una fuente gratuita. Si se archivara solo el libro P2P, dentro de seis meses
habria precios en lira sin nada contra lo que compararlos, y la serie entera seria
inservible. Por eso cada registro crudo guarda el libro Y el tipo oficial del mismo momento,
los dos intactos: la pareja es el dato, no cada parte por separado.

`pit='forward_capture'` en estado puro: NADIE publica el libro P2P de hace tres meses. Lo
que no se capture hoy no existira nunca, y es la razon de que la captura arrancase antes
que ningun adaptador.

LA MEDIANA DEL LIBRO, NO LA MEJOR OFERTA
----------------------------------------
El mejor anuncio de la pagina suele ser un cebo de importe minusculo. La mediana de los
primeros anuncios es mas barata de defender y mucho mas estable; y como el libro entero
queda archivado, cualquier otro estimador se puede recalcular sin volver a capturar.

Y se capturan LOS DOS LADOS. En la jerga de Binance, `tradeType=BUY` devuelve los anuncios
contra los que el usuario compra USDT (la demanda paga ahi) y `SELL` los del lado
contrario; la feature publicada promedia los dos, asi que es la prima del PUNTO MEDIO y no
la del lado caro. Es la eleccion conservadora —el diferencial entre lados es coste de
transaccion, no senal— y como los dos lados quedan en el archivo con su marca en
`request.trade_type`, la prima de solo compra se puede re-derivar sin volver a capturar.
"""
from __future__ import annotations

import logging
import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime

import pandas as pd

from ai_trader.data.providers.http import JsonHttpClient
from ai_trader.shared.signals import DAY, ENTITY
from ai_trader.signals.adapters.common import iso_day, numeric
from ai_trader.signals.source import (
    RAW_DAY,
    RAW_FETCHED_AT,
    RAW_REQUEST,
    BaseJsonAdapter,
    RawRecord,
)

logger = logging.getLogger(__name__)

BINANCE_P2P_BASE = "https://p2p.binance.com"
SEARCH_PATH = "/bapi/c2c/v2/friendly/c2c/adv/search"

# Tipo de cambio oficial de referencia: abierto, sin credencial y con actualizacion diaria.
FX_BASE = "https://open.er-api.com"
FX_PATH = "/v6/latest/USD"

# Monedas en crisis, tres regimenes distintos: inflacion alta con tipo administrado (TRY),
# control de capitales (ARS) y escasez de divisa (NGN).
#
# MEDIDO: TRY y ARS devuelven libro lleno (10 anuncios por lado); NGN devuelve CERO
# anuncios. Se queda en la lista igualmente —la consulta es barata y el dia que vuelva a
# haber libro se captura solo— y por eso `p2p_currencies` vale 2 y no 3: esa columna existe
# precisamente para que un hueco de cobertura se vea como tal en vez de disolverse dentro de
# una media de dos monedas que parece de tres.
CURRENCIES: tuple[str, ...] = ("TRY", "ARS", "NGN")

# Lado del libro. `BUY` en la jerga de Binance es el lado en el que el usuario COMPRA
# USDT pagando fiat, que es el que mide la demanda de dolares.
TRADE_TYPES: tuple[str, ...] = ("BUY", "SELL")

ROWS_PER_PAGE = 10


class BinanceP2PPremium(BaseJsonAdapter):
    """Un registro crudo por (activo, moneda, lado), con el tipo oficial pareado."""

    def __init__(self, source, *, fx_client: JsonHttpClient | None = None, **kwargs) -> None:
        super().__init__(source, base_url=BINANCE_P2P_BASE, **kwargs)
        self._fx_client = fx_client

    @property
    def fx_client(self) -> JsonHttpClient:
        if self._fx_client is None:
            self._fx_client = JsonHttpClient(FX_BASE, self._http_config)
        return self._fx_client

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        official = self.fx_client.get_json(FX_PATH) or {}
        rates = official.get("rates") or {}

        out: list[RawRecord] = []
        for asset in entities:
            for currency in CURRENCIES:
                rate = numeric(rates.get(currency))
                if rate is None:
                    logger.info("· p2p: sin tipo oficial para %s, se salta", currency)
                    continue
                for trade_type in TRADE_TYPES:
                    book = self.client.post_json(
                        SEARCH_PATH,
                        body={
                            "asset": asset,
                            "fiat": currency,
                            "tradeType": trade_type,
                            "page": 1,
                            "rows": ROWS_PER_PAGE,
                            "payTypes": [],
                            "publisherType": None,
                        },
                    )
                    payload = {
                        "p2p": book,
                        "fx": {
                            "rate": rate,
                            "currency": currency,
                            "source": FX_BASE,
                            "as_of": official.get("time_last_update_utc"),
                        },
                    }
                    out.append(
                        self.record(
                            asset,
                            payload,
                            request={"currency": currency, "trade_type": trade_type},
                        )
                    )
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        rows: list[dict] = []
        currencies_by_day: dict[tuple[str, str], set[str]] = {}

        for record in records or ():
            row = _premium_row(record)
            if row is None:
                continue
            rows.append(row)
            key = (row[ENTITY], row[DAY])
            currency = (record.get(RAW_REQUEST) or {}).get("currency")
            if currency:
                currencies_by_day.setdefault(key, set()).add(str(currency))

        frame = self.to_daily(rows)
        if frame.empty:
            return frame
        # Cuantas monedas cotizaron ese dia es un recuento DISTINTO, y un recuento distinto
        # no sobrevive a una agregacion por media: se calcula antes y se pega despues.
        frame["p2p_currencies"] = [
            float(len(currencies_by_day.get((entity, day.date().isoformat()), ())))
            for entity, day in frame.index
        ]
        return frame


def _premium_row(record: Mapping) -> dict | None:
    payload = record.get("payload") or {}
    book = payload.get("p2p") or {}
    rate = numeric((payload.get("fx") or {}).get("rate"))
    if rate is None or rate <= 0:
        return None

    prices = [
        price
        for advert in book.get("data") or []
        if (price := numeric((advert.get("adv") or {}).get("price"))) is not None and price > 0
    ]
    if not prices:
        return None

    # No hay `day` en el registro: el libro P2P es la foto de AHORA, asi que el dia de la
    # observacion ES el de la captura.
    day = iso_day(record.get(RAW_DAY)) or iso_day(record.get(RAW_FETCHED_AT))
    if not day:
        return None

    premium_pct = 100.0 * (statistics.median(prices) / rate - 1.0)
    return {
        ENTITY: record.get("entity") or "",
        DAY: day,
        "p2p_premium_pct": premium_pct,
        "p2p_currencies": None,
    }


def register() -> None:
    from ai_trader.signals.source import register_adapter

    register_adapter("p2p_premium", BinanceP2PPremium)


__all__ = [
    "CURRENCIES",
    "FX_BASE",
    "SEARCH_PATH",
    "TRADE_TYPES",
    "BinanceP2PPremium",
    "register",
]
