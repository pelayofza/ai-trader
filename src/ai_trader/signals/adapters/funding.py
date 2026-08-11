"""
Dispersion del funding entre venues (CCXT). La DISPERSION, no el nivel.

POR QUE NO EL NIVEL
-------------------
El funding de un perpetuo es el precio de mantener la posicion, y es el numero mas
arbitrado del mercado: hay mesas dedicadas a cobrarlo, asi que el NIVEL agregado ya esta
descontado en cuanto se publica. Lo que no se arbitra igual de rapido es la DISCREPANCIA
entre venues: cuando Binance paga una cosa y OKX otra, es que el arbitraje esta topado
—colateral atrapado, limites de riesgo, un venue con el libro roto— y eso ocurre justo en
los momentos que importan.

`funding_median` esta al lado, y no como senal: es el contexto que hace interpretable la
dispersion. Una dispersion de 3 pb con la mediana en 1 pb no es lo mismo que la misma
dispersion con la mediana en 30.

LA DISPERSION SE CALCULA ANTES DE AGREGAR EL DIA
------------------------------------------------
Es una estadistica ENTRE VENUES en un instante, no a lo largo del dia. Si se archivaran los
niveles y se dejara que el esquema agregara, la desviacion tipica saldria de mezclar horas
distintas y mediria otra cosa. Por eso la capa 2 agrupa primero por (entidad, sello de
funding) —los venues comparten ventana de 8 h— calcula ahi la dispersion, y solo entonces
el dia promedia esas fotos.

FORWARD CAPTURE, Y DESIGUAL POR VENUE
-------------------------------------
`fetch_funding_rate` devuelve la foto de AHORA. Algunos venues sirven historico y otros no,
asi que la profundidad real es desigual y no hay un `history_from` unico honesto: se mide
capturando. Los tres venues estan MEDIDOS —los tres respondieron con `fundingRate` para
BTC/USDT:USDT— y un venue que falle no tumba a los demas.
"""
from __future__ import annotations

import logging
import os
import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime

import pandas as pd

from ai_trader.shared.signals import DAY, ENTITY
from ai_trader.signals.adapters.common import UTC, numeric, unix_day
from ai_trader.signals.source import RAW_FETCHED_AT, RAW_REQUEST, BaseJsonAdapter, RawRecord

logger = logging.getLogger(__name__)

# Venues MEDIDOS: los tres devolvieron funding de BTC/USDT:USDT via CCXT. Se pueden cambiar
# por entorno sin tocar codigo, porque la lista correcta depende de que este operable.
VENUES: tuple[str, ...] = tuple(
    v.strip() for v in os.environ.get("FUNDING_VENUES", "binance,bybit,okx").split(",") if v.strip()
)

# Sufijo del perpetuo lineal en la nomenclatura unificada de CCXT.
PERPETUAL_SUFFIX = "/USDT:USDT"

BPS = 10_000.0

# Minimo de venues para que una dispersion signifique algo. Con uno no hay dispersion; con
# dos, la desviacion tipica es la mitad de la diferencia y cualquier caida de un venue la
# cambia de escala. Tres es el minimo honesto.
MIN_VENUES = 3


class FundingDispersion(BaseJsonAdapter):
    """Un registro crudo por (entidad, venue). La dispersion se deriva en la capa 2."""

    def __init__(self, source, *, venues: Sequence[str] = VENUES, exchanges=None, **kwargs) -> None:
        super().__init__(source, **kwargs)
        self.venues = tuple(venues)
        self._exchanges = dict(exchanges or {})

    def exchange(self, venue: str):
        """Cliente CCXT del venue, construido PEREZOSAMENTE (como en `CCXTCrypto`)."""
        if venue not in self._exchanges:
            import ccxt

            self._exchanges[venue] = getattr(ccxt, venue)(
                {"enableRateLimit": True, "options": {"defaultType": "swap"}}
            )
        return self._exchanges[venue]

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        out: list[RawRecord] = []
        for entity in entities:
            symbol = f"{entity}{PERPETUAL_SUFFIX}"
            for venue in self.venues:
                try:
                    rate = self.exchange(venue).fetch_funding_rate(symbol)
                except Exception as exc:  # noqa: BLE001 - un venue sin ese par no es un fallo
                    logger.info("· funding: %s no sirve %s (%s)", venue, symbol, exc)
                    continue
                out.append(
                    self.record(entity, rate, request={"venue": venue, "symbol": symbol})
                )
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        # (entidad, dia, ventana de funding) -> {venue: tipo en bps}. El venue es la clave y
        # no una lista porque el mismo venue archivado dos veces en la misma ventana es la
        # misma observacion, no dos, y contarla dos veces estrecharia la dispersion.
        snapshots: dict[tuple[str, str, str], dict[str, float]] = {}

        for record in records or ():
            payload = record.get("payload") or {}
            rate = numeric(payload.get("fundingRate"))
            entity = str(record.get("entity") or "")
            if rate is None or not entity:
                continue
            # OJO con el sello: `fundingTimestamp` de CCXT es el PROXIMO cobro, no el
            # momento de la observacion. Fechar el dia con el se descubrio midiendo —la
            # sonda devolvio una serie que empezaba MANANA— y habria metido en el archivo
            # observaciones fechadas en el futuro. El sello del cobro sigue sirviendo para
            # AGRUPAR (todos los venues comparten ventana), que es para lo que vale.
            window = payload.get("fundingTimestamp") or payload.get("fundingDatetime")
            day = unix_day(payload.get("timestamp")) or _day_of(record)
            if not day:
                continue
            venue = str((record.get(RAW_REQUEST) or {}).get("venue") or "")
            snapshots.setdefault((entity, day, str(window)), {})[venue] = rate * BPS

        rows = []
        for (entity, day, _stamp), by_venue in snapshots.items():
            rates = list(by_venue.values())
            if len(rates) < MIN_VENUES:
                # Se archivo igual (el crudo esta intacto), pero no se publica una
                # dispersion de dos puntos como si fuera comparable con una de cinco.
                continue
            rows.append(
                {
                    ENTITY: entity,
                    DAY: day,
                    "funding_dispersion": float(statistics.stdev(rates)),
                    "funding_median": float(statistics.median(rates)),
                    "funding_venues": float(len(rates)),
                }
            )
        return self.to_daily(rows)


def _day_of(record: Mapping) -> str | None:
    fetched = record.get(RAW_FETCHED_AT)
    if not fetched:
        return None
    try:
        return datetime.fromisoformat(str(fetched)).astimezone(UTC).date().isoformat()
    except ValueError:
        return None


def register() -> None:
    from ai_trader.signals.source import register_adapter

    register_adapter("funding_dispersion", FundingDispersion)


__all__ = ["BPS", "MIN_VENUES", "PERPETUAL_SUFFIX", "VENUES", "FundingDispersion", "register"]
