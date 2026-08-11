"""
CFTC COT (informe TFF): posicionamiento en los futuros de CME, por categoria.

QUE APORTA
----------
El TFF ("Traders in Financial Futures") reparte el interes abierto entre categorias con
funcion economica distinta: `dealer` es el creador de mercado, que suele estar al otro lado
por oficio, y `leveraged funds` son los fondos apalancados, que es el dinero direccional.
Los dos netos juntos dicen quien lleva el riesgo, algo que ni el precio ni el volumen dicen.

EL DESFASE DE PUBLICACION ES LA DECISION DE DISENO DE ESTE ADAPTADOR
--------------------------------------------------------------------
El informe se toma el MARTES y se publica el VIERNES a las 15:30 ET. Un dato fechado el
martes que en realidad no se conoce hasta el viernes es tres dias de futuro metidos en la
serie, y el recorte por dia de observacion (`shared/signals.py::visible_signals`) no puede
verlo: para el, el martes es pasado.

Asi que `day` en esta fuente es **el dia de PUBLICACION, no el de referencia**. El martes
original no se pierde —queda en el payload crudo y en el propio registro— pero la fecha con
la que la serie se cruza con cualquier otra cosa es aquella en la que el dato ERA CONOCIBLE.
Es la unica eleccion que no exige que cada consumidor futuro se acuerde de decalar a mano;
la contraria pone una trampa que solo se descubre cuando un backtest sale demasiado bien.

Cuando el viernes es festivo, la CFTC publica el lunes. Aqui se aplican tres dias naturales
siempre: la aproximacion es conservadora en el sentido correcto (nunca adelanta el dato) y
la alternativa seria mantener un calendario de festivos federales, que es mas superficie de
error que la que ahorra.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

import pandas as pd

from ai_trader.shared.signals import DAY, ENTITY
from ai_trader.signals.adapters.common import iso_day, numeric, rows_from_records, safe_call
from ai_trader.signals.source import BaseJsonAdapter, RawRecord

logger = logging.getLogger(__name__)

SOCRATA_BASE = "https://publicreporting.cftc.gov"
TFF_PATH = "/resource/gpe5-46if.json"

# entidad -> nombre EXACTO del contrato en el fichero de la CFTC. Medido: son los dos con
# serie larga (435 y 279 semanas). Los micro y los de otros exchanges se dejan fuera a
# proposito —mezclar CME con CBOE en una misma serie sumaria contratos de tamano distinto—.
CONTRACTS: dict[str, str] = {
    "BTC": "BITCOIN - CHICAGO MERCANTILE EXCHANGE",
    "ETH": "ETHER CASH SETTLED - CHICAGO MERCANTILE EXCHANGE",
}

# Dias naturales entre la foto (martes) y su publicacion (viernes). Ver el docstring.
PUBLICATION_LAG_DAYS = 3

PAGE_SIZE = 5000


class CftcCot(BaseJsonAdapter):
    """Un registro crudo por semana e instrumento, fechado el dia en que se publico."""

    def __init__(self, source, **kwargs) -> None:
        super().__init__(source, base_url=SOCRATA_BASE, **kwargs)

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        out: list[RawRecord] = []
        for entity in entities:
            contract = CONTRACTS.get(entity)
            if contract is None:
                continue
            where = f"market_and_exchange_names='{contract}'"
            if start is not None:
                where += f" AND report_date_as_yyyy_mm_dd >= '{start.date().isoformat()}'"
            rows = safe_call(
                lambda: self.client.get_json(
                    TFF_PATH,
                    params={
                        "$where": where,
                        "$order": "report_date_as_yyyy_mm_dd ASC",
                        "$limit": PAGE_SIZE,
                    },
                ),
                what=f"cftc {contract}",
                logger=logger,
            )
            for row in rows or []:
                day = published_day(row.get("report_date_as_yyyy_mm_dd"))
                if day:
                    out.append(
                        self.record(entity, row, day=day, request={"contract": contract})
                    )
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        return self.to_daily(rows_from_records(records, row_of=_cot_row))


def published_day(report_date) -> str | None:
    """Dia de PUBLICACION a partir del dia de referencia. Ver el docstring del modulo."""
    day = iso_day(report_date)
    if not day:
        return None
    return (datetime.fromisoformat(day) + timedelta(days=PUBLICATION_LAG_DAYS)).date().isoformat()


def _cot_row(record: Mapping) -> dict | None:
    payload = record.get("payload") or {}
    day = published_day(payload.get("report_date_as_yyyy_mm_dd"))
    if not day:
        return None

    leveraged = _net(payload, "lev_money_positions_long", "lev_money_positions_short")
    dealer = _net(payload, "dealer_positions_long_all", "dealer_positions_short_all")
    open_interest = numeric(payload.get("open_interest_all"))
    if leveraged is None and dealer is None and open_interest is None:
        return None

    return {
        ENTITY: record.get("entity") or "",
        DAY: day,
        "cot_leveraged_net": leveraged,
        "cot_dealer_net": dealer,
        "cot_open_interest": open_interest,
    }


def _net(payload: Mapping, long_field: str, short_field: str) -> float | None:
    """Neto = largo - corto. None si falta cualquiera de las dos patas: un neto calculado
    con una sola pata no es un neto pequeno, es un numero inventado."""
    long_side = numeric(payload.get(long_field))
    short_side = numeric(payload.get(short_field))
    if long_side is None or short_side is None:
        return None
    return long_side - short_side


def register() -> None:
    from ai_trader.signals.source import register_adapter

    register_adapter("cftc_cot", CftcCot)


__all__ = [
    "CONTRACTS",
    "PUBLICATION_LAG_DAYS",
    "SOCRATA_BASE",
    "TFF_PATH",
    "CftcCot",
    "published_day",
    "register",
]
