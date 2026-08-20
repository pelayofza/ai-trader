"""
FRED: las series macro que el generador sintetico ya modela como FACTORES.

EL MAPEO EXPLICITO, QUE ES EL MOTIVO DE QUE ESTA FUENTE MEREZCA LA PENA
-----------------------------------------------------------------------
`research/synthetic/universe.py` genera precios con cinco factores: EQUITY, RATES, USD, COMMODITY y
CRYPTO. Los cuatro primeros son observables y tienen serie publica y gratuita; el quinto no
es macro. `FACTOR_OF` mapea cada serie de FRED sobre su factor, uno a uno y por escrito:

    DTWEXBGS    dolar amplio                 -> USD
    DFII10      tipo real a 10 anos (TIPS)   -> RATES
    DGS2, DGS10 nominales 2a y 10a           -> RATES
    T10Y2Y      pendiente 10a-2a             -> RATES
    SP500       S&P 500                      -> EQUITY
    NASDAQ100   Nasdaq 100                   -> EQUITY
    VIXCLS      volatilidad implicita        -> EQUITY
    DCOILWTICO  petroleo WTI                 -> COMMODITY

Sin ese mapeo, esta fuente son ocho columnas mas. Con el, es la unica pieza que permite
preguntar si el escenario sintetico se parecio al mundo que hubo ese dia.

EL ORO NO ESTA, Y ES UNA MEDICION
---------------------------------
FRED discontinuo las series de la LBMA (`GOLDAMGBD228NLBM`, `GOLDPMGBD228NLBM`): hoy
devuelven 404, comprobado. El factor COMMODITY se cubre con WTI, que sigue viva y es
diaria. Escribir el id del oro "porque siempre estuvo" habria dejado una serie que falla en
silencio en cada captura.

REVISABLE HACIA ATRAS, Y CON CALENDARIO DE DIA HABIL
----------------------------------------------------
FRED reescribe observaciones pasadas cuando la fuente original las revisa. El campo `pit`
es `archive_revisable` por eso, y el archivo crudo con su `fetched_at` es lo unico que
permite saber, dentro de un ano, que valor se veia el dia de la decision.

`macro_change_1d` es la variacion contra la ULTIMA OBSERVACION, no contra el dia natural
anterior: estas series no publican fines de semana ni festivos, asi que un lunes se compara
con el viernes. Es la lectura correcta y esta declarada porque la alternativa —reindexar a
dias naturales y rellenar— inventaria variaciones de cero que nadie observo.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime

import pandas as pd

from ai_trader.shared.signals import DAY, ENTITY
from ai_trader.signals.adapters.common import (
    env_secret,
    iso_day,
    numeric,
    rows_from_records,
    safe_call,
)
from ai_trader.signals.source import BaseJsonAdapter, RawRecord

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org"
OBSERVATIONS_PATH = "/fred/series/observations"

# Serie -> factor del generador sintetico (`research/synthetic/universe.py`).
FACTOR_OF: dict[str, str] = {
    "DTWEXBGS": "USD",
    "DFII10": "RATES",
    "DGS2": "RATES",
    "DGS10": "RATES",
    "T10Y2Y": "RATES",
    "SP500": "EQUITY",
    "NASDAQ100": "EQUITY",
    "VIXCLS": "EQUITY",
    "DCOILWTICO": "COMMODITY",
}

# El valor que usa FRED para "ese dia no hubo observacion".
MISSING = "."


class FredMacro(BaseJsonAdapter):
    """Una entidad por SERIE. Necesita `FRED_API_KEY`: sin ella no captura y lo dice."""

    def __init__(self, source, **kwargs) -> None:
        super().__init__(source, base_url=FRED_BASE, **kwargs)

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        key = env_secret(self.source.auth_env)
        if not key:
            raise RuntimeError(
                f"FRED necesita la variable de entorno {self.source.auth_env} (key gratuita). "
                "Sin ella la fuente no se captura; el resto del catalogo sigue su curso."
            )

        out: list[RawRecord] = []
        for series_id in entities:
            params = {
                "series_id": series_id,
                "api_key": key,
                "file_type": "json",
            }
            if start is not None:
                params["observation_start"] = start.date().isoformat()
            if end is not None:
                params["observation_end"] = end.date().isoformat()

            payload = safe_call(
                lambda params=params: self.client.get_json(OBSERVATIONS_PATH, params=params),
                what=f"fred {series_id}",
                logger=logger,
            )
            for observation in (payload or {}).get("observations") or []:
                day = iso_day(observation.get("date"))
                if not day or observation.get("value") == MISSING:
                    continue
                # La key va en `params` pero NO en el `request` que se archiva: el crudo se
                # comparte, se copia y se sube a un backup, y una credencial dentro de una
                # linea de archivo es una credencial filtrada para siempre.
                out.append(
                    self.record(
                        series_id,
                        observation,
                        day=day,
                        request={"series_id": series_id, "factor": FACTOR_OF.get(series_id, "")},
                    )
                )
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        rows = rows_from_records(records, row_of=_macro_row)
        frame = self.to_daily(rows)
        if frame.empty:
            return frame
        frame["macro_change_1d"] = frame["macro_value"].groupby(level=ENTITY).diff()
        return frame


def _macro_row(record: Mapping) -> dict | None:
    payload = record.get("payload") or {}
    day = iso_day(payload.get("date"))
    value = numeric(payload.get("value"))
    if not day or value is None:
        return None
    return {
        ENTITY: record.get("entity") or "",
        DAY: day,
        "macro_value": value,
        "macro_change_1d": None,
    }


def register() -> None:
    from ai_trader.signals.source import register_adapter

    register_adapter("fred_macro", FredMacro)


__all__ = ["FACTOR_OF", "FRED_BASE", "FredMacro", "register"]
