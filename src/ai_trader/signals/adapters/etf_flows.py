"""
Flujos diarios de los ETF spot, POR EMISOR (TFTC, CC BY 4.0).

POR QUE POR EMISOR Y NO EL AGREGADO
-----------------------------------
El neto del dia es el numero que publica todo el mundo y el que ya esta en el precio. Lo
que no se publica masticado es el DESGLOSE: un dia con neto cero puede ser un dia en el
que no paso nada, o uno en el que GBTC solto 300 millones y IBIT los absorbio. Son dos
mercados distintos y el agregado los cuenta igual.

De ahi `etf_issuer_dispersion`, que es bruto entre neto: la suma de los valores absolutos
de los emisores dividida por el valor absoluto de la suma. Vale 1 cuando todos empujan en
la misma direccion —flujo neto puro— y crece sin techo cuando unos entran y otros salen
—rotacion—. Es una razon, no una z: normalizarla contra su propia historia es cosa de
`signals/normalize.py`, que es donde se normaliza TODO.

LO QUE HAY Y LO QUE NO, MEDIDO
------------------------------
TFTC publica la serie completa de los ETF spot de BITCOIN en JSON abierto: 662 dias desde
2024-01-11, con desglose por emisor y licencia CC BY 4.0 (comprobado). De ETHEREUM no
publica dataset equivalente —las tres rutas evidentes devuelven 404, medido— asi que ETH
NO tiene cobertura por esta puerta. No se rellena con el agregado de otro sitio: un ETH
con flujo agregado y sin desglose no es la misma serie que el BTC de aqui, y mezclarlas
haria que `etf_issuer_dispersion` significase una cosa para un activo y otra para el otro.
Farside y SoSoValue quedan como contraste manual, no como fallback automatico.

REVISABLE, Y SE NOTA
--------------------
El flujo del dia se revisa al alza al dia siguiente segun reportan los emisores. Por eso
la fuente es `archive_revisable` y por eso el archivo es append-only: las dos lineas del
mismo dia, con su `fetched_at`, son la unica forma de medir cuanto revisa TFTC.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime

import pandas as pd

from ai_trader.shared.signals import DAY, ENTITY
from ai_trader.signals.adapters.common import (
    day_or_none,
    iso_day,
    numeric,
    rows_from_records,
    safe_call,
)
from ai_trader.signals.source import BaseJsonAdapter, RawRecord

logger = logging.getLogger(__name__)

TFTC_BASE = "https://www.tftc.io"

# entidad -> ruta del dataset abierto. Solo BTC: ver el docstring.
DATASETS: dict[str, str] = {"BTC": "/bitcoin-etf-flows/data.json"}

# Clave del array de dias dentro del JSON de TFTC.
DAYS_KEY = "days"


class TftcEtfFlows(BaseJsonAdapter):
    """Un registro crudo por dia, con el desglose por emisor intacto."""

    def __init__(self, source, **kwargs) -> None:
        super().__init__(source, base_url=TFTC_BASE, **kwargs)

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        since = day_or_none(start)
        out: list[RawRecord] = []
        for entity in entities:
            path = DATASETS.get(entity)
            if path is None:
                continue
            payload = safe_call(
                lambda path=path: self.client.get_json(path),
                what=f"tftc {entity}",
                logger=logger,
            )
            for row in (payload or {}).get(DAYS_KEY) or []:
                day = iso_day(row.get("date"))
                if day and (since is None or day >= since):
                    out.append(self.record(entity, row, day=day))
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        return self.to_daily(rows_from_records(records, row_of=_flow_row))


def _flow_row(record: Mapping) -> dict | None:
    payload = record.get("payload") or {}
    day = iso_day(payload.get("date"))
    if not day:
        return None

    per_issuer = {
        name: value
        for name, raw in (payload.get("perEtfUsd") or {}).items()
        if (value := numeric(raw)) is not None
    }
    # El neto declarado manda sobre la suma del desglose: son la misma cifra salvo cuando
    # TFTC ya tiene el agregado del dia y aun no el reparto, y en ese caso el agregado es
    # el dato real y la suma seria un cero inventado.
    net = numeric(payload.get("netFlowUsd"))
    if net is None:
        net = sum(per_issuer.values()) if per_issuer else None
    if net is None:
        return None

    return {
        ENTITY: record.get("entity") or "",
        DAY: day,
        "etf_netflow_usd": net,
        "etf_issuer_dispersion": _dispersion(per_issuer.values()),
        "etf_issuers_reporting": float(sum(1 for v in per_issuer.values() if v != 0.0)),
    }


def _dispersion(flows) -> float | None:
    """Bruto / |neto|: 1 = todos en la misma direccion, alto = rotacion entre emisores.

    None cuando el neto es cero exacto y el bruto no —el dia de rotacion perfecta— porque
    la razon se va a infinito. Publicar None y no un numero enorme deja que quien consuma
    decida; rellenarlo con un tope arbitrario meteria una constante inventada en la cola de
    la distribucion, que es justo donde esta la senal.
    """
    values = [v for v in flows if v is not None]
    if not values:
        return None
    gross = sum(abs(v) for v in values)
    net = abs(sum(values))
    if gross == 0.0:
        return 1.0  # dia sin flujo en ningun emisor: ni rotacion ni neto
    return gross / net if net > 0 else None


def register() -> None:
    from ai_trader.signals.source import register_adapter

    register_adapter("etf_flows", TftcEtfFlows)


__all__ = ["DATASETS", "TFTC_BASE", "TftcEtfFlows", "register"]
