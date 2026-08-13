"""
PRESTAMOS ON-CHAIN: el otro lado del apalancamiento, y una medicion incomoda.

QUE APORTA QUE NO APORTE HYPERLIQUID
------------------------------------
Hyperliquid dice donde revienta el apalancamiento del PERPETUO. Esto dice donde revienta el
del COLATERAL SPOT, y no son el mismo riesgo: liquidar un perp cierra un contrato y liquidar
una posicion de Aave VENDE el colateral en el mercado al contado. La segunda mueve el precio
del subyacente de forma mecanica; la primera, solo a traves de la base.

Por eso las dos se codifican igual —mapa de precios, `events.py::PRICE_SPECS`— con la misma
forma: una distancia en porcentaje de precio y un notional acumulado hasta ella. Aqui la
distancia sale de la distribucion de HEALTH FACTORS: un prestamo con health factor 1,25
aguanta una caida del 20% del colateral, y agrupando por esa caida sale el mismo mapa.

LA MEDICION, QUE ES EL RESULTADO DE ESTE MODULO
-----------------------------------------------
MEDIDO 2026-08-13, cuatro vias y ninguna gratuita:

    api.thegraph.com (servicio alojado)     el DNS ya no resuelve: se retiro.
    gateway.thegraph.com (descentralizado)  200 con {"errors":[{"message":"auth error:
                                            missing authorization header"}]}
    api.compound.finance/api/v2/account     410 Gone.
    datasets de liquidaciones de DefiLlama  404.

Asi que hoy esta fuente no tiene dato, y el adaptador se escribe igualmente por lo mismo que
el de unlocks de DefiLlama: para que "no hay dato" sea una MEDICION fechada y con su error
en el registro, y no un hueco que dentro de seis meses se lea como "nadie lo intento".

EL 200 CON ERROR DENTRO ES LO QUE HAY QUE VIGILAR
-------------------------------------------------
El gateway responde 200 y mete el fallo en el cuerpo, que es la forma mas silenciosa de
romper una ingesta: `get_json` no levanta nada, el adaptador archivaria una respuesta sin
datos y la sonda declararia "cero registros" en vez de "sin credencial". Por eso
`graphql_error` mira dentro y `fetch_raw` LEVANTA. Es la misma decision que en unlocks
(402) y en la cola de staking (401): el error de credencial es el resultado y tiene que
llegar entero a la sonda.

LO QUE NO SE PUEDE AFIRMAR DE LA CAPA 2
---------------------------------------
El mapeo esta escrito contra el ESQUEMA documentado de los subgrafos de Aave v3, no contra
una respuesta observada. Los adaptadores que se escribieron con el payload real delante lo
dicen; este no puede, y esa diferencia es real. El dia que haya clave, lo primero es
comprobar el mapeo contra una respuesta de verdad.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

import pandas as pd

from ai_trader.data.providers.http import JsonHttpConfig
from ai_trader.shared.signals import DAY, ENTITY
from ai_trader.signals.adapters.common import env_secret, iso_day, numeric, rows_from_records
from ai_trader.signals.source import BaseJsonAdapter, RawRecord

logger = logging.getLogger(__name__)

GATEWAY_BASE = "https://gateway.thegraph.com"

# Subgrafo -> id en la red descentralizada. Los ids son publicos; lo que falta es la clave.
SUBGRAPHS: dict[str, str] = {
    "aave-v3-ethereum": "Cd2gEDVeqnjBn1hSeqFMitw8Q1iiEV9klFqGWTNZfvY",
}

# Colateral cuyo mapa interesa: el que tiene mercado spot profundo y esta en el universo.
COLLATERAL: tuple[str, ...] = ("WETH", "WBTC")

# Entidad con la que se archiva cada colateral. WETH es ETH y WBTC es BTC en todo lo que
# importa aqui —el precio que dispara la liquidacion es el del subyacente—, asi que se
# archivan con la clave del subyacente para que crucen con el resto del catalogo.
COLLATERAL_ENTITY: dict[str, str] = {"WETH": "ETH", "WBTC": "BTC"}

# Cuantos prestamos se piden por pagina. Mil es el tope del propio The Graph.
PAGE_SIZE = 1000

# Health factor por encima del cual un prestamo no entra en el mapa. 3,0 es una caida del
# 67% del colateral: mas alla, el prestamo no describe el riesgo de nadie.
MAX_HEALTH_FACTOR = 3.0

# La consulta. Pide posiciones con deuda viva y su colateral, ordenadas por health factor.
POSITIONS_QUERY = """
query Positions($first: Int!, $skip: Int!) {
  accounts(first: $first, skip: $skip) {
    id
    positions(where: {balance_gt: 0}) {
      side
      balance
      asset { symbol decimals }
    }
  }
}
""".strip()


class LendingHealth(BaseJsonAdapter):
    """Distribucion de health factors -> mapa de liquidacion del colateral spot."""

    def __init__(self, source, *, subgraphs: Mapping[str, str] | None = None, **kwargs) -> None:
        token = env_secret(source.auth_env)
        config = kwargs.pop("http_config", None) or JsonHttpConfig(
            timeout_seconds=60.0,
            headers={"Authorization": f"Bearer {token}"} if token else {},
        )
        super().__init__(source, base_url=GATEWAY_BASE, http_config=config, **kwargs)
        self._subgraphs = dict(subgraphs or SUBGRAPHS)
        self._token = token

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        if not self._token:
            # Sin credencial no se pide nada: el gateway contestaria 200 con el error
            # dentro y eso archivaria una respuesta vacia con pinta de exito.
            raise RuntimeError(
                f"La fuente '{self.source.key}' necesita {self.source.auth_env} en el entorno "
                "(el gateway de The Graph responde 'auth error' sin cabecera)"
            )

        day = datetime.now(timezone.utc).date().isoformat()
        wanted = {str(e).upper() for e in entities}
        out: list[RawRecord] = []
        for name, subgraph_id in sorted(self._subgraphs.items()):
            payload = self.client.post_json(
                f"/api/subgraphs/id/{subgraph_id}",
                body={"query": POSITIONS_QUERY, "variables": {"first": PAGE_SIZE, "skip": 0}},
            )
            error = graphql_error(payload)
            if error:
                # Sin safe_call, igual que unlocks: el error de credencial ES el resultado.
                raise RuntimeError(f"{name}: {error}")
            for symbol, block in collateral_blocks(payload, COLLATERAL).items():
                entity = COLLATERAL_ENTITY.get(symbol, symbol)
                if wanted and entity not in wanted:
                    continue
                out.append(
                    self.record(entity, {"subgraph": name, "symbol": symbol, **block}, day=day)
                )
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        return self.to_daily(rows_from_records(records, row_of=_lending_row))


def graphql_error(payload) -> str | None:
    """El mensaje de error de una respuesta GraphQL, o None. Ver el docstring del modulo."""
    if not isinstance(payload, Mapping):
        return "respuesta que no es un objeto JSON"
    errors = payload.get("errors")
    if not errors:
        return None
    first = errors[0] if isinstance(errors, list) and errors else errors
    if isinstance(first, Mapping):
        return str(first.get("message") or first)
    return str(first)


def collateral_blocks(payload, symbols: Sequence[str]) -> dict[str, dict]:
    """Respuesta del subgrafo -> `{simbolo: {levels: [[caida %, notional], ...]}}`.

    Cada prestamo aporta UN nivel: la caida del colateral que lo lleva a health factor 1, y
    el notional de ese colateral. Es la misma forma que produce el mapa de Hyperliquid
    —niveles con notional— para que el codificador sea literalmente el mismo.
    """
    accounts = ((payload or {}).get("data") or {}).get("accounts") or []
    wanted = {s.upper() for s in symbols}
    out: dict[str, dict] = {}
    for account in accounts:
        for symbol, drop, notional in _account_levels(account, wanted):
            block = out.setdefault(symbol, {"levels": [], "loans": 0})
            block["levels"].append([drop, notional])
            block["loans"] += 1
    return out


def _account_levels(account: Mapping, wanted: set[str]):
    """Los niveles que aporta UNA cuenta. Vacio si no tiene deuda o no tiene el colateral."""
    collateral: dict[str, float] = {}
    debt = 0.0
    for position in (account or {}).get("positions") or []:
        symbol = str(((position or {}).get("asset") or {}).get("symbol") or "").upper()
        balance = numeric(position.get("balance")) or 0.0
        side = str(position.get("side") or "").upper()
        if side.startswith("BORROW"):
            debt += balance
        elif symbol:
            collateral[symbol] = collateral.get(symbol, 0.0) + balance

    total = sum(collateral.values())
    if debt <= 0 or total <= 0:
        return []

    health = total / debt
    if health <= 1.0 or health > MAX_HEALTH_FACTOR:
        return []
    # Caida del colateral que lleva el prestamo a health factor 1, con signo negativo: el
    # colateral se liquida cuando el precio BAJA, y el mapa habla de distancias con signo.
    drop = -(1.0 - 1.0 / health) * 100.0
    return [(symbol, drop, amount) for symbol, amount in collateral.items() if symbol in wanted]


def _lending_row(record: Mapping) -> dict | None:
    from ai_trader.signals.adapters.hyperliquid import nearest_cluster

    payload = record.get("payload") or {}
    day = iso_day(record.get("day")) or iso_day(record.get("fetched_at"))
    levels = payload.get("levels") or []
    if not day or not levels:
        return None

    # EL MISMO ALGORITMO QUE EL MAPA DEL PERPETUO, importado y no copiado: son el mismo
    # objeto (niveles con notional) y dos implementaciones divergirian en cuanto alguien
    # tocase una. La diferencia esta en la ENTRADA —aqui los niveles ya vienen en % de
    # caida, alli en precio— y por eso se pasa un precio de referencia de 100.
    distance, notional = nearest_cluster(
        [(100.0 + numeric(drop), numeric(size) or 0.0) for drop, size in levels], 100.0
    )
    return {
        ENTITY: record.get("entity") or "",
        DAY: day,
        "lending_liq_distance_pct": distance,
        "lending_liq_notional_usd": notional,
    }


ADAPTERS = {"lending_health": LendingHealth}


def register() -> None:
    from ai_trader.signals.source import register_adapter

    for key, factory in ADAPTERS.items():
        try:
            register_adapter(key, factory)
        except ValueError:
            logger.debug("%s ya estaba registrado", key)


__all__ = [
    "ADAPTERS",
    "COLLATERAL",
    "COLLATERAL_ENTITY",
    "MAX_HEALTH_FACTOR",
    "POSITIONS_QUERY",
    "SUBGRAPHS",
    "LendingHealth",
    "collateral_blocks",
    "graphql_error",
    "register",
]
