"""
HYPERLIQUID: lo unico que ningun CEX permite calcular.

QUE HAY AQUI QUE NO HAYA EN BINANCE
-----------------------------------
En un CEX se publica el interes abierto AGREGADO y el funding. Eso responde "cuanto
apalancamiento hay" y no responde ninguna de las tres preguntas que de verdad deciden si un
movimiento se convierte en cascada:

    ¿como esta REPARTIDO ese apalancamiento?     -> la distribucion, no la media
    ¿a que precios revienta?                     -> el mapa de liquidacion
    ¿cuantas manos lo sostienen?                 -> la concentracion

Hyperliquid liquida on-chain y publica el estado de cada cuenta, asi que las tres se pueden
calcular. La API es gratuita, sin credencial y sin KYC. Lo que cuesta es el TRABAJO: hay que
enumerar cuentas, pedir el estado de cada una y reconstruir el agregado. Eso es todo lo que
mantiene esto sin arbitrar.

DE DONDE SALEN LAS CUENTAS, Y POR QUE LA MUESTRA ESTA SESGADA A PROPOSITO
------------------------------------------------------------------------
No existe un endpoint que devuelva "todas las posiciones". Existe `clearinghouseState`, que
devuelve las de UNA direccion, y existe el leaderboard, que devuelve direcciones. MEDIDO
2026-08-13: el leaderboard trae 41.714 cuentas y la mediana de valor de cuenta es de 763
dolares, asi que pedirlas todas seria pagar 41.714 peticiones para que la cola de abajo
aporte ruido.

Se piden las `SAMPLE_ACCOUNTS` mayores por valor de cuenta. La muestra cubrio el 21,7% del
OI del venue (1.578 M$ de 7.285 M$) y esta SESGADA hacia las cuentas grandes. El sesgo es
deliberado y hay que decirlo en los dos sentidos: es la mitad que decide un gap —una cuenta
de 763 dolares no mueve el precio al ser liquidada— y significa que "distribucion del
apalancamiento" aqui quiere decir "distribucion en la cola de arriba", no "del libro
entero". Una fuente que dijera lo segundo estaria mintiendo con un numero correcto.

UNA SOLA DESCARGA PARA DOS FUENTES
----------------------------------
El catalogo declara dos fuentes sobre estos mismos datos —`hyperliquid_leverage` (continua)
y `hyperliquid_liqmap` (mapa de precios)— porque se CODIFICAN distinto, no porque se
descarguen distinto. Sin cuidado, una captura pediria las 200 cuentas dos veces. `snapshot()`
memoriza la foto durante `SNAPSHOT_TTL_SECONDS` y las dos fuentes comparten la descarga.

El memo tiene caducidad y no es eterno a proposito: un proceso en vivo que corriera durante
dias con la misma foto publicaria un mapa de liquidacion viejo con pinta de fresco, que es
exactamente el fallo contra el que `events.PRICE_MAP_STALE_DAYS` protege un nivel mas
arriba. Aqui la caducidad es de minutos porque el coste de refrescar es de minutos.

LO QUE SE ARCHIVA NO ES LA RESPUESTA ENTERA, Y ESTA DECLARADO
-------------------------------------------------------------
Guardar las 200 respuestas de `clearinghouseState` intactas serian ~365 MB al ano de
resumenes de margen, ordenes y fondeo acumulado que ninguna feature mira, y ademas
archivados bajo la direccion como entidad, que no es el eje de esta fuente (el eje es el
ACTIVO). Se archiva una linea por activo y dia con las posiciones de ese activo, cada una
con lo que hace falta para re-derivar sus features. Es la misma excepcion razonada que en
`ofac_sdn`: el criterio de recorte esta escrito aqui y es re-aplicable, y lo que no se puede
es recuperar manana la respuesta de hoy.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ai_trader.data.providers.http import JsonHttpClient, JsonHttpConfig
from ai_trader.shared.signals import DAY, ENTITY
from ai_trader.signals.adapters.common import (
    certifi_bundle,
    iso_day,
    numeric,
    rows_from_records,
    safe_call,
)
from ai_trader.signals.source import BaseJsonAdapter, RawRecord

logger = logging.getLogger(__name__)

HYPERLIQUID_BASE = "https://api.hyperliquid.xyz"
INFO_PATH = "/info"
LEADERBOARD_URL = "https://stats-data.hyperliquid.xyz"
LEADERBOARD_PATH = "/Mainnet/leaderboard"

# Cuantas cuentas se piden, ordenadas por valor de cuenta. Doscientas es donde se cruzan las
# dos curvas: cubren el 21,7% del OI del venue (MEDIDO) y cuestan doscientas peticiones, que
# a la tarifa de Hyperliquid (1.200 de peso por minuto, peso 2 esta llamada) caben de sobra
# en una captura diaria. Subirlo a mil multiplicaria el coste por cinco para anadir cuentas
# cuyo valor mediano ya esta por debajo de los cien mil dolares.
SAMPLE_ACCOUNTS = 200

# Caducidad del memo compartido entre las dos fuentes. Ver el docstring.
SNAPSHOT_TTL_SECONDS = 900.0

# Apalancamiento a partir del cual una posicion cuenta como "apalancada". Diez veces no es
# un umbral estadistico: es donde una caida del 10% se lleva la posicion entera, que es lo
# que la convierte en combustible de cascada y no en una postura direccional.
HIGH_LEVERAGE = 10.0

# Cuantas cuentas definen la concentracion. Cinco, como dice la pregunta que responde: un
# perp cuyo OI esta en cinco manos es un perp con riesgo de gap, porque a esas cinco se las
# puede liquidar a la vez y no hay ley de los grandes numeros que lo suavice.
TOP_ACCOUNTS = 5

# Anchura del cubo con el que se buscan los clusters, en % de precio. Un punto porcentual:
# mas fino y cada posicion es su propio cubo (no habria clusters, solo posiciones); mas
# grueso y dos niveles que un operador distingue perfectamente se funden en uno.
CLUSTER_BUCKET_PCT = 1.0

# Que fraccion del notional mapeado tiene que concentrar un cubo para llamarse CLUSTER. Una
# decima parte en un solo punto porcentual es una acumulacion real y no la cola de una
# distribucion suave. Si ningun cubo llega, la respuesta honesta es que NO HAY CLUSTER —el
# mapa es liso— y las dos features salen NaN, que aguas arriba se lee como "hay mapa y no
# hay nada cerca" y no como "no se nada de este activo".
CLUSTER_MIN_SHARE = 0.10

# Hasta donde se busca, en % de precio. Se lee junto a `events.PRICE_DISTANCE_CAP_PCT`, que
# es el que decide despues cuanto pesa lo que se encuentre; este solo acota la busqueda.
CLUSTER_SEARCH_PCT = 25.0


# --- la foto compartida ---------------------------------------------------------------

_snapshot_cache: dict[int, tuple[float, dict]] = {}


# Los dos hosts de Hyperliquid fallan la validacion con el almacen del sistema en Windows
# (CERTIFICATE_VERIFY_FAILED, MEDIDO 2026-08-13), igual que el de OFAC. Ver `certifi_bundle`.
def _client() -> JsonHttpClient:
    return JsonHttpClient(
        HYPERLIQUID_BASE, JsonHttpConfig(timeout_seconds=30.0, ca_bundle=certifi_bundle())
    )


def snapshot(
    *,
    accounts: int = SAMPLE_ACCOUNTS,
    ttl: float = SNAPSHOT_TTL_SECONDS,
    client: JsonHttpClient | None = None,
    leaderboard_client: JsonHttpClient | None = None,
    now: float | None = None,
) -> dict:
    """
    Contextos de mercado + posiciones de las `accounts` cuentas mayores. Memorizado.

    Devuelve `{"day", "contexts": {coin: {...}}, "positions": {coin: [ ... ]}}`. Toca red;
    la capa pura no lo llama nunca.
    """
    stamp = time.monotonic() if now is None else now
    cached = _snapshot_cache.get(accounts)
    if cached is not None and (stamp - cached[0]) < ttl:
        return cached[1]

    api = client or _client()
    board = leaderboard_client or JsonHttpClient(
        LEADERBOARD_URL, JsonHttpConfig(timeout_seconds=120.0, ca_bundle=certifi_bundle())
    )

    payload = safe_call(
        lambda: api.post_json(INFO_PATH, body={"type": "metaAndAssetCtxs"}),
        what="hyperliquid metaAndAssetCtxs",
        logger=logger,
    )
    contexts = asset_contexts(payload)

    rows = safe_call(
        lambda: board.get_json(LEADERBOARD_PATH), what="hyperliquid leaderboard", logger=logger
    )
    addresses = top_accounts(rows, accounts)

    positions: dict[str, list[dict]] = {}
    for address in addresses:
        state = safe_call(
            lambda a=address: api.post_json(
                INFO_PATH, body={"type": "clearinghouseState", "user": a}
            ),
            what=f"hyperliquid clearinghouseState {address[:10]}",
            logger=logger,
        )
        for row in account_positions(state, address):
            positions.setdefault(row["coin"], []).append(row)

    out = {
        "day": datetime.now(timezone.utc).date().isoformat(),
        "accounts": len(addresses),
        "contexts": contexts,
        "positions": positions,
    }
    _snapshot_cache[accounts] = (stamp, out)
    logger.info(
        "· hyperliquid: %s cuentas, %s activos con posicion", len(addresses), len(positions)
    )
    return out


def asset_contexts(payload) -> dict[str, dict]:
    """`metaAndAssetCtxs` -> `{coin: {mark_px, open_interest_usd, day_ntl_vlm}}`.

    Es una tupla de dos listas alineadas por indice —el universo y sus contextos— y esa
    alineacion posicional es todo el contrato: si llegan con longitudes distintas, lo unico
    honesto es quedarse con el prefijo comun en vez de emparejar a ciegas.
    """
    if not isinstance(payload, (list, tuple)) or len(payload) < 2:
        return {}
    universe = (payload[0] or {}).get("universe") or []
    contexts = payload[1] or []
    out: dict[str, dict] = {}
    for meta, ctx in zip(universe, contexts):
        name = str((meta or {}).get("name") or "").upper()
        mark = numeric((ctx or {}).get("markPx"))
        if not name or not mark:
            continue
        open_interest = numeric(ctx.get("openInterest")) or 0.0
        out[name] = {
            "mark_px": mark,
            "open_interest_usd": open_interest * mark,
            "day_ntl_vlm": numeric(ctx.get("dayNtlVlm")),
            "max_leverage": numeric((meta or {}).get("maxLeverage")),
        }
    return out


def top_accounts(rows, limit: int) -> list[str]:
    """Las `limit` direcciones con mas valor de cuenta. El leaderboard viene sin ordenar."""
    if isinstance(rows, Mapping):
        rows = rows.get("leaderboardRows") or []
    ranked = sorted(
        (r for r in rows or [] if (r or {}).get("ethAddress")),
        key=lambda r: numeric(r.get("accountValue")) or 0.0,
        reverse=True,
    )
    return [str(r["ethAddress"]) for r in ranked[:limit]]


def account_positions(state, address: str) -> list[dict]:
    """`clearinghouseState` -> una fila por posicion, con lo que hace falta re-derivar."""
    out: list[dict] = []
    for entry in (state or {}).get("assetPositions") or []:
        position = (entry or {}).get("position") or {}
        coin = str(position.get("coin") or "").upper()
        size = numeric(position.get("szi"))
        value = numeric(position.get("positionValue"))
        if not coin or size is None or value is None:
            continue
        out.append(
            {
                "coin": coin,
                "account": address,
                "szi": size,
                "notional_usd": abs(value),
                "leverage": numeric((position.get("leverage") or {}).get("value")),
                # None es un valor con significado: el venue no publica precio de
                # liquidacion para una posicion cruzada con holgura. Ver el catalogo.
                "liquidation_px": numeric(position.get("liquidationPx")),
                "entry_px": numeric(position.get("entryPx")),
            }
        )
    return out


# =====================================================================================
# 1. Distribucion del apalancamiento y concentracion (serie continua)
# =====================================================================================


class HyperliquidLeverage(BaseJsonAdapter):
    """Una linea por activo y dia con las posiciones muestreadas de ese activo."""

    def __init__(self, source, *, accounts: int = SAMPLE_ACCOUNTS, **kwargs) -> None:
        super().__init__(source, base_url=HYPERLIQUID_BASE, **kwargs)
        self._accounts = accounts

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        foto = snapshot(accounts=self._accounts)
        wanted = {str(e).upper() for e in entities}
        out: list[RawRecord] = []
        for coin, rows in sorted(foto["positions"].items()):
            if wanted and coin not in wanted:
                continue
            context = foto["contexts"].get(coin) or {}
            payload = {
                "coin": coin,
                "open_interest_usd": context.get("open_interest_usd"),
                "sampled_accounts": foto["accounts"],
                "positions": [
                    {k: row[k] for k in ("account", "notional_usd", "leverage")} for row in rows
                ],
            }
            out.append(
                self.record(coin, payload, day=foto["day"], request={"series": "leverage"})
            )
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        return self.to_daily(rows_from_records(records, row_of=_leverage_row))


def _leverage_row(record: Mapping) -> dict | None:
    payload = record.get("payload") or {}
    day = iso_day(record.get("day")) or iso_day(record.get("fetched_at"))
    positions = payload.get("positions") or []
    if not day or not positions:
        return None

    notionals = np.array([numeric(p.get("notional_usd")) or 0.0 for p in positions], dtype=float)
    total = float(notionals.sum())
    if total <= 0:
        return None

    levered = [
        (numeric(p.get("leverage")), numeric(p.get("notional_usd")) or 0.0)
        for p in positions
        if numeric(p.get("leverage"))
    ]

    return {
        ENTITY: record.get("entity") or payload.get("coin") or "",
        DAY: day,
        # PONDERADAS POR NOTIONAL, no por numero de posiciones. Cien posiciones de mil
        # dolares a 3x y una de cincuenta millones a 25x tienen mediana simple 3 y no
        # describen el riesgo del activo. La pregunta es que apalancamiento sostiene el
        # DINERO, no cuantas cuentas hay.
        "hl_leverage_median": _weighted_quantile(levered, 0.5),
        "hl_leverage_p90": _weighted_quantile(levered, 0.9),
        "hl_high_leverage_share": (
            sum(n for lev, n in levered if lev >= HIGH_LEVERAGE) / total if levered else None
        ),
        "hl_oi_top5_share": _top_accounts_share(positions, total),
        "hl_sampled_oi_usd": total,
    }


def _weighted_quantile(pairs: Sequence[tuple[float, float]], q: float) -> float | None:
    """Cuantil de los apalancamientos ponderado por notional. None si no hay muestra."""
    rows = sorted((lev, weight) for lev, weight in pairs if lev and weight > 0)
    if not rows:
        return None
    weights = np.array([w for _, w in rows], dtype=float)
    cumulative = np.cumsum(weights) / weights.sum()
    index = int(np.searchsorted(cumulative, q, side="left"))
    return float(rows[min(index, len(rows) - 1)][0])


def _top_accounts_share(positions: Sequence[Mapping], total: float) -> float | None:
    """Fraccion del notional muestreado que esta en `TOP_ACCOUNTS` cuentas.

    Se agrupa POR CUENTA antes de ordenar: una cuenta con tres posiciones en el mismo activo
    es una sola mano, y contarlas por separado repartiria su concentracion en tres."""
    if total <= 0:
        return None
    by_account: dict[str, float] = {}
    for row in positions:
        account = str(row.get("account") or "")
        if not account:
            continue
        by_account[account] = by_account.get(account, 0.0) + (
            numeric(row.get("notional_usd")) or 0.0
        )
    if not by_account:
        return None
    largest = sorted(by_account.values(), reverse=True)[:TOP_ACCOUNTS]
    return float(sum(largest) / total)


# =====================================================================================
# 2. Mapa de precios de liquidacion (codificacion `price_map`)
# =====================================================================================


class HyperliquidLiquidationMap(BaseJsonAdapter):
    """Una linea por activo y dia con los niveles de liquidacion y su notional."""

    def __init__(self, source, *, accounts: int = SAMPLE_ACCOUNTS, **kwargs) -> None:
        super().__init__(source, base_url=HYPERLIQUID_BASE, **kwargs)
        self._accounts = accounts

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        foto = snapshot(accounts=self._accounts)
        wanted = {str(e).upper() for e in entities}
        out: list[RawRecord] = []
        for coin, rows in sorted(foto["positions"].items()):
            if wanted and coin not in wanted:
                continue
            context = foto["contexts"].get(coin) or {}
            mark = context.get("mark_px")
            levels = [
                [row["liquidation_px"], row["notional_usd"]]
                for row in rows
                if row.get("liquidation_px")
            ]
            if not mark or not levels:
                continue
            payload = {
                "coin": coin,
                "mark_px": mark,
                # Las dos cifras que hacen legible el hueco: cuantas posiciones tenian
                # precio de liquidacion y cuantas habia en total. Sin ellas, un mapa de
                # tres niveles y uno de trescientos se leen igual.
                "with_liquidation_px": len(levels),
                "positions": len(rows),
                "levels": levels,
            }
            out.append(self.record(coin, payload, day=foto["day"], request={"series": "liqmap"}))
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        return self.to_daily(rows_from_records(records, row_of=_liqmap_row))


def _liqmap_row(record: Mapping) -> dict | None:
    payload = record.get("payload") or {}
    day = iso_day(record.get("day")) or iso_day(record.get("fetched_at"))
    mark = numeric(payload.get("mark_px"))
    levels = payload.get("levels") or []
    if not day or not mark or mark <= 0 or not levels:
        return None

    distance, notional = nearest_cluster(
        [(numeric(price), numeric(size) or 0.0) for price, size in levels], mark
    )
    return {
        ENTITY: record.get("entity") or payload.get("coin") or "",
        DAY: day,
        "liq_cluster_distance_pct": distance,
        "liq_cluster_notional_usd": notional,
    }


def nearest_cluster(
    levels: Sequence[tuple[float | None, float]],
    mark_px: float,
    *,
    bucket_pct: float = CLUSTER_BUCKET_PCT,
    min_share: float = CLUSTER_MIN_SHARE,
    search_pct: float = CLUSTER_SEARCH_PCT,
) -> tuple[float | None, float | None]:
    """
    El cluster mas cercano y el notional acumulado HASTA el. `(distancia %, notional USD)`.

    El algoritmo, entero:

    1. cada nivel se pasa a distancia en % con SIGNO (negativo = por debajo del precio);
    2. se descarta lo que este mas lejos que `search_pct`, que no es un cluster sino paisaje;
    3. se agrupa en cubos de `bucket_pct`;
    4. se recorre hacia AFUERA desde el precio, y el primer cubo que concentre al menos
       `min_share` del notional mapeado es el cluster;
    5. el notional que se devuelve es el ACUMULADO desde el precio hasta ese cubo, por su
       lado. Es la cifra que responde a la pregunta util —cuanto se liquida si el precio
       llega hasta ahi— y no cuanto hay exactamente en ese punto.

    `(None, None)` cuando el mapa es LISO: ningun cubo concentra lo suficiente. No es un
    fallo ni un hueco de datos, y por eso se distingue de "no hay mapa" (que aguas arriba
    es `_seen = 0`): es un mapa sin acumulaciones, que es informacion.
    """
    if mark_px <= 0 or bucket_pct <= 0:
        return None, None

    buckets: dict[int, float] = {}
    total = 0.0
    for price, size in levels:
        if not price or price <= 0 or not size or size <= 0:
            continue
        distance = (price / mark_px - 1.0) * 100.0
        if abs(distance) > search_pct:
            continue
        # El redondeo antes del suelo NO es cosmetico: (95/100 - 1) * 100 da
        # -5.000000000000004 en coma flotante, y `floor` de eso cae en el cubo [-6, -5) en
        # vez de en [-5, -4). Un nivel puesto justo en el borde saltaria un cubo entero
        # segun como se hubiera calculado el precio, que es la clase de inestabilidad que
        # hace que dos capturas identicas den respuestas distintas.
        index = int(np.floor(round(distance, 9) / bucket_pct))
        buckets[index] = buckets.get(index, 0.0) + float(size)
        total += float(size)

    if total <= 0:
        return None, None

    threshold = total * min_share
    # Hacia afuera desde el precio: por |distancia| y, a igualdad, primero el de abajo, que
    # es el lado que fuerza ventas. Que el desempate sea explicito y no el orden del dict es
    # lo que hace que dos capturas con las mismas posiciones den la misma respuesta.
    for index in sorted(buckets, key=lambda i: (min(abs(i), abs(i + 1)), i)):
        if buckets[index] < threshold:
            continue
        # El acumulado es el de SU LADO: lo que hay entre el precio y el cluster. Sumar los
        # dos lados diria "cuanto se liquida si el precio va arriba Y abajo", que no es una
        # situacion.
        if index < 0:
            covered = [size for other, size in buckets.items() if index <= other < 0]
        else:
            covered = [size for other, size in buckets.items() if 0 <= other <= index]
        center = (index + 0.5) * bucket_pct
        return float(center), float(sum(covered))

    return None, None


# =====================================================================================

ADAPTERS = {
    "hyperliquid_leverage": HyperliquidLeverage,
    "hyperliquid_liqmap": HyperliquidLiquidationMap,
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
    "CLUSTER_BUCKET_PCT",
    "CLUSTER_MIN_SHARE",
    "CLUSTER_SEARCH_PCT",
    "HIGH_LEVERAGE",
    "SAMPLE_ACCOUNTS",
    "SNAPSHOT_TTL_SECONDS",
    "TOP_ACCOUNTS",
    "HyperliquidLeverage",
    "HyperliquidLiquidationMap",
    "account_positions",
    "asset_contexts",
    "nearest_cluster",
    "register",
    "snapshot",
    "top_accounts",
]
