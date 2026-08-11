"""
DefiLlama: tres fuentes distintas que comparten proveedor y nada mas.

    stablecoins  oferta POR CADENA        -> polvora seca y rotacion entre ecosistemas
    fees         comisiones/ingresos/TVL  -> el denominador de P/F y P/S
    volumes      cuota DEX vs CEX         -> donde se forma el precio

LOS TRES PARTEN LA SERIE EN UN REGISTRO POR DIA
-----------------------------------------------
Los tres endpoints devuelven la historia COMPLETA en cada llamada (la de Ethereum son
2.840 puntos, medidos). Archivar esa respuesta como una sola linea la colocaria entera en
el mes en que se descargo, y leer una ventana de backtest obligaria a descomprimir el
archivo entero. Partida por dia (`common.chart_records`), un backfill de siete anos cae en
sus 80 ficheros mensuales y cada uno se lee solo.

LOS MAPAS ENTIDAD -> SLUG ESTAN MEDIDOS, NO SUPUESTOS
-----------------------------------------------------
Cada entrada de `FEE_SLUGS` y `DEX_CHAINS` se comprobo contra el endpoint antes de
escribirla, y las que devolvian 400/500 NO estan: `polkadot`, `celestia` y `cosmos` no
tienen serie de volumen DEX (500 del proveedor), y Dogecoin no tiene cadena DEX —
`dogechain` es otra cosa y confundirlas seria cruzar el volumen de una cadena EVM con el
token equivocado—. Un hueco declarado es mejor que un cruce plausible y falso.

LA CUOTA DEX/CEX NECESITA LAS DOS PATAS
---------------------------------------
`dex_share` sin denominador no significa nada, y DefiLlama no publica volumen de CEX. La
pata de CEX sale de CCXT —el mismo proveedor que ya trae las barras— y se archiva como un
registro crudo mas, marcado en `request.series`. Es la unica fuente del lote con dos
proveedores detras, y por eso los dos quedan escritos en el crudo: si manana la referencia
de CEX cambia de venue, la serie vieja sigue siendo interpretable.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

import pandas as pd

from ai_trader.shared.signals import DAY, ENTITY
from ai_trader.signals.adapters.common import (
    UTC,
    chart_records,
    day_or_none,
    numeric,
    rows_from_records,
    safe_call,
    unix_day,
)
from ai_trader.signals.source import RAW_DAY, RAW_REQUEST, BaseJsonAdapter, RawRecord

logger = logging.getLogger(__name__)

STABLECOINS_BASE = "https://stablecoins.llama.fi"
LLAMA_BASE = "https://api.llama.fi"

# Marca de que serie es cada registro crudo, dentro de `request`. Un solo directorio de
# archivo por fuente y varias series dentro: la alternativa —una fuente de catalogo por
# serie— multiplicaria por tres las entradas sin anadir una sola decision.
SERIES = "series"


class DefiLlamaStablecoins(BaseJsonAdapter):
    """Oferta de stablecoins por cadena. La entidad es la CADENA, no un token."""

    def __init__(self, source, **kwargs) -> None:
        super().__init__(source, base_url=STABLECOINS_BASE, **kwargs)

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        since = day_or_none(start)
        out: list[RawRecord] = []
        for chain in entities:
            payload = safe_call(
                lambda chain=chain: self.client.get_json(f"/stablecoincharts/{chain}"),
                what=f"defillama stablecoins {chain}",
                logger=logger,
            )
            points = chart_records(
                payload or (), day_of=lambda p: unix_day(p.get("date")), since=since
            )
            for day, point in points:
                out.append(self.record(chain, point, day=day))
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        rows = rows_from_records(records, row_of=_stablecoin_row)
        frame = self.to_daily(rows)
        if frame.empty:
            return frame
        # La emision neta es la VARIACION de la oferta, y solo se puede calcular con la
        # serie ya ordenada por dia: por eso se anade despues de agregar y no fila a fila.
        # Con un hueco en medio, el primer dia tras el hueco acumula lo emitido durante
        # todo el hueco. Es la lectura correcta —"emitido desde la ultima observacion"— y
        # la cobertura del dia esta al lado, en `observed`, para saber cuando fiarse.
        supply = frame["stablecoin_supply_usd"]
        frame["stablecoin_issuance_usd"] = supply.groupby(level=ENTITY).diff()
        return frame


def _stablecoin_row(record: Mapping) -> dict | None:
    payload = record.get("payload") or {}
    day = unix_day(payload.get("date"))
    circulating = payload.get("totalCirculatingUSD") or {}
    supply = numeric(circulating.get("peggedUSD"))
    if not day or supply is None:
        return None
    return {
        ENTITY: record.get("entity") or "",
        DAY: day,
        "stablecoin_supply_usd": supply,
        "stablecoin_issuance_usd": None,
    }


class DefiLlamaFees(BaseJsonAdapter):
    """Comisiones, ingresos y TVL por protocolo (o por cadena, cuando el token ES la cadena)."""

    # entidad -> slug de DefiLlama. MEDIDO contra el endpoint: cada uno devolvia serie el
    # dia que se escribio esta tabla. XRP y ATOM entran por su cadena ('xrpl', 'cosmos-hub')
    # porque el slug obvio ('ripple', 'cosmos') devuelve 400.
    FEE_SLUGS: dict[str, str] = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "BNB": "bsc",
        "XRP": "xrpl",
        "ADA": "cardano",
        "DOGE": "doge",
        "AVAX": "avalanche",
        "DOT": "polkadot",
        "LINK": "chainlink",
        "LTC": "litecoin",
        "UNI": "uniswap",
        "ATOM": "cosmos-hub",
        "NEAR": "near",
        "APT": "aptos",
        "ARB": "arbitrum",
        "OP": "op-mainnet",
        "INJ": "injective",
        "FIL": "filecoin",
        "ETC": "ethereum-classic",
        "AAVE": "aave",
        "SUI": "sui",
        "SEI": "sei",
        "TIA": "celestia",
    }

    # Los slugs que son una CADENA y no un protocolo. La diferencia no es cosmetica: el TVL
    # de una cadena vive en `/v2/historicalChainTvl/<slug>` y `/protocol/<slug>` devuelve
    # 400 para casi todas ellas (medido: bsc, xrpl, doge, polkadot, cosmos-hub, arbitrum,
    # op-mainnet, sui, ethereum-classic). Las que responden algo devuelven una ficha sin
    # serie de TVL, que es peor: un 200 vacio no se nota.
    CHAIN_SLUGS: frozenset[str] = frozenset(
        {
            "bitcoin", "ethereum", "solana", "bsc", "xrpl", "cardano", "doge", "avalanche",
            "polkadot", "litecoin", "cosmos-hub", "near", "aptos", "arbitrum", "op-mainnet",
            "injective", "filecoin", "ethereum-classic", "sui", "sei", "celestia",
        }
    )

    def __init__(self, source, **kwargs) -> None:
        super().__init__(source, base_url=LLAMA_BASE, **kwargs)

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        since = day_or_none(start)
        out: list[RawRecord] = []
        for entity in entities:
            slug = self.FEE_SLUGS.get(entity)
            if slug is None:
                continue
            # Cada llamada por separado: `litecoin` tiene comisiones pero no ingresos (400
            # medido), y perder su serie de fees por eso seria absurdo.
            out.extend(self._series(entity, slug, "fees", "dailyFees", since))
            out.extend(self._series(entity, slug, "revenue", "dailyRevenue", since))
            out.extend(self._tvl(entity, slug, since))
        return out

    def _series(
        self, entity: str, slug: str, name: str, data_type: str, since: str | None = None
    ) -> list[RawRecord]:
        payload = safe_call(
            lambda: self.client.get_json(f"/summary/fees/{slug}", params={"dataType": data_type}),
            what=f"defillama {name} {slug}",
            logger=logger,
        )
        points = (payload or {}).get("totalDataChart") or []
        return [
            self.record(entity, point, day=day, request={SERIES: name, "slug": slug})
            for day, point in chart_records(points, day_of=lambda p: unix_day(p[0]), since=since)
        ]

    def _tvl(self, entity: str, slug: str, since: str | None = None) -> list[RawRecord]:
        is_chain = slug in self.CHAIN_SLUGS
        path = f"/v2/historicalChainTvl/{slug}" if is_chain else f"/protocol/{slug}"
        body = safe_call(
            lambda: self.client.get_json(path), what=f"defillama tvl {slug}", logger=logger
        ) or ({} if not is_chain else [])
        points = body if is_chain else (body.get("tvl") or [])
        return [
            self.record(entity, point, day=day, request={SERIES: "tvl", "slug": slug})
            for day, point in chart_records(
                points if isinstance(points, list) else [],
                day_of=lambda p: unix_day(p.get("date")),
                since=since,
            )
        ]

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        rows = rows_from_records(records, row_of=_fees_row)
        return self.to_daily(rows)


def _fees_row(record: Mapping) -> dict | None:
    series = (record.get(RAW_REQUEST) or {}).get(SERIES)
    payload = record.get("payload")
    entity = record.get("entity") or ""

    if series == "tvl":
        point = payload or {}
        day = unix_day(point.get("date"))
        # Una cadena lo llama `tvl` y un protocolo `totalLiquidityUSD`. Las dos formas
        # conviven en el archivo desde el dia que la fuente distingue cadenas de
        # protocolos, y la capa 2 tiene que saber leer las dos para siempre.
        value = numeric(point.get("totalLiquidityUSD"))
        if value is None:
            value = numeric(point.get("tvl"))
        column = "tvl_usd"
    elif series in {"fees", "revenue"}:
        if not isinstance(payload, (list, tuple)) or len(payload) < 2:
            return None
        day = unix_day(payload[0])
        value = numeric(payload[1])
        column = "fees_usd" if series == "fees" else "revenue_usd"
    else:
        return None

    if not day or value is None:
        return None
    row = {ENTITY: entity, DAY: day, "fees_usd": None, "revenue_usd": None, "tvl_usd": None}
    row[column] = value
    return row


class DefiLlamaVolumes(BaseJsonAdapter):
    """Volumen DEX (DefiLlama) contra volumen CEX (CCXT): donde se forma el precio."""

    # entidad -> cadena en DefiLlama. Solo estan las MEDIDAS: 'polkadot', 'celestia' y
    # 'cosmos' devuelven 500 y por eso DOT, TIA y ATOM no tienen pata DEX. BTC tampoco:
    # su volumen on-chain no es comparable con el de una cadena de contratos.
    DEX_CHAINS: dict[str, str] = {
        "ETH": "ethereum",
        "SOL": "solana",
        "BNB": "bsc",
        "XRP": "xrpl",
        "ADA": "cardano",
        "AVAX": "avalanche",
        "NEAR": "near",
        "APT": "aptos",
        "ARB": "arbitrum",
        "OP": "optimism",
        "INJ": "injective",
        "FIL": "filecoin",
        "SUI": "sui",
        "SEI": "sei",
    }

    # Par y venue de referencia para la pata de CEX. Es una CONVENCION declarada, no una
    # medida del mercado entero: un venue y un par, el mismo para todos los activos, para
    # que la cuota sea comparable entre ellos aunque no sea el volumen global de CEX.
    CEX_QUOTE = "USDT"
    CEX_DEFAULT_DAYS = 365

    def __init__(self, source, *, cex_provider=None, **kwargs) -> None:
        super().__init__(source, base_url=LLAMA_BASE, **kwargs)
        self._cex_provider = cex_provider

    @property
    def cex_provider(self):
        if self._cex_provider is None:  # perezoso, como el cliente HTTP
            from ai_trader.data.providers.ccxt_crypto import CCXTCrypto

            self._cex_provider = CCXTCrypto()
        return self._cex_provider

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        finish = end or datetime.now(UTC)
        begin = start or finish - timedelta(days=self.CEX_DEFAULT_DAYS)

        since = day_or_none(begin)
        out: list[RawRecord] = []
        for entity in entities:
            chain = self.DEX_CHAINS.get(entity)
            if chain is None:
                continue
            out.extend(self._dex(entity, chain, since))
            out.extend(self._cex(entity, begin, finish))
        return out

    def _dex(self, entity: str, chain: str, since: str | None = None) -> list[RawRecord]:
        payload = safe_call(
            lambda: self.client.get_json(
                f"/overview/dexs/{chain}",
                params={
                    "excludeTotalDataChart": "false",
                    "excludeTotalDataChartBreakdown": "true",
                },
            ),
            what=f"defillama dexs {chain}",
            logger=logger,
        )
        points = (payload or {}).get("totalDataChart") or []
        return [
            self.record(entity, point, day=day, request={SERIES: "dex", "chain": chain})
            for day, point in chart_records(points, day_of=lambda p: unix_day(p[0]), since=since)
        ]

    def _cex(self, entity: str, start: datetime, end: datetime) -> list[RawRecord]:
        symbol = f"{entity}/{self.CEX_QUOTE}"
        try:
            bars = self.cex_provider.get_daily_bars(symbol, start, end)
        except Exception as exc:  # noqa: BLE001 - un par sin listar no puede tumbar la fuente
            logger.info("· volumes: sin pata CEX para %s (%s)", symbol, exc)
            return []

        out: list[RawRecord] = []
        for timestamp, bar in bars.iterrows():
            day = timestamp.date().isoformat()
            # El volumen de CCXT es en unidades BASE: pasarlo a USD exige un precio, y el
            # cierre del dia es el unico que esta en la misma barra. Aproximacion declarada.
            payload = {"close": float(bar["close"]), "volume_base": float(bar["volume"])}
            out.append(
                self.record(
                    entity, payload, day=day, request={SERIES: "cex", "symbol": symbol}
                )
            )
        return out

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        rows = rows_from_records(records, row_of=_volume_row)
        frame = self.to_daily(rows)
        if frame.empty:
            return frame
        dex = frame["dex_volume_usd"].fillna(0.0)
        total = dex + frame["cex_volume_usd"].fillna(0.0)
        # Sin ninguna de las dos patas la cuota no existe: NaN, no 0.5 ni 0. Un 0 diria
        # "todo el volumen es de CEX", que es una afirmacion que no se ha medido.
        frame["dex_share"] = (frame["dex_volume_usd"] / total).where(total > 0)
        return frame


def _volume_row(record: Mapping) -> dict | None:
    series = (record.get(RAW_REQUEST) or {}).get(SERIES)
    payload = record.get("payload")
    entity = record.get("entity") or ""
    row = {
        ENTITY: entity,
        DAY: None,
        "dex_volume_usd": None,
        "cex_volume_usd": None,
        "dex_share": None,
    }

    if series == "dex":
        if not isinstance(payload, (list, tuple)) or len(payload) < 2:
            return None
        row[DAY] = unix_day(payload[0])
        row["dex_volume_usd"] = numeric(payload[1])
    elif series == "cex":
        row[DAY] = record.get(RAW_DAY)
        close = numeric((payload or {}).get("close"))
        volume = numeric((payload or {}).get("volume_base"))
        row["cex_volume_usd"] = None if close is None or volume is None else close * volume
    else:
        return None

    if not row[DAY]:
        return None
    return row


def register() -> None:
    """Da de alta las tres fuentes de DefiLlama en el registro de adaptadores."""
    from ai_trader.signals.source import register_adapter

    register_adapter("defillama_stablecoins", DefiLlamaStablecoins)
    register_adapter("defillama_fees", DefiLlamaFees)
    register_adapter("defillama_volumes", DefiLlamaVolumes)


__all__ = [
    "DefiLlamaFees",
    "DefiLlamaStablecoins",
    "DefiLlamaVolumes",
    "register",
]
