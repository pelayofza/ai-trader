"""
TESORERIAS COTIZADAS (DATs): el indice de estres de vendedores forzados.

QUE ES ESTO, Y POR QUE NO ES EL mNAV DE UNA COMPANIA
-----------------------------------------------------
Una tesoreria cotizada es una empresa cuyo balance ES un tesoro de cripto y cuya accion
cotiza a un multiplo de ese tesoro (mNAV = capitalizacion / valor del tesoro). Por encima
de 1 la maquina funciona hacia adelante: emitir acciones es acretivo, se emite y se compra
mas cripto. Por DEBAJO de 1 funciona en reversa, y es aritmetica, no sentimiento: emitir
diluye, asi que la via barata para levantar caja pasa a ser VENDER EL TESORO.

Lo que se publica aqui no es el mNAV de nadie: es la DISTRIBUCION del mNAV por activo
subyacente. La cifra que importa es la fraccion de la cohorte por debajo de 1 —cada
compania ahi es oferta futura estructural sobre ese activo— y la distancia de la mediana a
esa frontera, que dice cuanto falta para que la cola engorde. La compresion RELATIVA entre
grupos (BTC frente a ETH frente a SOL) sale de comparar esa distancia entre entidades, que
es lo que el frame publica una fila por activo y dia.

NO HAY API, Y ESA ES LA SENAL
-----------------------------
bitcointreasuries.net, mnav.io, bitcoinquant y Artemis son cuadros de mando: se miran, no
se descargan. La serie hay que COMPONERLA, y esa friccion es exactamente lo que la mantiene
sin arbitrar. Tres patas, las tres gratuitas y las tres medidas el 2026-08-13:

    (a) TENENCIAS   XBRL de la SEC. `CryptoAssetFairValue` (valor razonable, en USD) y
                    `CryptoAssetNumberOfUnits` (unidades). Los dos vienen con `end` —la
                    fecha a la que se refiere— y `filed` —el dia en que se publico—, que
                    es lo que hace medible el retraso en vez de suponerlo.
    (b) ACCIONES    `dei:EntityCommonStockSharesOutstanding`, del mismo XBRL.
    (c) PRECIOS     El cierre diario de la accion Y el del activo, del MISMO proveedor y
                    de la MISMA sesion. Ver `PRICE_BASE`.

La pata (c) no sale de CCXT, que es lo que el repo ya tiene para cripto, y el motivo no es
comodidad: el mNAV es un COCIENTE entre una pata de renta variable y una cripto. Un cierre
de CCXT es medianoche UTC y un cierre de bolsa son las 21:00 UTC, asi que mezclarlos mete
nueve horas de desfase DENTRO del cociente —justo en los dias de gap, que son los unicos en
los que esta senal dice algo—. Es el mismo cuidado que en `p2p.py`, donde la prima se
archiva con su tipo de cambio del mismo momento porque ninguna de las dos patas se puede
re-descargar con fecha pasada.

QUIEN ESTA EN LA COHORTE: TRES FILTROS, Y NINGUNO MIRA EL mNAV
---------------------------------------------------------------
Definir la cohorte con el propio mNAV seria circular: truncaria justo la cola que se
publica. Los tres filtros son estructurales y salen del balance o del registro de la SEC:

  1. NO ES UN VEHICULO PASIVO. Se descartan los SIC 6221 (los ETF y trusts al contado:
     iShares Bitcoin Trust, ARK 21Shares) y 6211 (brokers que custodian cripto de sus
     clientes: Robinhood, Galaxy). MEDIDO: 25 y 3 de las 138 declarantes. Un trust no puede
     emitir por encima del NAV —crea y redime AL NAV—, asi que su mNAV esta clavado en 1
     por arbitraje y no contiene ninguna informacion sobre venta forzada.
  2. EL TESORO ES EL BALANCE. `CryptoAssetFairValue / us-gaap:Assets` por encima de
     `TREASURY_MIN_ASSET_SHARE`. Sale del balance y no de la cotizacion, luego no es
     circular. MEDIDO: es el filtro que separa a BitMine (93% del activo) de CleanSpark
     (3,7%) y de Riot (22%), que son mineras con algo de cripto y no tesorerias.
  3. SE SABE QUE ACTIVO TIENE. Ver `identify_asset`, que es la parte dificil.

QUE ACTIVO TIENE CADA UNA: UN NOMBRE IDENTIFICA, EL PRECIO VERIFICA
--------------------------------------------------------------------
Es la parte dificil y la que decide si esto sirve para algo. `CryptoAssetNumberOfUnits`
trae una etiqueta de unidad, y la tentacion es leerla. MEDIDO sobre las 108 candidatas: las
etiquetas realmente usadas son `Integer` (12 veces), `Bitcoin` (8), `item` (5), `pure` (4),
`token` (4), y despues `bitcoin`, `BITCOIN`, `Bitcoins`, `Btcoin` (con la errata dentro),
`cryptoAsset`, `unit`, `decimal`, `Quantity`, `canton_coin`, `InfiniBand` y `USD/shares`.
Dos tercios no nombran nada.

La salida evidente —identificar por el PRECIO IMPLICITO (valor razonable entre unidades) y
quedarse con el unico activo del universo que cuadre— se implemento, se midio y SE RETIRO.
Sobre la cohorte del 2026-08-13 daba dos falsos positivos de ocho: TON Strategy Co (Toncoin,
~1,60 $) salia NEAR, e Hyperion DeFi (HYPE) salia LTC. El fallo no es la tolerancia: es que
"el unico que cuadra" solo significa algo si el conjunto de candidatos esta COMPLETO, y no
puede estarlo —hay miles de tokens y el universo tiene veinticuatro—. Anadir senuelos
tampoco lo arregla: Toncoin tiene DOS dias de serie en el proveedor de precios, asi que ni
compite en las fechas que importan. Una distribucion con un cuarto de las filas mal
asignadas seria una mentira con forma de medicion.

Lo que hay hoy reparte los papeles:

    IDENTIFICA un NOMBRE, en la etiqueta de unidad (`Bitcoin`, `Injective`) o en la razon
               social del emisor (`Solana Co`, `TON Strategy Co`). La tabla incluye a
               proposito activos que NO se operan, para que nombrar uno de ellos sea un
               rechazo EXPLICITO y no una asignacion al vecino de precio mas parecido.
    VERIFICA   el precio implicito, contra el precio de mercado de ese activo y en TODAS las
               fechas en que la companıa declara las dos patas.

Los dos casos que la verificacion atrapa estan medidos y ninguno daria error por si solo:

    CleanSpark  declara 1.719.000 unidades etiquetadas `Bitcoin` con 100,6 M$ de valor
                razonable -> 58,53 $ por unidad. No son bitcoins: son 1.719 con un error de
                escala de mil en el propio filing (el trimestre anterior declaro 1.641).
                Creyendo la etiqueta, la cohorte de BTC habria sumado un tesoro cien veces
                mayor que el de Strategy.
    Bit Digital declara unidades etiquetadas `ETH` con un valor razonable que cubre TODA su
                cartera -> 349,85 $ por unidad, que no es el precio de nada. Es el caso de
                la companıa con mas de un activo, y ahi el cociente no significa nada.

El precio DEJO DE IDENTIFICAR y ahora solo sabe rechazar, y eso se paga en cobertura:
Forward Industries, UPEXI y BitMine son tesorerias reales, etiquetan `Integer`/`token`, su
razon social no dice nada y por tanto NO ENTRAN. Es la eleccion deliberada entre una
cobertura baja y declarada y una cobertura alta con un cuarto de las filas mal asignadas.
Todo lo que se cae sale en `rejections` con su motivo, que es la diferencia entre una
cobertura declarada y un numero inventado.

LO QUE NO SE PUEDE COMPONER, MEDIDO Y DECLARADO
------------------------------------------------
Las APIs XBRL de la SEC solo exponen hechos SIN DIMENSIONES. Una companıa con varias clases
de accion etiqueta su recuento por clase, y entonces el hecho no existe sin dimension.
MEDIDO 2026-08-13: `companyfacts` de Strategy (CIK 1050446) tiene UN solo tag en el espacio
`dei` —`EntityPublicFloat`— y ningun recuento de acciones ordinarias. La consecuencia hay
que decirla entera: **la tesoreria mas grande que existe no esta en esta cohorte**.

Y no se sustituye por `WeightedAverageNumberOfSharesOutstandingBasic`, que si esta sin
dimension. Es una MEDIA del periodo, y estas companias emiten acciones contra el mercado
todos los dias: la media subestima el recuento de hoy justo en las que mas emiten, o sea
que meteria un sesgo a la baja en la capitalizacion y por tanto en el mNAV, empujando hacia
la cola inferior precisamente a las mas activas. Un hueco declarado es preferible a un
numero que sesga en la direccion de la senal.

EL RETRASO ES UN DATO, NO UNA SUPOSICION
-----------------------------------------
Las tenencias se publican tarde y de forma irregular. Aqui eso no se estima: cada hecho
trae su `end` y su `filed`, la fila se fecha en `filed` —el dia en que la informacion
EXISTE, igual que el COT se fecha el dia de publicacion y no el martes al que se refiere— y
el retraso realizado (`filed - end`) se publica como una feature mas
(`dat_disclosure_lag_days`). El catalogo declara la mediana MEDIDA en `disclosure_lag_days`.
Sin eso, un backtest usaria el 30 de junio una tenencia que no se publico hasta el 3 de
agosto.
"""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ai_trader.data.providers.http import JsonHttpClient, JsonHttpConfig
from ai_trader.shared.clock import utc_now
from ai_trader.shared.reports import load_report, write_report
from ai_trader.shared.signals import DAY, ENTITY
from ai_trader.signals.adapters.common import (
    UTC,
    chart_records,
    iso_day,
    numeric,
    safe_call,
    unique_records,
)
from ai_trader.signals.adapters.legal import SEC_USER_AGENT
from ai_trader.signals.source import BaseJsonAdapter, RawRecord

logger = logging.getLogger(__name__)

# --- los tres proveedores ------------------------------------------------------------

SEC_DATA_BASE = "https://data.sec.gov"
# El mapa ticker -> CIK vive en el otro host de la SEC (www, no data). MEDIDO: 10.396
# companias, 795 KB, sin credencial y con el mismo User-Agent identificable que exige EDGAR.
SEC_FILES_BASE = "https://www.sec.gov"
TICKERS_PATH = "/files/company_tickers.json"

FRAMES_PATH = "/api/xbrl/frames/us-gaap/CryptoAssetFairValue/USD/{period}.json"
SUBMISSIONS_PATH = "/submissions/CIK{cik:010d}.json"
CONCEPT_PATH = "/api/xbrl/companyconcept/CIK{cik:010d}/{namespace}/{tag}.json"

# Los cuatro conceptos, con su espacio de nombres. `Assets` esta para el filtro de
# materialidad y no para ninguna feature: sin el, una minera con dos bitcoins en el balance
# entra en la distribucion con un mNAV de cinco cifras.
CONCEPTS: tuple[tuple[str, str, str], ...] = (
    ("units", "us-gaap", "CryptoAssetNumberOfUnits"),
    ("fair_value", "us-gaap", "CryptoAssetFairValue"),
    ("assets", "us-gaap", "Assets"),
    ("shares", "dei", "EntityCommonStockSharesOutstanding"),
)

# Las dos patas de precio salen de aqui, del mismo endpoint y del mismo cierre de sesion.
PRICE_BASE = "https://query1.finance.yahoo.com"
PRICE_PATH = "/v8/finance/chart/{symbol}"
# Sufijo del ticker de un activo cripto en este proveedor. La accion va sin sufijo.
CRYPTO_SUFFIX = "-USD"

# Activos cuyo simbolo en este proveedor NO es `<TICKER>-USD`. Cuando dos tokens comparten
# ticker, el proveedor desempata con un sufijo numerico y el simbolo obvio o no existe o es
# OTRA moneda. MEDIDO 2026-08-13, y las dos formas de fallar estan aqui: `SUI-USD` y
# `HYPE-USD` devuelven 200 sin serie, mientras que `TON-USD` SI existe y cotiza a 0,0053 $
# —no es Toncoin, que es `TON11419-USD` a 1,33 $—. La segunda es la peligrosa: un simbolo
# equivocado que responde no se distingue de uno correcto.
CRYPTO_SYMBOL_OVERRIDES: dict[str, str] = {
    "SUI": "SUI20947-USD",
    "TON": "TON11419-USD",
    "HYPE": "HYPE32196-USD",
    "TAO": "TAO22974-USD",
}

def price_symbol(asset: str) -> str:
    """El simbolo de un activo cripto en el proveedor de precios."""
    return CRYPTO_SYMBOL_OVERRIDES.get(asset.upper(), f"{asset.upper()}{CRYPTO_SUFFIX}")

# --- politica declarada --------------------------------------------------------------

# SIC que NO son tesorerias por construccion. Ver el docstring: un vehiculo pasivo cotiza
# al NAV por arbitraje y un broker custodia cripto que no es suya. Son codigos del registro
# de la SEC, no una clasificacion nuestra.
EXCLUDED_SIC: frozenset[str] = frozenset({"6221", "6211"})

# Que fraccion del ACTIVO TOTAL tiene que ser el tesoro para que la companıa sea una
# tesoreria. La mitad: por encima, el balance ES el tesoro y el mNAV describe la companıa
# entera; por debajo, el mNAV es el cociente entre un negocio operativo y una partida del
# balance, que no es la magnitud de la que habla esta fuente. Sale del BALANCE y no de la
# cotizacion, asi que no es circular con lo que se publica.
TREASURY_MIN_ASSET_SHARE = 0.5

# Cuanto puede alejarse el precio implicito (valor razonable / unidades) del precio de
# mercado del activo ese dia antes de que la identificacion se rechace. Una cuarta parte, y
# es un orden de magnitud razonado y no un parametro ajustado: entre el cierre del periodo
# fiscal de la companıa y el cierre diario del mercado hay horas, y en el mismo epigrafe
# puede ir una posicion pequena de otro activo. Lo que un 25% NO cubre es un error de escala
# ni una cartera mezclada, que son los dos fallos medidos.
UNIT_PRICE_TOLERANCE = 1.25

# NOMBRES DE ACTIVO, en la etiqueta de unidad o en la razon social del emisor.
#
# Es la unica via de identificacion, y la tabla incluye a proposito activos que NO se
# operan: nombrar uno de esos es lo que hace que la companıa se rechace EXPLICITAMENTE en
# vez de acabar asignada al vecino de precio mas parecido. Las erratas del emisor estan
# dentro —`Btcoin` viene de un filing real— porque lo que hay que reconocer es lo que los
# emisores escriben, no lo que deberian escribir.
#
# Cada linea sale de una companıa OBSERVADA en la cohorte del 2026-08-13, con el mismo
# criterio que `shared/entities.py::ENTITY_OVERRIDES`: una tabla no se rellena por
# adelantado, crece con los casos que se ven.
ASSET_NAMES: dict[str, str] = {
    # --- del universo que se opera ---
    "bitcoin": "BTC", "bitcoins": "BTC", "btc": "BTC", "btcoin": "BTC",
    "ethereum": "ETH", "eth": "ETH",
    "solana": "SOL", "sol": "SOL",
    "injective": "INJ", "inj": "INJ",
    "avalanche": "AVAX", "avax": "AVAX",
    "cardano": "ADA", "ada": "ADA",
    "chainlink": "LINK", "link": "LINK",
    "dogecoin": "DOGE", "doge": "DOGE",
    "litecoin": "LTC", "ltc": "LTC",
    "polkadot": "DOT", "dot": "DOT",
    "ripple": "XRP", "xrp": "XRP",
    "sui": "SUI", "aptos": "APT", "apt": "APT", "near": "NEAR", "sei": "SEI",
    "celestia": "TIA", "tia": "TIA", "cosmos": "ATOM", "atom": "ATOM",
    "filecoin": "FIL", "fil": "FIL", "aave": "AAVE", "uniswap": "UNI", "uni": "UNI",
    "arbitrum": "ARB", "arb": "ARB", "optimism": "OP", "op": "OP", "bnb": "BNB",
}

# Activos que se nombran y NO se operan. Una companıa que nombre uno de estos se rechaza
# con su motivo, que es una respuesta distinta —y mucho mas util— que "no se sabe".
FOREIGN_ASSET_NAMES: dict[str, str] = {
    "ton": "TON", "toncoin": "TON",            # TON Strategy Co
    "hyperliquid": "HYPE", "hype": "HYPE",     # Hyperliquid Strategies Inc
    "canton": "CANTON",                        # Canton Strategic Holdings, Inc.
    "stablecoin": "USDC", "usdc": "USDC", "usdt": "USDT", "tether": "USDT",
    "tron": "TRX", "trx": "TRX",
    "hedera": "HBAR", "hbar": "HBAR",
    "bittensor": "TAO", "tao": "TAO",
    "ethena": "ENA", "ena": "ENA",
    "worldcoin": "WLD", "wld": "WLD",
    "cronos": "CRO",
}

# Longitud minima de un token para buscarlo en la RAZON SOCIAL. Tres: por debajo, un ticker
# de dos letras aparece dentro de nombres que no tienen nada que ver. En la ETIQUETA de
# unidad no hace falta, porque ahi el token es la etiqueta entera.
MIN_NAME_TOKEN = 3

# Cuantos dias puede tener el ultimo hecho publicado antes de que la companıa deje de
# contar. Ciento ochenta: un declarante trimestral publica cada ~91 dias con un retraso
# medido de ~44, asi que entre dos hechos visibles pasan ~135 dias en el caso regular mas
# lento. Por encima de dos trimestres, lo que hay no es un dato viejo: es una companıa que
# dejo de declarar, y arrastrarla congelaria su mNAV con el ultimo tesoro conocido.
FACT_STALE_DAYS = 180

# Cuantas companias se componen por pasada. La cohorte MEDIDA son 138 declarantes y cada
# una cuesta cinco peticiones (submissions + cuatro conceptos) mas una de precio; el tope
# esta para que ampliar la ventana no convierta una captura diaria en una hora de red, y
# para que lo que quede fuera quede fuera POR UNA DECISION VISIBLE y no porque el bucle se
# canso. Se recorta por valor razonable descendente: lo que se cae es la cola pequena.
MAX_COMPANIES = 160

# Trimestres hacia atras que se piden al endpoint de frames para enumerar la cohorte. Cuatro
# porque un trimestre recien cerrado esta a medio llenar: MEDIDO 2026-08-13, CY2026Q2I
# tenia 65 declarantes y CY2026Q1I, 118. Enumerar solo con el ultimo publicaria una cohorte
# que encoge cada vez que empieza un trimestre.
FRAME_QUARTERS = 4

# Pausa entre peticiones a la SEC. Su politica publica son diez por segundo; con 0,12 s en
# serie no se roza, y el coste de pasarse es un bloqueo por IP que no avisa.
SEC_PAUSE_SECONDS = 0.12

# Ventana de precio que se pide por simbolo. Dos anos: es lo que cubre la vida de esta
# cohorte (los primeros `CryptoAssetFairValue` son de 2025) sin pedir una serie que no se
# va a usar.
PRICE_RANGE = "2y"


# =====================================================================================
# capa 1: red
# =====================================================================================


class DigitalAssetTreasuries(BaseJsonAdapter):
    """
    Compone las tres patas y las archiva por separado, cada una con su marca de tiempo.

    Se archivan TRES series distintas (marcadas en `request.series`) y no un mNAV ya
    calculado, por lo de siempre en este puerto: el mapeo es lo que se equivoca. Que un
    tesoro sea de SOL y no de AAVE, que un recuento de acciones sea el bueno, que un valor
    razonable cubra un activo o cinco —todo eso se decide en la capa PURA y se puede
    re-derivar sobre el archivo el dia que se corrija. Un mNAV archivado no se corrige: se
    vuelve a descargar, y el precio de la accion de hace seis meses tampoco es re-pedible
    con garantias.
    """

    def __init__(
        self,
        source,
        *,
        max_companies: int = MAX_COMPANIES,
        quarters: int = FRAME_QUARTERS,
        pause_seconds: float = SEC_PAUSE_SECONDS,
        price_client: JsonHttpClient | None = None,
        files_client: JsonHttpClient | None = None,
        **kwargs,
    ) -> None:
        config = kwargs.pop("http_config", None) or JsonHttpConfig(
            timeout_seconds=60.0,
            user_agent=SEC_USER_AGENT,
            headers={"Accept": "application/json"},
        )
        super().__init__(source, base_url=SEC_DATA_BASE, http_config=config, **kwargs)
        self._max_companies = max_companies
        self._quarters = quarters
        self._pause = pause_seconds
        self._price_client = price_client
        self._files_client = files_client

    @property
    def files(self) -> JsonHttpClient:
        """El otro host de la SEC (www), donde vive el mapa ticker -> CIK."""
        if self._files_client is None:
            self._files_client = JsonHttpClient(
                SEC_FILES_BASE,
                JsonHttpConfig(timeout_seconds=60.0, user_agent=SEC_USER_AGENT),
            )
        return self._files_client

    @property
    def prices(self) -> JsonHttpClient:
        if self._price_client is None:
            self._price_client = JsonHttpClient(
                PRICE_BASE,
                JsonHttpConfig(timeout_seconds=30.0, user_agent=SEC_USER_AGENT),
            )
        return self._price_client

    # --- pata (a) y (b): la SEC ------------------------------------------------------

    def fetch_raw(
        self, entities: Sequence[str], start: datetime | None = None, end: datetime | None = None
    ) -> list[RawRecord]:
        stop = (end or utc_now()).date()
        begin = start.date() if start is not None else stop - timedelta(days=365)
        assets = tuple(str(e).upper() for e in entities if str(e).strip())

        cohort = self._cohort(stop)
        logger.info("· dat: %s declarantes de cripto en el registro XBRL", len(cohort))
        if not cohort:
            return []

        out: list[RawRecord] = []
        tickers: list[str] = []
        for cik, fair_value in cohort[: self._max_companies]:
            payload = self._company(cik)
            if payload is None:
                continue
            payload["frame_fair_value_usd"] = fair_value
            out.append(
                self.record(
                    payload["ticker"], payload, request={"series": SERIES_FACTS, "cik": cik}
                )
            )
            tickers.append(payload["ticker"])

        logger.info("· dat: %s companias con ticker y sin excluir por SIC", len(tickers))

        # --- pata (c): los dos precios, del mismo proveedor y del mismo cierre --------
        since = begin.isoformat()
        for ticker in tickers:
            out.extend(self._price_records(ticker, ticker, SERIES_EQUITY_PX, since))
        for asset in assets:
            out.extend(
                self._price_records(price_symbol(asset), asset, SERIES_ASSET_PX, since)
            )
        return out

    def _cohort(self, stop: date) -> list[tuple[int, float]]:
        """CIKs que declaran cripto en el balance, por valor razonable descendente.

        Sale del endpoint de FRAMES —una peticion por trimestre devuelve a todos los
        declarantes de ese periodo— y no de una lista escrita a mano. Lo que se hereda de
        eso es que la cohorte la define el registro de la SEC y no nuestra idea de quien es
        una tesoreria; los tres filtros que la acotan estan declarados arriba.
        """
        best: dict[int, float] = {}
        for period in _quarters_before(stop, self._quarters):
            payload = safe_call(
                lambda p=period: self.client.get_json(FRAMES_PATH.format(period=p)),
                what=f"dat frames {period}",
                logger=logger,
            )
            rows = (payload or {}).get("data") or []
            logger.info("·   %s: %s declarantes", period, len(rows))
            for row in rows:
                cik = int(row.get("cik") or 0)
                value = numeric(row.get("val")) or 0.0
                if cik and value > best.get(cik, 0.0):
                    best[cik] = value
            time.sleep(self._pause)
        return sorted(best.items(), key=lambda pair: -pair[1])

    def _company(self, cik: int) -> dict | None:
        """Metadatos + los cuatro conceptos de UNA companıa. None si no procede archivarla.

        El filtro por SIC se aplica AQUI, en la capa de red, y es la unica decision de
        cohorte que no vive en la capa pura. El motivo es de coste, no de diseno: descartar
        un ETF antes de pedirle cuatro conceptos ahorra el 20% de las peticiones de la
        pasada, y el codigo SIC se archiva igualmente dentro del payload para que la
        decision se pueda revisar sobre el archivo.
        """
        meta = safe_call(
            lambda: self.client.get_json(SUBMISSIONS_PATH.format(cik=cik)),
            what=f"dat submissions {cik}",
            logger=logger,
        )
        time.sleep(self._pause)
        if not isinstance(meta, Mapping):
            return None

        sic = str(meta.get("sic") or "")
        tickers = [str(t).upper() for t in (meta.get("tickers") or []) if t]
        if sic in EXCLUDED_SIC or not tickers:
            return None

        facts: dict[str, dict] = {}
        for name, namespace, tag in CONCEPTS:
            payload = safe_call(
                lambda ns=namespace, t=tag: self.client.get_json(
                    CONCEPT_PATH.format(cik=cik, namespace=ns, tag=t)
                ),
                what=f"dat {tag} {cik}",
                logger=logger,
            )
            time.sleep(self._pause)
            facts[name] = _concept_facts(payload)

        return {
            "cik": cik,
            # El PRIMER ticker es la clase ordinaria: la SEC los publica en el orden del
            # registro y las preferentes van detras (Strategy: MSTR, STRC, STRD, STRF, STRK).
            "ticker": tickers[0],
            "tickers": tickers,
            "name": str(meta.get("name") or ""),
            "sic": sic,
            "sic_description": str(meta.get("sicDescription") or ""),
            "facts": facts,
        }

    def _price_records(
        self, symbol: str, entity: str, series: str, since: str
    ) -> list[RawRecord]:
        """La serie de cierres de UN simbolo, partida en un registro por dia.

        Partida y no archivada de una pieza por lo que explica `common.chart_records`: el
        archivo se shardea por el mes de la OBSERVACION, y un bloque de dos anos caeria
        entero en el mes en que se descargo.
        """
        payload = safe_call(
            lambda: self.prices.get_json(
                PRICE_PATH.format(symbol=symbol),
                params={"range": PRICE_RANGE, "interval": "1d"},
            ),
            what=f"dat precio {symbol}",
            logger=logger,
        )
        points = closes_from_chart(payload)
        if not points:
            logger.info("· dat: %s sin serie de precio", symbol)
            return []
        return [
            self.record(
                entity,
                {"symbol": symbol, "day": day, "close": close},
                day=day,
                request={"series": series},
            )
            for day, close in chart_records(
                points, day_of=lambda p: p[0], payload_of=lambda p: p[1], since=since
            )
        ]

    # =================================================================================
    # capa 2: pura
    # =================================================================================

    def daily_from_raw(self, records: Sequence[Mapping]) -> pd.DataFrame:
        return self.to_daily(cohort_rows(records))


# Marca de que serie es cada registro crudo. Va en `request.series`, como en
# `defillama_fees` (tres llamadas) y `deribit_volatility` (dvol y libro).
SERIES_FACTS = "facts"
SERIES_EQUITY_PX = "equity_px"
SERIES_ASSET_PX = "asset_px"


def _quarters_before(stop: date, count: int) -> list[str]:
    """Los `count` trimestres instantaneos (`CY2026Q2I`) CERRADOS antes de `stop`.

    Se empieza en el anterior al que contiene `stop` y no en el que lo contiene: un frame
    instantaneo se refiere al ULTIMO DIA del trimestre, asi que el trimestre en curso se
    refiere a una fecha que todavia no ha ocurrido y devuelve cero declarantes siempre. No
    es un error —la peticion contesta 200 con la lista vacia— y ese es el problema: gastaria
    una de las cuatro peticiones en un hueco garantizado y la ventana de cohorte seria un
    trimestre mas corta de lo que dice el codigo.
    """
    quarter = (stop.month - 1) // 3 + 1
    year = stop.year
    out: list[str] = []
    for _ in range(max(0, count)):
        quarter -= 1
        if quarter == 0:
            quarter, year = 4, year - 1
        out.append(f"CY{year}Q{quarter}I")
    return out


def _concept_facts(payload) -> dict:
    """`companyconcept` -> `{unidad: [{end, val, filed, form}]}`, sin tocar nada mas.

    Se conservan TODOS los hechos y no solo el ultimo: el mismo `end` se vuelve a declarar
    en filings posteriores (MEDIDO: las unidades de Strategy a 2025-12-31 aparecen otra vez
    en el 10-Q publicado el 2026-08-03) y quedarse con uno haria imposible reconstruir que
    se veia en una fecha pasada, que es justo lo que esta fuente tiene que poder hacer.
    """
    units = (payload or {}).get("units") if isinstance(payload, Mapping) else None
    if not isinstance(units, Mapping):
        return {}
    out: dict[str, list[dict]] = {}
    for unit, rows in units.items():
        kept = [
            {
                "end": iso_day(row.get("end")),
                "val": numeric(row.get("val")),
                "filed": iso_day(row.get("filed")),
                "form": str(row.get("form") or ""),
            }
            for row in rows or []
            if iso_day(row.get("filed")) and numeric(row.get("val")) is not None
        ]
        if kept:
            out[str(unit)] = kept
    return out


def closes_from_chart(payload) -> list[tuple[str, float]]:
    """La respuesta del proveedor de precios -> `[(dia, cierre)]`. Vacio si no la trae.

    PURA a proposito, aunque solo la llame la capa 1: la forma de esta respuesta (dos
    listas alineadas por indice, con nulos dentro de los cierres) es de las que cambian, y
    poder ejercitarla con un payload copiado es mas barato que descubrirlo en produccion.
    """
    result = (((payload or {}).get("chart") or {}).get("result") or [None])[0]
    if not isinstance(result, Mapping):
        return []
    stamps = result.get("timestamp") or []
    quotes = ((result.get("indicators") or {}).get("quote") or [{}])[0] or {}
    closes = quotes.get("close") or []
    out: list[tuple[str, float]] = []
    for stamp, close in zip(stamps, closes):
        value = numeric(close)
        if value is None or value <= 0:
            continue  # sesion sin cierre: el proveedor manda null y no cero
        day = datetime.fromtimestamp(float(stamp), tz=UTC).date().isoformat()
        out.append((day, float(value)))
    return out


# --- los hechos de una companıa, ya legibles ------------------------------------------


@dataclass(frozen=True, slots=True)
class Fact:
    """Un hecho XBRL con las DOS fechas que lo hacen utilizable.

    `as_of` es el dia al que se refiere y `filed` el dia en que se publico. Todo lo que
    esta fuente hace bien depende de no confundirlos.
    """

    as_of: str
    filed: str
    value: float
    unit: str = ""
    form: str = ""

    @property
    def lag_days(self) -> int:
        """El retraso REALIZADO de este hecho. Ni estimado ni declarado: restado."""
        return (date.fromisoformat(self.filed) - date.fromisoformat(self.as_of)).days


@dataclass(frozen=True, slots=True)
class Treasury:
    """Una companıa de la cohorte, con sus hechos ordenados y su activo YA identificado."""

    cik: int
    ticker: str
    name: str
    asset: str
    units: tuple[Fact, ...]
    shares: tuple[Fact, ...]
    treasury_share_of_assets: float
    implied_unit_price: float
    identified_by: str

    def visible(self, facts: Sequence[Fact], day: str) -> Fact | None:
        """El hecho que estaba PUBLICADO el dia `day`, o None si ninguno o si caduco.

        Es el nucleo anti look-ahead de esta fuente y por eso no esta en linea en el bucle:
        el criterio es `filed <= day` —no `as_of <= day`, que es el error que mete cinco
        semanas de futuro— y despues la caducidad se mide contra `as_of`, que es la fecha a
        la que el numero de verdad se refiere.
        """
        candidates = [f for f in facts if f.filed <= day]
        if not candidates:
            return None
        # Gana el publicado mas tarde y, a igualdad de publicacion, el que se refiere a la
        # fecha mas reciente: un mismo filing trae el trimestre y su comparativo.
        latest = max(candidates, key=lambda f: (f.filed, f.as_of))
        if (date.fromisoformat(day) - date.fromisoformat(latest.as_of)).days > FACT_STALE_DAYS:
            return None
        return latest


def facts_of(block: Mapping, name: str) -> tuple[Fact, ...]:
    """Los hechos de un concepto, ordenados por publicacion. Vacio si no hay."""
    out: list[Fact] = []
    for unit, rows in ((block or {}).get(name) or {}).items():
        for row in rows or []:
            as_of, filed, value = row.get("end"), row.get("filed"), numeric(row.get("val"))
            if as_of and filed and value is not None:
                out.append(Fact(as_of, filed, float(value), str(unit), str(row.get("form") or "")))
    return tuple(sorted(out, key=lambda f: (f.filed, f.as_of)))


def asset_named_in(text: str, *, whole_text: bool) -> tuple[str | None, bool]:
    """
    El activo que NOMBRA un texto, y si es de los que no se operan. `(clave, es_ajeno)`.

    `whole_text=True` para la ETIQUETA de unidad, donde el token es la etiqueta entera;
    `False` para la RAZON SOCIAL, donde hay que buscar por palabras y con una longitud
    minima (`MIN_NAME_TOKEN`): un ticker de dos letras aparece dentro de nombres que no
    tienen nada que ver con el.
    """
    value = (text or "").strip().lower()
    if not value:
        return None, False
    if whole_text:
        return ASSET_NAMES.get(value) or FOREIGN_ASSET_NAMES.get(value), (
            value in FOREIGN_ASSET_NAMES and value not in ASSET_NAMES
        )

    tokens = {t for t in re.split(r"[^a-z0-9]+", value) if len(t) >= MIN_NAME_TOKEN}
    for token in sorted(tokens):
        if token in ASSET_NAMES:
            return ASSET_NAMES[token], False
        if token in FOREIGN_ASSET_NAMES:
            return FOREIGN_ASSET_NAMES[token], True
    return None, False


def identify_asset(
    label: str,
    name: str,
    observations: Sequence[tuple[float, str]],
    prices: Mapping[str, Mapping[str, float]],
    *,
    tolerance: float = UNIT_PRICE_TOLERANCE,
) -> tuple[str | None, str]:
    """
    Que activo tiene esta tesoreria. Devuelve `(activo, motivo)`; `None` = no se sabe.

    DOS PIEZAS, Y EL REPARTO DE PAPELES ES LA DECISION DE TODO EL MODULO:

        IDENTIFICA un NOMBRE: la etiqueta de unidad (`Bitcoin`, `Injective`) o la razon
                   social del emisor (`Solana Co`, `TON Strategy Co`). Es lo unico que dice
                   de que activo se habla.
        VERIFICA   el PRECIO IMPLICITO (valor razonable / unidades) contra el precio de
                   mercado de ese activo, en TODAS las fechas en que la companıa declara las
                   dos patas. Es lo que atrapa los errores de escala y las carteras mezcladas.

    LA VERSION ANTERIOR IDENTIFICABA POR PRECIO, Y ESTABA MAL. La regla era "si un unico
    activo del universo cae dentro de la tolerancia, es ese", y sobre la cohorte MEDIDA del
    2026-08-13 produjo dos falsos positivos de ocho: TON Strategy Co (Toncoin, ~1,60 $)
    salio NEAR, e Hyperion DeFi (HYPE) salio LTC. El fallo no es la tolerancia: es que la
    afirmacion "unico" solo vale si el conjunto de candidatos esta COMPLETO, y no puede
    estarlo —hay miles de tokens y el universo tiene veinticuatro—. Anadir senuelos no lo
    arregla: Toncoin tiene DOS dias de serie en el proveedor de precios (medido), asi que ni
    siquiera compite en las fechas que importan. Con un 25% de identificaciones erroneas, la
    distribucion publicada seria una mentira con forma de medicion.

    Asi que el precio dejo de identificar. Lo que queda solo puede RECHAZAR: una companıa
    cuyo nombre no dice nada se queda fuera aunque su tesoro sea evidente para una persona
    (Forward Industries, UPEXI y BitMine son tesorerias reales y no entran), y esa perdida
    esta publicada en `rejections`. Es la eleccion entre una cobertura baja y declarada y una
    cobertura alta con un cuarto de las filas mal asignadas.

    PURA y sin reloj: se ejercita con los nombres y los precios implicitos MEDIDOS de las
    companias reales.
    """
    asset, foreign = asset_named_in(label, whole_text=True)
    source = "la etiqueta de unidad"
    if asset is None:
        asset, foreign = asset_named_in(name, whole_text=False)
        source = "la razon social"
    if asset is None:
        return None, "ni la etiqueta ni la razon social nombran ningun activo"
    if foreign:
        return None, f"tesoro en {asset}, que no esta en el universo"

    usable = [
        (float(price), day)
        for price, day in observations or ()
        if price is not None and np.isfinite(price) and price > 0 and day
    ]
    if not usable:
        return None, "sin precio implicito con el que verificar el nombre"

    series = prices.get(asset) or {}
    checked = 0
    for implied, day in usable:
        quote = price_on(series, day)
        if not quote or quote <= 0:
            continue  # hueco de la serie: ni a favor ni en contra
        checked += 1
        if max(implied / quote, quote / implied) > tolerance:
            return None, f"{source} dice {asset} y el precio implicito lo contradice"
    if not checked:
        return None, f"{source} dice {asset} y no hay precio con el que comprobarlo"

    plural = "" if checked == 1 else f" en las {checked} fechas"
    return asset, f"{source}, verificada por el precio implicito{plural}"


def price_on(series: Mapping[str, float], day: str) -> float | None:
    """El ultimo cierre en `day` o antes. None si la serie no llega.

    Hacia atras y nunca hacia adelante: un festivo o un fin de semana usa el cierre
    anterior, que es el ultimo precio que existio de verdad.
    """
    if not series:
        return None
    keys = [d for d in series if d <= day]
    return float(series[max(keys)]) if keys else None


# --- de los registros crudos a la distribucion ----------------------------------------


def price_series(records: Sequence[Mapping], series: str) -> dict[str, dict[str, float]]:
    """`entidad -> {dia: cierre}` de una de las dos patas de precio."""
    out: dict[str, dict[str, float]] = {}
    for record in records:
        if (record.get("request") or {}).get("series") != series:
            continue
        payload = record.get("payload") or {}
        day = iso_day(payload.get("day")) or iso_day(record.get("day"))
        close = numeric(payload.get("close"))
        entity = str(record.get("entity") or "")
        if not (day and entity) or close is None or close <= 0:
            continue
        out.setdefault(entity, {})[day] = float(close)
    return out


def treasuries_from_records(
    records: Sequence[Mapping], asset_prices: Mapping[str, Mapping[str, float]]
) -> tuple[list[Treasury], list[dict]]:
    """
    Las companias que forman la cohorte, y las que NO con su motivo.

    Devolver las dos listas es el punto. Una cobertura del 60% sin decir que paso con el
    otro 40% es indistinguible de un filtro mal escrito, y aqui los motivos son
    exactamente lo que hay que poder citar: cuantas se caen por no declarar acciones
    (multiclase), cuantas por no ser una tesoreria (el tesoro no es el balance) y cuantas
    por tener un activo que no se opera.
    """
    latest = unique_records(
        records=[r for r in records if (r.get("request") or {}).get("series") == SERIES_FACTS],
        key_of=lambda r: (r.get("payload") or {}).get("cik"),
    )

    kept: list[Treasury] = []
    dropped: list[dict] = []
    for record in latest:
        payload = record.get("payload") or {}
        ticker = str(payload.get("ticker") or "")
        row = {
            "cik": payload.get("cik"),
            "ticker": ticker,
            "name": str(payload.get("name") or ""),
            "sic": str(payload.get("sic") or ""),
        }
        block = payload.get("facts") or {}
        units = facts_of(block, "units")
        shares = facts_of(block, "shares")
        fair = facts_of(block, "fair_value")
        assets = facts_of(block, "assets")

        if not units:
            dropped.append({**row, "reason": "no declara unidades, solo valor razonable"})
            continue
        if not shares:
            # El caso multiclase, que se lleva por delante a la tesoreria mas grande que
            # existe. Ver el docstring del modulo: no se sustituye por la media ponderada.
            dropped.append({**row, "reason": "sin recuento de acciones sin dimension (multiclase)"})
            continue

        # Un trimestre con unidades Y valor razonable a la MISMA fecha es una OBSERVACION.
        # Solo esas sirven: el cociente valor/unidades unicamente significa un precio si las
        # dos patas son del mismo cierre contable. Se usan todas, y eso es lo que convierte
        # una coincidencia de precio en una trayectoria.
        observations = [
            (fact, value)
            for fact in units
            for value in [_same_day(fair, fact.as_of)]
            if value and value > 0 and fact.value
        ]
        if not observations:
            dropped.append({**row, "reason": "sin valor razonable en la fecha de las unidades"})
            continue
        implied = [(value / fact.value, fact.as_of) for fact, value in observations]

        # La materialidad se mide en la ULTIMA observacion completa: una companıa puede
        # convertirse en tesoreria (y dejar de serlo), asi que lo que decide si hoy lo es no
        # es lo que tenia en el balance hace un ano. Sobre la observacion y no sobre el
        # ultimo hecho de unidades a secas, para que el numerador y el denominador de la
        # cuota vengan del mismo cierre contable.
        anchor, value = observations[-1]
        total = _same_day(assets, anchor.as_of)
        share = (value / total) if (total and total > 0) else None
        if share is None:
            dropped.append({**row, "reason": "sin activo total en la fecha de las unidades"})
            continue
        if share < TREASURY_MIN_ASSET_SHARE:
            dropped.append({
                **row,
                "reason": f"el tesoro es el {share:.0%} del activo, no es una tesoreria",
            })
            continue

        asset, why = identify_asset(anchor.unit, row["name"], implied, asset_prices)
        last_implied = implied[-1][0]
        if asset is None:
            dropped.append({**row, "reason": why, "implied_unit_price": last_implied})
            continue

        kept.append(
            Treasury(
                cik=int(payload.get("cik") or 0),
                ticker=ticker,
                name=row["name"],
                asset=asset,
                units=units,
                shares=shares,
                treasury_share_of_assets=float(share),
                implied_unit_price=float(last_implied or 0.0),
                identified_by=why,
            )
        )
    return kept, dropped


def _same_day(facts: Sequence[Fact], as_of: str) -> float | None:
    """El valor de un concepto en EXACTAMENTE esa fecha de referencia. None si no hay.

    Exacta y no la mas cercana: el cociente valor/unidades solo significa un precio si las
    dos patas son del mismo cierre contable. Con fechas distintas seria un numero con la
    forma correcta y sin significado, que es como se cuela un activo mal identificado.
    """
    matching = [f for f in facts if f.as_of == as_of]
    return max(matching, key=lambda f: f.filed).value if matching else None


def cohort_rows(records: Sequence[Mapping]) -> list[dict]:
    """
    Los registros crudos -> una fila por (activo, dia de publicacion) con la distribucion.

    UNA FILA POR DIA DE PUBLICACION, y no una por dia natural. El mNAV se mueve todos los
    dias porque los precios se mueven, pero lo que esta fuente aporta no es el precio —eso
    el sistema ya lo tiene— sino la TENENCIA, y la tenencia solo cambia cuando alguien la
    publica. Fechar en `filed` hace ademas que la serie sea de EVENTO de verdad: entre
    temporadas de resultados no hay filas, que es el estado honesto.
    """
    asset_prices = price_series(records, SERIES_ASSET_PX)
    equity_prices = price_series(records, SERIES_EQUITY_PX)
    kept, _ = treasuries_from_records(records, asset_prices)
    if not kept:
        return []

    by_asset: dict[str, list[Treasury]] = {}
    for treasury in kept:
        by_asset.setdefault(treasury.asset, []).append(treasury)

    rows: list[dict] = []
    for asset, cohort in sorted(by_asset.items()):
        days = sorted({fact.filed for t in cohort for fact in t.units})
        for day in days:
            snapshot = [
                m
                for m in (mnav_at(t, day, asset_prices, equity_prices) for t in cohort)
                if m is not None
            ]
            row = distribution_row(asset, day, snapshot)
            if row is not None:
                rows.append(row)
    return rows


@dataclass(frozen=True, slots=True)
class Reading:
    """El mNAV de UNA companıa en UN dia, con lo que hace falta para auditarlo."""

    ticker: str
    mnav: float
    market_cap_usd: float
    treasury_usd: float
    lag_days: int


def mnav_at(
    treasury: Treasury,
    day: str,
    asset_prices: Mapping[str, Mapping[str, float]],
    equity_prices: Mapping[str, Mapping[str, float]],
) -> Reading | None:
    """
    El mNAV de una companıa en un dia, con SOLO lo que era publico ese dia. None si falta.

    El tesoro se marca a mercado —unidades por precio de HOY— y no se toma el valor
    razonable publicado, que esta marcado al cierre del trimestre. Las dos patas estan
    obsoletas, pero de forma distinta: las unidades lo estan en CANTIDAD y el valor
    razonable lo esta en cantidad Y en precio. Mezclar en la misma distribucion companias
    marcadas de las dos maneras produciria una cola inferior que se mueve con la fecha del
    filing y no con el mercado.
    """
    units = treasury.visible(treasury.units, day)
    shares = treasury.visible(treasury.shares, day)
    if units is None or shares is None or units.value <= 0 or shares.value <= 0:
        return None

    asset_price = price_on(asset_prices.get(treasury.asset) or {}, day)
    equity_price = price_on(equity_prices.get(treasury.ticker) or {}, day)
    if not asset_price or not equity_price:
        return None

    treasury_usd = units.value * asset_price
    market_cap = shares.value * equity_price
    if treasury_usd <= 0:
        return None
    return Reading(
        ticker=treasury.ticker,
        mnav=market_cap / treasury_usd,
        market_cap_usd=market_cap,
        treasury_usd=treasury_usd,
        lag_days=(date.fromisoformat(day) - date.fromisoformat(units.as_of)).days,
    )


def distribution_row(asset: str, day: str, readings: Sequence[Reading]) -> dict | None:
    """
    La distribucion de la cohorte de un activo en un dia -> la fila del frame.

    Se exige `MIN_COHORT` companias. Con una sola, "la fraccion por debajo de 1" solo puede
    valer 0 o 1 y la palabra distribucion sobra; publicarlo daria una feature que salta
    entre los dos extremos y que no describe ninguna cola.
    """
    if len(readings) < MIN_COHORT:
        return None
    values = np.array([r.mnav for r in readings], dtype=float)
    return {
        ENTITY: asset,
        DAY: day,
        # LA FEATURE. Cada companıa por debajo de 1 es oferta futura estructural sobre este
        # activo: emitir la diluye, asi que la via barata para levantar caja es vender.
        "dat_below_nav_share": float((values < 1.0).mean()),
        # LA DISTANCIA A LA FRONTERA, con signo y en tanto por uno. Negativa = la mediana de
        # la cohorte ya esta por debajo. Es lo que hace legible la fraccion: un 0% con la
        # mediana en 1,05 esta a un paso, y con la mediana en 3,0 no lo esta.
        "dat_mnav_gap": float(np.median(values) - 1.0),
        # EL CUARTIL INFERIOR: donde vive la cola que engorda.
        "dat_mnav_p25": float(np.percentile(values, 25)),
        # LA MUESTRA, en la propia fila. El N de esta fuente no lo dan los eventos de una
        # companıa sino el POOLING sobre la cohorte, y sin esta columna no hay forma de
        # distinguir una fraccion sobre tres companias de una sobre cuarenta.
        "dat_companies": float(len(readings)),
        # EL RETRASO REALIZADO, medido y no supuesto. Ver el docstring del modulo.
        "dat_disclosure_lag_days": float(np.median([r.lag_days for r in readings])),
    }


# Companias minimas para que una fila exista. Tres: es el minimo con el que un cuartil
# inferior distingue algo de la mediana. Es una constante declarada y no un parametro:
# subirlo o bajarlo cambia la cobertura de la fuente, y eso no puede ser sorteable.
MIN_COHORT = 3


# =====================================================================================
# el informe: la cifra que hay que publicar
# =====================================================================================

# En data/ y no en .cache/ por lo mismo que el registro de profundidad y el de ADV: es la
# EVIDENCIA de con cuantas companias se compuso la distribucion y por que se cayeron las
# demas, y eso tiene que poder citarse con fecha.
COHORT_REPORT = Path("data") / "signals" / "dat_cohort.json"


def cohort_report(records: Sequence[Mapping]) -> dict:
    """
    Con cuantas companias se compone esto, y que paso con las que no. La cifra publicable.

    `pooled_observations` es el N de verdad de esta fuente y por eso va delante: no es el
    numero de eventos de una companıa —cada una publica cuatro veces al ano— sino el de
    OBSERVACIONES DE COMPANIA agrupadas sobre toda la cohorte, que es lo que pone a las
    doscientas en la misma distribucion. Al lado va `companies`, porque doscientas
    observaciones de tres companias y doscientas de cuarenta sostienen inferencias
    distintas y el agregado no las distingue (misma leccion que `events.pool_report`).
    """
    asset_prices = price_series(records, SERIES_ASSET_PX)
    equity_prices = price_series(records, SERIES_EQUITY_PX)
    kept, dropped = treasuries_from_records(records, asset_prices)

    per_asset: dict[str, dict] = {}
    pooled = 0
    lags: list[int] = []
    for treasury in sorted(kept, key=lambda t: (t.asset, t.ticker)):
        days = sorted({f.filed for f in treasury.units})
        readings = [
            m
            for m in (mnav_at(treasury, day, asset_prices, equity_prices) for day in days)
            if m is not None
        ]
        pooled += len(readings)
        lags.extend(r.lag_days for r in readings)
        row = per_asset.setdefault(
            treasury.asset,
            {"companies": [], "n_companies": 0, "latest": None, "observations": 0},
        )
        row["companies"].append(
            {
                "ticker": treasury.ticker,
                "name": treasury.name,
                "treasury_share_of_assets": round(treasury.treasury_share_of_assets, 3),
                "implied_unit_price": round(treasury.implied_unit_price, 2),
                "identified_by": treasury.identified_by,
                "observations": len(readings),
                "last_mnav": round(readings[-1].mnav, 3) if readings else None,
            }
        )
        row["n_companies"] += 1
        row["observations"] += len(readings)

    frame_rows = cohort_rows(records)
    for row in frame_rows:
        block = per_asset.get(str(row[ENTITY]))
        if block is not None and (block["latest"] is None or row[DAY] >= block["latest"]["day"]):
            block["latest"] = {
                "day": row[DAY],
                "below_nav_share": round(row["dat_below_nav_share"], 3),
                "mnav_gap": round(row["dat_mnav_gap"], 3),
                "mnav_p25": round(row["dat_mnav_p25"], 3),
                "companies": int(row["dat_companies"]),
            }

    reasons: dict[str, int] = {}
    for row in dropped:
        key = str(row.get("reason") or "").split(" entre ")[0]
        reasons[key] = reasons.get(key, 0) + 1

    return {
        "generated_at": utc_now().isoformat(),
        "policy": cohort_policy(),
        # EL N. Ver el docstring.
        "pooled_observations": pooled,
        "companies": len(kept),
        "companies_examined": len(kept) + len(dropped),
        "rows": len(frame_rows),
        "median_disclosure_lag_days": float(np.median(lags)) if lags else None,
        "assets": dict(sorted(per_asset.items())),
        # Por que se cayo cada una. Sin esto, la cobertura es indistinguible de un filtro
        # mal escrito.
        "rejections": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "rejected": sorted(dropped, key=lambda r: str(r.get("ticker") or "")),
    }


def cohort_policy() -> dict:
    """Los umbrales declarados, publicados con el dato. Igual que `normalization_spec`."""
    return {
        "excluded_sic": sorted(EXCLUDED_SIC),
        "treasury_min_asset_share": TREASURY_MIN_ASSET_SHARE,
        "unit_price_tolerance": UNIT_PRICE_TOLERANCE,
        "fact_stale_days": FACT_STALE_DAYS,
        "min_cohort": MIN_COHORT,
        "frontier": 1.0,
        "dated_at": "filed (el dia en que la tenencia se publica), nunca `end`",
        "names": dict(sorted(ASSET_NAMES.items())),
        "foreign_names": dict(sorted(FOREIGN_ASSET_NAMES.items())),
    }


def write_cohort_report(report: Mapping, path: Path | str = COHORT_REPORT) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return write_report(dict(report), target, indent=2, ensure_ascii=True)


def load_cohort_report(path: Path | str = COHORT_REPORT) -> dict | None:
    """El ultimo informe, o None si nunca se ha compuesto. Misma politica que sus hermanos
    de `signals/`: un fichero que no esta y uno corrupto son los dos 'no hay dato', porque
    los escribe una pasada que se puede quedar a medias."""
    try:
        return load_report(path)
    except Exception:  # noqa: BLE001 - fichero corrupto; ver el docstring
        return None


def declared_vs_measured_lag(path: Path | str = COHORT_REPORT) -> dict:
    """Compara el retraso que DECLARA el catalogo con el que dice el informe.

    La misma disciplina que `depth.declared_vs_measured` y `liquidity.declared_vs_measured_adv`:
    el catalogo copia de un fichero medido y un test falla si alguien escribe una cifra que
    el fichero no respalda.
    """
    from ai_trader.signals.catalog import get_source

    report = load_cohort_report(path) or {}
    measured = report.get("median_disclosure_lag_days")
    declared = get_source("dat_mnav").disclosure_lag_days
    return {
        "declared_days": declared,
        "measured_days": measured,
        "generated_at": report.get("generated_at"),
        "matches": (
            None
            if (declared is None or measured is None)
            else abs(float(declared) - float(measured)) <= LAG_TOLERANCE_DAYS
        ),
    }

# Cuanto puede alejarse el retraso declarado del medido. Quince dias, y la tolerancia sale
# de LA AFIRMACION que el campo hace, no de la dispersion de los datos: lo que dice es "las
# tenencias que estas leyendo tienen alrededor de siete semanas", y cualquier cifra entre 34
# y 64 dias dice exactamente eso. Exigir mas precision convertiria el test en una alarma que
# salta porque este mes han publicado otras companias, que es calendario y no un error.
LAG_TOLERANCE_DAYS = 15.0


# =====================================================================================

ADAPTERS = {"dat_mnav": DigitalAssetTreasuries}


def register() -> None:
    from ai_trader.signals.source import register_adapter

    for key, factory in ADAPTERS.items():
        try:
            register_adapter(key, factory)
        except ValueError:
            logger.debug("%s ya estaba registrado", key)


__all__ = [
    "ADAPTERS",
    "COHORT_REPORT",
    "CONCEPTS",
    "EXCLUDED_SIC",
    "FACT_STALE_DAYS",
    "LAG_TOLERANCE_DAYS",
    "MAX_COMPANIES",
    "MIN_COHORT",
    "SERIES_ASSET_PX",
    "SERIES_EQUITY_PX",
    "SERIES_FACTS",
    "TREASURY_MIN_ASSET_SHARE",
    "ASSET_NAMES",
    "FOREIGN_ASSET_NAMES",
    "UNIT_PRICE_TOLERANCE",
    "DigitalAssetTreasuries",
    "Fact",
    "Reading",
    "Treasury",
    "closes_from_chart",
    "cohort_policy",
    "cohort_report",
    "cohort_rows",
    "declared_vs_measured_lag",
    "distribution_row",
    "facts_of",
    "asset_named_in",
    "identify_asset",
    "load_cohort_report",
    "mnav_at",
    "price_on",
    "price_series",
    "register",
    "treasuries_from_records",
    "write_cohort_report",
]
