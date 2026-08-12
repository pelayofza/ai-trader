"""
CATALOGO DE FUENTES DE SENAL: que se puede llegar a mirar, y con que honestidad.

Una entrada de este catalogo no es una promesa de dato: es una DECLARACION. Dice que
produce la fuente, cada cuanto, sobre que clase de entidad, con que licencia y —los dos
campos que sostienen todo lo demas— desde cuando hay historia MEDIDA (`history_from`) y de
que naturaleza es esa historia (`pit`).

LOS DOS CAMPOS QUE HACEN HONESTO EL RESTO
-----------------------------------------
`history_from` es la primera fecha con dato COMPROBADO POR NOSOTROS, no la que anuncia el
proveedor. `None` significa exactamente una cosa: **solo hacia adelante**, y esa fuente no
puede participar en un backtest. Lo que el proveedor promete se escribe en `notes`, en
prosa y como lo que es —una afirmacion suya—, para que ningun codigo pueda confundir una
promesa con una medicion.

QUIEN TIENE DERECHO A ESCRIBIR UNA FECHA AQUI. Solo `signals/depth.py`: la sonda descarga
la serie, la deriva con el adaptador REAL y apunta lo que encuentra en
`data/signals/history_depth.json`. La fecha del catalogo se copia de ahi a mano y un test
(`tests/test_signal_adapters.py`) falla si alguien declara una que el registro no respalda.

Y con una condicion mas: la sonda tiene que haber medido al menos `MIN_MEASURED_DAYS` dias.
El motivo es que en una fuente `forward_capture` el "primer dia con dato" es simplemente el
dia que arranco la captura; declararlo pondria `backtestable=True` en una serie de un dia,
que es exactamente la clase de verdad literal y engañosa que este catalogo existe para
evitar. Para esas, el registro de mediciones guarda desde cuando se captura —que es el dato
util— y el catalogo espera a que la ventana exista.

`pit` (point-in-time) dice si la historia, cuando la haya, vale para backtest:

  - `forward_capture`: solo es point-in-time lo que capturemos nosotros en vivo. Nadie
    publica el pasado, o lo publica reescrito. Es el caso mas comun y el motivo por el que
    `signals/capture.py` se arranca antes de que nada este cableado: cada dia sin capturar
    es profundidad que no se recupera nunca.
  - `archive_revisable`: hay backfill, pero el proveedor puede reetiquetar, borrar o
    rellenar hacia atras. Sirve, con la advertencia declarada; el archivo crudo con su
    `fetched_at` es lo que permite detectar la revision a posteriori.
  - `derived_from_price`: se deriva de datos que ya tenemos (barras, libro). Es
    point-in-time por construccion y su profundidad es la de nuestro historico de precios.

Juntos responden la unica pregunta que importa antes de puntuar nada: **que fuentes pueden
entrar en un backtest y cuales solo en vivo**. El dia que se construyo el esqueleto la
respuesta era "ninguna", con el catalogo entero en `None`. Hoy, con las diecisiete
conectadas y sondeadas, son diez, y las demas siguen a `None` por motivos distintos que
conviene no confundir: la credencial no esta en el entorno (Guavy, FRED, beaconcha.in), el
proveedor cerro el endpoint detras de un muro de pago (unlocks de DefiLlama: 402 medido), o
la fuente no tiene pasado que descargar y su profundidad la compra el calendario (P2P,
funding, OFAC).

EL `tier` DESCRIBE; YA NO ENRUTA (2026-08-11)
---------------------------------------------
Durante una fase este campo fue una BIFURCACION: el tier `A` (mecanico) iba a entrar como
ELEGIBILIDAD —una guarda que veta operar— y solo el `B` como features. La bifurcacion se
retira, y no por comodidad: la defensa que la justificaba —"muestras de DECENAS, un CEM
suelto sobre catorce observaciones construye una estrategia preciosa y falsa"— se apoyaba en
un numero que NADIE HABIA MEDIDO. Al medirlo (2026-08-12) resulto falso por un factor de
diez: 463 ajustes de dificultad desde 2009 —321 dentro de la ventana de doce anos que pide
la sonda—, 621 hacks fechados desde 2016 y el calendario del FOMC desde 2017. Y donde la
muestra SI es corta, la razon medida es otra: el endpoint de unlocks pasa a ser de pago
(402) y el de colas de staking pide credencial (401). Ademas, contando el EVENTO NORMALIZADO
(% del float, % del ADV) y no el token, la unidad de observacion se multiplica por el numero
de activos: el recuento pooled se publica en `data/signals/event_pool.json`.

Hoy TODAS las fuentes van por la misma via: features al espacio de observacion, en backtest
y en vivo (`observation/signal_radar.py`). El tier ya no separa el destino sino que DESCRIBE
la naturaleza economica de la fuente:

`A` = OFERTA MECANICA, calendarizada y determinista: unlocks, colas de staking, ajustes de
dificultad, hacks, sanciones, calendario macro.

`B` = EFECTO ESTADISTICO, pequeno y decadente: sentimiento, flujos de ETF, macro, actividad
de desarrollo, dispersion de funding.

Y LA CODIFICACION LA DECIDE LA CADENCIA, NO EL TIER. `cadence == 'event'` -> `events.py`
(dias-al-evento acotado, magnitud sobre su escala declarada, ventana activa), porque una z
contra una serie que es 99% ceros no significa nada. Cualquier otra cadencia -> las dos
varas de `normalize.py`. Coinciden en cinco de las seis mecanicas; la sexta,
`staking_queue`, es mecanica y publica un NIVEL diario, y enrutar por el tier le habria
puesto un dias-al-evento a una cola de validadores.

QUE SUSTITUYE A LA PUERTA, sin fingir que es equivalente: NINGUNA feature —de ningun tier—
entra en `scoring/search_space.py`, y los umbrales de las puertas son constantes declaradas
y razonadas en codigo, no parametros sorteables. Eso limita los grados de libertad, pero no
es un test.

EL TEST YA EXISTE (2026-08-12), y conviene saber que contesta y que no.
`scoring/signal_study.py` barre la capacidad predictiva de un canal de observacion
SINTETICO —con rho=0 como grupo de control— y publica el rho de BREAK-EVEN: a partir de
que IC una senal hace que la estrategia bata al baseline despues de costes (evidencia en
`data/signal_channel/`). Es una propiedad del DISENO, no de los datos, y por eso no se
puede sobreajustar a ningun historico. Lo que NO contesta es cuanto rho tiene ninguna de
las diecisiete fuentes de aqui abajo: eso se mide en el sustrato REAL, con la profundidad
que la captura vaya comprando. Hasta que se mida, el break-even es un liston contra el que
comparar, no un aprobado.

ESTE MODULO NO CONECTA NADA. No importa `requests` ni ningun cliente: es una lista de
declaraciones. Los adaptadores viven en `signals/adapters/` y se registran en
`signals/source.py`; el catalogo es lo que permite medir cuantos faltan (hoy, ninguno).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ai_trader.shared.entities import EntityKind
from ai_trader.shared.signals import (
    AGG_LAST,
    AGG_MEAN,
    AGG_SUM,
    AGGREGATIONS,
    DEFAULT_AGGREGATION,
)

# --- vocabulario cerrado ------------------------------------------------------------

TIER_MECHANICAL = "A"
TIER_STATISTICAL = "B"
TIERS: tuple[str, ...] = (TIER_MECHANICAL, TIER_STATISTICAL)

SCOPE_ASSET = "asset"
SCOPE_MARKET = "market"
SCOPE_CHAIN = "chain"
SCOPE_MACRO = "macro"
SCOPE_VENUE = "venue"
SCOPES: tuple[str, ...] = (SCOPE_ASSET, SCOPE_MARKET, SCOPE_CHAIN, SCOPE_MACRO, SCOPE_VENUE)

PIT_FORWARD_CAPTURE = "forward_capture"
PIT_ARCHIVE_REVISABLE = "archive_revisable"
PIT_DERIVED_FROM_PRICE = "derived_from_price"
# Reconstruible desde un registro INMUTABLE que no es nuestro (las cabeceras de bloque de
# una cadena). Comparte con `derived_from_price` lo que importa —point-in-time por
# construccion, nadie puede reescribir el pasado— y se distingue de el en que su
# profundidad no la limita nuestro historico de precios sino la del propio registro.
PIT_CHAIN_IMMUTABLE = "chain_immutable"
PIT_KINDS: tuple[str, ...] = (
    PIT_FORWARD_CAPTURE,
    PIT_ARCHIVE_REVISABLE,
    PIT_DERIVED_FROM_PRICE,
    PIT_CHAIN_IMMUTABLE,
)

# Cadencia -> cada cuantos DIAS NATURALES se espera una observacion nueva. Es lo que
# convierte "hay 40 dias con dato" en un porcentaje de cobertura (`signals/audit.py`).
# `event` es None a proposito: una fuente por evento no tiene cadencia esperada, y darle
# una convertiria "no paso nada" en "falta dato".
CADENCE_DAYS: dict[str, float | None] = {
    "hourly": 1.0,
    "daily": 1.0,
    "weekly": 7.0,
    "monthly": 30.0,
    "event": None,
}
CADENCES: tuple[str, ...] = tuple(CADENCE_DAYS)

# Entidades que puede tener una fuente segun su alcance. Un alcance de mercado tiene UNA
# entidad (`MARKET_ENTITY`); uno de activo, una por activo del universo.
SCOPE_ENTITY_KINDS: dict[str, tuple[EntityKind, ...]] = {
    SCOPE_ASSET: (EntityKind.TOKEN, EntityKind.EQUITY, EntityKind.PREDICTION),
    SCOPE_MARKET: (EntityKind.MARKET,),
    SCOPE_CHAIN: (EntityKind.CHAIN,),
    SCOPE_MACRO: (EntityKind.MACRO, EntityKind.MARKET),
    SCOPE_VENUE: (EntityKind.TOKEN, EntityKind.VENUE),
}


@dataclass(frozen=True, slots=True)
class SignalFeature:
    """
    Una columna que produce la fuente, con su politica de agregacion a dia.

    `agg` no es un detalle de implementacion: sumar un ESTADO (oferta de stablecoins) o
    quedarse con el ultimo de un FLUJO (entradas de ETF) son errores que no se ven en la
    grafica. Por eso la politica es un dato del catalogo y `shared/signals.py` la aplica,
    en vez de dejarla en manos de cada adaptador.
    """

    name: str
    agg: str = DEFAULT_AGGREGATION
    unit: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"Nombre de feature invalido: '{self.name}'")
        if self.name != self.name.lower():
            raise ValueError(f"Los nombres de feature van en minusculas: '{self.name}'")
        if self.agg not in AGGREGATIONS:
            raise ValueError(f"Agregacion desconocida en '{self.name}': '{self.agg}'")

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "agg": self.agg,
            "unit": self.unit,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class SignalSource:
    """Una fuente declarada. Ver el docstring del modulo para `history_from` y `pit`."""

    key: str
    title: str
    tier: str
    scope: str
    cadence: str
    entity_kind: EntityKind
    features: tuple[SignalFeature, ...]
    pit: str
    license: str
    # MEDIDO. None = no se ha medido -> solo hacia adelante -> fuera de cualquier backtest.
    history_from: date | None = None
    endpoint: str = ""
    # Variable de entorno con la credencial, o None si la fuente es abierta. Es un NOMBRE
    # de variable, nunca un valor: aqui no vive ningun secreto.
    auth_env: str | None = None
    # Entidades DECLARADAS, para las fuentes cuyo eje no sale del universo de simbolos: una
    # cadena, una serie macro. Vacio = las entidades se derivan del universo configurado
    # (`signals/capture.py::entities_for`), que es el caso de todo lo que es por activo.
    fixed_entities: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.key or not _is_slug(self.key):
            raise ValueError(f"Clave de fuente invalida: '{self.key}' (minusculas, [a-z0-9_])")
        if self.tier not in TIERS:
            raise ValueError(f"Tier desconocido en '{self.key}': '{self.tier}'")
        if self.scope not in SCOPES:
            raise ValueError(f"Scope desconocido en '{self.key}': '{self.scope}'")
        if self.cadence not in CADENCES:
            raise ValueError(f"Cadencia desconocida en '{self.key}': '{self.cadence}'")
        if self.pit not in PIT_KINDS:
            raise ValueError(f"pit desconocido en '{self.key}': '{self.pit}'")
        if not self.license:
            raise ValueError(f"La fuente '{self.key}' no declara licencia")
        if not self.features:
            raise ValueError(f"La fuente '{self.key}' no produce ninguna feature")
        names = [f.name for f in self.features]
        if len(set(names)) != len(names):
            raise ValueError(f"Features repetidas en '{self.key}': {names}")
        allowed = SCOPE_ENTITY_KINDS[self.scope]
        if self.entity_kind not in allowed:
            raise ValueError(
                f"'{self.key}': entity_kind '{self.entity_kind.value}' no encaja con "
                f"scope '{self.scope}' (admitidos: {[k.value for k in allowed]})"
            )

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Orden canonico de las columnas de esta fuente."""
        return tuple(f.name for f in self.features)

    @property
    def backtestable(self) -> bool:
        """¿Puede participar en un backtest? Solo si hay historia MEDIDA. Sin `history_from`
        la fuente existe unicamente hacia adelante, por muy profundo que sea el backfill
        que anuncie el proveedor."""
        return self.history_from is not None

    @property
    def expected_days_between_observations(self) -> float | None:
        """Dias naturales entre observaciones esperadas. None = fuente por evento."""
        return CADENCE_DAYS[self.cadence]

    @property
    def is_market_scoped(self) -> bool:
        """True si la fuente tiene UNA entidad para todo el mercado, no una por activo."""
        return self.entity_kind is EntityKind.MARKET

    def aggregations(self) -> dict[str, str]:
        """Politica feature -> agregacion, tal y como la consume `normalize_signals`."""
        return {f.name: f.agg for f in self.features}

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "tier": self.tier,
            "scope": self.scope,
            "cadence": self.cadence,
            "entity_kind": self.entity_kind.value,
            "features": [f.as_dict() for f in self.features],
            "pit": self.pit,
            "license": self.license,
            "history_from": self.history_from.isoformat() if self.history_from else None,
            "backtestable": self.backtestable,
            "endpoint": self.endpoint,
            "auth_env": self.auth_env,
            "fixed_entities": list(self.fixed_entities),
            "notes": self.notes,
        }


def _is_slug(value: str) -> bool:
    # La clave es tambien un NOMBRE DE DIRECTORIO en data/signals_raw/: nada de barras,
    # espacios ni mayusculas, que en Windows y en Linux no se comportan igual.
    return bool(value) and all(c.islower() or c.isdigit() or c == "_" for c in value)


# --- el catalogo --------------------------------------------------------------------
#
# Ninguna fecha de aqui se escribe a mano sin respaldo: `history_from` se copia del
# registro de mediciones y un test falla si no coincide. Lo que el proveedor PROMETE va en
# `notes`, en prosa y como lo que es, para que ningun codigo lo confunda con una medicion.

CATALOG: tuple[SignalSource, ...] = (
    # -------- Tier B: efecto estadistico; cadencia diaria/semanal -> las dos z --------
    SignalSource(
        key="guavy_sentiment",
        title="Sentimiento social por token (Guavy)",
        tier=TIER_STATISTICAL,
        scope=SCOPE_ASSET,
        cadence="daily",
        entity_kind=EntityKind.TOKEN,
        features=(
            SignalFeature("sentiment_positive", AGG_SUM, "mensajes", "Menciones positivas."),
            SignalFeature("sentiment_negative", AGG_SUM, "mensajes", "Menciones negativas."),
            SignalFeature("sentiment_neutral", AGG_SUM, "mensajes", "Menciones neutras."),
            SignalFeature("sentiment_total", AGG_SUM, "mensajes", "Volumen total de mencion."),
        ),
        pit=PIT_ARCHIVE_REVISABLE,
        license="Comercial, plan gratuito. Bearer token.",
        endpoint="https://guavy.com/api/v1/sentiment/get-sentiment-history/<symbol>",
        auth_env="GUAVY_API_KEY",
        notes="Se consumen SOLO los conteos crudos. Los endpoints trend/signal son el "
              "output de SU modelo: pueden estar recalculados con informacion posterior y "
              "no son auditables (el adaptador los rechaza por codigo). El host esta "
              "MEDIDO: guavy.com responde 401 sin token y api.guavy.com da 404. Sin "
              "credencial en el entorno la sonda no pudo medir profundidad, y por eso "
              "history_from sigue a None.",
    ),
    SignalSource(
        key="etf_flows",
        title="Flujos diarios de los ETF spot (BTC/ETH), por emisor",
        tier=TIER_STATISTICAL,
        scope=SCOPE_ASSET,
        cadence="daily",
        entity_kind=EntityKind.TOKEN,
        features=(
            SignalFeature("etf_netflow_usd", AGG_SUM, "USD", "Flujo neto del dia."),
            SignalFeature("etf_issuer_dispersion", AGG_LAST, "ratio",
                          "Bruto/|neto| entre emisores: 1 = flujo neto puro, alto = rotacion."),
            SignalFeature("etf_issuers_reporting", AGG_LAST, "n",
                          "Emisores que reportaron ese dia (cobertura, no senal)."),
        ),
        pit=PIT_ARCHIVE_REVISABLE,
        history_from=date(2024, 1, 11),  # MEDIDO: 662 dias, 1 entidad (BTC)
        license="TFTC: CC BY 4.0. Farside/SoSoValue como contraste.",
        endpoint="https://www.tftc.io/bitcoin-etf-flows/data.json",
        notes="MEDIDO: solo BITCOIN. TFTC no publica dataset equivalente de ETH (las rutas "
              "evidentes dan 404), asi que ETH no tiene cobertura por esta puerta y NO se "
              "rellena con un agregado de otro sitio: sin desglose por emisor, "
              "etf_issuer_dispersion significaria otra cosa para ese activo. Los flujos se "
              "revisan al alza el dia siguiente, de ahi archive_revisable.",
    ),
    SignalSource(
        key="fred_macro",
        title="Series macro de FRED (DXY, tipos reales, 2y/10y, oro, indices)",
        tier=TIER_STATISTICAL,
        scope=SCOPE_MACRO,
        cadence="daily",
        entity_kind=EntityKind.MACRO,
        features=(
            SignalFeature("macro_value", AGG_LAST, "nativa", "Valor de la serie ese dia."),
            SignalFeature("macro_change_1d", AGG_LAST, "nativa", "Variacion diaria."),
        ),
        pit=PIT_ARCHIVE_REVISABLE,
        license="FRED: uso publico, key gratuita.",
        endpoint="https://api.stlouisfed.org/fred/series/observations",
        auth_env="FRED_API_KEY",
        fixed_entities=(
            "DTWEXBGS", "DFII10", "DGS2", "DGS10", "T10Y2Y",
            "SP500", "NASDAQ100", "VIXCLS", "DCOILWTICO",
        ),
        notes="La entidad es la SERIE, no un activo, y cada una mapea sobre un factor del "
              "generador sintetico (adapters/fred.py::FACTOR_OF; universe.py: EQUITY, "
              "RATES, USD, COMMODITY). Las nueve existen: COMPROBADO. El ORO no esta y es "
              "una medicion, no un olvido: FRED discontinuo las series de la LBMA "
              "(GOLDAMGBD228NLBM da 404 hoy) y el factor COMMODITY se cubre con WTI. FRED "
              "revisa series hacia atras: sin el archivo crudo con fetched_at no hay forma "
              "de saber que se veia aquel dia. Sin FRED_API_KEY en el entorno la sonda no "
              "pudo medir profundidad.",
    ),
    SignalSource(
        key="defillama_stablecoins",
        title="Oferta de stablecoins por cadena (DefiLlama)",
        tier=TIER_STATISTICAL,
        scope=SCOPE_CHAIN,
        cadence="daily",
        entity_kind=EntityKind.CHAIN,
        features=(
            SignalFeature("stablecoin_supply_usd", AGG_LAST, "USD", "Oferta circulante."),
            SignalFeature("stablecoin_issuance_usd", AGG_SUM, "USD", "Emision neta del dia."),
        ),
        pit=PIT_ARCHIVE_REVISABLE,
        history_from=date(2017, 11, 29),  # MEDIDO: 12.497 dias-cadena sobre 6 cadenas
        license="DefiLlama: abierta, sin auth.",
        endpoint="https://stablecoins.llama.fi/stablecoincharts/<chain>",
        fixed_entities=("ethereum", "solana", "arbitrum", "base", "tron", "bsc"),
        notes="La emision neta es polvora seca entrando; el desglose POR CADENA es rotacion "
              "de capital entre ecosistemas, que es la parte que no se publica agregada. La "
              "emision se deriva como VARIACION de la oferta: tras un hueco, el primer dia "
              "acumula lo emitido durante todo el hueco, y `observed` esta al lado para "
              "saber cuando fiarse.",
    ),
    SignalSource(
        key="defillama_fees",
        title="Comisiones, ingresos y TVL por protocolo (DefiLlama)",
        tier=TIER_STATISTICAL,
        scope=SCOPE_ASSET,
        cadence="daily",
        entity_kind=EntityKind.TOKEN,
        features=(
            SignalFeature("fees_usd", AGG_SUM, "USD", "Comisiones del dia."),
            SignalFeature("revenue_usd", AGG_SUM, "USD", "Ingresos del protocolo."),
            SignalFeature("tvl_usd", AGG_LAST, "USD", "Valor bloqueado al cierre."),
        ),
        pit=PIT_ARCHIVE_REVISABLE,
        history_from=date(2011, 1, 31),  # MEDIDO: 54.129 dias-entidad sobre 24 entidades
        license="DefiLlama: abierta, sin auth.",
        endpoint="https://api.llama.fi/summary/fees/<protocol>",
        notes="Con el precio ya en el sistema salen P/F y P/S. El universo con ingresos "
              "reales es pequeno e ignorado, que es lo que lo hace interesante. Los 24 "
              "slugs estan COMPROBADOS: XRP y ATOM entran por su cadena (xrpl, cosmos-hub) "
              "porque el slug obvio da 400. La serie llega en tres llamadas (dailyFees, "
              "dailyRevenue, y el TVL del protocolo) marcadas en request.series.",
    ),
    SignalSource(
        key="defillama_volumes",
        title="Cuota DEX/CEX por activo (DefiLlama)",
        tier=TIER_STATISTICAL,
        scope=SCOPE_ASSET,
        cadence="daily",
        entity_kind=EntityKind.TOKEN,
        features=(
            SignalFeature("dex_volume_usd", AGG_SUM, "USD", "Volumen en DEX (cadena del activo)."),
            SignalFeature("cex_volume_usd", AGG_SUM, "USD", "Volumen en CEX (par de referencia)."),
            SignalFeature("dex_share", AGG_LAST, "fraccion", "DEX / (DEX + CEX)."),
        ),
        pit=PIT_ARCHIVE_REVISABLE,
        history_from=date(2017, 8, 17),  # MEDIDO: 30.590 dias-entidad sobre 14 activos
        license="DefiLlama: abierta, sin auth. Pata CEX via CCXT (endpoints publicos).",
        endpoint="https://api.llama.fi/overview/dexs/<chain>",
        notes="Donde se forma el precio. La cuota necesita las DOS patas y DefiLlama no "
              "publica volumen de CEX: la de CEX sale de CCXT y se archiva como un crudo "
              "mas, marcado en request.series. Solo estan las cadenas MEDIDAS: polkadot, "
              "celestia y cosmos devuelven 500, asi que DOT, TIA y ATOM no tienen pata DEX. "
              "Un activo cuyo volumen migra a DEX cambia de microestructura y el modelo de "
              "costes lo trata igual (limite conocido).",
    ),
    SignalSource(
        key="wikipedia_pageviews",
        title="Atencion por idioma (Wikipedia Pageviews)",
        tier=TIER_STATISTICAL,
        scope=SCOPE_ASSET,
        cadence="hourly",
        entity_kind=EntityKind.TOKEN,
        features=(
            SignalFeature("pageviews", AGG_SUM, "visitas", "Visitas del dia, todos los idiomas."),
            SignalFeature("pageviews_lang_concentration", AGG_LAST, "HHI",
                          "Concentracion por idioma: descomposicion GEOGRAFICA de la atencion."),
        ),
        pit=PIT_ARCHIVE_REVISABLE,
        history_from=date(2015, 7, 1),  # MEDIDO: el limite de la propia API, 15 activos
        license="Wikimedia REST: CC BY-SA, sin auth.",
        endpoint="https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article",
        notes="Desglosado por idioma, que es lo que Google Trends no da limpio: seis "
              "proyectos (en/es/de/ja/ru/zh) dan una descomposicion geografica de la "
              "atencion. Dos limites MEDIDOS: el User-Agent generico se come un 403 y el "
              "limite de peticiones salta a 429 incluso en serie a 0,7 s, asi que el "
              "adaptador va secuencial con pausa de 2 s. Los titulos son una TABLA (el "
              "articulo de SOL es 'Solana (blockchain platform)'), comprobada uno a uno: lo "
              "que daba 404 no esta.",
    ),
    SignalSource(
        key="p2p_premium",
        title="Prima P2P en monedas en crisis (Binance P2P)",
        tier=TIER_STATISTICAL,
        scope=SCOPE_VENUE,
        cadence="hourly",
        entity_kind=EntityKind.TOKEN,
        features=(
            SignalFeature("p2p_premium_pct", AGG_MEAN, "%",
                          "Prima media del dia sobre el precio oficial."),
            SignalFeature("p2p_currencies", AGG_LAST, "n", "Divisas con cotizacion ese dia."),
        ),
        pit=PIT_FORWARD_CAPTURE,
        license="Endpoint publico de Binance. Tipo oficial: open.er-api.com (abierto).",
        endpoint="https://p2p.binance.com/bapi/c2c/v2/friendly/c2c/adv/search",
        fixed_entities=("USDT",),
        notes="TRY/ARS/NGN contra USDT (MEDIDO: TRY y ARS tienen libro lleno, NGN devuelve "
              "cero anuncios y por eso p2p_currencies vale 2). Demanda por crisis monetaria: un driver distinto y "
              "no correlacionado con el ciclo especulativo. Cada registro crudo guarda el "
              "libro Y el tipo oficial del mismo momento, porque la prima es un cociente y "
              "ninguna de las dos patas se puede re-descargar con fecha pasada. NADIE "
              "publica el historico de un libro P2P: lo que no se capture hoy no existe "
              "manana, y por eso la profundidad crece un dia por dia real.",
    ),
    SignalSource(
        key="funding_dispersion",
        title="Dispersion de funding entre venues (CCXT)",
        tier=TIER_STATISTICAL,
        scope=SCOPE_VENUE,
        cadence="hourly",
        entity_kind=EntityKind.TOKEN,
        features=(
            SignalFeature("funding_dispersion", AGG_MEAN, "bps",
                          "Desviacion tipica del funding entre venues."),
            SignalFeature("funding_median", AGG_MEAN, "bps", "Nivel mediano (contexto)."),
            SignalFeature("funding_venues", AGG_LAST, "n", "Venues con dato (cobertura)."),
        ),
        pit=PIT_FORWARD_CAPTURE,
        license="Endpoints publicos de cada venue via CCXT.",
        endpoint="ccxt.fetch_funding_rate (binance, bybit, okx)",
        notes="La DISPERSION, no el nivel: el nivel esta muy arbitrado. Los tres venues "
              "estan MEDIDOS (los tres devolvieron funding de BTC/USDT:USDT). La "
              "dispersion se calcula ENTRE VENUES en cada ventana de funding y solo "
              "despues se promedia el dia: al reves mediria la variacion horaria, que es "
              "otra cosa. Se exigen 3 venues como minimo. fetch_funding_rate da la foto de "
              "AHORA, asi que la profundidad crece con la captura.",
    ),
    SignalSource(
        key="github_activity",
        title="Velocidad de desarrollo (GitHub API)",
        tier=TIER_STATISTICAL,
        scope=SCOPE_ASSET,
        cadence="daily",
        entity_kind=EntityKind.TOKEN,
        features=(
            SignalFeature("commits", AGG_SUM, "commits", "Commits del dia en el repo principal."),
            SignalFeature("contributors", AGG_LAST, "n", "Contribuidores unicos activos."),
        ),
        pit=PIT_ARCHIVE_REVISABLE,
        # MEDIDO: 15.877 dias-entidad sobre 24 repos. OJO a la asimetria, declarada abajo:
        # esta fecha la alcanza `contributors`, no `commits`.
        history_from=date(2009, 8, 23),
        license="GitHub API, token gratuito (rate limit 5000/h).",
        endpoint="https://api.github.com/repos/<owner>/<repo>/stats/commit_activity",
        auth_env="GITHUB_TOKEN",
        notes="Santiment cobra por esto y el dato crudo es publico. LAS DOS FEATURES NO "
              "TIENEN LA MISMA PROFUNDIDAD, y por eso hay que decirlo aqui: history_from "
              "(2009, la creacion del repo mas viejo) la alcanza contributors, que llega "
              "con cubos semanales desde el principio; commits solo tiene 52 semanas por "
              "captura, asi que hacia atras es NaN —no cero— y la historia larga se acumula "
              "captura a captura en el archivo. commit_activity reparte la semana en curso "
              "en siete dias incluidos los que AUN NO HAN PASADO: la capa 2 los corta con el "
              "fetched_at del registro, porque un cero en un dia que no ha ocurrido no es "
              "'no hubo commits'. contributors es SEMANAL y se ancla al inicio de su semana; "
              "repetirlo siete veces convertiria una observacion en siete. Sin GITHUB_TOKEN "
              "son 60 peticiones/hora y el universo necesita 48: la sonda se quedo sin cuota "
              "en el primer intento (medido). "
              "contributors es SEMANAL y se ancla al inicio de su semana; repetirlo siete "
              "veces convertiria una observacion en siete. El 202 con cuerpo vacio (GitHub "
              "calculando) se trata como 'hoy no hay dato'. Revisable: un force-push "
              "reescribe el historico de commits.",
    ),
    SignalSource(
        key="cftc_cot",
        title="Posicionamiento en futuros CME (CFTC COT / TFF)",
        tier=TIER_STATISTICAL,
        scope=SCOPE_ASSET,
        cadence="weekly",
        entity_kind=EntityKind.TOKEN,
        features=(
            SignalFeature("cot_leveraged_net", AGG_LAST, "contratos",
                          "Neto de fondos apalancados."),
            SignalFeature("cot_dealer_net", AGG_LAST, "contratos", "Neto de dealers."),
            SignalFeature("cot_open_interest", AGG_LAST, "contratos", "Interes abierto."),
        ),
        pit=PIT_ARCHIVE_REVISABLE,
        history_from=date(2018, 4, 13),  # MEDIDO: 714 publicaciones (BTC y ETH)
        license="CFTC: dominio publico (Socrata).",
        endpoint="https://publicreporting.cftc.gov/resource/gpe5-46if.json",
        notes="Semanal: foto del MARTES, publicada el VIERNES. El dia con el que se archiva "
              "es el de PUBLICACION, no el de referencia —el martes queda en el payload—, "
              "porque el recorte por dia de observacion no ve un desfase de tres dias y "
              "fechar el martes meteria tres dias de futuro en cualquier cruce. Solo los "
              "dos contratos de CME con serie larga (BTC 435 semanas, ETH 279, medidas); "
              "los micro y los de otros exchanges quedan fuera para no sumar contratos de "
              "tamano distinto.",
    ),
    # ------ Tier A: eventos discretos; features como todas, con otra codificacion -----
    #
    # NINGUNA declara ya un `days_to_*`. No es una perdida: los dias-al-evento son
    # EXACTAMENTE lo que publica `signals/events.py` para las seis a la vez. Declararlos
    # aqui decia que los producia el proveedor, y lo que el proveedor produce es el
    # CALENDARIO; los dias que faltan son una cuenta nuestra contra el 'ahora', y hacerla
    # una vez en un sitio es lo que la hace testeable.
    SignalSource(
        key="token_unlocks",
        title="Desbloqueos y vesting (DefiLlama emissions)",
        tier=TIER_MECHANICAL,
        scope=SCOPE_ASSET,
        cadence="event",
        entity_kind=EntityKind.TOKEN,
        features=(
            SignalFeature("unlock_pct_float", AGG_SUM, "%", "Desbloqueo como % del float."),
            SignalFeature("unlock_pct_adv", AGG_SUM, "%", "Desbloqueo como % del volumen diario."),
        ),
        pit=PIT_ARCHIVE_REVISABLE,
        license="DefiLlama: el endpoint de emisiones dejo de ser abierto (402 medido).",
        endpoint="https://api.llama.fi/emissions",
        notes="Lo valioso NO es 'hay unlock' (explotado) sino el desbloqueo NORMALIZADO: "
              "un 3% sobre un token con 40 dias de volumen en circulacion es un evento; el "
              "mismo 3% sobre uno liquido no lo es. Esa normalizacion es ademas lo que "
              "permite agrupar: la unidad de observacion es el evento comparable entre "
              "activos, no el token. MEDIDO 2026-08-12: la API responde 402 Payment "
              "Required, asi que hoy no hay muestra que contar y history_from sigue a None. "
              "El adaptador existe para que eso sea una MEDICION y no un hueco. El "
              "calendario futuro se reescribe cuando el equipo cambia el vesting: revisable "
              "por naturaleza.",
    ),
    SignalSource(
        key="staking_queue",
        title="Cola de entrada/salida del staking (Ethereum)",
        tier=TIER_MECHANICAL,
        scope=SCOPE_CHAIN,
        cadence="daily",
        entity_kind=EntityKind.CHAIN,
        features=(
            SignalFeature("staking_exit_queue", AGG_LAST, "validadores",
                          "Oferta futura en cola."),
            SignalFeature("staking_entry_queue", AGG_LAST, "validadores",
                          "Demanda futura en cola."),
        ),
        pit=PIT_FORWARD_CAPTURE,
        license="beaconcha.in: ahora exige credencial (401 medido).",
        endpoint="https://beaconcha.in/api/v1/validators/queue",
        auth_env="BEACONCHAIN_API_KEY",
        fixed_entities=("ethereum",),
        notes="Oferta futura con FECHA CONOCIDA dias antes. Tres correcciones MEDIDAS el "
              "2026-08-12: (1) el endpoint devuelve 401 sin credencial, asi que la fuente "
              "declara auth_env y sin la variable en el entorno no hay medicion; (2) solo "
              "cubre Ethereum —Solana y Cosmos no tienen este endpoint— y por eso "
              "fixed_entities baja a una; (3) la espera en DIAS no se publica y no se "
              "estima: convertir la cola en dias exige el limite de churn vigente, que "
              "cambia con cada fork. Forward_capture puro: devuelve la foto de AHORA y no "
              "guarda la de ayer, el caso que mas penaliza retrasar la captura.",
    ),
    SignalSource(
        key="btc_difficulty",
        title="Ajuste de dificultad de Bitcoin (mempool.space)",
        tier=TIER_MECHANICAL,
        scope=SCOPE_CHAIN,
        cadence="event",
        entity_kind=EntityKind.CHAIN,
        features=(
            SignalFeature("difficulty_change_pct", AGG_LAST, "%", "Ajuste aplicado."),
        ),
        pit=PIT_CHAIN_IMMUTABLE,
        history_from=date(2014, 8, 19),  # MEDIDO: 321 retargets en la ventana de la sonda
        license="mempool.space: abierta, sin auth.",
        endpoint="https://mempool.space/api/v1/mining/difficulty-adjustments/all",
        fixed_entities=("bitcoin",),
        notes="LA HIPOTESIS DE 'DECENAS' ERA FALSA AQUI POR UN FACTOR DE DIEZ: 321 ajustes "
              "MEDIDOS en la ventana de la sonda (doce anos, 2014-08-19 en adelante), uno "
              "cada 2016 bloques. El limite lo pone PROBE_YEARS y no el proveedor: el "
              "endpoint devuelve los 463 retargets desde el bloque 0 (2009-01-03, "
              "COMPROBADO), y el catalogo declara lo medido, no lo disponible. Las "
              "cabeceras de bloque no se reescriben, de ahi el pit. El endpoint que se "
              "sondea no es el que estaba declarado: /difficulty-adjustment da la foto de "
              "AHORA y /mining/difficulty-adjustments/all da la serie entera en 21 KB. LIMITE "
              "DECLARADO: los dias-al-evento del pasado se cuentan contra la fecha en que "
              "el retarget OCURRIO, que aquel dia era una estimacion (bloques restantes x "
              "10 min) con error de horas. El HASHPRICE queda fuera: exige una pata en USD "
              "que esta capa no tiene, y publicarlo a medias seria peor que no publicarlo.",
    ),
    SignalSource(
        key="defillama_hacks",
        title="Hacks y exploits fechados (DefiLlama)",
        tier=TIER_MECHANICAL,
        scope=SCOPE_MARKET,
        cadence="event",
        entity_kind=EntityKind.MARKET,
        features=(
            SignalFeature("hack_amount_usd", AGG_SUM, "USD", "Importe comprometido ese dia."),
            SignalFeature("hack_count", AGG_SUM, "n", "Numero de incidentes."),
        ),
        pit=PIT_ARCHIVE_REVISABLE,
        history_from=date(2016, 6, 17),  # MEDIDO: 621 incidentes fechados
        license="DefiLlama: abierta, sin auth.",
        endpoint="https://api.llama.fi/hacks",
        notes="Era 'el caso mas duro del catalogo' y la sonda dice otra cosa: 621 "
              "incidentes MEDIDOS desde 2016-06-17, la segunda muestra mas grande de las "
              "seis de evento. El importe llega en MILLONES de dolares y el adaptador lo "
              "pasa a dolares, para que no se compare un 1,5 de aqui con un 1,5 de otra "
              "serie. El evento no se anuncia, asi que aqui los dias-al-evento no miran "
              "adelante: lo que codifica es la ventana POSTERIOR. Revisable: DefiLlama "
              "corrige importes y clasificaciones despues del incidente.",
    ),
    SignalSource(
        key="ofac_sdn",
        title="Lista OFAC SDN (direcciones sancionadas)",
        tier=TIER_MECHANICAL,
        scope=SCOPE_MARKET,
        cadence="event",
        entity_kind=EntityKind.MARKET,
        features=(
            SignalFeature("sdn_crypto_addresses", AGG_LAST, "n", "Direcciones en la lista."),
            SignalFeature("sdn_new_entries", AGG_SUM, "n", "Altas desde la captura anterior."),
        ),
        pit=PIT_FORWARD_CAPTURE,
        license="US Treasury: dominio publico.",
        endpoint="https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN.XML",
        notes="Se publica la lista de HOY, no la de una fecha pasada. Reconstruir el pasado "
              "exige haber guardado cada version: forward_capture, y por eso el archivo es "
              "append-only con fetched_at. MEDIDO: 28 MB de XML, 19.199 registros y 977 "
              "direcciones de cripto (529 XBT, 198 TRX, 100 ETH...). Se archiva SOLO el "
              "subconjunto cripto —la unica excepcion del catalogo a 'el payload intacto', "
              "razonada en adapters/events.py— y el certificado del host no lo valida el "
              "almacen de Windows, asi que esta fuente usa el bundle de certifi.",
    ),
    SignalSource(
        key="macro_calendar",
        title="Calendario macro (FOMC y publicaciones del IPC)",
        tier=TIER_MECHANICAL,
        scope=SCOPE_MACRO,
        cadence="event",
        entity_kind=EntityKind.MARKET,
        features=(
            SignalFeature("fomc_meeting", AGG_LAST, "0/1", "Dia de reunion del FOMC."),
            SignalFeature("cpi_release", AGG_LAST, "0/1", "Dia de publicacion del IPC."),
        ),
        pit=PIT_FORWARD_CAPTURE,
        history_from=date(2017, 2, 1),  # MEDIDO: la primera reunion del calendario de la Fed
        license="Fuentes oficiales (Fed, BLS): dominio publico.",
        endpoint="https://www.federalreserve.gov/json/calendar.json",
        notes="Determinista y conocido con meses de antelacion: las fechas son exactas y no "
              "revisables, que es la mejor propiedad que puede tener una feature de evento. "
              "De una reunion de dos dias se toma el SEGUNDO, que es cuando sale el "
              "comunicado. DOS PROFUNDIDADES DISTINTAS, como en GitHub: el calendario de la "
              "Fed llega a 2017 (MEDIDO), pero el del BLS es una ventana RODANTE de ~13 "
              "meses, asi que la historia larga del IPC solo se acumula captura a captura y "
              "un cpi_release vacio en 2019 no significa que no hubiera IPC. El BLS no "
              "publica esto en JSON: se archiva el HTML crudo para poder re-derivar.",
    ),
)


def _check_catalog(sources: tuple[SignalSource, ...]) -> None:
    keys = [s.key for s in sources]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Claves de fuente repetidas en el catalogo: {sorted(keys)}")

    seen: dict[str, str] = {}
    for source in sources:
        for name in source.feature_names:
            if name in seen:
                # Global y no por fuente: el dia que estas columnas se junten en un solo
                # vector de observacion, dos features homonimas se pisarian en silencio.
                raise ValueError(
                    f"Feature '{name}' declarada dos veces: '{seen[name]}' y '{source.key}'"
                )
            seen[name] = source.key


_check_catalog(CATALOG)

CATALOG_BY_KEY: dict[str, SignalSource] = {s.key: s for s in CATALOG}


def get_source(key: str) -> SignalSource:
    """La fuente con esa clave. KeyError con la lista completa si no existe."""
    try:
        return CATALOG_BY_KEY[key]
    except KeyError:
        raise KeyError(
            f"Fuente de senal desconocida: '{key}'. Catalogo: {sorted(CATALOG_BY_KEY)}"
        ) from None


def sources_by_tier(tier: str) -> tuple[SignalSource, ...]:
    if tier not in TIERS:
        raise ValueError(f"Tier desconocido: '{tier}'")
    return tuple(s for s in CATALOG if s.tier == tier)


def backtestable_sources() -> tuple[SignalSource, ...]:
    """Las que pueden entrar en un backtest HOY. Que hoy sea vacio es la medicion."""
    return tuple(s for s in CATALOG if s.backtestable)


def catalog_summary() -> dict:
    """Recuento del catalogo tal y como se publica en el dashboard y en la metodologia."""
    return {
        "n_sources": len(CATALOG),
        "n_features": sum(len(s.features) for s in CATALOG),
        "by_tier": {tier: len(sources_by_tier(tier)) for tier in TIERS},
        "by_scope": {scope: sum(1 for s in CATALOG if s.scope == scope) for scope in SCOPES},
        "by_pit": {kind: sum(1 for s in CATALOG if s.pit == kind) for kind in PIT_KINDS},
        "n_backtestable": len(backtestable_sources()),
        "n_forward_only": sum(1 for s in CATALOG if not s.backtestable),
        "n_open": sum(1 for s in CATALOG if s.auth_env is None),
    }


__all__ = [
    "CADENCES",
    "CADENCE_DAYS",
    "CATALOG",
    "CATALOG_BY_KEY",
    "PIT_ARCHIVE_REVISABLE",
    "PIT_DERIVED_FROM_PRICE",
    "PIT_FORWARD_CAPTURE",
    "PIT_KINDS",
    "SCOPES",
    "SCOPE_ASSET",
    "SCOPE_CHAIN",
    "SCOPE_MACRO",
    "SCOPE_MARKET",
    "SCOPE_VENUE",
    "TIERS",
    "TIER_MECHANICAL",
    "TIER_STATISTICAL",
    "SignalFeature",
    "SignalSource",
    "backtestable_sources",
    "catalog_summary",
    "get_source",
    "sources_by_tier",
]
