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
respuesta era "ninguna", con el catalogo entero en `None`. Hoy, con las treinta
conectadas y sondeadas, son trece, y las demas siguen a `None` por motivos distintos que
conviene no confundir: la credencial no esta en el entorno (Guavy, FRED, beaconcha.in,
Naver, Yandex, los subgrafos de prestamos), el proveedor cerro el endpoint detras de un
muro de pago (unlocks de DefiLlama: 402 medido), o la fuente no tiene pasado que descargar
y su profundidad la compra el calendario (P2P, funding, OFAC, el estado de las cuentas de
Hyperliquid, el OI por strike de Deribit).

EL TERCER CAMPO, QUE LLEGA CON EL LOTE CARO: `typical_adv_usd`
--------------------------------------------------------------
Una senal puede ser genuina y no servir para nada, y la razon casi nunca es estadistica:
es que vive en activos donde no cabe tamano. Este campo declara el ADV TIPICO —la mediana
en dolares— de las entidades donde la senal existe, MEDIDO por `signals/liquidity.py` y
apuntado en `data/signals/entity_adv.json` con la misma disciplina que `history_from`: un
test falla si el catalogo declara una cifra que el registro no respalda.

Las dos cifras que salieron de la primera medicion son el motivo de que el campo exista: el
perp MEDIANO de Hyperliquid mueve 307.000 dolares al dia —el medio, no el pequeno, y de los
232 listados solo 177 negocian algo— y el mercado KRW mediano de Upbit, 248.000. Las dos
fuentes que cuelgan de ellos son buenas y no admiten tamano, y eso hay que saberlo ANTES de
escalar y no despues.

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

Con UNA excepcion declarada desde 2026-08-13, y declarada fuente a fuente en el campo
`encoding`: el MAPA DE PRECIOS (`price_map`). Un mapa de liquidacion se observa todos los
dias —luego no es un evento fechado— y lo que dice es una DISTANCIA EN PRECIO, no un nivel
comparable con su propia historia. Las dos codificaciones anteriores lo leerian mal de dos
maneras distintas y ninguna daria error, que es el peor de los casos. Son dos fuentes:
`hyperliquid_liqmap` y `lending_health`.

QUE SUSTITUYE A LA PUERTA, sin fingir que es equivalente: NINGUNA feature —de ningun tier—
entra en `scoring/search_space.py`, y los umbrales de las puertas son constantes declaradas
y razonadas en codigo, no parametros sorteables. Eso limita los grados de libertad, pero no
es un test.

EL TEST YA EXISTE (2026-08-12), y conviene saber que contesta y que no.
`research/signal_study.py` barre la capacidad predictiva de un canal de observacion
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
"""  # noqa: RUF002 - las notas citan titulos y unidades tal y como los publica cada fuente
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

# Como se codifica la fuente para el espacio de observacion. La regla sigue siendo la de
# siempre —la CADENCIA decide— y este campo es la EXCEPCION DECLARADA, no un sustituto:
# `encoding_kind` deriva de la cadencia salvo que la fuente escriba otra cosa a mano.
#
# La tercera codificacion entra con el mapa de liquidacion, y entra porque las otras dos no
# saben leerlo. Un mapa de liquidacion es una observacion DIARIA (luego no es un evento
# fechado: no tiene fecha futura que anticipar) cuyo contenido es una DISTANCIA EN PRECIO y
# un notional acumulado hasta ella. Con las dos z, la feature seria "¿es hoy la distancia
# alta PARA ESTE ACTIVO?", que no es la pregunta: la pregunta es si el cluster esta cerca,
# y "cerca" es un hecho absoluto en porcentaje de precio, no un percentil de su historia.
# Con la codificacion de evento, el `_ahead` mediria dias hasta una fecha que no existe.
ENCODING_CONTINUOUS = "continuous"
ENCODING_EVENT = "event"
ENCODING_PRICE_MAP = "price_map"
ENCODINGS: tuple[str, ...] = (ENCODING_CONTINUOUS, ENCODING_EVENT, ENCODING_PRICE_MAP)

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
    # Codificacion DECLARADA. None = la decide la cadencia, que es el caso de todas menos
    # las de mapa de precios. Ver `ENCODINGS`.
    encoding: str | None = None
    # ADV TIPICO (mediana, USD) de las entidades donde esta senal existe. MEDIDO por
    # `signals/liquidity.py`, que lo apunta en `data/signals/entity_adv.json`; un test falla
    # si alguien escribe aqui una cifra que el registro no respalda, igual que con
    # `history_from`. None = no aplica (fuentes de alcance mercado, cuyo eje no es un
    # activo) o no se ha podido medir; las dos cosas se distinguen en `adv_note`.
    typical_adv_usd: float | None = None
    adv_note: str = ""
    # RETRASO TIPICO DE PUBLICACION, en dias: cuanto tarda el dato en existir desde la fecha
    # a la que se refiere. MEDIDO, con la misma disciplina que los dos campos anteriores (el
    # registro lo escribe la propia fuente y un test compara). None = la fuente publica en
    # el dia y no hay desfase que declarar, que es el caso de casi todas; `lag_note` lo dice
    # para que None no signifique dos cosas.
    #
    # No es cosmetico y no lo cubre `pit`: `pit` dice si el pasado se puede descargar, y
    # esto dice si el dato de un dia ESTABA ese dia. Una fuente cuyo dato se publica cinco
    # semanas tarde y se fecha en su fecha de referencia mete cinco semanas de futuro en
    # cualquier backtest, y no da ningun error.
    disclosure_lag_days: float | None = None
    lag_note: str = ""
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
        if self.encoding is not None and self.encoding not in ENCODINGS:
            raise ValueError(f"Codificacion desconocida en '{self.key}': '{self.encoding}'")
        if self.typical_adv_usd is not None and self.typical_adv_usd <= 0:
            raise ValueError(f"ADV invalido en '{self.key}': {self.typical_adv_usd}")
        if self.disclosure_lag_days is not None and self.disclosure_lag_days < 0:
            raise ValueError(
                f"Retraso de publicacion negativo en '{self.key}': {self.disclosure_lag_days}"
            )
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
    def encoding_kind(self) -> str:
        """Como se codifica. La cadencia decide; el campo `encoding` es la excepcion.

        Que el default se DERIVE y no se escriba en cada fuente es lo que impide que
        anadir una fuente de evento y olvidarse del campo la mande en silencio a las dos z,
        que es exactamente el fallo que `events.py` existe para evitar."""
        if self.encoding is not None:
            return self.encoding
        return ENCODING_EVENT if self.cadence == "event" else ENCODING_CONTINUOUS

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
            "encoding": self.encoding_kind,
            "typical_adv_usd": self.typical_adv_usd,
            "adv_note": self.adv_note,
            "disclosure_lag_days": self.disclosure_lag_days,
            "lag_note": self.lag_note,
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
    # ================== Lote de alta friccion (2026-08-13) ==========================
    #
    # Lo que separa a estas doce de las diecisiete anteriores no es el precio —todas menos
    # tres son gratuitas y sin credencial— sino el TRABAJO: hay que reconstruir estado a
    # partir de posiciones sueltas, calcular deltas para encontrar un strike, parsear
    # titulos en coreano o pedir el mismo endpoint un dia detras de otro porque no acepta
    # rangos. Nadie las conecta porque da trabajo, no porque sean secretas.
    #
    # Y traen el campo que faltaba: `typical_adv_usd`. Varias son senales genuinas que
    # viven en activos donde no cabe tamano —la mediana de los perps que negocian en
    # Hyperliquid mueve 307.000 $ al dia y la de los 283 mercados KRW de Upbit, 248.000— y
    # esa cifra tiene que estar delante ANTES de que a alguien se le ocurra escalar.
    SignalSource(
        key="hyperliquid_leverage",
        title="Distribucion del apalancamiento y concentracion del OI (Hyperliquid)",
        tier=TIER_STATISTICAL,
        scope=SCOPE_ASSET,
        cadence="daily",
        entity_kind=EntityKind.TOKEN,
        features=(
            SignalFeature("hl_leverage_median", AGG_LAST, "x",
                          "Mediana del apalancamiento declarado, ponderada por posicion."),
            SignalFeature("hl_leverage_p90", AGG_LAST, "x", "Percentil 90 del apalancamiento."),
            SignalFeature("hl_high_leverage_share", AGG_LAST, "0-1",
                          "Fraccion del notional muestreado en posiciones de 10x o mas."),
            SignalFeature("hl_oi_top5_share", AGG_LAST, "0-1",
                          "Fraccion del notional muestreado que esta en 5 cuentas."),
            SignalFeature("hl_sampled_oi_usd", AGG_LAST, "USD", "Notional de la muestra."),
        ),
        pit=PIT_FORWARD_CAPTURE,
        license="Hyperliquid: abierta, sin auth ni KYC.",
        endpoint="https://api.hyperliquid.xyz/info",
        typical_adv_usd=307_370.0,
        adv_note="MEDIDO 2026-08-13: mediana de `dayNtlVlm` sobre los 177 perps que "
                 "NEGOCIAN (de 232 listados; los otros 55 no movieron nada en 24 horas). "
                 "Decil inferior 93.597 $, maximo 1.748 M$. La media enganaria por tres "
                 "ordenes de magnitud: la distribucion del apalancamiento existe en los "
                 "232, y el tamano solo cabe en la docena de arriba.",
        notes="LO QUE NINGUN CEX PERMITE: Binance publica el OI agregado; aqui el estado de "
              "cada cuenta es publico y la DISTRIBUCION se puede calcular. MEDIDO "
              "2026-08-13: el leaderboard trae 41.714 cuentas y la muestra de las 200 "
              "mayores cubre el 21,7% del OI del venue (1.578 M$ de 7.285 M$). Es una "
              "MUESTRA SESGADA por construccion —las cuentas grandes— y esa es justo la "
              "mitad que decide un gap, pero la palabra 'distribucion' aqui significa "
              "'distribucion en la cola de arriba' y no 'del libro entero'. Forward capture "
              "puro: nadie publica el estado de ayer, asi que cada dia sin capturar es "
              "profundidad perdida.",
    ),
    SignalSource(
        key="hyperliquid_liqmap",
        title="Mapa de precios de liquidacion (Hyperliquid)",
        tier=TIER_MECHANICAL,
        scope=SCOPE_ASSET,
        cadence="daily",
        entity_kind=EntityKind.TOKEN,
        features=(
            SignalFeature("liq_cluster_distance_pct", AGG_LAST, "%",
                          "Distancia al cluster mas cercano, con signo (- = por debajo)."),
            SignalFeature("liq_cluster_notional_usd", AGG_LAST, "USD",
                          "Notional acumulado entre el precio y ese cluster, incluido."),
        ),
        pit=PIT_FORWARD_CAPTURE,
        encoding=ENCODING_PRICE_MAP,
        license="Hyperliquid: abierta, sin auth ni KYC.",
        endpoint="https://api.hyperliquid.xyz/info",
        typical_adv_usd=307_370.0,
        adv_note="El mismo venue y la misma medicion que `hyperliquid_leverage`.",
        notes="LA TERCERA CODIFICACION DEL SISTEMA, y entra porque las otras dos no saben "
              "leer esto: un mapa de liquidacion no tiene fecha (no es un evento fechado) ni "
              "es un nivel comparable con su propia historia (una z de 'distancia al "
              "cluster' no responde a la pregunta). Lo que tiene es una DISTANCIA EN PRECIO "
              "y un notional, que es exactamente lo que codifica `events.py::PRICE_SPECS`. "
              "LIMITE MEDIDO 2026-08-13: solo el 75,7% de las posiciones muestreadas trae "
              "`liquidationPx`; en las de margen cruzado con holgura el campo llega nulo, "
              "asi que el mapa esta INCOMPLETO por abajo y el notional publicado es una "
              "cota inferior. No se estima el hueco: estimarlo exigiria replicar el motor "
              "de margen del venue.",
    ),
    SignalSource(
        key="deribit_volatility",
        title="Superficie de volatilidad: DVOL, skew de 25 delta y term structure (Deribit)",
        tier=TIER_STATISTICAL,
        scope=SCOPE_ASSET,
        cadence="daily",
        entity_kind=EntityKind.TOKEN,
        features=(
            SignalFeature("dvol_index", AGG_LAST, "vol %", "Indice DVOL de la moneda."),
            SignalFeature("skew_25d", AGG_LAST, "vol %",
                          "IV del put de 25 delta menos la del call de 25 delta."),
            SignalFeature("atm_iv_30d", AGG_LAST, "vol %", "IV at-the-money a ~30 dias."),
            SignalFeature("iv_term_slope", AGG_LAST, "vol %",
                          "IV ATM del vencimiento largo menos la del corto."),
        ),
        pit=PIT_ARCHIVE_REVISABLE,
        history_from=date(2021, 3, 24),  # MEDIDO: primer punto de DVOL en BTC y ETH
        license="Deribit: API publica, sin auth.",
        endpoint="https://www.deribit.com/api/v2/public/get_volatility_index_data",
        typical_adv_usd=100_937_354.0,
        adv_note="MEDIDO 2026-08-13: mediana del volumen 24h en USD del perpetuo de las DOS "
                 "monedas con libro de opciones (BTC 152,2 M$, ETH 49,7 M$). Aqui el ADV no "
                 "es la restriccion —es la unica fuente del lote que vive en los dos "
                 "activos mas profundos que existen— y por eso se declara: para que se vea "
                 "que la restriccion esta en otras.",
        notes="DOS PROFUNDIDADES EN LA MISMA FUENTE, como en `macro_calendar`. El DVOL tiene "
              "backfill REAL y medido —2021-03-24 en BTC y en ETH, con paginacion hacia "
              "atras porque la API devuelve 1.000 puntos por peticion— y el skew y la "
              "pendiente salen del LIBRO DE HOY, que no tiene historia: esas dos son "
              "forward_capture dentro de una fuente declarada revisable, y el "
              "`history_from` de arriba solo respalda al DVOL. MEDIDO ademas: solo BTC y "
              "ETH tienen libro de opciones vivo (820 y 688 instrumentos); SOL tuvo DVOL "
              "entre 2022-05 y 2022-11 y dejo de publicarse, y XRP nunca lo tuvo. EL DELTA "
              "SE CALCULA AQUI: el libro no publica griegas, asi que el strike de 25 delta "
              "se busca con un Black-Scholes de tipo cero sobre `mark_iv` y se interpola en "
              "delta. Es una APROXIMACION declarada (las opciones de Deribit son inversas), "
              "y se hace en la capa pura para que sea testeable sin red.",
    ),
    SignalSource(
        key="deribit_expiries",
        title="Calendario de vencimientos de opciones con OI por strike (Deribit)",
        tier=TIER_MECHANICAL,
        scope=SCOPE_ASSET,
        cadence="event",
        entity_kind=EntityKind.TOKEN,
        features=(
            SignalFeature("expiry_oi_usd", AGG_LAST, "USD",
                          "Notional de interes abierto que vence ese dia."),
            SignalFeature("expiry_oi_share", AGG_LAST, "0-1",
                          "Fraccion del interes abierto total que vence ese dia."),
        ),
        pit=PIT_FORWARD_CAPTURE,
        license="Deribit: API publica, sin auth.",
        endpoint="https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
        typical_adv_usd=100_937_354.0,
        adv_note="El mismo venue y la misma medicion que `deribit_volatility`.",
        notes="LA MEJOR PROPIEDAD QUE PUEDE TENER UNA FEATURE DE EVENTO: la fecha es fija y "
              "no la revisa nadie —el ultimo viernes del mes a las 08:00 UTC—, asi que los "
              "dias-al-evento son exactos y el calendario de hoy es el que estaba publicado "
              "hace un ano. MEDIDO 2026-08-13: 12 vencimientos vivos en BTC repartidos en "
              "820 instrumentos; el trimestral de septiembre concentra 108.894 contratos y "
              "el semanal siguiente 19.888. Forward capture: el OI por strike es la foto de "
              "HOY y nadie publica la de una fecha pasada, asi que la profundidad de la "
              "MAGNITUD se compra capturando. Las FECHAS futuras, en cambio, ya estan.",
    ),
    SignalSource(
        key="lending_health",
        title="Health factors y mapa de liquidacion del colateral (Aave / Compound)",
        tier=TIER_MECHANICAL,
        scope=SCOPE_ASSET,
        cadence="daily",
        entity_kind=EntityKind.TOKEN,
        features=(
            SignalFeature("lending_liq_distance_pct", AGG_LAST, "%",
                          "Caida del colateral que empieza a liquidar, con signo."),
            SignalFeature("lending_liq_notional_usd", AGG_LAST, "USD",
                          "Colateral liquidable hasta ese nivel."),
        ),
        pit=PIT_FORWARD_CAPTURE,
        encoding=ENCODING_PRICE_MAP,
        license="Subgrafos: la red descentralizada de The Graph exige clave (auth error medido).",
        endpoint="https://gateway.thegraph.com/api/subgraphs/id",
        auth_env="THEGRAPH_API_KEY",
        adv_note="Sin medicion: sin clave no hay respuesta y el eje de entidades no se "
                 "puede resolver. Declararlo por analogia con Hyperliquid seria inventarlo.",
        notes="EL OTRO LADO DEL APALANCAMIENTO: Hyperliquid dice donde revienta el "
              "apalancamiento del perpetuo y esto dice donde revienta el del colateral "
              "SPOT, que es el que fuerza ventas de verdad en el mercado al contado. Se "
              "codifica igual —mapa de precios— a proposito: son la misma clase de objeto. "
              "MEDIDO 2026-08-13, y el resultado es que hoy NO HAY VIA GRATUITA: "
              "api.thegraph.com ya no resuelve (el servicio alojado se retiro), el gateway "
              "descentralizado responde 'auth error: missing authorization header', la API "
              "de cuentas de Compound v2 devuelve 410 Gone y los datasets de liquidaciones "
              "de DefiLlama, 404. El adaptador se escribe igualmente por lo mismo que el de "
              "unlocks: para que eso sea una MEDICION fechada y no un hueco que se lea como "
              "'nadie lo ha intentado'.",
    ),
    SignalSource(
        key="appstore_rank",
        title="Atencion retail por geografia: ranking en App Store (Corea frente a EE.UU.)",
        tier=TIER_STATISTICAL,
        scope=SCOPE_MARKET,
        cadence="event",
        entity_kind=EntityKind.MARKET,
        features=(
            SignalFeature("app_visibility_kr", AGG_LAST, "0-1",
                          "Visibilidad de las apps cripto en la lista coreana."),
            SignalFeature("app_visibility_us", AGG_LAST, "0-1",
                          "Visibilidad de las apps cripto en la lista estadounidense."),
            SignalFeature("app_visibility_gap", AGG_LAST, "-1..1",
                          "Corea menos EE.UU.: hacia donde esta entrando el retail."),
        ),
        pit=PIT_FORWARD_CAPTURE,
        license="Apple RSS (marketing tools): abierta, sin auth.",
        endpoint="https://rss.applemarketingtools.com/api/v2/kr/apps/top-free/100/apps.json",
        adv_note="Alcance de mercado: el eje no es un activo, asi que el ADV que limita es "
                 "el del universo que se opere y no uno de la fuente.",
        notes="LA VERSION BUENA ES EL DIFERENCIAL, no el ranking de Coinbase en EE.UU.: la "
              "señal es que Corea se caliente ANTES, y el nivel absoluto de cada lista lo "
              "domina lo que no tiene nada que ver con cripto. DOS LIMITES MEDIDOS "
              "2026-08-13 que cambian el diseno: (1) la lista es de CIEN puestos y no mas "
              "—pedir 200 devuelve 500— y el parametro de genero se IGNORA, asi que la "
              "unica lista disponible es la general; (2) ese dia NINGUNA de las cuatro apps "
              "(Upbit, Coinbase, Binance, Bitget) estaba en el top 100 de Corea ni en el de "
              "EE.UU. Una serie continua de ceros no la puede normalizar `normalize.py` "
              "—mediana 0, IQR 0, z indefinida— asi que la fuente se declara de EVENTO: el "
              "evento es 'una app cripto entra en la lista general', que es exactamente el "
              "hecho que interesa, y los dias en que no entra ninguna no producen fila.",
    ),
    SignalSource(
        key="naver_datalab",
        title="Interes de busqueda en Corea (Naver DataLab)",
        tier=TIER_STATISTICAL,
        scope=SCOPE_MARKET,
        cadence="daily",
        entity_kind=EntityKind.MARKET,
        features=(
            SignalFeature("naver_search_index", AGG_LAST, "0-100",
                          "Indice relativo de busquedas cripto en Naver."),
        ),
        pit=PIT_ARCHIVE_REVISABLE,
        license="Naver Developers: gratuito, exige client id y secret (401 medido).",
        endpoint="https://openapi.naver.com/v1/datalab/search",
        auth_env="NAVER_CLIENT_ID",
        adv_note="Alcance de mercado: igual que `appstore_rank`.",
        notes="Naver es el buscador de Corea y Corea pesa de forma desproporcionada en las "
              "altcoins; el dato es gratuito y practicamente nadie lo mira fuera del pais. "
              "MEDIDO 2026-08-13: 401 sin credencial. Hacen falta DOS variables "
              "(NAVER_CLIENT_ID y NAVER_CLIENT_SECRET) y el catalogo solo declara el nombre "
              "de una porque el campo es uno; el adaptador lee las dos y lo dice si falta "
              "cualquiera. La API devuelve un indice RELATIVO a la ventana pedida: dos "
              "ventanas distintas no son comparables entre si, y por eso se pide siempre la "
              "misma longitud de ventana y la fuente se declara revisable.",
    ),
    SignalSource(
        key="yandex_wordstat",
        title="Interes de busqueda en Rusia (Yandex Wordstat)",
        tier=TIER_STATISTICAL,
        scope=SCOPE_MARKET,
        cadence="monthly",
        entity_kind=EntityKind.MARKET,
        features=(
            SignalFeature("yandex_search_shows", AGG_LAST, "impresiones",
                          "Impresiones mensuales de las frases cripto en Yandex."),
        ),
        pit=PIT_FORWARD_CAPTURE,
        license="Yandex Direct API: gratuito con cuenta de anunciante, exige token OAuth.",
        endpoint="https://api.direct.yandex.ru/v4/json/",
        auth_env="YANDEX_OAUTH_TOKEN",
        adv_note="Alcance de mercado: igual que `appstore_rank`.",
        notes="MEDIDO 2026-08-13: sin token la API contesta con error propio (codigo 501) en "
              "vez de 401, que es su forma de decir que no hay sesion. Wordstat es MENSUAL "
              "—no publica serie diaria— y ademas es una foto: el informe se genera contra "
              "el momento en que se pide, asi que su pasado no se descarga. Es la fuente "
              "menos util de las tres de atencion y esta aqui porque el hueco geografico "
              "que cubre no lo cubre ninguna otra.",
    ),
    SignalSource(
        key="sec_edgar_fts",
        title="Menciones en filings de la SEC (EDGAR full-text search)",
        tier=TIER_MECHANICAL,
        scope=SCOPE_MARKET,
        cadence="event",
        entity_kind=EntityKind.MARKET,
        features=(
            SignalFeature("edgar_filings", AGG_SUM, "n",
                          "Filings con la consulta cripto ese dia."),
            SignalFeature("edgar_institutional", AGG_SUM, "n",
                          "De esos, los de tenencia institucional (13F/13G)."),
        ),
        pit=PIT_ARCHIVE_REVISABLE,
        history_from=date(2025, 6, 20),  # MEDIDO: los 420 dias que compra una pasada
        license="SEC: dominio publico. Exige User-Agent identificable.",
        endpoint="https://efts.sec.gov/LATEST/search-index",
        adv_note="Alcance de mercado: igual que `appstore_rank`.",
        notes="LLEGA ANTES QUE LA NOTICIA porque ES la noticia: un 8-K esta publicado en "
              "EDGAR horas antes de que nadie lo escriba. MEDIDO 2026-08-13: la API "
              "responde sin credencial, cubre desde 2001 (consultas de 2001, 2005, 2011 y "
              "2014 devuelven resultados) y hay 5.784 filings 13F-HR que mencionan bitcoin "
              "en lo que va de 2026. DOS LIMITES MEDIDOS: la respuesta trae como mucho ~100 "
              "hits y el total se corta en 10.000, asi que el recuento DIARIO se pide un dia "
              "por peticion; y por eso el adaptador declara un tope de dias por pasada "
              "(EDGAR_MAX_DAYS) en vez de fingir que un backfill de doce anos cabe en una "
              "llamada. Revisable: un filing se puede enmendar y reindexar. Y por eso el "
              "history_from de arriba es 2025-06-20 y no 2001: es lo que UNA pasada ha "
              "medido, no lo que el proveedor tiene. Cada pasada nueva lo empuja hacia "
              "atras otros catorce meses, y el registro de profundidad lo ira diciendo.",
    ),
    SignalSource(
        key="federal_register",
        title="Actividad regulatoria fechada (Federal Register API)",
        tier=TIER_MECHANICAL,
        scope=SCOPE_MACRO,
        cadence="event",
        entity_kind=EntityKind.MARKET,
        features=(
            SignalFeature("fedreg_documents", AGG_SUM, "n",
                          "Documentos publicados ese dia con la consulta cripto."),
            SignalFeature("fedreg_rules", AGG_SUM, "n",
                          "De esos, los que son norma o propuesta de norma."),
        ),
        pit=PIT_ARCHIVE_REVISABLE,
        history_from=date(2015, 11, 2),  # MEDIDO: primer documento dentro de la ventana sondeada
        license="Federal Register: dominio publico, sin auth.",
        endpoint="https://www.federalregister.gov/api/v1/documents.json",
        adv_note="Alcance de mercado: igual que `appstore_rank`.",
        notes="La fuente mas comoda de las tres legales: acepta rangos de fecha, devuelve "
              "hasta 1.000 documentos por peticion y no pide credencial. DOS MEDICIONES "
              "2026-08-13 que conviene no confundir: una consulta suelta sin limite de "
              "fecha encuentra 398 documentos con 'digital asset' desde 2010-01-11 (y el "
              "archivo entero del Federal Register llega a 1994), pero la SONDA, que pide "
              "los doce anos que declara PROBE_YEARS, encontro el primero en 2015-11-02. "
              "El catalogo declara lo segundo: lo que el adaptador REAL trajo, no lo que "
              "el proveedor tiene. Se fecha por `publication_date` —el dia en que el "
              "EXISTE— y no por `effective_on`, que suele ser posterior y a veces nulo. Las "
              "fechas futuras que el propio documento trae (fin del plazo de comentarios, "
              "entrada en vigor) quedan en el payload crudo para el dia que se quieran "
              "usar: hoy no se anticipan, asi que la codificacion no mira adelante.",
    ),
    SignalSource(
        key="courtlistener_dockets",
        title="Litigios y dockets federales (CourtListener / RECAP)",
        tier=TIER_MECHANICAL,
        scope=SCOPE_MARKET,
        cadence="event",
        entity_kind=EntityKind.MARKET,
        features=(
            SignalFeature("court_dockets", AGG_SUM, "n",
                          "Dockets nuevos ese dia con la consulta cripto."),
        ),
        pit=PIT_ARCHIVE_REVISABLE,
        license="Free Law Project: API publica; v4 responde sin token (v3 da 403).",
        endpoint="https://www.courtlistener.com/api/rest/v4/search/",
        adv_note="Alcance de mercado: igual que `appstore_rank`.",
        notes="MEDIDO 2026-08-13: la version 4 de la API contesta SIN token (9.569 dockets "
              "con 'cryptocurrency') y la version 3 devuelve 403, que es al reves de lo que "
              "documenta media internet. Pagina de veinte en veinte, asi que el adaptador "
              "declara un tope de paginas por pasada: lo que hay dentro de la ventana "
              "cubierta es exacto y lo que queda fuera se declara, en vez de publicar un "
              "recuento truncado como si fuera completo.",
    ),
    SignalSource(
        key="cex_listings",
        title="Listados y deslistados de CEX (Upbit)",
        tier=TIER_MECHANICAL,
        scope=SCOPE_ASSET,
        cadence="event",
        entity_kind=EntityKind.TOKEN,
        features=(
            SignalFeature("listing_change", AGG_SUM, "+1/-1",
                          "Alta (+1) o baja (-1) de negociacion en el venue."),
            SignalFeature("listing_warning", AGG_SUM, "n",
                          "Designaciones de 'valor de inversion en vigilancia'."),
        ),
        pit=PIT_ARCHIVE_REVISABLE,
        history_from=date(2018, 8, 2),  # MEDIDO: 523 eventos sobre 343 tokens, 33 paginas
        license="Upbit: endpoint publico de anuncios, sin auth.",
        endpoint="https://api-manager.upbit.com/api/v1/announcements",
        typical_adv_usd=247_645.0,
        adv_note="MEDIDO 2026-08-13: mediana del volumen 24h de los 283 mercados KRW, con "
                 "el cambio KRW/USD sacado del propio venue (p10 = 39.450 $, maximo = "
                 "99,0 M$). El 'efecto Upbit' es real y vive en activos cuya mediana mueve "
                 "un cuarto de millon al dia: el evento es limpio y el tamano que admite, "
                 "no.",
        notes="EL EVENTO MAS LIMPIO DE CRIPTO, y ademas el unico del lote que pide una "
              "guarda OPERATIVA y no solo una feature: un deslistado es oferta forzada mas "
              "riesgo de liquidez, y eso no se arregla con una feature (ver la evolucion "
              "'Guarda operativa por simbolo', que sigue pendiente a proposito). MEDIDO "
              "2026-08-13: el endpoint tiene 38 paginas de categoria 'trade' que llegan a "
              "2017-10-27, y la sonda bajo 33 de ellas —523 eventos fechados sobre 343 "
              "tokens desde 2018-08-02— antes de topar con el limite de tasa; el catalogo "
              "declara ESO y no el fondo del archivo. Los tres tipos que importan estan: "
              "alta, baja y designacion de vigilancia previa a la baja. SE FECHA EL "
              "ANUNCIO, no la ejecucion, por lo "
              "mismo que el COT se fecha el dia de publicacion: el anuncio es cuando la "
              "informacion existe. Y por eso `announced=False`: un listado no se preanuncia, "
              "asi que mirar hacia adelante aqui seria futuro. LIMITE DECLARADO: el "
              "endpoint de listado solo da el TITULO, y la fecha de ejecucion de una baja "
              "vive en el cuerpo del anuncio; leerla exigiria una peticion por anuncio.",
    ),
    # ============ Tesorerias cotizadas: la fuente COMPUESTA (2026-08-13) =============
    #
    # La primera del catalogo que no tiene proveedor: no existe API de mNAV —los cuatro
    # sitios que lo publican son cuadros de mando— y la serie se compone de tres patas
    # (tenencias y acciones del XBRL de la SEC, precio de la accion y del activo del mismo
    # cierre de sesion). Esa friccion es lo que la mantiene sin arbitrar, y es tambien lo
    # que obliga a declarar aqui el campo nuevo: `disclosure_lag_days`.
    SignalSource(
        key="dat_mnav",
        title="Estres de vendedores forzados: distribucion del mNAV de las tesorerias cotizadas",
        tier=TIER_MECHANICAL,
        scope=SCOPE_ASSET,
        cadence="event",
        entity_kind=EntityKind.TOKEN,
        features=(
            SignalFeature("dat_below_nav_share", AGG_LAST, "0-1",
                          "Fraccion de la cohorte que cotiza por debajo de su tesoro."),
            SignalFeature("dat_mnav_gap", AGG_LAST, "x",
                          "Mediana del mNAV menos 1: distancia del activo a la frontera."),
            SignalFeature("dat_mnav_p25", AGG_LAST, "x",
                          "Cuartil inferior del mNAV: donde vive la cola que engorda."),
            SignalFeature("dat_companies", AGG_LAST, "n",
                          "Companias de la cohorte ese dia (la muestra, en la propia fila)."),
            SignalFeature("dat_disclosure_lag_days", AGG_LAST, "dias",
                          "Retraso REALIZADO de las tenencias que sostienen la fila."),
        ),
        pit=PIT_ARCHIVE_REVISABLE,
        license="SEC: dominio publico, exige User-Agent identificable. Precios: endpoint publico.",
        endpoint="https://data.sec.gov/api/xbrl/frames/us-gaap/CryptoAssetFairValue/USD",
        adv_note="El eje es el activo subyacente y son los mas profundos que existen, asi que "
                 "aqui el ADV no restringe. Lo que restringe es el otro lado del cociente "
                 "—la accion— y ese venue no lo mide `signals/liquidity.py`, cuyas tres "
                 "sondas son de cripto: declararlo por analogia seria inventarlo.",
        disclosure_lag_days=49.0,
        lag_note="MEDIDO 2026-08-13 sobre las observaciones de la cohorte: mediana de "
                 "`filed - end`, o sea cuanto tarda una tenencia en existir para el publico "
                 "desde la fecha a la que se refiere. Coherente con el plazo que fija la SEC "
                 "para un 10-Q (40 dias para un declarante acelerado grande, 45 para el "
                 "resto), que es el suelo estructural. El registro esta en "
                 "data/signals/dat_cohort.json y un test compara las dos cifras, igual que "
                 "con history_from y typical_adv_usd. Lo que PROTEGE del look-ahead no es "
                 "este campo sino que la fila se fecha en `filed`; el campo es lo que hace "
                 "que la magnitud del desfase se pueda citar.",
        notes="LO QUE PUBLICA NO ES EL mNAV DE NADIE: es la DISTRIBUCION por activo "
              "subyacente. Por debajo de 1 la maquina va en reversa y es aritmetica, no "
              "sentimiento —emitir diluye, asi que la via barata para levantar caja es "
              "vender el tesoro— y cada companıa en esa cola es oferta futura estructural. "
              "TRES FILTROS Y NINGUNO MIRA EL mNAV, porque definir la cohorte con el propio "
              "mNAV truncaria justo la cola que se publica: fuera los SIC 6221 (ETF y trusts "
              "al contado, que cotizan al NAV por arbitraje) y 6211 (brokers que custodian "
              "cripto ajena); dentro solo si el tesoro pasa del 50% del ACTIVO TOTAL del "
              "balance; y solo si se sabe que activo es. "
              "EL RESULTADO DE COMPONERLA, MEDIDO 2026-08-13, Y NO ES EL QUE SE ESPERABA: de "
              "138 declarantes de cripto en el registro XBRL quedan TRES companias "
              "publicables, repartidas en tres activos distintos, asi que HOY NO HAY NINGUNA "
              "DISTRIBUCION QUE PUBLICAR y la fuente produce cero filas. El desglose es la "
              "medicion: 25 trusts y 3 brokers fuera por SIC, 61 que declaran el valor "
              "razonable en dolares y NO el numero de unidades (sin unidades no se puede "
              "marcar a mercado ni identificar el activo), 26 cuyo tesoro no llega al 50% "
              "del balance —mineras, sobre todo—, 6 multiclase, 12 que no nombran su activo "
              "y 4 cuyo activo no se opera. "
              "LA IDENTIFICACION ES LO QUE MAS CUESTA, y la primera version estaba mal: "
              "identificar por el PRECIO IMPLICITO (valor razonable / unidades) y quedarse "
              "con el unico activo del universo que cuadrase daba DOS falsos positivos de "
              "ocho (TON Strategy Co salia NEAR, Hyperion DeFi salia LTC), porque 'el unico "
              "que cuadra' solo significa algo con el conjunto de candidatos completo y hay "
              "miles de tokens. Hoy IDENTIFICA UN NOMBRE —la etiqueta de unidad o la razon "
              "social— y el precio implicito solo VERIFICA, que es lo que atrapa a CleanSpark "
              "declarando 1.719.000 unidades `Bitcoin` a 58,53 $ (error de escala de mil en "
              "el propio filing) y a Bit Digital con un valor razonable que cubre toda su "
              "cartera. Se paga en cobertura y esta contado. "
              "HUECO DECLARADO, y es el mayor: las APIs XBRL solo exponen hechos SIN "
              "DIMENSIONES, asi que una companıa con varias clases de accion no publica "
              "recuento de ordinarias y LA TESORERIA MAS GRANDE QUE EXISTE queda fuera "
              "(COMPROBADO: el `companyfacts` de Strategy tiene un solo tag `dei`, "
              "EntityPublicFloat). No se sustituye por la media ponderada del periodo, que si "
              "esta: es una MEDIA, y estas companias emiten contra el mercado todos los dias, "
              "asi que sesgaria a la baja la capitalizacion justo de las mas activas y las "
              "empujaria hacia la cola inferior. history_from sigue a None porque no hay "
              "ninguna fila que fechar, y eso es la medicion y no un pendiente.",
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


def sources_by_encoding(kind: str) -> tuple[SignalSource, ...]:
    if kind not in ENCODINGS:
        raise ValueError(f"Codificacion desconocida: '{kind}'")
    return tuple(s for s in CATALOG if s.encoding_kind == kind)


def catalog_summary() -> dict:
    """Recuento del catalogo tal y como se publica en el dashboard y en la metodologia."""
    with_adv = [s.typical_adv_usd for s in CATALOG if s.typical_adv_usd]
    return {
        "n_sources": len(CATALOG),
        "n_features": sum(len(s.features) for s in CATALOG),
        "by_tier": {tier: len(sources_by_tier(tier)) for tier in TIERS},
        "by_scope": {scope: sum(1 for s in CATALOG if s.scope == scope) for scope in SCOPES},
        "by_pit": {kind: sum(1 for s in CATALOG if s.pit == kind) for kind in PIT_KINDS},
        "by_encoding": {kind: len(sources_by_encoding(kind)) for kind in ENCODINGS},
        "n_backtestable": len(backtestable_sources()),
        "n_forward_only": sum(1 for s in CATALOG if not s.backtestable),
        "n_open": sum(1 for s in CATALOG if s.auth_env is None),
        # El ADV no se resume con una media: la pregunta que contesta es "¿cabe tamano en
        # las entidades donde esto existe?", y la respuesta util es la mas pequena.
        "n_with_adv": len(with_adv),
        "min_adv_usd": min(with_adv) if with_adv else None,
    }


__all__ = [
    "CADENCES",
    "CADENCE_DAYS",
    "CATALOG",
    "CATALOG_BY_KEY",
    "ENCODINGS",
    "ENCODING_CONTINUOUS",
    "ENCODING_EVENT",
    "ENCODING_PRICE_MAP",
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
    "sources_by_encoding",
    "sources_by_tier",
]
