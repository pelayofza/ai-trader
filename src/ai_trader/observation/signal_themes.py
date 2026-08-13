"""
EL RADAR TEMATICO: los mismos treinta numeros, agrupados por la pregunta que contestan.

POR QUE HACE FALTA, SI YA HAY UN RADAR
--------------------------------------
`observation/signal_radar.py` colapsa las treinta fuentes en seis numeros por DOS bloques,
activo y mercado. Esa particion existe porque, sin temas, es la unica estructura disponible:
sobre treinta fuentes heterogeneas la unica pregunta contestable es "¿esto habla de mi
activo o del mundo?". Y basta mientras la senal sea una PUERTA que confirma o desmiente lo
que el precio ya dijo, que es todo lo que momentum y reversion le piden.

No basta en cuanto la senal quiere ser la TESIS. Cinco estrategias que lean `signal_tone`
leen el mismo numero: no son cinco apuestas, son una repetida cinco veces. Un mapa de
liquidaciones que dice "hay 200 M$ un 4% mas abajo" y un flujo de creaciones de ETF que
lleva tres semanas entrando contestan preguntas distintas, con horizontes distintos, y
promediarlos en un solo tono los cancela.

Asi que este modulo publica QUINCE numeros mas —una terna por tema— SIN tocar los seis de
siempre, que siguen saliendo del metodo de la clase base, ejecutando su mismo codigo.

POR QUE LA TABLA VIVE AQUI Y NO EN EL CATALOGO
---------------------------------------------
`signals/catalog.py` es una capa de DECLARACION: dice lo que cada fuente publica, con que
cadencia y desde cuando, y su docstring insiste en que "no conecta nada". Un tema no es una
propiedad de la fuente —Deribit no publica "soy del tema volatilidad"— sino una agrupacion
que hace el OBSERVADOR. Es exactamente el argumento por el que `POLARITY` vive en
`observation/signal_radar.py` y no en el catalogo, y merece el mismo sitio.

Hay ademas tres razones mecanicas: un campo nuevo en un dataclass frozen cambia `as_dict()`
y con el los goldens del catalogo, a cambio de nada; un `theme: str` forzaria UN tema por
fuente y `deribit_volatility` esta legitimamente en dos; y `synthetic/signal_channel.py`
construye `SignalSource` al vuelo para los canales simulados, a los que habria que exigirles
declarar un tema del catalogo real.

LA CLASIFICACION ES EXHAUSTIVA; LA PARTICION NO
-----------------------------------------------
Toda fuente del catalogo tiene que aparecer o en un tema o en `THEMELESS` con su motivo
escrito: anadir una fuente y no clasificarla FALLA al importar. Una fuente puede estar en
dos temas como maximo. Asi "no esta" es siempre una decision fechada y nunca un olvido, que
es la misma regla que ya gobierna `history_from`.

EL MINIMO DE FUENTES, Y POR QUE VA EN EL DENOMINADOR
----------------------------------------------------
`MIN_SIGNAL_COVERAGE = 0,25` se eligio para un bloque de catorce o dieciseis fuentes, donde
un cuarto son cuatro. Sobre un tema de cuatro, un cuarto es UNA, y una sola fuente pasando
por lectura del conjunto es literalmente lo que ese minimo existe para impedir.

La correccion NO es un segundo umbral. Es un SUELO del denominador: un tema pequeno se
compara contra un bloque hipotetico de seis fuentes equivalentes. Con eso sigue habiendo UN
solo umbral en todo el sistema, la puerta tematica se mide contra el mismo numero que la
global, y "hacen falta al menos dos fuentes" queda demostrado por aritmetica en vez de por
un `if` que alguien pueda relajar por separado.

El precio, que hay que saber: la cobertura de un tema pequeno nunca llega a 1,0
(`vol_surface` satura en 2/6 = 0,333). Es honesto en otro sentido: dice "de un bloque de
seis equivalentes, tengo dos".
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from ai_trader.observation.signal_radar import (
    MAX_STALE_DAYS,
    MIN_SIGNAL_COVERAGE,
    POLARITY,
    SIGNAL_FEATURES,
    SignalRadarProvider,
    _aggregate,
    _collapse,
    _Reading,
    gate_reason_for_blocks,
    is_market_scoped,
)
from ai_trader.shared.clock import Clock, visible_cutoff
from ai_trader.shared.entities import resolve_entity
from ai_trader.signals.catalog import CATALOG, SignalSource

# Cuantas fuentes CUBIERTAS hace falta que tenga un tema para que su puerta se evalue.
# Constante declarada, NO parametro, por el mismo motivo que `MIN_SIGNAL_COVERAGE`: si fuera
# sorteable, el optimizador podria subirla hasta convertir el radar en un filtro de
# disponibilidad de datos.
MIN_THEME_SOURCES = 2

# Tope de temas por fuente. Dos es "puede describir dos mecanismos"; tres ya es una fuente
# que no describe ninguno en particular.
MAX_THEMES_PER_SOURCE = 2

# Prefijo y sufijos de las claves que publica un tema.
_PREFIX = "signal_"
_SUFFIXES = ("tone", "intensity", "coverage")


def effective_denominator(n_declared: int, min_sources: int = MIN_THEME_SOURCES) -> int:
    """
    Denominador de la cobertura de un tema: el numero de fuentes declaradas, con un SUELO.

        suelo = (min_sources - 0.5) / MIN_SIGNAL_COVERAGE = 1,5 / 0,25 = 6

    La MEDIA fuente de mas no es un ajuste fino: coloca el suelo entre "una cubierta" y "dos
    cubiertas" en vez de justo encima de una division exacta, de modo que la comparacion
    `covered / denominador >= MIN_SIGNAL_COVERAGE` no dependa de como represente el hardware
    un cociente que cae exactamente en el umbral.
    """
    if min_sources < 1:
        raise ValueError("min_sources must be >= 1")
    floor = math.ceil((min_sources - 0.5) / MIN_SIGNAL_COVERAGE)
    return max(int(n_declared), floor)


@dataclass(frozen=True, slots=True)
class ThemeSpec:
    """Un tema: un nombre, las fuentes que lo componen y por que estan juntas."""

    name: str
    sources: tuple[str, ...]
    reason: str = ""
    min_sources: int = MIN_THEME_SOURCES

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("ThemeSpec needs a name")
        # Un tema SIN fuentes se admite aqui y se rechaza en `_check_themes`, que es donde se
        # valida la tabla real. El motivo es el panel sintetico vacio: "el mundo sin senales"
        # es el estado por defecto del sistema, no un caso degradado, y tiene que poder
        # construir su tabla de temas y publicar quince ceros como cualquier otro.
        if len(set(self.sources)) != len(self.sources):
            raise ValueError(f"Theme '{self.name}' repeats a source")
        if self.min_sources < 1:
            raise ValueError(f"Theme '{self.name}': min_sources must be >= 1")

    @property
    def features(self) -> tuple[str, str, str]:
        return theme_features(self.name)


def theme_features(theme: str) -> tuple[str, str, str]:
    """Las tres claves que publica un tema, en su orden canonico."""
    return tuple(f"{_PREFIX}{theme}_{suffix}" for suffix in _SUFFIXES)  # type: ignore[return-value]


# --- la tabla ------------------------------------------------------------------------
#
# Cada tema agrupa fuentes que contestan LA MISMA pregunta economica, no fuentes del mismo
# proveedor ni del mismo alcance. Por eso un tema mezcla libremente fuentes de activo y de
# mercado: el eje de entidades es un detalle de implementacion de cada proveedor
# (`p2p_premium` es de mercado porque Binance solo publica USDT, no porque la prima P2P sea
# un hecho de mercado y no de un activo).

THEMES: dict[str, ThemeSpec] = {
    "liquidation": ThemeSpec(
        "liquidation",
        (
            "funding_dispersion",
            "hyperliquid_leverage",
            "hyperliquid_liqmap",
            "deribit_volatility",
        ),
        reason=(
            "Donde esta el combustible del apalancamiento y cuanto queda. El mapa dice a que "
            "distancia de precio revienta cuanto notional, el apalancamiento dice cuanta "
            "fragilidad hay detras, el funding dice si los largos estan hacinados y DVOL es "
            "el termometro del mismo mercado que revienta. Es el unico tema cuyo contenido "
            "es un MECANISMO y no una correlacion."
        ),
    ),
    "vol_surface": ThemeSpec(
        "vol_surface",
        ("deribit_volatility", "deribit_expiries"),
        reason=(
            "Lo unico del catalogo que cotiza el FUTURO en vez de resumir el pasado. El skew "
            "es cuanto mas cara esta la proteccion que la apuesta; la estructura temporal, si "
            "la tension es de hoy o de dentro de un trimestre; el vencimiento, que dia "
            "concentra la gamma."
        ),
    ),
    "macro": ThemeSpec(
        "macro",
        (
            "fred_macro",
            "ofac_sdn",
            "macro_calendar",
            "sec_edgar_fts",
            "federal_register",
            "courtlistener_dockets",
        ),
        reason=(
            "Los dias que se saben con meses de antelacion y el flujo regulatorio y judicial "
            "que reprecia la clase de activo entera a la vez, sobre el fondo macro (DXY, tipos "
            "reales, VIX) contra el que se leen. AVISO ESTRUCTURAL: de sus seis fuentes solo "
            "`ofac_sdn` tiene polaridad declarada, asi que el TONO de este tema es ~0 por "
            "construccion y ademas las demas lo DILUYEN (entran como tono 0 en la media, no se "
            "saltan). Lo que aporta es CUANDO, no hacia donde: quien lo lea para decidir "
            "direccion esta leyendo ruido con nombre."
        ),
    ),
    "attention": ThemeSpec(
        "attention",
        (
            "guavy_sentiment",
            "wikipedia_pageviews",
            "p2p_premium",
            "appstore_rank",
            "naver_datalab",
            "yandex_wordstat",
            "cex_listings",
        ),
        reason=(
            "La demanda de ultimo recurso de cripto, que llega tarde, lenta e insensible al "
            "precio. Cuatro angulos que casi nadie mira juntos: el listado en Upbit (el evento "
            "mas limpio que existe), el diferencial de visibilidad Corea-EEUU en la App Store, "
            "las busquedas en Naver y Yandex, las visitas por idioma, y la prima P2P de quien "
            "compra por necesidad monetaria."
        ),
    ),
    "flow": ThemeSpec(
        "flow",
        (
            "etf_flows",
            "defillama_stablecoins",
            "defillama_fees",
            "defillama_volumes",
            "github_activity",
            "cftc_cot",
            "token_unlocks",
            "staking_queue",
            "btc_difficulty",
            "defillama_hacks",
            "lending_health",
            "dat_mnav",
        ),
        reason=(
            "Capital que entra y oferta que va a llegar, que es lo unico del catalogo con "
            "PERSISTENCIA: una creacion de ETF, una emision de stablecoins o una rotacion de "
            "fondos apalancados no se agota en un dia. Es tambien el tema con mejor materia "
            "prima: once de sus doce fuentes tienen polaridad declarada y ocho tienen historia "
            "medida, la mas antigua desde 2011."
        ),
    ),
}

# Fuentes del catalogo deliberadamente SIN tema, con el motivo. Hoy vacia: las treinta estan
# clasificadas. Existe para que anadir una fuente y no decidir su tema falle en vez de
# desaparecer en silencio de todos los bloques tematicos.
THEMELESS: tuple[tuple[str, str], ...] = ()

THEME_NAMES: tuple[str, ...] = tuple(THEMES)

# Orden canonico del bloque tematico: por tema en el orden de la tabla, y dentro de cada uno
# tono, intensidad y cobertura.
THEME_FEATURES: tuple[str, ...] = tuple(
    name for theme in THEME_NAMES for name in theme_features(theme)
)


def _check_themes(
    themes: Mapping[str, ThemeSpec],
    themeless: Sequence[tuple[str, str]],
    catalog: Sequence[SignalSource],
) -> None:
    """
    Las invariantes de la tabla, comprobadas al importar.

    Misma disciplina que `signals/catalog.py::_check_catalog`: si la tabla es incoherente el
    proceso no arranca, en vez de publicar unos numeros que nadie puede interpretar.
    """
    declared = {source.key for source in catalog}
    reserved = set(SIGNAL_FEATURES)

    for name, spec in themes.items():
        if name != spec.name:
            raise ValueError(f"Theme key '{name}' does not match its spec name '{spec.name}'")
        if not spec.sources:
            raise ValueError(f"Theme '{name}' declares no sources")
        # Un tema llamado 'market' generaria `signal_market_tone` y PISARIA el bloque de
        # mercado del radar base al fusionar los diccionarios.
        clash = reserved.intersection(theme_features(name))
        if clash:
            raise ValueError(
                f"Theme '{name}' would shadow the base radar features {sorted(clash)}"
            )
        unknown = [key for key in spec.sources if key not in declared]
        if unknown:
            raise ValueError(f"Theme '{name}' declares sources absent from the catalog: {unknown}")

    counts: dict[str, int] = {}
    for spec in themes.values():
        for key in spec.sources:
            counts[key] = counts.get(key, 0) + 1
    crowded = sorted(k for k, n in counts.items() if n > MAX_THEMES_PER_SOURCE)
    if crowded:
        raise ValueError(
            f"Sources in more than {MAX_THEMES_PER_SOURCE} themes: {crowded}"
        )

    excused = {key for key, _ in themeless}
    overlap = excused.intersection(counts)
    if overlap:
        raise ValueError(f"Sources both themed and excused: {sorted(overlap)}")
    unclassified = sorted(declared - set(counts) - excused)
    if unclassified:
        raise ValueError(
            "Every catalog source must be in a theme or listed in THEMELESS with a reason. "
            f"Unclassified: {unclassified}"
        )


_check_themes(THEMES, THEMELESS, CATALOG)


# --- la lectura de un tema -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThemeReading:
    """Lo que un tema dice hoy, y si se puede leer siquiera."""

    theme: str
    tone: float
    intensity: float
    coverage: float

    @property
    def readable(self) -> bool:
        """El MISMO umbral que gobierna todas las puertas del sistema."""
        return self.coverage >= MIN_SIGNAL_COVERAGE


def theme_reading(features: Mapping[str, float], theme: str) -> ThemeReading:
    """Extrae la terna de un tema de un diccionario de features del proveedor."""
    tone_key, intensity_key, coverage_key = theme_features(theme)
    return ThemeReading(
        theme=theme,
        tone=float(features.get(tone_key, 0.0)),
        intensity=float(features.get(intensity_key, 0.0)),
        coverage=float(features.get(coverage_key, 0.0)),
    )


def themed_gate_reason(
    features: Mapping[str, float],
    theme: str,
    *,
    min_tone: float,
    min_intensity: float | None = None,
    max_intensity: float | None = None,
) -> str | None:
    """
    Puerta de UN tema. Falla ABIERTA por debajo de `MIN_SIGNAL_COVERAGE`, igual que la
    global, y por el mismo cuerpo: `gate_reason_for_blocks` es el unico sitio donde ese
    invariante esta escrito.
    """
    return gate_reason_for_blocks(
        features,
        ((theme, theme_features(theme)),),
        min_tone=min_tone,
        min_intensity=min_intensity,
        max_intensity=max_intensity,
    )


# --- el proveedor -----------------------------------------------------------------------


class ThemedSignalRadarProvider(SignalRadarProvider):
    """
    El radar de siempre, MAS una terna por tema.

    Es una SUBCLASE y no una rama dentro de `SignalRadarProvider`, y la razon es que asi la
    invariancia de los seis numeros de siempre deja de ser algo que un test comprueba y pasa
    a ser algo que la estructura garantiza: `features()` llama a `super().features(symbol)`
    y fusiona. Los seis salen del metodo del padre, ejecutando su codigo, sin una sola rama
    nueva en el camino caliente de todo el sistema.

    Ademas hereda el trabajo caro del constructor —normalizar cada frame UNA vez y trocearlo
    por entidad—, que es justo lo que una implementacion alternativa (componer cinco
    proveedores filtrados, uno por tema) repetiria cinco veces.

    El orden de agregacion es el del CATALOGO, no el de la tupla del tema ni el de un `set`:
    el orden cambia los ultimos bits de una media en coma flotante, asi que se fija.
    """

    def __init__(
        self,
        frames: Mapping[str, pd.DataFrame] | None,
        clock: Clock,
        *,
        sources: Sequence[SignalSource] = CATALOG,
        max_stale_days: int = MAX_STALE_DAYS,
        polarity: Mapping[str, float] = POLARITY,
        themes: Mapping[str, ThemeSpec] = THEMES,
    ) -> None:
        super().__init__(
            frames, clock, sources=sources, max_stale_days=max_stale_days, polarity=polarity
        )
        self._themes = dict(themes)

        catalog_order = [source.key for source in sources]
        declared = set(catalog_order)
        market_keys = {source.key for source in sources if is_market_scoped(source)}
        loaded = set(self._asset_keys) | set(self._market_keys)

        self._theme_keys: dict[str, tuple[str, ...]] = {}
        self._theme_market: dict[str, frozenset[str]] = {}
        self._theme_denominator: dict[str, int] = {}
        for name, spec in self._themes.items():
            in_theme = set(spec.sources)
            # Denominador: lo que el CATALOGO declara en el tema, no lo que hoy tiene
            # fichero. Es la misma regla que `_n_declared` del padre y por el mismo motivo.
            self._theme_denominator[name] = effective_denominator(
                len(in_theme & declared), spec.min_sources
            )
            # Solo se recorren las fuentes CARGADAS: una fuente sin frame no produce lectura,
            # que es exactamente lo mismo que producir una no cubierta.
            self._theme_keys[name] = tuple(
                key for key in catalog_order if key in in_theme and key in loaded
            )
            self._theme_market[name] = frozenset(
                key for key in self._theme_keys[name] if key in market_keys
            )

        self._themed_asof: datetime | None = None
        self._themed_market_readings: dict[str, _Reading] = {}
        self._themed_by_symbol: dict[str, dict[str, float]] = {}

    # --- API ---------------------------------------------------------------------------

    def features(self, symbol: str) -> dict[str, float]:
        """Los seis de siempre MAS los quince tematicos. Los seis los produce el padre."""
        base = super().features(symbol)
        base.update(self._theme_block(symbol))
        return base

    def coverage_report(self) -> dict:
        report = super().coverage_report()
        report["themes"] = {
            name: {
                "declared": sorted(spec.sources),
                "loaded": sorted(self._theme_keys[name]),
                "denominator": self._theme_denominator[name],
                "min_sources": spec.min_sources,
            }
            for name, spec in sorted(self._themes.items())
        }
        report["min_theme_sources"] = MIN_THEME_SOURCES
        return report

    # --- calculo -----------------------------------------------------------------------

    def _refresh_themes(self, cutoff: pd.Timestamp) -> None:
        """El bloque de mercado de cada tema se calcula UNA vez por dia, como en el padre."""
        as_of = self._clock.now()
        if self._themed_asof == as_of:
            return
        self._themed_asof = as_of
        self._themed_by_symbol = {}
        shared = {key for keys in self._theme_market.values() for key in keys}
        self._themed_market_readings = {
            key: _collapse(self._readings_for_source(key, None, cutoff)) for key in sorted(shared)
        }

    def _theme_block(self, symbol: str) -> dict[str, float]:
        cutoff = visible_cutoff(self._clock.now())
        self._refresh_themes(cutoff)
        cached = self._themed_by_symbol.get(symbol)
        if cached is not None:
            return dict(cached)

        # Un simbolo que no resuelve a ninguna entidad no tiene lecturas de activo. Se
        # diferencia del padre —que devuelve el bloque entero a cero— en que las fuentes de
        # MERCADO del tema si aportan: son las mismas para todos y no dependen del simbolo.
        entity = resolve_entity(symbol)
        entity_key = entity.key if entity.resolved else None

        block: dict[str, float] = {}
        for name in self._themes:
            market = self._theme_market[name]
            readings: list[_Reading] = []
            for key in self._theme_keys[name]:
                if key in market:
                    readings.append(self._themed_market_readings[key])
                elif entity_key is not None:
                    readings.append(_collapse(self._readings_for_source(key, entity_key, cutoff)))
            block.update(
                zip(theme_features(name), _aggregate(readings, self._theme_denominator[name]))
            )

        self._themed_by_symbol[symbol] = block
        return dict(block)


__all__ = [
    "MAX_THEMES_PER_SOURCE",
    "MIN_THEME_SOURCES",
    "THEMELESS",
    "THEMES",
    "THEME_FEATURES",
    "THEME_NAMES",
    "ThemeReading",
    "ThemeSpec",
    "ThemedSignalRadarProvider",
    "effective_denominator",
    "theme_features",
    "theme_reading",
    "themed_gate_reason",
]
