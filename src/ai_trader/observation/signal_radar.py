"""
EL RADAR DE SENALES: diecisiete fuentes -> seis numeros que una decision puede usar.

QUE ES ESTO Y QUE NO
--------------------
Es el gemelo de `observation/regime.py`: misma forma (`features(symbol) -> dict[str,
float]`), mismo memo por el 'ahora' del reloj, mismo anti look-ahead importado de
`shared/clock.py`, y la misma costura —colaborador OPCIONAL inyectado en las primitivas—.
Que la forma sea identica no es estetica: es lo que hace que el motor de backtest y el
proceso en vivo lo enganchen con el MISMO bucle duck-typed que ya usan para el regimen, sin
una linea de caso especial.

Lo que NO es: un veto. Ninguna senal de este modulo impide operar nada. Todo lo que sale de
aqui es una feature; lo que una estrategia haga con ella es cosa suya, y lo que ninguna
puede hacer es bloquear por falta de datos (ver `signal_gate_reason`).

SEIS NUMEROS, Y POR QUE SEIS
----------------------------
Cuarenta y cinco columnas crudas no son un espacio de observacion: son cuarenta y cinco
grados de libertad esperando a que alguien los pondere. El radar las reduce a tres ejes,
por dos BLOQUES que se mantienen separados EN CODIGO:

    signal_tone / signal_intensity / signal_coverage                  -> del ACTIVO
    signal_market_tone / signal_market_intensity / signal_market_coverage -> del MERCADO

- TONO: hacia donde apunta lo que se sabe. Positivo = flujo, atencion o actividad a favor;
  negativo = oferta entrando, dinero saliendo, algo roto. Solo suman las features con
  POLARIDAD declarada (`POLARITY`), y la declaracion esta razonada una por una.
- INTENSIDAD: cuanto esta pasando, sin signo. Es el eje que momentum y mean-reversion
  quieren en direcciones OPUESTAS, y el unico que lo es.
- COBERTURA: que fraccion de las fuentes del bloque tiene dato hoy. Ver mas abajo.

La separacion mercado/activo es un INVARIANTE TESTEABLE —dos simbolos distintos tienen el
mismo bloque de mercado el mismo dia— y no un comentario. Ademas es lo que el generador
sintetico va a necesitar el dia que simule el canal de observacion: un factor comun y un
componente idiosincratico no se simulan igual.

LA TRAMPA DE LOS TRES ESTADOS
-----------------------------
La convencion del espacio de observacion es "feature no disponible = 0.0 neutro"
(`observation/features.py`). Para el tono, 0 es neutral y esta bien. Pero "no tengo datos"
NO es "no hay senal", y en una serie de EVENTOS las dos cosas se escriben exactamente
igual: `unlock_ahead = 0` significa "no hay desbloqueo a la vista" y tambien "de este token
no se nada". `signal_coverage` es lo que las distingue, y con ella el invariante central de
esta evolucion:

    UNA PUERTA DE SENALES NUNCA BLOQUEA POR FALTA DE DATOS.

Por debajo de `MIN_SIGNAL_COVERAGE` la puerta NO SE EVALUA —falla abierta— en vez de leer
un cero de ausencia como si fuera una lectura neutra. Y ese umbral es una CONSTANTE, no un
parametro: si fuera sorteable, el optimizador podria subirlo hasta convertir el radar en un
filtro de disponibilidad de datos, que rankearia por que fuentes tenian cobertura en cada
tramo de historia en vez de por lo que dicen.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from ai_trader.shared.clock import Clock, visible_cutoff
from ai_trader.shared.entities import resolve_entity
from ai_trader.shared.signals import DAY, ENTITY
from ai_trader.signals.catalog import CATALOG, SignalSource
from ai_trader.signals.events import (
    EVENT_SPECS,
    MAGNITUDE_CLIP,
    SUFFIX_ACTIVE,
    SUFFIX_AHEAD,
    SUFFIX_MAG,
    EventSpec,
    as_naive,
    day_values,
    encode_arrays,
    is_event_source,
)
from ai_trader.signals.normalize import SUFFIX_CROSS, SUFFIX_SELF, Z_CLIP, normalize_features

logger = logging.getLogger(__name__)

# Orden canonico del bloque de senales del vector de observacion. Los tres primeros son del
# ACTIVO; los tres ultimos, del MERCADO (iguales para todos los simbolos ese dia).
ASSET_SIGNAL_FEATURES: tuple[str, ...] = (
    "signal_tone",
    "signal_intensity",
    "signal_coverage",
)
MARKET_SIGNAL_FEATURES: tuple[str, ...] = (
    "signal_market_tone",
    "signal_market_intensity",
    "signal_market_coverage",
)
SIGNAL_FEATURES: tuple[str, ...] = ASSET_SIGNAL_FEATURES + MARKET_SIGNAL_FEATURES

# Cobertura minima para que una puerta de senales se EVALUE. Constante declarada, NO
# parametro: ver el docstring. Un cuarto de las fuentes del bloque es poco exigente a
# proposito —el objetivo no es garantizar una lectura buena, es impedir que una lectura de
# UNA sola fuente pase por lectura del conjunto— y sobre todo es un numero que no depende
# de ningun resultado.
MIN_SIGNAL_COVERAGE = 0.25

# Cuantos dias puede tener una observacion continua antes de dejar de contar como cobertura.
# Sin esto, una fuente que dejo de publicar en 2019 seguiria "cubriendo" en 2026 con su
# ultimo valor congelado, que es la forma mas silenciosa de que la cobertura mienta.
MAX_STALE_DAYS = 14

# --- los valores INERTES de las puertas ----------------------------------------------
#
# El tono vive en [-Z_CLIP, +Z_CLIP] y la intensidad en [0, Z_CLIP], y no por convenio: las
# z estan recortadas (`normalize.Z_CLIP`), las magnitudes de evento tambien
# (`events.MAGNITUDE_CLIP`, el mismo numero) y una media de valores dentro de un intervalo
# se queda dentro del intervalo. Asi que un umbral puesto EN EL BORDE EXACTO no puede
# bloquear nada: la inercia del default queda DEMOSTRADA por el rango, no confiada.
#
# Es la leccion de `min_relative_strength = -1.0`, que se documento como "sin filtro" y no
# lo era: la fuerza relativa es un log-retorno diferencial sin acotar y un -1.0 es
# alcanzable, asi que aquel default SI podia bloquear.
INERT_MIN_TONE = -Z_CLIP
INERT_MIN_INTENSITY = 0.0
INERT_MAX_INTENSITY = Z_CLIP


# --- polaridad declarada -------------------------------------------------------------
#
# feature -> signo con el que entra en el TONO. +1 = a favor del precio, -1 = en contra,
# ausente = NO entra en el tono (solo en la intensidad).
#
# La tabla es explicita y razonada una a una, como `SPREAD_BPS_BY_SYMBOL` en
# `execution/microstructure.py`. Lo que NO esta aqui no es un olvido: es una feature cuya
# direccion no se sostiene sin una hipotesis que nadie ha medido (¿un DXY alto es malo para
# cripto? probablemente; ¿cuanto, y con que retardo? no lo sabemos), y meterla con signo
# seria colar esa hipotesis dentro de un numero. Esas siguen contando como INTENSIDAD, que
# es lo que si se puede afirmar de ellas: que estan pasando.
#
# ES EL DEFAULT DEL PROVEEDOR, NO UN GLOBAL. `SignalRadarProvider` acepta otra tabla junto
# con otro catalogo de fuentes, y hay exactamente un caso que la usa: el canal de
# observacion SINTETICO (`synthetic/signal_channel.py`), cuya polaridad es +1 por
# construccion —el canal se emite correlacionado con el retorno futuro, asi que su signo lo
# fija el generador y no una hipotesis sobre el mundo—. Se inyecta en vez de escribirse
# aqui para que ninguna fuente simulada pueda aparecer en la tabla de las reales.
POLARITY: dict[str, float] = {
    # --- flujo y posicionamiento ---
    "etf_netflow_usd": +1.0,          # dinero entrando en el vehiculo regulado
    "cot_leveraged_net": +1.0,        # el dinero direccional, largo
    "stablecoin_issuance_usd": +1.0,  # polvora seca entrando en el ecosistema
    "funding_median": -1.0,           # funding muy positivo = largos hacinados pagando
    # --- fundamentales de protocolo ---
    "fees_usd": +1.0,
    "revenue_usd": +1.0,
    "tvl_usd": +1.0,
    # --- desarrollo y atencion ---
    "commits": +1.0,
    "contributors": +1.0,
    "sentiment_positive": +1.0,
    "sentiment_negative": -1.0,
    # --- demanda por crisis monetaria ---
    "p2p_premium_pct": +1.0,
    # --- oferta mecanica (las de evento entran por su `_mag`) ---
    "staking_exit_queue": -1.0,       # oferta futura saliendo del staking
    "staking_entry_queue": +1.0,      # demanda futura entrando
    f"token_unlocks{SUFFIX_MAG}": -1.0,     # oferta nueva con fecha
    f"defillama_hacks{SUFFIX_MAG}": -1.0,   # capital destruido y riesgo repreciado
    f"ofac_sdn{SUFFIX_MAG}": -1.0,          # perimetro operable encogiendo
    f"btc_difficulty{SUFFIX_MAG}": +1.0,    # hashrate creciendo; negativo = capitulacion
    # `macro_calendar` no aparece: lo que aporta es CUANDO, no en que direccion. Un FOMC
    # no es bueno ni malo antes de ocurrir, y darle signo seria inventarse el resultado.
}


def _sanitize(value: float) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


@dataclass(frozen=True, slots=True)
class _Reading:
    """Lo que aporta UNA fuente para UNA entidad en un instante."""

    tone: float
    intensity: float
    covered: bool


class SignalRadarProvider:
    """
    Ensamblador del bloque de SENALES EXTERNAS de la observacion.

    Se construye sobre los frames diarios que devuelve cada adaptador (los mismos que
    publica `signals features`), y hace en el constructor todo lo que se puede hacer una
    vez: normaliza las series continuas —la z propia es EXPANSIVA y causal, asi que
    calcularla entera de golpe da exactamente el mismo numero que se habria calculado dia a
    dia— y trocea cada fuente en arrays por entidad. Despues, cada consulta es una busqueda
    binaria por fuente, que es lo que la hace viable dentro de un backtest de 2.700 dias por
    24 simbolos.

    Sin frames utiles (sin credenciales, sin archivo, `[signals] enabled = false`) el radar
    existe igual y devuelve ceros con cobertura 0. No es un caso degradado que haya que
    evitar: es el estado por defecto del sistema, y las puertas se saltan solas.
    """

    def __init__(
        self,
        frames: Mapping[str, pd.DataFrame] | None,
        clock: Clock,
        *,
        sources: Sequence[SignalSource] = CATALOG,
        max_stale_days: int = MAX_STALE_DAYS,
        polarity: Mapping[str, float] = POLARITY,
    ) -> None:
        self._clock = clock
        self._max_stale_days = max_stale_days
        self._polarity = polarity
        self._catalog = {s.key: s for s in sources}
        self._asset_keys: list[str] = []
        self._market_keys: list[str] = []
        self._series: dict[str, dict[str, _Series]] = {}
        self._events: dict[str, dict[str, _Calendar]] = {}
        # Denominador de la cobertura: las fuentes que el CATALOGO declara en cada bloque,
        # no las que hoy tienen fichero. Es la diferencia entre "tengo 1 de las 8 fuentes de
        # mercado" (cobertura 0,13, la puerta se salta) y "tengo la unica que cargue"
        # (cobertura 1,0), que es justo la lectura de una sola fuente haciendose pasar por
        # la del conjunto.
        self._n_declared = {
            True: sum(1 for s in sources if is_market_scoped(s)),
            False: sum(1 for s in sources if not is_market_scoped(s)),
        }

        for source in sources:
            frame = (frames or {}).get(source.key)
            if frame is None or frame.empty:
                continue
            (self._market_keys if is_market_scoped(source) else self._asset_keys).append(
                source.key
            )
            if is_event_source(source):
                self._events[source.key] = _index_events(frame, source)
            else:
                self._series[source.key] = _index_series(frame, source, polarity)

        # Memo por el 'ahora' del reloj, igual que el regimen: el bloque de mercado se
        # calcula UNA vez por dia y lo reutilizan los 24 simbolos.
        self._cache_asof: datetime | None = None
        self._market: dict[str, float] = dict.fromkeys(MARKET_SIGNAL_FEATURES, 0.0)
        self._by_symbol: dict[str, dict[str, float]] = {}

    # --- API (identica a MarketRegimeProvider) -----------------------------------

    def features(self, symbol: str) -> dict[str, float]:
        """Bloque de senales para `symbol` en el 'ahora' del reloj."""
        self._refresh()
        cached = self._by_symbol.get(symbol)
        if cached is None:
            cached = {**self._asset_block(symbol), **self._market}
            self._by_symbol[symbol] = cached
        return dict(cached)

    @property
    def is_empty(self) -> bool:
        """True si no hay ninguna fuente con dato detras. El sistema arranca asi."""
        return not (self._series or self._events)

    def coverage_report(self) -> dict:
        """Que fuentes sostienen el radar, para publicarlo al lado de las cifras."""
        return {
            "asset_sources": sorted(self._asset_keys),
            "market_sources": sorted(self._market_keys),
            "event_sources": sorted(self._events),
            "continuous_sources": sorted(self._series),
            "min_coverage": MIN_SIGNAL_COVERAGE,
            "max_stale_days": self._max_stale_days,
        }

    # --- calculo ------------------------------------------------------------------

    def _refresh(self) -> None:
        as_of = self._clock.now()
        if self._cache_asof == as_of:
            return
        self._cache_asof = as_of
        self._by_symbol = {}
        cutoff = visible_cutoff(as_of)
        readings = [
            _collapse(self._readings_for_source(key, None, cutoff))
            for key in self._market_keys
        ]
        self._market = dict(
            zip(MARKET_SIGNAL_FEATURES, _aggregate(readings, self._n_declared[True]))
        )

    def _asset_block(self, symbol: str) -> dict[str, float]:
        cutoff = visible_cutoff(self._clock.now())
        entity = resolve_entity(symbol)
        if not entity.resolved:
            return dict.fromkeys(ASSET_SIGNAL_FEATURES, 0.0)
        readings = [
            _collapse(self._readings_for_source(key, entity.key, cutoff))
            for key in self._asset_keys
        ]
        return dict(
            zip(ASSET_SIGNAL_FEATURES, _aggregate(readings, self._n_declared[False]))
        )

    def _readings_for_source(
        self, key: str, entity: str | None, cutoff: pd.Timestamp
    ) -> list[_Reading]:
        """Lecturas de una fuente. Una por entidad si la fuente es de mercado (FRED tiene
        nueve series, las stablecoins seis cadenas) y una sola si va por activo."""
        calendars = self._events.get(key)
        if calendars is not None:
            spec = EVENT_SPECS.get(key) or EventSpec(None, 1.0, False)
            targets = calendars if entity is None else {entity: calendars.get(entity)}
            return [
                _event_reading(key, calendar, cutoff, spec, self._polarity)
                for calendar in targets.values()
                if calendar is not None
            ]

        series = self._series.get(key)
        if series is None:
            return []
        targets = series if entity is None else {entity: series.get(entity)}
        return [
            reading
            for block in targets.values()
            if block is not None
            for reading in (_series_reading(block, cutoff, self._max_stale_days),)
        ]


# --- lectura de una fuente ------------------------------------------------------------


def _collapse(readings: Sequence[_Reading]) -> _Reading:
    """Las lecturas de UNA fuente (una por entidad) en una sola.

    Es lo que hace que cada fuente pese lo mismo. Sin este paso, FRED entraria nueve veces
    —una por serie macro— y los hacks una, de modo que el tono del mercado lo decidiria
    cual de las fuentes tiene mas entidades declaradas y no cual dice algo.
    """
    covered = [r for r in readings if r.covered]
    if not covered:
        return _Reading(0.0, 0.0, False)
    return _Reading(
        tone=float(np.mean([r.tone for r in covered])),
        intensity=float(np.mean([r.intensity for r in covered])),
        covered=True,
    )


def _aggregate(readings: Sequence[_Reading], n_declared: int) -> tuple[float, float, float]:
    """Tono, intensidad y cobertura de un bloque, con UNA lectura por fuente.

    La cobertura es la fraccion de las fuentes que el catalogo DECLARA en el bloque, no de
    las que hoy tienen fichero: si solo una de ocho esta cargada, la respuesta honesta es
    "cobertura 0,13" y no "cobertura total segun la unica que miro".
    """
    covered = [r for r in readings if r.covered]
    if not covered or n_declared <= 0:
        return 0.0, 0.0, 0.0
    tone = float(np.clip(np.mean([r.tone for r in covered]), -Z_CLIP, Z_CLIP))
    intensity = float(np.clip(np.mean([r.intensity for r in covered]), 0.0, Z_CLIP))
    coverage = min(1.0, len(covered) / float(n_declared))
    return _sanitize(tone), _sanitize(intensity), _sanitize(coverage)


def _event_reading(
    key: str,
    calendar: _Calendar,
    cutoff: pd.Timestamp,
    spec: EventSpec,
    polarity_table: Mapping[str, float] = POLARITY,
) -> _Reading:
    encoded = encode_arrays(calendar.days, calendar.values, cutoff, key, spec)
    weight = max(encoded[f"{key}{SUFFIX_AHEAD}"], encoded[f"{key}{SUFFIX_ACTIVE}"])
    magnitude = encoded[f"{key}{SUFFIX_MAG}"]
    polarity = polarity_table.get(f"{key}{SUFFIX_MAG}", 0.0)
    # La proximidad PONDERA la magnitud: un desbloqueo del 3% dentro de un mes no es el
    # mismo hecho que el mismo desbloqueo manana, y sin el peso serian el mismo numero.
    # Una fuente sin magnitud (el calendario macro) aporta intensidad = proximidad: algo va
    # a pasar, aunque no se sepa de que tamano.
    intensity = weight * (abs(magnitude) if spec.magnitude else 1.0)
    return _Reading(
        tone=polarity * magnitude * weight,
        intensity=float(np.clip(intensity, 0.0, MAGNITUDE_CLIP)),
        # Una fuente de evento CUBRE a una entidad si tiene calendario para ella, aunque
        # hoy no haya nada cerca. Es exactamente la distincion que `_seen` existe para
        # hacer: "no hay evento" no es "no se de eventos".
        covered=True,
    )


def _series_reading(block: _Series, cutoff: pd.Timestamp, max_stale_days: int) -> _Reading:
    index = int(np.searchsorted(block.days, as_naive(cutoff), side="left"))
    if index == 0:
        return _Reading(0.0, 0.0, False)
    last = index - 1
    stale = (cutoff - pd.Timestamp(block.days[last], tz="UTC")).days
    if stale > max_stale_days:
        return _Reading(0.0, 0.0, False)

    tones = block.tone[last]
    intensity = block.intensity[last]
    if not np.isfinite(intensity):
        return _Reading(0.0, 0.0, False)
    return _Reading(
        tone=_sanitize(tones), intensity=_sanitize(abs(intensity)), covered=True
    )


# --- indexado (se hace una vez, en el constructor) -------------------------------------


@dataclass(frozen=True, slots=True)
class _Series:
    """Serie continua ya normalizada y reducida a dos numeros por dia."""

    days: np.ndarray
    tone: np.ndarray
    intensity: np.ndarray


@dataclass(frozen=True, slots=True)
class _Calendar:
    """Calendario de eventos de una entidad: dias ordenados y su magnitud cruda."""

    days: np.ndarray
    values: np.ndarray


def _index_series(
    frame: pd.DataFrame, source: SignalSource, polarity: Mapping[str, float] = POLARITY
) -> dict[str, _Series]:
    """Normaliza la fuente y precalcula tono e intensidad fila a fila."""
    names = [name for name in source.feature_names if name in frame.columns]
    if not names:
        return {}
    normalized = normalize_features(frame, names, keep_raw=False)

    tone_columns: list[tuple[str, float]] = []
    value_columns: list[str] = []
    for name in names:
        # La z PROPIA responde "¿es alto para este activo?" y la TRANSVERSAL "¿es alto hoy
        # comparado con los demas?". Se usa la propia y, donde no exista —un listado sin
        # historia, que es el caso que motivo publicar las dos—, la transversal.
        column = _merge_z(normalized, name)
        if column is None:
            continue
        normalized[name] = column
        value_columns.append(name)
        sign = polarity.get(name)
        if sign:
            tone_columns.append((name, sign))

    if not value_columns:
        return {}

    # Media sobre las columnas DISPONIBLES (skipna), no sobre todas: una feature sin z ese
    # dia —constante, o sin historia todavia— no puede anular el tono de las que si la
    # tienen. Sumar y dividir por el total convertiria un NaN en un cero contagioso.
    tone = (
        pd.concat([normalized[name] * sign for name, sign in tone_columns], axis=1).mean(
            axis=1, skipna=True
        )
        if tone_columns
        else pd.Series(0.0, index=normalized.index)
    )
    intensity = normalized[value_columns].abs().mean(axis=1, skipna=True)

    out: dict[str, _Series] = {}
    for entity, block in normalized.assign(_tone=tone, _intensity=intensity).groupby(
        level=ENTITY, sort=False
    ):
        days = day_values(block.index.get_level_values(DAY))
        out[str(entity)] = _Series(
            days=days,
            tone=block["_tone"].fillna(0.0).to_numpy(dtype=float),
            intensity=block["_intensity"].to_numpy(dtype=float),
        )
    return out


def _merge_z(normalized: pd.DataFrame, name: str) -> pd.Series | None:
    self_z = normalized.get(f"{name}{SUFFIX_SELF}")
    cross_z = normalized.get(f"{name}{SUFFIX_CROSS}")
    if self_z is None and cross_z is None:
        return None
    if self_z is None:
        return cross_z
    return self_z if cross_z is None else self_z.fillna(cross_z)


def _index_events(frame: pd.DataFrame, source: SignalSource) -> dict[str, _Calendar]:
    spec = EVENT_SPECS.get(source.key)
    magnitude = spec.magnitude if spec and spec.magnitude in frame.columns else None
    out: dict[str, _Calendar] = {}
    for entity, block in frame.sort_index().groupby(level=ENTITY, sort=False):
        values = (
            pd.to_numeric(block[magnitude], errors="coerce").to_numpy(dtype=float)
            if magnitude
            else np.zeros(len(block), dtype=float)
        )
        out[str(entity)] = _Calendar(
            days=day_values(block.index.get_level_values(DAY)),
            values=np.nan_to_num(values, nan=0.0),
        )
    return out


def is_market_scoped(source: SignalSource) -> bool:
    """
    ¿La fuente habla del MERCADO o de cada activo?

    Es la misma regla que decide a que entidades se le pide el dato
    (`signals/capture.py::entities_for`): alcance de mercado o entidades DECLARADAS en el
    catalogo (cadenas, series macro, divisas) -> mercado; entidades derivadas del universo
    -> activo. Tenerla escrita una vez es lo que hace que el bloque de mercado sea
    realmente identico para todos los simbolos, en vez de parecerlo.
    """
    return source.is_market_scoped or bool(source.fixed_entities)


# --- la puerta ------------------------------------------------------------------------


def signal_gate_reason(
    features: Mapping[str, float],
    *,
    min_tone: float,
    min_intensity: float | None = None,
    max_intensity: float | None = None,
) -> str | None:
    """
    ¿Bloquea la puerta de senales? Devuelve el motivo, o None si deja pasar.

    Vive aqui y no en cada estrategia porque el invariante que implementa es el mismo para
    todas y es el que no se puede perder de vista: **sin cobertura suficiente no se
    evalua**. Cada bloque —activo y mercado— se comprueba por separado y solo si SU
    cobertura llega a `MIN_SIGNAL_COVERAGE`; el que no llegue, no bloquea. Un sistema sin
    credenciales tiene cobertura 0 en los dos y opera exactamente igual que antes de que
    este modulo existiera.
    """
    blocks = (
        ("activo", ASSET_SIGNAL_FEATURES),
        ("mercado", MARKET_SIGNAL_FEATURES),
    )
    for label, (tone_key, intensity_key, coverage_key) in blocks:
        if float(features.get(coverage_key, 0.0)) < MIN_SIGNAL_COVERAGE:
            continue  # falla ABIERTA: sin datos no se decide nada
        tone = float(features.get(tone_key, 0.0))
        intensity = float(features.get(intensity_key, 0.0))
        if tone < min_tone:
            return f"tono {label} {tone:+.2f} < {min_tone:+.2f}"
        if min_intensity is not None and intensity < min_intensity:
            return f"intensidad {label} {intensity:.2f} < {min_intensity:.2f}"
        if max_intensity is not None and intensity > max_intensity:
            return f"intensidad {label} {intensity:.2f} > {max_intensity:.2f}"
    return None


__all__ = [
    "ASSET_SIGNAL_FEATURES",
    "INERT_MAX_INTENSITY",
    "INERT_MIN_INTENSITY",
    "INERT_MIN_TONE",
    "MARKET_SIGNAL_FEATURES",
    "MAX_STALE_DAYS",
    "MIN_SIGNAL_COVERAGE",
    "POLARITY",
    "SIGNAL_FEATURES",
    "SignalRadarProvider",
    "is_market_scoped",
    "signal_gate_reason",
]
