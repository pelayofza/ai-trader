"""
CODIFICACION DE EVENTOS: lo que `normalize.py` no puede hacer, y por que.

EL PROBLEMA, QUE NO ES DE MATIZ
-------------------------------
Las dos varas de `normalize.py` —z contra la propia historia, z contra la seccion cruzada—
suponen una serie que existe todos los dias. Una serie de eventos no lo es: es 99% ceros y
un dia con algo. La mediana de esa serie es 0, su rango intercuartilico es 0, y una z
contra ella es o NaN o un numero enorme sin significado. Peor: rellenar los huecos con
ceros y normalizar produce una feature que parece razonable, no falla en ningun test de
forma y no significa NADA.

Asi que las de evento se codifican de otra manera, y la diferencia vive en el codigo:

    `<fuente>_ahead`    proximidad al PROXIMO evento, en [0, 1]. 1 = hoy; 0 = no hay
                        ninguno dentro del tope. Es el "dias-al-evento" del catalogo, dado
                        la vuelta y ACOTADO a proposito (ver DAYS_AHEAD_CAP).
    `<fuente>_active`   frescura del ULTIMO evento, en [0, 1]. 1 = hoy; 0 = hace mas de la
                        ventana. Es lo unico que se puede codificar de un evento que no se
                        anuncia (un hack).
    `<fuente>_mag`      magnitud del evento que domina —el mas cercano en el tiempo—,
                        NORMALIZADA por su escala declarada y recortada. Con signo.
    `<fuente>_seen`     1 si esa entidad tiene calendario en esa fuente, 0 si no. Es la
                        pieza que distingue "no hay evento" de "no se de eventos".

EL TOPE NO ES UN DETALLE DE IMPLEMENTACION
------------------------------------------
"Dias hasta el proximo desbloqueo" es infinito cuando no hay ninguno, y el infinito no
entra en un vector de observacion. La salida perezosa es escribir 0 —o 9999— y las dos son
mentiras distintas: el 0 dice "es hoy" y el 9999 aplasta la escala de todo lo demas. Con un
tope declarado (`DAYS_AHEAD_CAP`) y la cuenta invertida, "no hay nada a la vista" es un 0
que significa exactamente eso, y el rango es finito por construccion.

Y el tope hace ademas un trabajo que no se ve: es lo que mantiene HONESTO usar el
calendario de hoy para fechar el pasado. Las reuniones del FOMC de 2019 estan en el fichero
de hoy, pero en 2019 tambien estaban publicadas —se anuncian con mas de un ano— asi que a
30 dias vista la respuesta es la misma que se habria tenido entonces. A dos anos vista no
lo seria.

LO QUE SE ANUNCIA Y LO QUE NO
-----------------------------
Un desbloqueo, un ajuste de dificultad y una reunion del FOMC tienen fecha conocida ANTES.
Un hack y una sancion, no. Mirar hacia adelante en las dos ultimas seria futuro puro, asi
que `announced=False` apaga `_ahead` por codigo y no por convencion: lo unico que queda de
un evento no anunciado es su estela.

QUE FUENTE SE CODIFICA COMO EVENTO, Y POR QUE NO LO DECIDE EL TIER
------------------------------------------------------------------
Lo decide la CADENCIA (`cadence == 'event'`), que es lo que el catalogo ya declaraba. El
tier dice de que NATURALEZA es la fuente —oferta mecanica frente a efecto estadistico— y
acierta en cinco de las seis mecanicas; la sexta, `staking_queue`, es mecanica y sin embargo
publica un NIVEL diario (cuantos validadores hay en cola), que se codifica con las dos z
como cualquier serie continua. Enrutar por el tier habria puesto una z de evento sobre un
nivel y un dias-al-evento sobre una cola. La cadencia no se equivoca en ese caso.

EL MAPA DE PRECIOS: LA TERCERA CODIFICACION (2026-08-13)
--------------------------------------------------------
El lote de alta friccion trae un objeto que ninguna de las dos anteriores sabe leer: un
MAPA DE LIQUIDACION. Se observa todos los dias (luego no es un evento fechado: no hay
ninguna fecha futura que anticipar) y lo que dice no es un nivel sino una DISTANCIA EN
PRECIO —"hay 180 M$ de posiciones que revientan un 4% mas abajo"— mas el notional acumulado
hasta ahi.

Las dos codificaciones anteriores lo leerian mal, cada una a su manera, y NINGUNA DE LAS
DOS DARIA ERROR:

  - con las dos z, la feature contestaria "¿es hoy la distancia alta PARA ESTE ACTIVO?",
    y esa no es la pregunta. Que un cluster este al 4% es un hecho absoluto: no mejora ni
    empeora porque el mes pasado estuviera al 9%.
  - con la codificacion de evento, `_ahead` contaria dias hasta una fecha que no existe.

Asi que la proximidad se mide en la unidad en la que el hecho vive —porcentaje de precio—
con el mismo patron que ya estaba: un tope declarado (`PRICE_DISTANCE_CAP_PCT`), la cuenta
invertida para que "no hay nada cerca" sea un 0 que significa eso, y una magnitud
normalizada por su escala y recortada al mismo `MAGNITUDE_CLIP` que todo lo demas.

    `<fuente>_near`  proximidad al cluster mas cercano, en [0, 1]. 1 = en el precio;
                     0 = mas lejos que el tope, o no hay mapa.
    `<fuente>_mag`   notional acumulado hasta ese cluster, normalizado y CON SIGNO:
                     negativo si el cluster esta POR DEBAJO (largos que revientan vendiendo)
                     y positivo si esta por encima (cortos que revientan comprando).
    `<fuente>_seen`  1 si esa entidad tiene mapa, 0 si no.

No hay `_active`: un mapa no tiene estela. La foto de ayer no es el rastro de nada, es
simplemente una foto vieja, y por eso lo que la gobierna es la CADUCIDAD
(`PRICE_MAP_STALE_DAYS`) y no una ventana de decaimiento.

Que enruta a esto: el campo `encoding` del catalogo, declarado fuente a fuente. La regla
general sigue siendo la cadencia; esto es la excepcion, y es explicita para que se vea.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ai_trader.shared.clock import visible_cutoff
from ai_trader.shared.signals import DAY, ENTITY, OBSERVED
from ai_trader.signals.catalog import (
    CATALOG,
    ENCODING_EVENT,
    ENCODING_PRICE_MAP,
    SignalSource,
)

# --- politica declarada -------------------------------------------------------------

# Cadencia que dispara la codificacion de evento. Ver el docstring.
EVENT_CADENCE = "event"

SUFFIX_AHEAD = "_ahead"
SUFFIX_ACTIVE = "_active"
SUFFIX_MAG = "_mag"
SUFFIX_SEEN = "_seen"
SUFFIX_NEAR = "_near"

# Tope de dias-al-evento. Treinta dias naturales: mas alla, la anticipacion de un evento de
# oferta no se distingue del ruido, y ademas es la ventana dentro de la cual el calendario
# de hoy coincide con el que estaba publicado entonces (ver el docstring).
DAYS_AHEAD_CAP = 30.0

# Ventana posterior. Mas corta que la de anticipacion a proposito: la reaccion a un evento
# que ya paso se agota antes que la anticipacion del que viene, y una estela de treinta dias
# convertiria cualquier fuente activa en un uno permanente.
DAYS_ACTIVE_CAP = 10.0

# Recorte de la magnitud, en las MISMAS unidades que `normalize.Z_CLIP`. Que coincidan no es
# casualidad: el radar mezcla magnitudes de evento con z de series continuas, y dos escalas
# distintas harian que una domine a la otra por construccion y no por informacion.
MAGNITUDE_CLIP = 4.0


@dataclass(frozen=True, slots=True)
class EventSpec:
    """
    Como se lee UNA fuente de evento. Todo lo que aqui hay es una decision declarada.

    `magnitude` es la columna que mide "cuanto"; `scale` es cuanto de esa columna vale UNA
    unidad de magnitud —el equivalente a una sigma en la escala de las z— y `announced` dice
    si el evento tiene fecha conocida de antemano.

    `scale` es un orden de magnitud razonado, no un parametro ajustado: si se calibrara
    contra el resultado, seria un grado de libertad mas, que es exactamente lo que esta
    evolucion se compromete a no anadir.
    """

    magnitude: str | None
    scale: float
    announced: bool
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "magnitude": self.magnitude,
            "scale": self.scale,
            "announced": self.announced,
            "reason": self.reason,
        }


EVENT_SPECS: dict[str, EventSpec] = {
    "token_unlocks": EventSpec(
        magnitude="unlock_pct_float",
        scale=1.0,
        announced=True,
        reason="Un 1% del float en un dia es una unidad. El calendario de vesting se "
               "publica con meses de antelacion (y se reescribe: por eso es revisable).",
    ),
    "btc_difficulty": EventSpec(
        magnitude="difficulty_change_pct",
        scale=5.0,
        announced=True,
        reason="Un ajuste del 5% es grande: la mediana historica esta en torno al 1-2%. La "
               "fecha se estima con los bloques que faltan, no la anuncia nadie.",
    ),
    "defillama_hacks": EventSpec(
        magnitude="hack_amount_usd",
        scale=1e8,
        announced=False,
        reason="100 M$ comprometidos es una unidad. Un hack NO se anuncia: mirar hacia "
               "adelante aqui seria futuro puro, asi que solo queda la estela.",
    ),
    "ofac_sdn": EventSpec(
        magnitude="sdn_new_entries",
        scale=10.0,
        announced=False,
        reason="Diez direcciones nuevas en una publicacion es un dia grande. La lista sale "
               "sin preaviso.",
    ),
    "macro_calendar": EventSpec(
        magnitude=None,
        scale=1.0,
        announced=True,
        reason="No tiene magnitud: lo que aporta es CUANDO, y el cuando se sabe con meses "
               "de antelacion. La magnitud queda en 0 y solo cuenta la proximidad.",
    ),
    # --- lote de alta friccion (2026-08-13) ---
    "deribit_expiries": EventSpec(
        magnitude="expiry_oi_share",
        scale=0.25,
        announced=True,
        reason="Un vencimiento que se lleva el 25% del interes abierto es una unidad. Es la "
               "fuente con la mejor propiedad de todo el catalogo: la fecha es el ultimo "
               "viernes del mes y no la revisa nadie, asi que los dias-al-evento del pasado "
               "son EXACTOS y no una estimacion, que es lo que si pasa en el retarget.",
    ),
    "cex_listings": EventSpec(
        magnitude="listing_change",
        scale=1.0,
        announced=False,
        reason="Un alta o una baja es una unidad, con signo. NO se anuncia: entre que Upbit "
               "publica y el mercado reacciona hay minutos, y mirar hacia adelante aqui "
               "seria futuro puro. Lo que queda es la estela, que es donde vive el efecto.",
    ),
    "appstore_rank": EventSpec(
        magnitude="app_visibility_gap",
        scale=0.5,
        announced=False,
        reason="Medio punto de diferencia de visibilidad entre Corea y EE.UU. es una "
               "unidad. Que la fuente sea de evento y no continua es una consecuencia "
               "MEDIDA: la lista solo tiene cien puestos y las apps cripto estan fuera casi "
               "siempre, asi que la serie continua seria ceros y su z no existiria.",
    ),
    "sec_edgar_fts": EventSpec(
        magnitude="edgar_institutional",
        scale=10.0,
        announced=False,
        reason="Diez filings de tenencia institucional en un dia es un dia grande. La "
               "magnitud es la pata 13F/13G y no el total: un 8-K de una empresa que "
               "menciona bitcoin de pasada y una posicion declarada no son el mismo hecho.",
    ),
    "federal_register": EventSpec(
        magnitude="fedreg_rules",
        scale=3.0,
        announced=False,
        reason="Tres normas o propuestas de norma en un dia es un dia de actividad "
               "regulatoria real. Se cuenta la pata normativa y no el total porque el total "
               "lo domina el aviso administrativo rutinario. Las fechas futuras que el "
               "propio documento trae no se anticipan todavia: por eso announced=False.",
    ),
    "courtlistener_dockets": EventSpec(
        magnitude="court_dockets",
        scale=5.0,
        announced=False,
        reason="Cinco dockets nuevos en un dia es una acumulacion visible. Una demanda no "
               "se preanuncia.",
    ),
    "dat_mnav": EventSpec(
        magnitude="dat_below_nav_share",
        scale=0.25,
        announced=False,
        reason="Una cuarta parte de la cohorte cotizando por debajo de su tesoro es una "
               "unidad: deja de ser el caso raro de una companıa mal gestionada y pasa a "
               "ser un regimen en el que la via barata de financiacion del grupo es vender. "
               "EL EVENTO ES LA PUBLICACION, no el cruce: la tenencia solo cambia cuando "
               "alguien la declara, y la fila se fecha en el dia en que se declara. "
               "announced=False porque lo que tiene fecha conocida es el PLAZO del 10-Q, no "
               "su contenido: anticipar la temporada de resultados encenderia la feature "
               "cuatro veces al ano hubiera o no estres, que es ruido con calendario. "
               "LIMITE DECLARADO: de las dos features que publica el frame, esta "
               "codificacion lleva la MAGNITUD (la fraccion bajo 1) y no la DISTANCIA "
               "(`dat_mnav_gap`), que se queda en el frame como la pata auditable y como lo "
               "que hace comparable un activo con otro. Llevar las dos exigiria la "
               "codificacion de mapa de precios, cuya frontera absoluta —el 1,0x— y cuyo "
               "signo encajan; no se hace hoy porque la fuente entra como esta declarada.",
    ),
}


# --- mapa de precios: la tercera codificacion -----------------------------------------

# Tope de distancia, en PORCENTAJE DE PRECIO. Un cuarto del precio es donde deja de tener
# sentido llamar "cercano" a un cluster: para llegar hasta ahi el precio tiene que hacer un
# recorrido que ya es la noticia por si mismo, y la aceleracion que aporta la liquidacion
# deja de ser lo que explica el movimiento. Cumple ademas el mismo papel que
# `DAYS_AHEAD_CAP`: convierte "no hay nada cerca" en un 0 que significa exactamente eso.
#
# Coincide con el horizonte de busqueda del adaptador (`CLUSTER_SEARCH_PCT`) a proposito: si
# el tope fuera mas estrecho, el adaptador gastaria trabajo en encontrar clusters que este
# modulo iba a pesar con cero, y el frame publicaria una distancia que no significa nada.
#
# MEDIDO 2026-08-13, y conviene tenerlo delante antes de leer la feature: en la muestra de
# ese dia los clusters de BTC caian a -25%, -32% y -69%. Con este tope, ninguno pesa, y esa
# es la lectura CORRECTA —las doscientas cuentas mayores no estaban cerca de reventar— y no
# un tope mal puesto. La feature vale precisamente los dias en que deja de ser cero.
PRICE_DISTANCE_CAP_PCT = 25.0

# Cuantos dias puede tener la ultima foto del mapa antes de dejar de contar. Un mapa de
# liquidacion de hace dos semanas no describe el libro de hoy —las posiciones que lo
# formaban ya se han cerrado, movido o liquidado— y usarlo seria peor que no tener mapa,
# porque tendria la misma pinta que uno fresco. Es la caducidad de `signal_radar.py`
# aplicada aqui, y mas corta: dos dias.
PRICE_MAP_STALE_DAYS = 2.0


@dataclass(frozen=True, slots=True)
class PriceMapSpec:
    """Como se lee UN mapa de precios. Igual que `EventSpec`: todo declarado."""

    distance: str  # columna con la distancia al cluster, en % y CON SIGNO
    notional: str  # columna con el notional acumulado hasta el
    scale: float  # cuanto notional vale UNA unidad de magnitud
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "distance": self.distance,
            "notional": self.notional,
            "scale": self.scale,
            "reason": self.reason,
        }


PRICE_SPECS: dict[str, PriceMapSpec] = {
    "hyperliquid_liqmap": PriceMapSpec(
        distance="liq_cluster_distance_pct",
        notional="liq_cluster_notional_usd",
        scale=1e8,
        reason="100 M$ de posiciones que revientan antes de llegar al cluster es una "
               "unidad. Es el mismo orden que la escala de los hacks, y a proposito: son "
               "dos formas de que salga capital de golpe y conviene que pesen igual.",
    ),
    "lending_health": PriceMapSpec(
        distance="lending_liq_distance_pct",
        notional="lending_liq_notional_usd",
        scale=1e9,
        reason="1.000 M$ de colateral liquidable es una unidad: un orden de magnitud MAS "
               "que el perpetuo porque el colateral spot de Aave se mide en decenas de "
               "miles de millones. Igualar las dos escalas haria que el mapa on-chain "
               "saturase el recorte todos los dias y dejara de distinguir nada.",
    ),
}


def day_values(index: pd.Index) -> np.ndarray:
    """Nivel `day` -> `datetime64[ns]` SIN huso.

    `DatetimeIndex.to_numpy()` sobre un indice con huso devuelve un array de objetos
    Timestamp, y comparar eso con una fecha en `searchsorted` revienta con "cannot compare
    tz-naive and tz-aware". Todo el sistema es UTC (`shared/signals.py` lo garantiza), asi
    que quitar el huso aqui no pierde informacion y es lo que hace comparable el array.
    """
    values = pd.DatetimeIndex(index)
    return (values.tz_convert(None) if values.tz is not None else values).to_numpy()


def as_naive(moment: pd.Timestamp) -> np.datetime64:
    """La frontera del reloj, en la misma escala que `day_values`."""
    stamp = pd.Timestamp(moment)
    return np.datetime64(stamp.tz_convert(None) if stamp.tz is not None else stamp)


def event_sources(catalog: Sequence[SignalSource] = CATALOG) -> tuple[SignalSource, ...]:
    """Las fuentes que se codifican como evento FECHADO. Es la cadencia, no el tier."""
    return tuple(s for s in catalog if is_event_source(s))


def is_event_source(source: SignalSource) -> bool:
    """Evento fechado. Se pregunta al catalogo, que deriva la respuesta de la cadencia
    salvo que la fuente declare otra codificacion (`price_map`)."""
    return source.encoding_kind == ENCODING_EVENT


def price_map_sources(catalog: Sequence[SignalSource] = CATALOG) -> tuple[SignalSource, ...]:
    """Las fuentes que se codifican como mapa de precios. Ver el docstring del modulo."""
    return tuple(s for s in catalog if is_price_map_source(s))


def is_price_map_source(source: SignalSource) -> bool:
    return source.encoding_kind == ENCODING_PRICE_MAP


def encoded_names(source_key: str) -> tuple[str, ...]:
    """Las columnas que produce una fuente de evento, en orden canonico."""
    return (
        f"{source_key}{SUFFIX_AHEAD}",
        f"{source_key}{SUFFIX_ACTIVE}",
        f"{source_key}{SUFFIX_MAG}",
        f"{source_key}{SUFFIX_SEEN}",
    )


def price_encoded_names(source_key: str) -> tuple[str, ...]:
    """Las columnas que produce un mapa de precios. Tres y no cuatro: no hay estela."""
    return (
        f"{source_key}{SUFFIX_NEAR}",
        f"{source_key}{SUFFIX_MAG}",
        f"{source_key}{SUFFIX_SEEN}",
    )


def event_encoding_spec() -> dict:
    """La politica, en un dict. Se publica con el dato, igual que `normalization_spec`."""
    return {
        "cadence": EVENT_CADENCE,
        "days_ahead_cap": DAYS_AHEAD_CAP,
        "days_active_cap": DAYS_ACTIVE_CAP,
        "magnitude_clip": MAGNITUDE_CLIP,
        "suffixes": {
            "ahead": SUFFIX_AHEAD,
            "active": SUFFIX_ACTIVE,
            "magnitude": SUFFIX_MAG,
            "seen": SUFFIX_SEEN,
        },
        "missing": "0.0 con `_seen` = 0 (la ausencia de dato se declara aparte)",
        "sources": {key: spec.as_dict() for key, spec in sorted(EVENT_SPECS.items())},
        "price_map": price_map_encoding_spec(),
    }


def price_map_encoding_spec() -> dict:
    """La politica del mapa de precios, publicada al lado de la de evento."""
    return {
        "routed_by": "campo `encoding` del catalogo (excepcion declarada a la cadencia)",
        "distance_cap_pct": PRICE_DISTANCE_CAP_PCT,
        "stale_days": PRICE_MAP_STALE_DAYS,
        "magnitude_clip": MAGNITUDE_CLIP,
        "suffixes": {
            "near": SUFFIX_NEAR,
            "magnitude": SUFFIX_MAG,
            "seen": SUFFIX_SEEN,
        },
        "sign": "la magnitud es negativa si el cluster esta POR DEBAJO del precio",
        "missing": "0.0 con `_seen` = 0; una foto de mas de `stale_days` no cuenta",
        "sources": {key: spec.as_dict() for key, spec in sorted(PRICE_SPECS.items())},
    }


# --- codificacion --------------------------------------------------------------------


def encode_at(
    frame: pd.DataFrame,
    source_key: str,
    as_of: datetime,
    *,
    entities: Sequence[str] | None = None,
    spec: EventSpec | None = None,
) -> dict[str, dict[str, float]]:
    """
    Codifica UNA fuente de evento en el 'ahora' del reloj. `entidad -> columnas`.

    Anti look-ahead, con el matiz que hace distinta a una fuente de evento: el pasado se
    recorta con la MISMA frontera que las barras (`visible_cutoff`), pero el FUTURO no se
    tira. Un calendario publicado hace meses es informacion disponible hoy, y borrarlo
    "por si acaso" no seria prudencia: seria eliminar justo la propiedad por la que una
    feature de evento vale algo. Lo que impide que eso sea futuro es `announced`, que
    apaga la mirada hacia adelante en las fuentes que no se anuncian.
    """
    spec = spec or EVENT_SPECS.get(source_key) or EventSpec(None, 1.0, False)
    names = encoded_names(source_key)
    empty = dict.fromkeys(names, 0.0)

    keys = list(entities or ())
    out: dict[str, dict[str, float]] = {key: dict(empty) for key in keys}
    if frame is None or frame.empty:
        return out

    cutoff = visible_cutoff(as_of)
    magnitude = spec.magnitude if spec.magnitude in frame.columns else None

    for entity, block in frame.groupby(level=ENTITY, sort=False):
        entity = str(entity)
        if keys and entity not in out:
            continue
        days = day_values(block.index.get_level_values(DAY))
        values = (
            pd.to_numeric(block[magnitude], errors="coerce").to_numpy(dtype=float)
            if magnitude
            else np.zeros(len(block), dtype=float)
        )
        out[entity] = encode_arrays(days, values, cutoff, source_key, spec)

    return out


def encode_arrays(
    days: np.ndarray,
    values: np.ndarray,
    cutoff: pd.Timestamp,
    source_key: str,
    spec: EventSpec,
) -> dict[str, float]:
    """
    EL NUCLEO de la codificacion: calendario ordenado de UNA entidad -> las cuatro columnas.

    Trabaja sobre arrays y no sobre un frame porque lo llaman dos sitios con exigencias
    distintas —`encode_at`, que es la API legible, y el radar, que pregunta una vez por
    simbolo y por dia y no puede permitirse un `groupby` por consulta— y tener DOS
    implementaciones de la misma cuenta seria tener dos codificaciones distintas en cuanto
    alguien tocase una.
    """
    names = encoded_names(source_key)
    encoded = dict.fromkeys(names, 0.0)
    if days is None or len(days) == 0:
        return encoded

    # `_seen` mira el calendario ENTERO de la entidad, no solo el pasado visible: que una
    # fuente cubra a este activo es un hecho de hoy, no una observacion fechada.
    encoded[f"{source_key}{SUFFIX_SEEN}"] = 1.0

    # El calendario llega ordenado (el frame canonico lo esta), asi que la frontera entre
    # pasado y futuro es una busqueda binaria y no un barrido.
    split = int(np.searchsorted(days, as_naive(cutoff), side="left"))

    weight_active, magnitude_active = 0.0, 0.0
    if split > 0:
        elapsed = (cutoff - pd.Timestamp(days[split - 1], tz="UTC")).days
        weight_active = max(0.0, 1.0 - elapsed / DAYS_ACTIVE_CAP)
        magnitude_active = values[split - 1]

    weight_ahead, magnitude_ahead = 0.0, 0.0
    if spec.announced and split < len(days):
        remaining = (pd.Timestamp(days[split], tz="UTC") - cutoff).days
        weight_ahead = max(0.0, 1.0 - remaining / DAYS_AHEAD_CAP)
        magnitude_ahead = values[split]

    encoded[f"{source_key}{SUFFIX_ACTIVE}"] = weight_active
    encoded[f"{source_key}{SUFFIX_AHEAD}"] = weight_ahead
    # DOMINA EL MAS CERCANO EN EL TIEMPO. En cualquier instante o se esta en la antesala de
    # un evento o en su estela; mezclar las dos magnitudes en una media produciria un
    # numero que no describe ninguna de las dos situaciones.
    chosen = magnitude_ahead if weight_ahead >= weight_active else magnitude_active
    encoded[f"{source_key}{SUFFIX_MAG}"] = scaled_magnitude(chosen, spec.scale)
    return encoded


def scaled_magnitude(value: float, scale: float) -> float:
    """Magnitud en unidades declaradas y recortada al mismo tope que las z."""
    if value is None or not np.isfinite(value) or scale <= 0:
        return 0.0
    return float(np.clip(value / scale, -MAGNITUDE_CLIP, MAGNITUDE_CLIP))


def encode_price_at(
    frame: pd.DataFrame,
    source_key: str,
    as_of: datetime,
    *,
    entities: Sequence[str] | None = None,
    spec: PriceMapSpec | None = None,
) -> dict[str, dict[str, float]]:
    """
    Codifica UN mapa de precios en el 'ahora' del reloj. `entidad -> columnas`.

    Gemelo de `encode_at` y con una diferencia que no es de forma: aqui el futuro NO se
    mira, porque no existe. Un mapa es la foto de un instante y la unica pregunta valida es
    cual es la ultima foto ANTERIOR a la frontera, y si sigue siendo fresca.
    """
    spec = spec or PRICE_SPECS.get(source_key)
    names = price_encoded_names(source_key)
    empty = dict.fromkeys(names, 0.0)

    keys = list(entities or ())
    out: dict[str, dict[str, float]] = {key: dict(empty) for key in keys}
    if frame is None or frame.empty or spec is None:
        return out

    cutoff = visible_cutoff(as_of)
    has_distance = spec.distance in frame.columns
    has_notional = spec.notional in frame.columns
    if not (has_distance and has_notional):
        return out

    for entity, block in frame.groupby(level=ENTITY, sort=False):
        entity = str(entity)
        if keys and entity not in out:
            continue
        out[entity] = encode_price_arrays(
            day_values(block.index.get_level_values(DAY)),
            pd.to_numeric(block[spec.distance], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(block[spec.notional], errors="coerce").to_numpy(dtype=float),
            cutoff,
            source_key,
            spec,
        )
    return out


def encode_price_arrays(
    days: np.ndarray,
    distances: np.ndarray,
    notionals: np.ndarray,
    cutoff: pd.Timestamp,
    source_key: str,
    spec: PriceMapSpec,
    *,
    stale_days: float = PRICE_MAP_STALE_DAYS,
) -> dict[str, float]:
    """
    EL NUCLEO del mapa de precios: la serie de fotos de UNA entidad -> las tres columnas.

    Sobre arrays y no sobre un frame por el mismo motivo que `encode_arrays`: lo llaman la
    API legible y el radar, y dos implementaciones de la misma cuenta son dos cuentas
    distintas en cuanto alguien toque una.
    """
    names = price_encoded_names(source_key)
    encoded = dict.fromkeys(names, 0.0)
    if days is None or len(days) == 0:
        return encoded

    # `_seen` mira si la entidad tiene mapa, no si hoy dice algo: es la misma distincion
    # entre "no hay cluster cerca" y "de este activo no se nada".
    encoded[f"{source_key}{SUFFIX_SEEN}"] = 1.0

    index = int(np.searchsorted(days, as_naive(cutoff), side="left"))
    if index == 0:
        return encoded  # todas las fotos son posteriores a la frontera

    last = index - 1
    stale = (cutoff - pd.Timestamp(days[last], tz="UTC")).days
    if stale > stale_days:
        return encoded  # foto caducada: ver PRICE_MAP_STALE_DAYS

    distance = distances[last]
    notional = notionals[last]
    if not np.isfinite(distance) or not np.isfinite(notional):
        return encoded

    encoded[f"{source_key}{SUFFIX_NEAR}"] = max(
        0.0, 1.0 - abs(float(distance)) / PRICE_DISTANCE_CAP_PCT
    )
    # EL SIGNO ES EL DE LA DISTANCIA, no el del notional: el notional es una cantidad y lo
    # que tiene direccion es donde esta el cluster. Un cluster por debajo son largos que
    # revientan VENDIENDO; por encima, cortos que revientan comprando.
    magnitude = scaled_magnitude(abs(float(notional)), spec.scale)
    encoded[f"{source_key}{SUFFIX_MAG}"] = magnitude * (-1.0 if distance < 0 else 1.0)
    return encoded


def encode_price_series(
    frame: pd.DataFrame,
    source_key: str,
    days: Sequence[datetime],
    *,
    entities: Sequence[str] | None = None,
) -> pd.DataFrame:
    """La version DENSA del mapa, para publicar e inspeccionar. Ver `encode_series`."""
    keys = list(entities or _entities_of(frame))
    rows: list[dict] = []
    encoded: dict[str, dict[str, float]] = {}
    for day in days:
        encoded = encode_price_at(frame, source_key, day, entities=keys)
        for entity, values in encoded.items():
            rows.append({ENTITY: entity, DAY: day, **values})
    if not rows:
        return pd.DataFrame(columns=[*price_encoded_names(source_key)])
    out = pd.DataFrame(rows).set_index([ENTITY, DAY]).sort_index()
    return out[list(price_encoded_names(source_key))]


def encode_series(
    frame: pd.DataFrame,
    source_key: str,
    days: Sequence[datetime],
    *,
    entities: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    La version DENSA: las mismas columnas, un dia por fila. Para publicar e inspeccionar.

    Llama a `encode_at` dia a dia en vez de reimplementar la cuenta vectorizada. Es mas
    lento y es deliberado: dos implementaciones de la misma codificacion terminan
    divergiendo, y la version que corre en la decision es la de `encode_at`.
    """
    keys = list(entities or _entities_of(frame))
    rows: list[dict] = []
    for day in days:
        encoded = encode_at(frame, source_key, day, entities=keys)
        for entity, values in encoded.items():
            rows.append({ENTITY: entity, DAY: day, **values})
    if not rows:
        return pd.DataFrame(columns=[*encoded_names(source_key)])
    out = pd.DataFrame(rows).set_index([ENTITY, DAY]).sort_index()
    return out[list(encoded_names(source_key))]


def _entities_of(frame: pd.DataFrame) -> tuple[str, ...]:
    if frame is None or frame.empty:
        return ()
    return tuple(sorted({str(e) for e in frame.index.get_level_values(ENTITY)}))


# --- el recuento que sustituye a la creencia ------------------------------------------


def pool_report(frames: Mapping[str, pd.DataFrame]) -> dict:
    """
    CUANTOS EVENTOS HAY, DE VERDAD, POR FUENTE. Es la cifra de la que colgaba todo.

    La unidad de observacion es el EVENTO NORMALIZADO, no el token: catorce desbloqueos de
    un token son catorce, pero catorce por veinte activos son doscientos ochenta eventos
    COMPARABLES entre si, porque la magnitud esta en % del float y no en unidades del
    token. Ese es el sentido de `pooled`: los eventos de todas las entidades de la fuente
    puestos en la misma distribucion.

    `per_entity` va al lado y no es decorativo: una fuente con 600 eventos repartidos en
    una sola entidad (los hacks son de mercado) y otra con 600 repartidos en 24 activos
    sostienen inferencias distintas, y el numero agregado no distingue las dos.
    """
    out: dict[str, dict] = {}
    for source in event_sources():
        frame = frames.get(source.key)
        spec = EVENT_SPECS.get(source.key)
        row = {
            "tier": source.tier,
            "cadence": source.cadence,
            "announced": bool(spec and spec.announced),
            "magnitude": spec.magnitude if spec else None,
            "pooled_events": 0,
            "entities": 0,
            "events_per_entity": 0.0,
            "with_magnitude": 0,
            "first_day": None,
            "last_day": None,
        }
        if frame is not None and not frame.empty:
            days = frame.index.get_level_values(DAY)
            per_entity = frame.groupby(level=ENTITY, sort=True).size()
            magnitude = spec.magnitude if spec else None
            row.update(
                pooled_events=int(len(frame)),
                entities=int(per_entity.size),
                events_per_entity=round(float(len(frame)) / float(per_entity.size), 2),
                # Un evento cuya magnitud no se pudo derivar (falta el denominador: el
                # float, el volumen) cuenta como evento y NO como magnitud. Son dos
                # muestras distintas y confundirlas es lo que hace creer que hay mas.
                with_magnitude=(
                    int(pd.to_numeric(frame[magnitude], errors="coerce").notna().sum())
                    if magnitude and magnitude in frame.columns
                    else 0
                ),
                first_day=days.min().date().isoformat(),
                last_day=days.max().date().isoformat(),
                observations=int(frame[OBSERVED].sum()) if OBSERVED in frame else int(len(frame)),
            )
        out[source.key] = row

    return {
        "spec": event_encoding_spec(),
        "n_event_sources": len(out),
        "pooled_events_total": sum(r["pooled_events"] for r in out.values()),
        "sources": dict(sorted(out.items())),
        # Los mapas de precios van APARTE y no dentro de `sources`: su unidad de
        # observacion es la FOTO DIARIA y no el evento, asi que sumarlos al recuento
        # pooled inflaria la muestra de eventos con algo que no lo es.
        "price_maps": price_map_pool(frames),
    }


def price_map_pool(frames: Mapping[str, pd.DataFrame]) -> dict:
    """Cuantas fotos de mapa hay, por fuente. La cifra que dice si el mapa existe."""
    out: dict[str, dict] = {}
    for source in price_map_sources():
        frame = frames.get(source.key)
        spec = PRICE_SPECS.get(source.key)
        row = {
            "tier": source.tier,
            "distance": spec.distance if spec else None,
            "snapshots": 0,
            "entities": 0,
            "first_day": None,
            "last_day": None,
        }
        if frame is not None and not frame.empty:
            days = frame.index.get_level_values(DAY)
            row.update(
                snapshots=int(len(frame)),
                entities=int(frame.index.get_level_values(ENTITY).nunique()),
                first_day=days.min().date().isoformat(),
                last_day=days.max().date().isoformat(),
            )
        out[source.key] = row
    return {"n_sources": len(out), "sources": dict(sorted(out.items()))}


# El recuento publicado. Va en data/ y no en .cache/ por el mismo motivo que el registro de
# profundidad: es la EVIDENCIA que sustituye a una creencia ("son muestras de decenas"), y
# lo que sustituye a una creencia tiene que poder citarse con fecha.
EVENT_POOL_REPORT = Path("data") / "signals" / "event_pool.json"


def write_pool_report(report: dict, path: Path | str = EVENT_POOL_REPORT) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return target


def load_pool_report(path: Path | str = EVENT_POOL_REPORT) -> dict | None:
    """El ultimo recuento, o None si nunca se ha corrido: los generadores de documentacion
    degradan a prosa sin cifras en vez de romperse."""
    target = Path(path)
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - fichero corrupto
        return None


__all__ = [
    "DAYS_ACTIVE_CAP",
    "EVENT_POOL_REPORT",
    "DAYS_AHEAD_CAP",
    "EVENT_CADENCE",
    "EVENT_SPECS",
    "MAGNITUDE_CLIP",
    "PRICE_DISTANCE_CAP_PCT",
    "PRICE_MAP_STALE_DAYS",
    "PRICE_SPECS",
    "SUFFIX_ACTIVE",
    "SUFFIX_AHEAD",
    "SUFFIX_MAG",
    "SUFFIX_NEAR",
    "SUFFIX_SEEN",
    "EventSpec",
    "PriceMapSpec",
    "as_naive",
    "day_values",
    "encode_arrays",
    "encode_at",
    "encode_price_arrays",
    "encode_price_at",
    "encode_price_series",
    "encode_series",
    "scaled_magnitude",
    "encoded_names",
    "event_encoding_spec",
    "event_sources",
    "is_event_source",
    "is_price_map_source",
    "load_pool_report",
    "pool_report",
    "price_encoded_names",
    "price_map_encoding_spec",
    "price_map_pool",
    "price_map_sources",
    "write_pool_report",
]
