"""
Forma canonica de una senal externa: `(entidad, dia) -> features`, mas la columna
`observed`.

Es a las senales lo que `shared/bars.py` es a las barras: el sitio donde toda fuente
tiene que haber pasado antes de que ningun consumidor la mire, para que nadie tenga que
adivinar si la columna del dia se llama `date`, `day` o `timestamp`, ni en que huso viene.

POR QUE NO SE REUSA `normalize_bars`, QUE ES LO OBVIO Y ES UN ERROR
------------------------------------------------------------------
`normalize_bars` termina con `df[~df.index.duplicated(keep="last")]`. Para barras es
correcto —dos filas con el mismo timestamp son el mismo dato descargado dos veces, y la
buena es la ultima—. Para senales es DESTRUCTIVO: un dia tiene varias observaciones
legitimas (un sondeo horario, un feed por evento, tres emisores publicando su flujo de
ETF) y `keep='last'` se quedaria con una y tiraria las demas EN SILENCIO. La diferencia
no es de matiz: la que sobrevive depende del orden de las filas, asi que el mismo archivo
leido en otro orden daria otro numero.

La agregacion a dia vive AQUI, una vez, y es explicita: se agrupa por `(entity, day)` y
cada feature declara COMO se agrega (`last` para un estado —TVL, oferta—, `sum` para un
flujo —entradas de ETF, comisiones—, `mean` para un nivel ruidoso, `max`/`min` para un
extremo). Sumar un estado o tomar el ultimo de un flujo son errores que no se ven en la
grafica, y por eso la politica es un dato del catalogo y no una decision del adaptador.

LA COLUMNA `observed`
---------------------
Cuenta las observaciones crudas que hay detras de esa celda. No es contabilidad: es la
unica forma de distinguir los TRES ESTADOS que un frame de senales tiene y una serie de
precios no:

  - hay dato y vale 0.0      -> `observed >= 1`, feature 0.0
  - no hay dato              -> no hay fila (o `observed == 0`)
  - hay dato pero es escaso  -> `observed == 1` donde otros dias hay 24

La convencion del espacio de observacion es "feature no disponible = 0.0 neutro"
(`observation/features.py`), y para senales es PELIGROSA por si sola: tono 0 es neutral,
pero "no tengo datos" no es "no hay senal". `observed` es lo que permite que quien cablee
esto a una estrategia pueda construir su feature de cobertura y fallar ABIERTO en vez de
confundir ausencia con neutralidad.

`observed` se SUMA al reagregar, de modo que normalizar dos veces, o fusionar frames ya
normalizados, da el mismo recuento que hacerlo de una vez.

EL DIA ES EL DE LA OBSERVACION, NO EL DE SU PUBLICACION
-------------------------------------------------------
`day` es el dia UTC al que se REFIERE el dato. `visible_signals` recorta con la misma
frontera que las barras (`shared/clock.py::visible_cutoff`), lo cual basta para una fuente
que publica en el dia. Para una que revisa hacia atras no basta, y ninguna funcion de este
modulo lo arregla: lo que protege ahi es el archivo crudo con su `fetched_at`
(`signals/store.py`) y el campo `pit` del catalogo, que declara de que tipo es cada
fuente.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime

import pandas as pd

from ai_trader.shared.clock import visible_cutoff

ENTITY = "entity"
DAY = "day"
OBSERVED = "observed"

# Indice canonico. Es un MultiIndex a proposito: con un indice de un solo nivel, un
# `duplicated(keep='last')` de otro modulo volveria a poder colapsar dos entidades del
# mismo dia sin que nadie lo note.
SIGNAL_INDEX: tuple[str, str] = (ENTITY, DAY)

# Politicas de agregacion intradia admitidas. Ver el docstring del modulo.
AGG_LAST = "last"
AGG_FIRST = "first"
AGG_MEAN = "mean"
AGG_SUM = "sum"
AGG_MAX = "max"
AGG_MIN = "min"
AGGREGATIONS: frozenset[str] = frozenset(
    {AGG_LAST, AGG_FIRST, AGG_MEAN, AGG_SUM, AGG_MAX, AGG_MIN}
)

# Politica por defecto: la ultima observacion del dia. Es la correcta para un ESTADO, que
# es el caso mas comun, y la unica que no inventa un numero que no se observo nunca.
DEFAULT_AGGREGATION = AGG_LAST


def empty_signals(features: Sequence[str] = ()) -> pd.DataFrame:
    """Frame canonico VACIO. Un adaptador sin datos devuelve esto, nunca None: 'no hay
    observaciones' y 'no se pregunto' se distinguen aguas arriba, no por un None."""
    index = pd.MultiIndex.from_arrays(
        [pd.Index([], dtype=object), pd.DatetimeIndex([], tz="UTC")], names=SIGNAL_INDEX
    )
    frame = pd.DataFrame(index=index)
    for name in features:
        frame[name] = pd.Series(dtype=float)
    frame[OBSERVED] = pd.Series(dtype=int)
    return frame


def normalize_signals(
    df: pd.DataFrame,
    *,
    features: Sequence[str] | None = None,
    agg: str | Mapping[str, str] = DEFAULT_AGGREGATION,
) -> pd.DataFrame:
    """
    Lleva un frame crudo de senales a la forma canonica, agregando a dia.

    Entrada: columnas `entity` y `day` (o ya un MultiIndex con esos nombres) mas una
    columna por feature. Si trae `observed`, se respeta y se SUMA; si no, cada fila cuenta
    como una observacion.

    Salida: MultiIndex `(entity, day)` unico y ordenado, `day` a medianoche UTC, features
    numericas en el orden declarado y `observed` entera al final.

    `agg` es una politica global o un mapa feature -> politica. Lo normal es que venga del
    catalogo (`SignalSource.aggregations()`), no escrito a mano en el adaptador.
    """
    frame = _with_index_columns(df)

    if features is None:
        features = [c for c in frame.columns if c not in {ENTITY, DAY, OBSERVED}]
    features = list(features)
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise KeyError(f"Faltan features en el frame de senales: {missing}")

    policy = _aggregation_policy(features, agg)

    frame = frame[[ENTITY, DAY, *features] + ([OBSERVED] if OBSERVED in frame.columns else [])]
    frame = frame.copy()
    frame[ENTITY] = frame[ENTITY].map(_clean_entity)
    frame[DAY] = _to_day(frame[DAY])
    for name in features:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")

    if OBSERVED in frame.columns:
        counts = pd.to_numeric(frame[OBSERVED], errors="coerce").fillna(0)
    else:
        counts = pd.Series(1, index=frame.index, dtype="int64")
    frame[OBSERVED] = counts.astype("int64")

    # Una fila sin entidad o sin dia no es una observacion pobre: es una observacion que no
    # se puede colocar en ningun sitio.
    frame = frame[(frame[ENTITY] != "") & frame[DAY].notna()]

    if frame.empty:
        return empty_signals(features)

    grouped = frame.groupby([ENTITY, DAY], sort=True, dropna=False)
    aggregated = grouped.agg({**policy, OBSERVED: "sum"})
    aggregated.index.names = list(SIGNAL_INDEX)
    aggregated = aggregated[[*features, OBSERVED]]
    aggregated[OBSERVED] = aggregated[OBSERVED].astype("int64")
    return aggregated.sort_index()


def merge_signals(
    frames: Iterable[pd.DataFrame],
    *,
    features: Sequence[str] | None = None,
    agg: str | Mapping[str, str] = DEFAULT_AGGREGATION,
) -> pd.DataFrame:
    """
    Une varios frames canonicos (varios meses del archivo, varias descargas) en uno.

    Se reagrega con la MISMA politica en vez de concatenar y quedarse con el ultimo: con
    `sum` esa diferencia es la de contar una vez o dos, y con `last` el resultado depende
    del orden. Para `last`/`first` el orden SI importa y es el de la iteracion: lo ultimo
    que entra es lo mas reciente, que es como los lee el archivo.
    """
    prepared = [f for f in frames if f is not None and not f.empty]
    if not prepared:
        return empty_signals(features or ())

    stacked = pd.concat([_with_index_columns(f) for f in prepared], ignore_index=True)
    if features is None:
        features = [c for c in stacked.columns if c not in {ENTITY, DAY, OBSERVED}]
    return normalize_signals(stacked, features=features, agg=agg)


def visible_signals(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """
    Recorte anti look-ahead: solo dias ESTRICTAMENTE anteriores al dia del reloj.

    Es la misma frontera que ven las barras (`shared/clock.py::visible_cutoff`), y tiene
    que serlo: si el regimen no puede mirar la barra de hoy y una senal si, la comparacion
    entre una estrategia con senales y una sin ellas deja de ser limpia.
    """
    if df is None or df.empty:
        return df if df is not None else empty_signals()
    days = df.index.get_level_values(DAY)
    return df[days < visible_cutoff(now)]


def entity_frame(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """Las filas de una entidad, indexadas por dia. Vacio si no hay ninguna."""
    key = _clean_entity(entity)
    if df is None or df.empty or key not in df.index.get_level_values(ENTITY):
        return pd.DataFrame(columns=list(df.columns) if df is not None else [])
    return df.xs(key, level=ENTITY)


def coverage(df: pd.DataFrame) -> dict[str, dict]:
    """
    Cuanto hay, por entidad: dias con observacion, primera, ultima y observaciones totales.

    Es el resumen que consume la auditoria. Vive aqui y no en `signals/audit.py` porque se
    calcula sobre la forma canonica y no sobre el archivo crudo.
    """
    if df is None or df.empty:
        return {}

    out: dict[str, dict] = {}
    for entity, block in df.groupby(level=ENTITY, sort=True):
        days = block.index.get_level_values(DAY)
        out[str(entity)] = {
            "days": int(len(block)),
            "observations": int(block[OBSERVED].sum()) if OBSERVED in block else int(len(block)),
            "first_day": days.min().date().isoformat(),
            "last_day": days.max().date().isoformat(),
        }
    return out


# --- interno ------------------------------------------------------------------------


def _with_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Acepta tanto un frame plano con columnas `entity`/`day` como uno ya canonico."""
    if df is None:
        raise ValueError("El frame de senales no puede ser None")

    frame = df
    if isinstance(frame.index, pd.MultiIndex) and set(SIGNAL_INDEX).issubset(
        set(frame.index.names or [])
    ):
        frame = frame.reset_index()

    for column in SIGNAL_INDEX:
        if column not in frame.columns:
            raise KeyError(
                f"Un frame de senales necesita la columna '{column}'. "
                f"Tiene: {list(frame.columns)}"
            )
    return frame


def _aggregation_policy(
    features: Sequence[str], agg: str | Mapping[str, str]
) -> dict[str, str]:
    if isinstance(agg, str):
        _check_aggregation(agg)
        return {name: agg for name in features}

    policy: dict[str, str] = {}
    for name in features:
        how = agg.get(name, DEFAULT_AGGREGATION)
        _check_aggregation(how)
        policy[name] = how
    return policy


def _check_aggregation(how: str) -> None:
    if how not in AGGREGATIONS:
        raise ValueError(f"Agregacion desconocida: '{how}'. Admitidas: {sorted(AGGREGATIONS)}")


def _clean_entity(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _to_day(values: pd.Series) -> pd.Series:
    """Dia UTC a medianoche. Acepta fechas, timestamps con hora, cadenas ISO y NaT.

    La hora se TIRA a proposito: el sistema entero es diario y una senal con hora es una
    observacion mas de su dia, no un dia distinto.

    El segundo intento con `format='mixed'` no es paranoia: pandas infiere el formato de la
    PRIMERA fila y convierte en NaT todo lo que no encaje, asi que una fuente que devuelve
    "2026-01-05T02:00:00Z" y "2026-01-06" en la misma respuesta —lo normal en cuanto hay
    dos endpoints detras— perderia la mitad de las filas EN SILENCIO. Que la primera pasada
    sea la rapida y la segunda solo cubra los huecos mantiene barato el caso comun.
    """
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        retry = pd.to_datetime(values, utc=True, errors="coerce", format="mixed")
        parsed = parsed.fillna(retry)
    return parsed.dt.normalize()


__all__ = [
    "AGGREGATIONS",
    "AGG_FIRST",
    "AGG_LAST",
    "AGG_MAX",
    "AGG_MEAN",
    "AGG_MIN",
    "AGG_SUM",
    "DAY",
    "DEFAULT_AGGREGATION",
    "ENTITY",
    "OBSERVED",
    "SIGNAL_INDEX",
    "coverage",
    "empty_signals",
    "entity_frame",
    "merge_signals",
    "normalize_signals",
    "visible_signals",
]
