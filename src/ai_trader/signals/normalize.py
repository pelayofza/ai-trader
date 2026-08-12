"""
PUBLICACION NORMALIZADA: la unica forma en que una senal sale de aqui.

EL PROBLEMA, QUE NO ES ESTETICO
-------------------------------
Las fuentes continuas producen numeros que no se pueden ni sumar ni comparar: 90.000
millones de dolares de oferta de stablecoins, 3,4 puntos basicos de dispersion de funding,
23.000 visitas, una prima del 12%, 47 commits. Un consumidor que reciba eso crudo tiene dos
salidas y las dos son malas: inventarse un factor de escala por columna, o dejar que la
columna con los numeros mas grandes domine cualquier combinacion lineal.

Peor todavia es la comparacion ENTRE ACTIVOS. BTC tiene 23.000 visitas y SEI tiene 40: la
diferencia no dice nada sobre cual esta recibiendo atencion INUSUAL hoy, que es la unica
pregunta interesante. Y un listado nuevo no tiene historia contra la que compararse, asi
que cualquier normalizacion que dependa solo de su pasado le da NaN durante meses.

LAS DOS Z, Y POR QUE HACEN FALTA LAS DOS
----------------------------------------
    `<feature>_z`  z contra la PROPIA HISTORIA de la entidad, expansiva y causal.
                   Responde: ¿esto es alto PARA ESTE ACTIVO?
    `<feature>_x`  z contra la SECCION CRUZADA de ese dia (todas las entidades).
                   Responde: ¿esto es alto HOY, comparado con los demas?

Son complementarias y ninguna sustituye a la otra. La propia historia no existe para un
listado de la semana pasada —y ahi la seccion cruzada sigue dando un numero util desde el
primer dia—. La seccion cruzada, a su vez, no distingue un dia en que TODO el mercado se
mueve de un dia normal, porque resta la mediana del dia; la propia historia si.

Que las dos esten publicadas y ninguna sea "la buena" es deliberado: cual usar depende de
la pregunta, y esa decision es del consumidor, no de la ingesta.

MEDIANA Y RANGO INTERCUARTILICO, NO MEDIA Y DESVIACION TIPICA
-------------------------------------------------------------
Estas series tienen colas gruesas por construccion: un dia de hack, un unlock, un pico de
menciones. Con media y desviacion tipica, UN dia extremo infla la escala y aplasta todo lo
que viene despues —la senal se apaga justo cuando empieza a pasar algo—. Con mediana y
IQR/1,349 (que es la desviacion tipica equivalente en una normal) el mismo dia extremo se
sale de rango sin cambiar la vara de medir.

CAUSAL, SIN EXCEPCIONES
-----------------------
La z propia usa una ventana EXPANSIVA hasta el dia t INCLUIDO: solo pasado y presente, en
ningun caso futuro. No es una z sobre la muestra completa disfrazada: normalizar con la
media de toda la serie es una de las formas mas silenciosas de meter futuro en un backtest,
porque no rompe nada y mejora los resultados.

EL RECORTE ESTA DECLARADO
-------------------------
`Z_CLIP = 4.0`. Todo lo que caiga fuera se recorta a la frontera, y `normalization_spec()`
publica el numero junto con el resto de la politica para que aparezca en el dashboard y en
la metodologia. El recorte existe porque una z de 300 —que sale sola cuando el IQR de una
serie casi constante es diminuto— no es una senal fortisima: es una division por casi cero
que arrastraria cualquier media ponderada detras de ella.

NaN SIGNIFICA "NO SE PUEDE CALCULAR", Y NO SE RELLENA
-----------------------------------------------------
Sin historia suficiente (`MIN_HISTORY`), sin suficientes entidades ese dia
(`MIN_CROSS_SECTION`) o con escala cero, el resultado es NaN. Rellenarlo con 0 diria
"normal, en la media", que es una afirmacion sobre el mundo que no se ha observado. La
convencion del espacio de observacion (`observation/features.py`) es 0.0 neutro, y para
senales eso es peligroso: por eso la columna `observed` viaja al lado, y quien cablee esto
tendra que decidir explicitamente que hace con los huecos.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ai_trader.shared.signals import DAY, ENTITY, OBSERVED, empty_signals

# --- politica declarada -------------------------------------------------------------

SUFFIX_SELF = "_z"
SUFFIX_CROSS = "_x"

# Recorte, en unidades de z. Ver el docstring.
Z_CLIP = 4.0

# Observaciones minimas de la propia entidad para publicar su z. Bajo a proposito: con 20
# dias la escala es mala pero honesta, y quien no llegue tiene la seccion cruzada.
MIN_HISTORY = 20

# Entidades minimas con dato ese dia para que una comparacion transversal signifique algo.
MIN_CROSS_SECTION = 5

# Factor que convierte el rango intercuartilico en desviacion tipica equivalente bajo
# normalidad: IQR = 1,349 sigma.
IQR_TO_SIGMA = 1.349


def normalization_spec() -> dict:
    """La politica, en un dict. Se publica con el dato: una senal normalizada sin su
    politica al lado no es interpretable dentro de seis meses."""
    return {
        "center": "mediana",
        "scale": "IQR/1.349",
        "self_window": "expansiva, causal (hasta t incluido)",
        "cross_section": "por dia, todas las entidades de la fuente",
        "clip": Z_CLIP,
        "min_history": MIN_HISTORY,
        "min_cross_section": MIN_CROSS_SECTION,
        "missing": "NaN (no se rellena con 0)",
        "suffix_self": SUFFIX_SELF,
        "suffix_cross": SUFFIX_CROSS,
    }


def normalized_names(features: Sequence[str]) -> tuple[str, ...]:
    """Las columnas que se publican para esas features, en orden canonico."""
    out: list[str] = []
    for name in features:
        out.append(f"{name}{SUFFIX_SELF}")
        out.append(f"{name}{SUFFIX_CROSS}")
    return tuple(out)


def normalize_features(
    frame: pd.DataFrame,
    features: Sequence[str] | None = None,
    *,
    clip: float = Z_CLIP,
    min_history: int = MIN_HISTORY,
    min_cross_section: int = MIN_CROSS_SECTION,
    keep_raw: bool = True,
) -> pd.DataFrame:
    """
    Anade las dos z a un frame canonico de senales. No toca las columnas crudas.

    Entrada: frame `(entity, day) -> features + observed` (el que devuelve cualquier
    adaptador). Salida: el mismo, mas `<feature>_z` y `<feature>_x` por cada feature, con
    `observed` siempre al final.
    """
    if frame is None or frame.empty:
        names = list(features or [])
        return empty_signals([*names, *normalized_names(names)]) if names else empty_signals()

    if features is None:
        features = [c for c in frame.columns if c != OBSERVED]
    features = [name for name in features if name in frame.columns]

    work = frame.sort_index()
    out = work.copy()

    for name in features:
        values = pd.to_numeric(work[name], errors="coerce")
        out[f"{name}{SUFFIX_SELF}"] = _self_z(values, clip=clip, min_history=min_history)
        out[f"{name}{SUFFIX_CROSS}"] = _cross_z(
            values, clip=clip, min_cross_section=min_cross_section
        )

    columns = [*(features if keep_raw else []), *normalized_names(features)]
    if OBSERVED in out.columns:
        columns.append(OBSERVED)
    return out[columns]


def _self_z(values: pd.Series, *, clip: float, min_history: int) -> pd.Series:
    """z contra la propia historia de cada entidad. Expansiva y causal."""
    grouped = values.groupby(level=ENTITY, sort=False)

    center = grouped.expanding().median()
    high = grouped.expanding().quantile(0.75)
    low = grouped.expanding().quantile(0.25)
    seen = grouped.expanding().count()

    # `groupby().expanding()` devuelve la entidad repetida como primer nivel del indice.
    # Se quita y se realinea: sin esto, la resta contra `values` alinearia por un indice de
    # tres niveles y saldria todo NaN sin avisar.
    center, high, low, seen = (_realign(s, values.index) for s in (center, high, low, seen))

    scale = (high - low) / IQR_TO_SIGMA
    z = (values - center) / scale.where(scale > 0)
    return _clip(z.where(seen >= min_history), clip)


def _cross_z(values: pd.Series, *, clip: float, min_cross_section: int) -> pd.Series:
    """z contra la seccion cruzada del dia. Un dia con pocas entidades no produce z."""
    grouped = values.groupby(level=DAY, sort=False)

    center = grouped.transform("median")
    scale = (grouped.transform(lambda s: s.quantile(0.75)) - grouped.transform(
        lambda s: s.quantile(0.25)
    )) / IQR_TO_SIGMA
    counts = grouped.transform("count")

    z = (values - center) / scale.where(scale > 0)
    return _clip(z.where(counts >= min_cross_section), clip)


def _realign(series: pd.Series, index: pd.Index) -> pd.Series:
    while isinstance(series.index, pd.MultiIndex) and series.index.nlevels > index.nlevels:
        series = series.droplevel(0)
    return series.reindex(index)


def _clip(series: pd.Series, clip: float) -> pd.Series:
    if clip is None or clip <= 0:
        return series
    return series.clip(lower=-clip, upper=clip).replace([np.inf, -np.inf], np.nan)


def panel(frames: Mapping[str, pd.DataFrame], *, clip: float = Z_CLIP) -> pd.DataFrame:
    """
    Todas las fuentes en UN panel `(entity, day) -> columnas normalizadas`.

    Une por indice, sin rellenar: una entidad que no aparece en una fuente tiene NaN en sus
    columnas, y eso es lo que hay que saber. `observed` se SUMA entre fuentes, porque la
    pregunta que responde ahi es "cuantas observaciones crudas sostienen esta fila", que es
    la suma de las de cada fuente.

    Las columnas crudas NO entran: el panel es la superficie de publicacion y publicar el
    crudo al lado invita a usarlo sin normalizar, que es exactamente el error que este
    modulo existe para evitar. El crudo sigue en el archivo y en el frame de cada fuente.
    """
    prepared: list[pd.DataFrame] = []
    observed: list[pd.Series] = []

    for key in sorted(frames):
        frame = frames[key]
        if frame is None or frame.empty:
            continue
        normalized = normalize_features(frame, clip=clip, keep_raw=False)
        if OBSERVED in normalized.columns:
            observed.append(normalized[OBSERVED])
            normalized = normalized.drop(columns=[OBSERVED])
        prepared.append(normalized)

    if not prepared:
        return empty_signals()

    joined = pd.concat(prepared, axis=1).sort_index()
    total = pd.concat(observed, axis=1).sum(axis=1) if observed else 0
    joined[OBSERVED] = pd.Series(total, index=joined.index).fillna(0).astype("int64")
    return joined


def normalization_coverage(frame: pd.DataFrame) -> dict:
    """
    Cuanto del panel es utilizable, en numeros.

    Las dos cifras que importan son distintas: `self_pct` cae con los activos jovenes (no
    tienen historia) y `cross_pct` cae con los dias flacos (pocas entidades). Verlas
    separadas es lo que dice si falta CALENDARIO o falta UNIVERSO.

    OJO CON EL DENOMINADOR: es la rejilla UNION de todas las fuentes, y las fuentes no
    comparten eje —una habla de cadenas, otra de series macro, otra de tokens—, asi que la
    mayor parte de las celdas estan vacias POR CONSTRUCCION y no por falta de dato. La
    cifra sirve para ver como evoluciona, no como nota sobre diez.
    """
    if frame is None or frame.empty:
        return {"rows": 0, "columns": 0, "self_pct": 0.0, "cross_pct": 0.0}

    self_cols = [c for c in frame.columns if c.endswith(SUFFIX_SELF)]
    cross_cols = [c for c in frame.columns if c.endswith(SUFFIX_CROSS)]

    def filled(columns: list[str]) -> float:
        if not columns:
            return 0.0
        block = frame[columns]
        return round(100.0 * float(block.notna().to_numpy().sum()) / block.size, 2)

    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "entities": int(frame.index.get_level_values(ENTITY).nunique()),
        "self_pct": filled(self_cols),
        "cross_pct": filled(cross_cols),
    }


__all__ = [
    "IQR_TO_SIGMA",
    "MIN_CROSS_SECTION",
    "MIN_HISTORY",
    "SUFFIX_CROSS",
    "SUFFIX_SELF",
    "Z_CLIP",
    "normalization_coverage",
    "normalization_spec",
    "normalize_features",
    "normalized_names",
    "panel",
]
