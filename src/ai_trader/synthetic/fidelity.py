"""
¿Se parece el mundo sintetico al mercado REAL?

El generador se construyo para tener colas gruesas, clustering de volatilidad y estructura
serial (ver synthetic/retrofit.py). Que esas propiedades EXISTAN se comprueba comparando
una libreria con la anterior; lo que no se puede comprobar asi es que esten en la MAGNITUD
del mercado real. Este modulo es la parte medible de esa pregunta: dado un conjunto de
series de retornos sinteticas y otro de series reales, calcula los mismos stylized-facts
sobre ambos y los compara. Y no solo los compara: los contrasta con umbrales de aceptacion
declarados (ver mas abajo), de modo que el estudio puede FALLAR. Un estudio que no puede
salir mal no es evidencia.

Cuatro familias de hechos estilizados, que son las que el generador dice reproducir:

- AUTOCORRELACION de retornos (lag 1). En el mercado real es practicamente cero (si no,
  habria dinero gratis). En el sintetico la fija `idio_ar` por fase, con signo segun el
  regimen: es la propiedad que hace que la reversion a la media sea ganable.
- CLUSTERING de volatilidad: autocorrelacion de |retorno|, a lag 1 y promediada hasta el
  lag 10. Es el hecho estilizado mas robusto de los mercados reales y el que mas cuesta
  falsificar: sobrevive a cualquier normalizacion.
- COLAS: exceedances mas alla de 3 sigma (%) y curtosis en exceso. Bajo una normal las
  exceedances serian 0,27%; los mercados reales dan varias veces mas.
- CORRELACIONES CRUZADAS entre activos, par a par. En el sintetico emergen del modelo de
  factores; en el real, de la realidad. Es el unico eje donde el generador SI diferencia
  entre activos (via las cargas), asi que es donde una correlacion de rangos tiene algo
  que medir.

Dos ejes de comparacion, deliberadamente distintos:

- NIVEL: mediana real vs mediana sintetica de cada metrica. Responde "¿la magnitud es la
  del mercado?".
- ORDENACION (`rank_corr`, Spearman): ¿ordena el sintetico los activos -o los pares- como
  los ordena el mercado? Es invariante a la escala, asi que responde a algo que el nivel
  no puede: si el mundo sintetico sabe QUE activo tiene mas cola o QUE par esta mas
  acoplado, aunque el nivel absoluto este mal calibrado.
- COBERTURA: que fraccion de los valores reales cae dentro del rango [p10, p90] que el
  ensemble sintetico produce para ese mismo activo. Un generador honesto no tiene que
  clavar el valor real: tiene que producirlo como una realizacion plausible.

Todo aqui es NUMERICO Y PURO: ni red, ni disco, ni aleatoriedad. La descarga de datos
reales (con cache en disco) y la escritura del informe viven en `fidelity_study.py`.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ai_trader.scoring.weight_calibration import spearman

# Evidencia publicada del estudio. La genera `fidelity_study` y la consumen los
# generadores de dashboard y documentacion, igual que el informe de calibracion: medir
# es caro (descarga + cientos de muestras), leer es gratis.
FIDELITY_DIR = Path("data") / "fidelity"

# Las DOS librerias que se publican juntas, y por que hay dos: ai_v2 es el generador cuyo
# hueco contra el mercado se midio (curtosis 0,37 contra 4,19) y ai_v3 el que lo cierra.
# Publicar solo la buena convertiria una correccion medida en una afirmacion sin control;
# el par es lo que hace auditable que el arreglo funciono y cuanto.
FIDELITY_LIBRARY = "ai_v3"
FIDELITY_BASELINE_LIBRARY = "ai_v2"


def fidelity_report_path(library_id: str = FIDELITY_LIBRARY) -> Path:
    return FIDELITY_DIR / f"report_{library_id}.json"


FIDELITY_REPORT = fidelity_report_path()

# Lags del clustering. 10 dias de mercado: suficiente para ver la persistencia sin
# entrar en la zona donde el estimador es puro ruido.
DEFAULT_MAX_LAG = 10
# Umbral de cola. 3 sigma bajo una normal = 0,27% de los dias.
DEFAULT_SIGMA = 3.0
GAUSSIAN_EXCEEDANCE_PCT = 100.0 * 2.0 * 0.001349898031630095  # P(|Z| > 3) exacto
# Minimo de observaciones para que una estimacion cuente. Por debajo, la autocorrelacion
# y sobre todo la curtosis son ruido con forma de numero.
MIN_OBSERVATIONS = 200
# Solape minimo entre dos series para estimar su correlacion cruzada.
MIN_PAIR_OVERLAP = 120

TRADING_DAYS_PER_YEAR = 365.0  # el universo comparado es cripto: cotiza todos los dias


@dataclass(frozen=True, slots=True)
class Metric:
    """Una metrica comparable, con como se muestra y si cuenta para el veredicto."""

    key: str
    label: str
    decimals: int
    # La volatilidad no es un hecho estilizado que el generador deba "acertar": la fija
    # el disenador escenario a escenario. Se reporta como contexto (sin ella no se sabe
    # si comparamos mundos de riesgo parecido) pero no entra en el veredicto.
    is_target: bool = True


METRICS: tuple[Metric, ...] = (
    Metric("ac1", "Autocorrelación de retornos (lag 1)", 3),
    Metric("ac_abs1", "Clustering de volatilidad (autocorr. |r|, lag 1)", 3),
    Metric("ac_abs_mean", f"Persistencia del clustering (|r|, lags 1-{DEFAULT_MAX_LAG})", 3),
    Metric("exceed_3sigma_pct", "Exceedances más allá de 3σ (%)", 2),
    Metric("excess_kurtosis", "Curtosis en exceso", 2),
    Metric("vol_annual_pct", "Volatilidad anualizada (%)", 1, is_target=False),
)

METRIC_KEYS: tuple[str, ...] = tuple(m.key for m in METRICS)
TARGET_METRIC_KEYS: tuple[str, ...] = tuple(m.key for m in METRICS if m.is_target)
CROSS_CORR_KEY = "cross_corr"
CROSS_CORR_LABEL = "Correlación cruzada entre activos (par a par)"


# ------------------------------------------------------------------ aceptacion --------
#
# El estudio deja de ser "un vistazo" cuando tiene umbrales que puede FALLAR. Estos son
# los del generador realista (ai_v3), y son deliberadamente dos cosas distintas:
#
# 1. COBERTURA: en al menos `min_coverage_pct` de los activos (o pares), el valor real
#    cae dentro del [p10, p90] del ensemble sintetico. Es el criterio honesto para un
#    generador estocastico: no tiene que clavar el numero, tiene que poder producirlo.
# 2. MEDIANA DE MERCADO: la mediana real de la seccion cruzada cae dentro del [p10, p90]
#    de TODAS las muestras sinteticas juntas. Es la promesa explicita del generador.
#
# LIMITE DECLARADO, y esta escrito aqui a proposito para que no se lea de mas: el p90 de
# la curtosis real de cripto va de 30 a 90 (DOGE y XRP en los anos de mania). Ni con
# dof=4 se reproduce eso. El generador cubre la MEDIANA del mercado -el cripto de un ano
# cualquiera-, no sus anos de mania, y por eso el umbral esta sobre la mediana y no sobre
# la cola de la seccion cruzada.

# Metricas cuya mediana de mercado tiene que caer dentro de la banda sintetica: son los
# tres hechos que el retrofit dice arreglar (colas, agrupamiento, exceedances).
MEDIAN_BAND_KEYS: tuple[str, ...] = ("excess_kurtosis", "ac_abs1", "exceed_3sigma_pct")

# Cobertura minima exigida a cada metrica objetivo. 60% no es "aprobado raspado": con 12
# activos significa que el mundo sintetico produce el valor real de 8 de ellos como una
# realizacion plausible. ai_v2 daba 35% de media (0% en curtosis).
MIN_COVERAGE_PCT = 60.0


def metric(key: str) -> Metric:
    for item in METRICS:
        if item.key == key:
            return item
    raise KeyError(f"Unknown fidelity metric '{key}'")


# ------------------------------------------------------------------ series -----------


def log_returns(closes: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    """Retornos logaritmicos de una serie de cierres. Descarta precios no positivos."""
    values = np.asarray(closes, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 2:
        return np.empty(0, dtype=float)
    return np.diff(np.log(values))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """
    Correlacion lineal (Pearson) entre dos series alineadas. NaN si alguna es constante o
    demasiado corta: NaN se propaga y se filtra explicitamente, en vez de fingir un 0 que
    se leeria como "no hay relacion".

    Es el cuerpo que estaba DENTRO de `autocorrelation`, extraido tal cual (mismo orden de
    operaciones en coma flotante, mismos golden) porque la autocorrelacion de una serie y
    el IC de un canal contra el retorno futuro son la misma cuenta con dos series
    distintas. Tener dos copias seria tener dos definiciones de correlacion que pueden
    derivar por separado.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size != y.size:
        raise ValueError("correlation needs two series of the same length")
    if x.size < 2:
        return float("nan")
    sa, sb = x.std(), y.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sa * sb))


def autocorrelation(values: np.ndarray, lag: int) -> float:
    """
    Autocorrelacion muestral al lag dado. NaN si la serie es demasiado corta o constante.
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")
    x = np.asarray(values, dtype=float)
    if x.size <= lag + 1:
        return float("nan")
    return correlation(x[:-lag], x[lag:])


@dataclass(frozen=True, slots=True)
class SeriesFacts:
    """Los stylized-facts de UNA serie de retornos (una ventana real o un path sintetico)."""

    n: int
    ac1: float
    ac_abs1: float
    ac_abs_mean: float
    exceed_3sigma_pct: float
    excess_kurtosis: float
    vol_annual_pct: float

    def value(self, key: str) -> float:
        return float(getattr(self, key))

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            **{key: round(self.value(key), 6) for key in METRIC_KEYS},
        }


def series_facts(
    returns: np.ndarray,
    *,
    max_lag: int = DEFAULT_MAX_LAG,
    sigma: float = DEFAULT_SIGMA,
    min_observations: int = MIN_OBSERVATIONS,
) -> SeriesFacts | None:
    """Mide una serie de retornos. Devuelve None si es demasiado corta o degenerada."""
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1")
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < min_observations:
        return None

    std = r.std()
    if std == 0:
        return None

    absr = np.abs(r)
    abs_acs = [autocorrelation(absr, lag) for lag in range(1, max_lag + 1)]
    usable = [v for v in abs_acs if not np.isnan(v)]
    z = (r - r.mean()) / std

    return SeriesFacts(
        n=int(r.size),
        ac1=autocorrelation(r, 1),
        ac_abs1=abs_acs[0],
        ac_abs_mean=float(np.mean(usable)) if usable else float("nan"),
        exceed_3sigma_pct=100.0 * float(np.mean(np.abs(z) > sigma)),
        excess_kurtosis=float(np.mean(z**4) - 3.0),
        vol_annual_pct=100.0 * float(std) * float(np.sqrt(TRADING_DAYS_PER_YEAR)),
    )


# ------------------------------------------------------- hechos de un CANAL ---------
#
# El loop de calibracion legitimo de este proyecto es este modulo: se cierra sobre
# PROPIEDADES DEL MUNDO medidas con estadisticos que ninguna estrategia optimiza, no sobre
# el rendimiento de las estrategias (eso seria sobreajuste con pasos extra). Cuando el
# mundo gana un canal de observacion, el loop tiene que ganar sus hechos: cuanta
# informacion lleva, con cuanto adelanto, y con que persistencia.

# Lags a cada lado del perfil lead-lag. Cinco dias bastan para ver donde esta la
# informacion con el horizonte mas largo que se barre y para que el lado negativo -el que
# no deberia existir- tenga suficientes puntos.
CHANNEL_MAX_LAG = 5

# Tolerancias de ACEPTACION de un canal, y las dos pueden fallar:
#
# - `CHANNEL_IC_TOLERANCE`: cuanto puede alejarse el IC medido del declarado. Con ~700
#   observaciones el error estandar de una correlacion es ~0,038, asi que 0,05 es poco mas
#   de un error estandar: exige que el canal entregue lo que dice sin exigirle que venza al
#   ruido de muestreo.
# - `CHANNEL_LEAK_TOLERANCE`: cuanta correlacion con retornos YA REALIZADOS se admite. Es
#   la MAXIMA sobre cinco lags, asi que el umbral tiene que dar margen a que el maximo de
#   cinco ruidos se salga: 0,10 son ~2,6 errores estandar.
CHANNEL_IC_TOLERANCE = 0.05
CHANNEL_LEAK_TOLERANCE = 0.10


@dataclass(frozen=True, slots=True)
class ChannelFacts:
    """
    Lo que un canal de observacion ENTREGA, medido, frente a lo que declara.

    Las tres cifras contestan tres preguntas distintas y ninguna sustituye a otra:

    - `ic`: correlacion entre la senal del dia t y el retorno que dice anticipar (t->t+h).
      Es el mando `rho` visto desde fuera. Si no coincide con `SignalChannel.expected_ic`,
      el canal no es el que se cree y el barrido esta etiquetando mal sus celdas.
    - `ac1`: autocorrelacion de la propia senal. Es lo que distingue una serie de confeti,
      y el numero que hay que mirar cuando el grupo de control (rho=0) puntua bien: si una
      estrategia gana con rho=0, esta explotando ESTO.
    - `lead_lag`: correlacion con el retorno de CADA dia alrededor, de -k a +k. Dice donde
      esta la informacion. El lado negativo es un test de falsacion gratis: si el canal
      correlaciona con retornos YA REALIZADOS, no es una senal adelantada sino una lectura
      del pasado, y la puerta estaria comprando informacion que la estrategia ya tiene en
      las barras.
    """

    n: int
    lead_days: int
    coverage_pct: float
    ic: float
    ac1: float
    lead_lag: tuple[tuple[int, float], ...]

    @property
    def past_leak(self) -> float:
        """Maxima |correlacion| con retornos ya realizados (lags <= 0 del perfil). Deberia
        ser ruido; si no lo es, el canal mira hacia atras y no hacia delante."""
        past = [abs(v) for lag, v in self.lead_lag if lag < 0 and not np.isnan(v)]
        return max(past) if past else float("nan")

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "lead_days": self.lead_days,
            "coverage_pct": round(self.coverage_pct, 2),
            "ic": _round_or_none(self.ic, 4),
            "ac1": _round_or_none(self.ac1, 4),
            "past_leak": _round_or_none(self.past_leak, 4),
            "lead_lag": {str(lag): _round_or_none(v, 4) for lag, v in self.lead_lag},
        }


def channel_facts(
    signal: Sequence[float] | np.ndarray,
    closes: Sequence[float] | np.ndarray,
    *,
    lead_days: int,
    max_lag: int = CHANNEL_MAX_LAG,
    min_observations: int = MIN_OBSERVATIONS,
) -> ChannelFacts | None:
    """
    Mide un canal contra los precios que dice anticipar. None si no hay muestra suficiente.

    `signal` y `closes` son la MISMA serie temporal alineada dia a dia; los NaN de la senal
    son dias sin observacion y se descartan por pares, nunca se rellenan.
    """
    if lead_days < 1:
        raise ValueError("lead_days must be >= 1")
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1")

    sig = np.asarray(signal, dtype=float)
    prices = np.asarray(closes, dtype=float)
    if sig.size != prices.size:
        raise ValueError("signal and closes must be aligned day by day")
    if sig.size <= lead_days + 1:
        return None

    valid = np.isfinite(prices) & (prices > 0)
    logp = np.where(valid, np.log(np.where(valid, prices, 1.0)), np.nan)
    n = sig.size
    observed = np.isfinite(sig)
    if int(observed.sum()) < min_observations:
        return None

    # Retorno del dia t -> t+1 (el que se realiza DESPUES de observar la senal de t) y
    # retorno acumulado del horizonte que el canal dice anticipar.
    daily = np.full(n, np.nan)
    daily[:-1] = logp[1:] - logp[:-1]
    horizon = np.full(n, np.nan)
    horizon[:-lead_days] = logp[lead_days:] - logp[:-lead_days]

    profile: list[tuple[int, float]] = []
    for lag in range(-max_lag, max_lag + 1):
        shifted = _shift(daily, lag)
        profile.append((lag, _paired_correlation(sig, shifted)))

    return ChannelFacts(
        n=int(observed.sum()),
        lead_days=lead_days,
        coverage_pct=100.0 * float(observed.mean()),
        ic=_paired_correlation(sig, horizon),
        # La autocorrelacion se mide sobre los dias observados CONSECUTIVOS: con cobertura
        # parcial, saltarse los huecos mediria la persistencia de otra serie.
        ac1=_paired_correlation(sig[:-1], sig[1:]),
        lead_lag=tuple(profile),
    )


def _shift(values: np.ndarray, lag: int) -> np.ndarray:
    """`values` desplazado para que la posicion t contenga el valor de t+lag."""
    out = np.full(values.size, np.nan)
    if lag == 0:
        return values.copy()
    if lag > 0:
        out[:-lag] = values[lag:]
    else:
        out[-lag:] = values[:lag]
    return out


def _paired_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Correlacion sobre los pares en los que las DOS series tienen dato."""
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return float("nan")
    return correlation(a[mask], b[mask])


def pair_key(a: str, b: str) -> str:
    """Clave estable (y serializable) de un par de activos."""
    first, second = sorted((a, b))
    return f"{first}|{second}"


def pair_correlations(
    returns: pd.DataFrame, *, min_overlap: int = MIN_PAIR_OVERLAP
) -> dict[str, float]:
    """
    Correlacion de Pearson par a par entre las columnas (un simbolo por columna).

    Alinea por indice y exige `min_overlap` observaciones comunes: comparar dos activos
    con veinte dias en comun no es medir su acoplamiento, es medir ruido.
    """
    if returns.shape[1] < 2:
        return {}
    matrix = returns.corr(min_periods=min_overlap)
    out: dict[str, float] = {}
    columns = sorted(matrix.columns)
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            value = matrix.at[a, b]
            if pd.notna(value):
                out[pair_key(a, b)] = float(value)
    return out


# --------------------------------------------------------------- agregacion ----------


@dataclass(frozen=True, slots=True)
class Estimate:
    """
    Una magnitud resumida sobre VARIAS estimaciones independientes (ventanas reales o
    paths sinteticos) del mismo activo o del mismo par.

    Mediana y no media: la curtosis de una ventana con un crash es un valor extremo
    legitimo que arrastraria la media entera. El rango [p10, p90] es lo que convierte la
    comparacion en algo mas honesto que "acerto / no acerto": el sintetico no tiene que
    clavar el numero real, tiene que poder producirlo.
    """

    median: float
    p10: float
    p90: float
    n: int

    def contains(self, value: float) -> bool:
        if np.isnan(value) or np.isnan(self.p10) or np.isnan(self.p90):
            return False
        return bool(self.p10 <= value <= self.p90)

    def as_dict(self) -> dict:
        return {
            "median": round(self.median, 6),
            "p10": round(self.p10, 6),
            "p90": round(self.p90, 6),
            "n": self.n,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> Estimate:
        return cls(
            median=float(data["median"]),
            p10=float(data["p10"]),
            p90=float(data["p90"]),
            n=int(data["n"]),
        )


def estimate(values: Sequence[float]) -> Estimate | None:
    """Resume una lista de estimaciones. None si no queda ningun valor utilizable."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return Estimate(
        median=float(np.median(arr)),
        p10=float(np.percentile(arr, 10)),
        p90=float(np.percentile(arr, 90)),
        n=int(arr.size),
    )


def aggregate_facts(facts: Sequence[SeriesFacts]) -> dict[str, Estimate]:
    """De N mediciones de un mismo activo a un Estimate por metrica."""
    out: dict[str, Estimate] = {}
    for key in METRIC_KEYS:
        agg = estimate([f.value(key) for f in facts])
        if agg is not None:
            out[key] = agg
    return out


def aggregate_pairs(pairs: Sequence[Mapping[str, float]]) -> dict[str, Estimate]:
    """De N matrices de correlacion (una por ventana/path) a un Estimate por par."""
    collected: dict[str, list[float]] = {}
    for snapshot in pairs:
        for key, value in snapshot.items():
            collected.setdefault(key, []).append(value)
    out: dict[str, Estimate] = {}
    for key, values in collected.items():
        agg = estimate(values)
        if agg is not None:
            out[key] = agg
    return out


# --------------------------------------------------------------- comparacion ---------


@dataclass(frozen=True, slots=True)
class ComparisonItem:
    """Un activo (o un par) con su valor real, su valor sintetico y si esta cubierto."""

    name: str
    real: float
    synth: float
    synth_p10: float
    synth_p90: float
    inside: bool

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "real": round(self.real, 6),
            "synth": round(self.synth, 6),
            "synth_p10": round(self.synth_p10, 6),
            "synth_p90": round(self.synth_p90, 6),
            "inside": self.inside,
        }


@dataclass(frozen=True, slots=True)
class Comparison:
    """El veredicto de UNA metrica: nivel, ordenacion y cobertura."""

    key: str
    label: str
    n: int
    rank_corr: float  # Spearman(real, sintetico) sobre la seccion cruzada
    real_median: float
    synth_median: float
    ratio: float  # sintetico / real (NaN si el real es ~0 y el cociente no significa nada)
    coverage_pct: float
    items: tuple[ComparisonItem, ...]

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "n": self.n,
            "rank_corr": _round_or_none(self.rank_corr, 4),
            "real_median": _round_or_none(self.real_median, 6),
            "synth_median": _round_or_none(self.synth_median, 6),
            "ratio": _round_or_none(self.ratio, 4),
            "coverage_pct": round(self.coverage_pct, 1),
            "items": [item.as_dict() for item in self.items],
        }


def _round_or_none(value: float, decimals: int):
    """NaN -> null en el JSON: un hueco se ve, un 0 inventado no."""
    if value is None or np.isnan(value):
        return None
    return round(float(value), decimals)


# Umbral por debajo del cual el cociente sintetico/real no significa nada (el
# denominador es ruido alrededor de cero, como pasa con la autocorrelacion real).
_RATIO_FLOOR = 1e-3


def compare(
    real: Mapping[str, Estimate],
    synth: Mapping[str, Estimate],
    *,
    key: str,
    label: str,
) -> Comparison:
    """
    Compara la seccion cruzada real contra la sintetica para UNA magnitud.

    Solo entran los elementos presentes en ambos lados: comparar activos distintos no es
    comparar. Los ausentes se declaran fuera, en el plan del informe.
    """
    names = sorted(set(real) & set(synth))
    items: list[ComparisonItem] = []
    for name in names:
        r, s = real[name], synth[name]
        if np.isnan(r.median) or np.isnan(s.median):
            continue
        items.append(
            ComparisonItem(
                name=name,
                real=r.median,
                synth=s.median,
                synth_p10=s.p10,
                synth_p90=s.p90,
                inside=s.contains(r.median),
            )
        )

    if not items:
        return Comparison(
            key=key, label=label, n=0, rank_corr=float("nan"),
            real_median=float("nan"), synth_median=float("nan"), ratio=float("nan"),
            coverage_pct=0.0, items=(),
        )

    real_values = [i.real for i in items]
    synth_values = [i.synth for i in items]
    real_median = float(np.median(real_values))
    synth_median = float(np.median(synth_values))
    ratio = (
        synth_median / real_median if abs(real_median) > _RATIO_FLOOR else float("nan")
    )

    return Comparison(
        key=key,
        label=label,
        n=len(items),
        rank_corr=spearman(real_values, synth_values),
        real_median=real_median,
        synth_median=synth_median,
        ratio=ratio,
        coverage_pct=100.0 * sum(1 for i in items if i.inside) / len(items),
        items=tuple(items),
    )


def compare_metrics(
    real: Mapping[str, Mapping[str, Estimate]],
    synth: Mapping[str, Mapping[str, Estimate]],
) -> list[Comparison]:
    """Una comparacion por metrica, con los activos como seccion cruzada."""
    out: list[Comparison] = []
    for item in METRICS:
        real_side = {
            symbol: facts[item.key] for symbol, facts in real.items() if item.key in facts
        }
        synth_side = {
            symbol: facts[item.key] for symbol, facts in synth.items() if item.key in facts
        }
        out.append(compare(real_side, synth_side, key=item.key, label=item.label))
    return out


def compare_cross_correlations(
    real: Mapping[str, Estimate], synth: Mapping[str, Estimate]
) -> Comparison:
    """La comparacion de correlaciones cruzadas, con los PARES como seccion cruzada."""
    return compare(real, synth, key=CROSS_CORR_KEY, label=CROSS_CORR_LABEL)


# ------------------------------------------------------------------ aceptacion --------


@dataclass(frozen=True, slots=True)
class AcceptanceCheck:
    """Un umbral concreto, con lo que se midio y si se cumple."""

    kind: str  # "coverage" | "market_median"
    key: str
    label: str
    value: float  # cobertura (%) o mediana real de la seccion cruzada
    threshold: float | None  # cobertura minima exigida
    band: tuple[float, float] | None  # [p10, p90] sintetico agregado
    passed: bool

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "key": self.key,
            "label": self.label,
            "value": _round_or_none(self.value, 4),
            "threshold": self.threshold,
            "band": None if self.band is None else [round(b, 6) for b in self.band],
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class Acceptance:
    """El veredicto binario del estudio: o el generador cumple los umbrales o no."""

    passed: bool
    min_coverage_pct: float
    checks: tuple[AcceptanceCheck, ...]

    @property
    def failures(self) -> tuple[AcceptanceCheck, ...]:
        return tuple(c for c in self.checks if not c.passed)

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "min_coverage_pct": self.min_coverage_pct,
            "median_band_keys": list(MEDIAN_BAND_KEYS),
            "checks": [c.as_dict() for c in self.checks],
        }


def channel_checks(
    declared: Mapping[str, float],
    measured: Mapping[str, Mapping[str, float]],
    *,
    ic_tolerance: float = CHANNEL_IC_TOLERANCE,
    leak_tolerance: float = CHANNEL_LEAK_TOLERANCE,
) -> list[AcceptanceCheck]:
    """
    Umbrales de aceptacion de los CANALES de observacion de una libreria.

    Extienden el loop de calibracion legitimo a las senales sin cambiar su naturaleza: las
    dos preguntas se contestan con estadisticos que NINGUNA estrategia optimiza.

    - ¿ENTREGA lo que declara? Un canal etiquetado rho=0,10 que entrega 0,02 haria que cada
      celda de un barrido dijera una cosa y midiera otra.
    - ¿MIRA hacia delante? La correlacion con retornos YA REALIZADOS tiene que ser ruido.
      Si no lo es, el canal no adelanta informacion: la repite, y la puerta estaria
      comprando algo que la estrategia ya tiene en las barras.

    `declared` es {canal -> IC esperado} y `measured` {canal -> {"ic":, "past_leak":}}.
    Un canal declarado y no medido FALLA; no medir no es aprobar.
    """
    checks: list[AcceptanceCheck] = []
    for name in sorted(declared):
        facts = measured.get(name) or {}
        ic = float(facts.get("ic", float("nan")))
        leak = float(facts.get("past_leak", float("nan")))
        target = float(declared[name])
        checks.append(
            AcceptanceCheck(
                kind="channel_ic",
                key=f"channel:{name}",
                label=f"Canal '{name}': IC entregado vs declarado ({target:+.3f})",
                value=ic,
                threshold=ic_tolerance,
                band=(target - ic_tolerance, target + ic_tolerance),
                passed=bool(np.isfinite(ic) and abs(ic - target) <= ic_tolerance),
            )
        )
        checks.append(
            AcceptanceCheck(
                kind="channel_leak",
                key=f"channel_leak:{name}",
                label=f"Canal '{name}': correlacion con retornos ya realizados",
                value=leak,
                threshold=leak_tolerance,
                band=None,
                passed=bool(np.isfinite(leak) and leak <= leak_tolerance),
            )
        )
    return checks


def evaluate_acceptance(
    comparisons: Sequence[Comparison],
    pooled_synth: Mapping[str, Estimate],
    *,
    min_coverage_pct: float = MIN_COVERAGE_PCT,
    extra_checks: Sequence[AcceptanceCheck] = (),
) -> Acceptance:
    """
    Aplica los umbrales publicados a un conjunto de comparaciones ya calculadas.

    `pooled_synth` es el [p10, p90] del ensemble sintetico ENTERO por metrica (todas las
    muestras de todos los activos juntas): es la banda contra la que se contrasta la
    mediana de mercado. Una comparacion sin elementos (n=0) falla; no medir no es aprobar.

    `extra_checks` son umbrales que no salen de comparar dos mundos de retornos: hoy, los
    de los canales de observacion (`channel_checks`). Entran en el MISMO veredicto binario
    a proposito —una libreria cuyo canal no entrega lo que declara no esta aceptada— en vez
    de publicarse en un bloque aparte que se pueda mirar solo cuando conviene.
    """
    checks: list[AcceptanceCheck] = []
    for item in comparisons:
        if item.key not in TARGET_METRIC_KEYS and item.key != CROSS_CORR_KEY:
            continue
        checks.append(
            AcceptanceCheck(
                kind="coverage",
                key=item.key,
                label=item.label,
                value=item.coverage_pct,
                threshold=min_coverage_pct,
                band=None,
                passed=bool(item.n) and item.coverage_pct >= min_coverage_pct,
            )
        )

    by_key = {c.key: c for c in comparisons}
    for key in MEDIAN_BAND_KEYS:
        item, band = by_key.get(key), pooled_synth.get(key)
        real_median = float("nan") if item is None else item.real_median
        inside = band is not None and band.contains(real_median)
        checks.append(
            AcceptanceCheck(
                kind="market_median",
                key=key,
                label=metric(key).label,
                value=real_median,
                threshold=None,
                band=None if band is None else (band.p10, band.p90),
                passed=bool(inside),
            )
        )

    checks.extend(extra_checks)

    return Acceptance(
        passed=all(c.passed for c in checks),
        min_coverage_pct=min_coverage_pct,
        checks=tuple(checks),
    )


# ------------------------------------------------------------------ informe ----------


def load_fidelity_report(path: Path | str = FIDELITY_REPORT) -> dict | None:
    """Lee el informe publicado. None si no esta, para que los generadores de dashboard y
    documentacion degraden a prosa sin cifras en vez de romperse."""
    report = Path(path)
    if not report.exists():
        return None
    return json.loads(report.read_text(encoding="utf-8"))
