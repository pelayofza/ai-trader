"""
¿Se parece el mundo sintetico al mercado REAL?

El generador de ai_v2 se construyo para tener colas gruesas, clustering de volatilidad y
estructura serial (ver synthetic/retrofit.py). Que esas propiedades EXISTAN ya esta
comprobado; lo que faltaba es comprobar que estan en la MAGNITUD del mercado real. Este
modulo es la parte medible de esa pregunta: dado un conjunto de series de retornos
sinteticas y otro de series reales, calcula los mismos stylized-facts sobre ambos y los
compara.

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
FIDELITY_REPORT = Path("data") / "fidelity" / "report_ai_v2.json"

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


def autocorrelation(values: np.ndarray, lag: int) -> float:
    """
    Autocorrelacion muestral al lag dado. NaN si la serie es demasiado corta o constante:
    NaN se propaga y se filtra explicitamente, en vez de fingir un 0 que se leeria como
    "no hay estructura".
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")
    x = np.asarray(values, dtype=float)
    if x.size <= lag + 1:
        return float("nan")
    a, b = x[:-lag], x[lag:]
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


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


# ------------------------------------------------------------------ informe ----------


def load_fidelity_report(path: Path | str = FIDELITY_REPORT) -> dict | None:
    """Lee el informe publicado. None si no esta, para que los generadores de dashboard y
    documentacion degraden a prosa sin cifras en vez de romperse."""
    report = Path(path)
    if not report.exists():
        return None
    return json.loads(report.read_text(encoding="utf-8"))
