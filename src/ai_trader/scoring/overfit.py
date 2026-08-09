"""
Descuento del sobreajuste por MULTIPLES PRUEBAS.

Buscar sobre un espacio de parametros garantiza encontrar algo que brilla aunque no
haya nada: con N intentos independientes sobre puro ruido, el mejor Sharpe esperado
crece con N. Estas dos piezas ponen numero a eso, cada una desde un angulo distinto:

- `deflated_sharpe_ratio` (DSR, Bailey & Lopez de Prado): dado el Sharpe del ganador y
  la dispersion de los Sharpe de TODOS los intentos, devuelve la probabilidad de que el
  Sharpe verdadero sea > 0 una vez descontado el maximo esperado bajo la hipotesis
  nula, corrigiendo ademas por asimetria y colas de los retornos.

- `probability_of_backtest_overfitting` (PBO por CSCV): con la matriz muestras x
  configuraciones, parte las muestras en bloques, en cada combinacion train/test elige
  la configuracion ganadora en train y mira su rango en test. PBO es la fraccion de
  combinaciones en las que la ganadora cae por debajo de la mediana fuera de muestra.
  Es la pregunta directa: "cuando elijo por backtest, ¿acierto?".

Ambas son puras y deterministas: mismos numeros de entrada, mismo veredicto.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from ai_trader.backtest.metrics import TRADING_DAYS_PER_YEAR

EULER_MASCHERONI = 0.5772156649015329
DEFAULT_PBO_BLOCKS = 10  # C(10,5) = 252 combinaciones train/test


@dataclass(slots=True, frozen=True)
class DeflatedSharpe:
    """Veredicto DSR. `computable=False` cuando no hay datos suficientes: se reporta el
    hueco en vez de fabricar una probabilidad."""

    dsr: float
    observed_sharpe: float
    expected_max_sharpe: float
    trial_sharpe_std: float
    n_trials: int
    n_observations: int
    computable: bool

    def as_dict(self) -> dict:
        return {
            "dsr": round(self.dsr, 4),
            "observed_sharpe": round(self.observed_sharpe, 4),
            "expected_max_sharpe": round(self.expected_max_sharpe, 4),
            "trial_sharpe_std": round(self.trial_sharpe_std, 4),
            "n_trials": self.n_trials,
            "n_observations": self.n_observations,
            "computable": self.computable,
        }


@dataclass(slots=True, frozen=True)
class PBOResult:
    """Veredicto PBO. `pbo` es la fraccion de particiones en las que la ganadora
    in-sample queda por debajo de la mediana out-of-sample; 0.5 es tirar una moneda."""

    pbo: float
    n_trials: int
    n_blocks: int
    n_splits: int
    n_samples_used: int
    median_logit: float
    computable: bool

    def as_dict(self) -> dict:
        return {
            "pbo": round(self.pbo, 4),
            "n_trials": self.n_trials,
            "n_blocks": self.n_blocks,
            "n_splits": self.n_splits,
            "n_samples_used": self.n_samples_used,
            "median_logit": round(self.median_logit, 4),
            "computable": self.computable,
        }


def deflated_sharpe_ratio(
    observed_sharpe: float,
    trial_sharpes: Sequence[float],
    n_observations: int,
    *,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> DeflatedSharpe:
    """
    DSR del Sharpe ganador dado el conjunto de Sharpe probados.

    `observed_sharpe` y `trial_sharpes` van ANUALIZADOS (como los devuelve
    metrics.sharpe_ratio); internamente se desanualizan porque la formula vive en la
    frecuencia de los retornos. `n_observations` es el numero de retornos del track
    record out-of-sample, y `kurtosis` NO es en exceso (3.0 = normal).
    """
    trials = [float(s) for s in trial_sharpes]
    n_trials = len(trials)
    scale = math.sqrt(periods_per_year)

    sr = observed_sharpe / scale
    std_trials = _std(trials) / scale
    sr0 = _expected_max_sharpe(std_trials, n_trials)

    if n_observations < 2:
        return DeflatedSharpe(
            dsr=0.0,
            observed_sharpe=observed_sharpe,
            expected_max_sharpe=sr0 * scale,
            trial_sharpe_std=std_trials * scale,
            n_trials=n_trials,
            n_observations=n_observations,
            computable=False,
        )

    # Varianza del estimador de Sharpe con retornos no normales (Mertens/Bailey).
    variance = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr * sr
    if variance <= 0:
        return DeflatedSharpe(
            dsr=0.0,
            observed_sharpe=observed_sharpe,
            expected_max_sharpe=sr0 * scale,
            trial_sharpe_std=std_trials * scale,
            n_trials=n_trials,
            n_observations=n_observations,
            computable=False,
        )

    z = (sr - sr0) * math.sqrt(n_observations - 1) / math.sqrt(variance)
    return DeflatedSharpe(
        dsr=_norm_cdf(z),
        observed_sharpe=observed_sharpe,
        expected_max_sharpe=sr0 * scale,
        trial_sharpe_std=std_trials * scale,
        n_trials=n_trials,
        n_observations=n_observations,
        computable=True,
    )


def probability_of_backtest_overfitting(
    matrix: Sequence[Sequence[float]],
    *,
    n_blocks: int = DEFAULT_PBO_BLOCKS,
) -> PBOResult:
    """
    PBO por Combinatorially Symmetric Cross-Validation.

    `matrix` es muestras x configuraciones: `matrix[s][c]` es el score de la config `c`
    en la muestra `s`. Las muestras se parten en `n_blocks` bloques contiguos (par); por
    cada eleccion de la mitad de los bloques como train, se toma la config ganadora en
    train y se mira su rango entre todas en test. PBO = P(logit del rango <= 0), o sea:
    con que frecuencia elegir por backtest deja a la ganadora en la mitad mala.
    """
    rows = [list(map(float, row)) for row in matrix]
    if not rows:
        return _pbo_unavailable(0, n_blocks, 0)

    n_trials = len(rows[0])
    if n_trials < 2 or any(len(r) != n_trials for r in rows):
        return _pbo_unavailable(n_trials, n_blocks, 0)

    blocks = _even_blocks(len(rows), n_blocks)
    if blocks < 2:
        return _pbo_unavailable(n_trials, blocks, 0)

    block_size = len(rows) // blocks
    used = block_size * blocks  # las filas sobrantes se descartan (bloques iguales)
    partition = [
        list(range(b * block_size, (b + 1) * block_size)) for b in range(blocks)
    ]

    logits: list[float] = []
    for train_blocks in combinations(range(blocks), blocks // 2):
        train_idx = [i for b in train_blocks for i in partition[b]]
        test_idx = [i for b in range(blocks) if b not in train_blocks for i in partition[b]]

        train_perf = _column_means(rows, train_idx, n_trials)
        test_perf = _column_means(rows, test_idx, n_trials)

        winner = min(range(n_trials), key=lambda c: (-train_perf[c], c))
        omega = _relative_rank(test_perf, winner)
        logits.append(math.log(omega / (1.0 - omega)))

    pbo = sum(1 for value in logits if value <= 0.0) / len(logits)
    return PBOResult(
        pbo=pbo,
        n_trials=n_trials,
        n_blocks=blocks,
        n_splits=len(logits),
        n_samples_used=used,
        median_logit=_median(logits),
        computable=True,
    )


# ------------------------------------------------------------------ internals --------


def _pbo_unavailable(n_trials: int, n_blocks: int, used: int) -> PBOResult:
    return PBOResult(
        pbo=0.0,
        n_trials=n_trials,
        n_blocks=n_blocks,
        n_splits=0,
        n_samples_used=used,
        median_logit=0.0,
        computable=False,
    )


def _even_blocks(n_samples: int, requested: int) -> int:
    """Bloques efectivos: par, al menos 2, y nunca mas que muestras disponibles."""
    blocks = min(requested, n_samples)
    if blocks % 2:
        blocks -= 1
    return max(blocks, 0)


def _column_means(rows: list[list[float]], idx: list[int], n_cols: int) -> list[float]:
    return [sum(rows[i][c] for i in idx) / len(idx) for c in range(n_cols)]


def _relative_rank(performance: list[float], target: int) -> float:
    """Rango relativo del ganador en test, en (0, 1). Empates a rango medio, para no
    premiar ni castigar por el orden accidental de las columnas."""
    value = performance[target]
    below = sum(1 for v in performance if v < value)
    ties = sum(1 for i, v in enumerate(performance) if v == value and i != target)
    rank = below + 0.5 * ties + 1.0
    return rank / (len(performance) + 1.0)


def _expected_max_sharpe(std_trials: float, n_trials: int) -> float:
    """Maximo Sharpe esperado bajo la nula (SR verdadero = 0) con `n_trials` intentos.
    Con un solo intento no hay nada que deflactar y vale 0."""
    if n_trials < 2 or std_trials <= 0:
        return 0.0
    a = _norm_ppf(1.0 - 1.0 / n_trials)
    b = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return std_trials * ((1.0 - EULER_MASCHERONI) * a + EULER_MASCHERONI * b)


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Coeficientes de la aproximacion racional de Acklam para la normal inversa; con un
# refinamiento de Halley el error queda por debajo de 1e-15. Se implementa aqui porque
# el proyecto no depende de scipy.
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)
_P_LOW = 0.02425


def _norm_ppf(p: float) -> float:
    """Inversa de la normal estandar. Fuera de (0, 1) se satura en +-8 sigma en vez de
    explotar: el DSR con un unico intento no debe romper el harness."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0

    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
        x /= (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
    elif p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        x = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q
        x /= ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
        x = -x / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)

    # Refinamiento de Halley sobre el residuo de la CDF.
    error = _norm_cdf(x) - p
    u = error * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)
