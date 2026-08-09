from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_CVAR_ALPHA = 0.25  # fraccion de cola (peor cuartil) que define la recompensa


@dataclass(slots=True, frozen=True)
class RewardStats:
    """
    Agregacion robusta del headline score out-of-sample sobre una DISTRIBUCION de
    muestras (900 = 30 escenarios x 30 paths, o el subconjunto evaluado).

    La recompensa ES el CVaR@25% (media del peor cuartil, Expected Shortfall): se
    optimiza y se rankea por la COLA MALA, no por el centro. Frente a `media -
    lambda*std`, el CVaR no premia la varianza al alza (una politica que a veces
    explota hacia arriba no compra con eso su cola mala) y no depende de un lambda
    arbitrario. Media, std y p25 se siguen reportando para no esconder la forma de la
    distribucion: la unidad de evaluacion es la distribucion, nunca un path.
    """

    reward: float
    mean: float
    std: float
    p25: float
    cvar25: float
    best: float
    worst: float
    n: int
    alpha: float

    def as_dict(self) -> dict:
        return {
            "reward": round(self.reward, 4),
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "p25": round(self.p25, 4),
            "cvar25": round(self.cvar25, 4),
            "best": round(self.best, 4),
            "worst": round(self.worst, 4),
            "n": self.n,
            "alpha": self.alpha,
        }


def aggregate_reward(
    scores: Sequence[float],
    *,
    alpha: float = DEFAULT_CVAR_ALPHA,
) -> RewardStats:
    """Agrega los headline scores por muestra en la recompensa (CVaR@alpha) y sus
    estadisticos de forma. `alpha` es la fraccion de cola promediada (0.25 = peor
    cuartil)."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")

    arr = np.asarray(list(scores), dtype=float)
    if arr.size == 0:
        return RewardStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, alpha)

    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    p25 = float(np.percentile(arr, 25))

    # CVaR@alpha: media del peor alpha-cuantil (Expected Shortfall). Es la recompensa.
    k = max(1, math.ceil(alpha * arr.size))
    cvar = float(np.sort(arr)[:k].mean())

    return RewardStats(
        reward=cvar,
        mean=mean,
        std=std,
        p25=p25,
        cvar25=cvar,
        best=float(arr.max()),
        worst=float(arr.min()),
        n=int(arr.size),
        alpha=alpha,
    )
