from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_LAMBDA = 0.5  # penalizacion de dispersion en la recompensa media - lambda*std


@dataclass(slots=True, frozen=True)
class RewardStats:
    """
    Agregacion robusta del Calmar out-of-sample sobre una DISTRIBUCION de muestras
    (900 = 30 escenarios x 30 paths, o el subconjunto evaluado).

    La recompensa es `mean - lambda*std`: premia el centro y castiga la dispersion,
    empujando hacia politicas robustas y no a picos de un solo arquetipo. Se reportan
    ADEMAS percentil 25 y CVaR@25% para no esconder la cola: la unidad de evaluacion
    es la distribucion, nunca un path.
    """

    reward: float
    mean: float
    std: float
    p25: float
    cvar25: float
    best: float
    worst: float
    n: int
    lam: float

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
            "lam": self.lam,
        }


def aggregate_reward(scores: Sequence[float], *, lam: float = DEFAULT_LAMBDA) -> RewardStats:
    """Agrega los Calmar por muestra en la recompensa robusta y sus estadisticos de cola."""
    arr = np.asarray(list(scores), dtype=float)
    if arr.size == 0:
        return RewardStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, lam)

    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    p25 = float(np.percentile(arr, 25))

    # CVaR@25%: media del peor cuartil (Expected Shortfall). Consciente de la cola mala.
    k = max(1, math.ceil(0.25 * arr.size))
    cvar25 = float(np.sort(arr)[:k].mean())

    reward = mean - lam * std
    return RewardStats(
        reward=reward,
        mean=mean,
        std=std,
        p25=p25,
        cvar25=cvar25,
        best=float(arr.max()),
        worst=float(arr.min()),
        n=int(arr.size),
        lam=lam,
    )
