from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # `activity` importa de aqui: la dependencia real va en un solo sentido
    from ai_trader.scoring.activity import ActivityStats

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

    `activity` viaja PEGADA a la recompensa y no como un dato que haya que ir a buscar a
    otro informe. Sin ella el numero es ambiguo por construccion: una ventana en la que la
    configuracion no abrio ninguna posicion puntua 0 EXACTO —Sharpe 0, rotacion 0, caida
    0—, y un cero le gana a cualquier cosa que arriesgue y pierda. Un CVaR de 0 puede
    significar "no perdio nunca" o "no jugo nunca", y solo la actividad lo distingue. Ver
    `scoring.activity`.
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
    activity: "ActivityStats | None" = None

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
            "activity": None if self.activity is None else self.activity.as_dict(),
        }


def aggregate_reward(
    scores: Sequence[float],
    *,
    alpha: float = DEFAULT_CVAR_ALPHA,
    trades: Sequence[int] | None = None,
) -> RewardStats:
    """Agrega los headline scores por muestra en la recompensa (CVaR@alpha) y sus
    estadisticos de forma. `alpha` es la fraccion de cola promediada (0.25 = peor
    cuartil).

    `trades` son las operaciones de CADA una de esas mismas ventanas, en el mismo orden.
    Se pide aqui, y no en un paso posterior, para que la actividad y la recompensa no
    puedan salir de conjuntos de ventanas distintos: emparejarlas mal seria peor que no
    reportarlas. Si se pasa una lista de longitud distinta, es un error, no un aviso."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")

    arr = np.asarray(list(scores), dtype=float)
    activity = None
    if trades is not None:
        # Import diferido a proposito: `activity` toma de aqui la fraccion de cola
        # (DEFAULT_CVAR_ALPHA) con la que define su tolerancia de ventanas vacias, asi que
        # importarlo arriba seria un ciclo. La dependencia real es de un solo sentido.
        from ai_trader.scoring.activity import measure_activity

        counts = list(trades)
        if len(counts) != arr.size:
            raise ValueError(
                f"trades ({len(counts)}) y scores ({arr.size}) tienen que venir de las "
                "mismas ventanas"
            )
        activity = measure_activity(counts)
    if arr.size == 0:
        return RewardStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, alpha, activity)

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
        activity=activity,
    )
