from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CEMConfig:
    """
    Cross-Entropy Method: muestrea de una gaussiana, se queda con la elite y reajusta
    la gaussiana hacia ella. Robusto, pocas evaluaciones, determinista con seed.
    """

    population: int = 24
    elite_frac: float = 0.25
    iterations: int = 8
    init_std_frac: float = 0.25  # std inicial como fraccion del rango de cada dimension
    min_std_frac: float = 0.02  # suelo de std para no colapsar antes de tiempo
    seed: int = 0

    def __post_init__(self) -> None:
        if self.population < 2:
            raise ValueError("population must be >= 2")
        if not 0.0 < self.elite_frac <= 1.0:
            raise ValueError("elite_frac must be in (0, 1]")
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")


@dataclass(slots=True)
class CEMResult:
    best_vector: np.ndarray
    best_score: float
    history: list[dict] = field(default_factory=list)


def maximize(
    objective: Callable[[np.ndarray], float],
    lows: np.ndarray,
    highs: np.ndarray,
    config: CEMConfig | None = None,
) -> CEMResult:
    """
    Maximiza `objective` sobre la caja [lows, highs] por CEM.

    Determinista: mismo (objective, lows, highs, config.seed) -> mismo resultado.
    La gaussiana se inicializa en el centro de la caja con std = init_std_frac*rango;
    cada iteracion reajusta media y std a la elite, con un suelo de std.
    """
    config = config or CEMConfig()
    lows = np.asarray(lows, dtype=float)
    highs = np.asarray(highs, dtype=float)
    if lows.shape != highs.shape:
        raise ValueError("lows and highs must have the same shape")

    span = highs - lows
    rng = np.random.default_rng(config.seed)

    mean = (lows + highs) / 2.0
    std = config.init_std_frac * span
    std_floor = config.min_std_frac * span
    n_elite = max(1, round(config.population * config.elite_frac))

    best_vector = mean.copy()
    best_score = -np.inf
    history: list[dict] = []

    for iteration in range(config.iterations):
        samples = rng.normal(mean, std, size=(config.population, mean.size))
        samples = np.clip(samples, lows, highs)

        scores = np.array([float(objective(sample)) for sample in samples])
        order = np.argsort(scores)[::-1]  # descendente: mejor primero
        elite = samples[order[:n_elite]]

        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), std_floor)

        iter_best_idx = int(order[0])
        if scores[iter_best_idx] > best_score:
            best_score = float(scores[iter_best_idx])
            best_vector = samples[iter_best_idx].copy()

        elite_mean_score = float(scores[order[:n_elite]].mean())
        history.append(
            {
                "iteration": iteration,
                "best_score": round(best_score, 4),
                "iter_best": round(float(scores[iter_best_idx]), 4),
                "elite_mean": round(elite_mean_score, 4),
            }
        )
        logger.info(
            "CEM iter %d | best=%.4f | iter_best=%.4f | elite_mean=%.4f",
            iteration, best_score, scores[iter_best_idx], elite_mean_score,
        )

    return CEMResult(best_vector=best_vector, best_score=best_score, history=history)
