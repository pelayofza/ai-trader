from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class ParamDim:
    """Una dimension del espacio de busqueda: rango [low, high] y si es entera."""

    name: str
    low: float
    high: float
    is_int: bool = False


@dataclass(slots=True, frozen=True)
class ParamSpace:
    """
    Espacio de busqueda de los parametros de una primitiva.

    El optimizador (CEM) trabaja en el vector continuo acotado por [lows, highs];
    `to_params` proyecta un vector a un dict de params VALIDO (redondea enteros,
    recorta a rango y aplica las restricciones de coherencia de la estrategia via
    `finalize`), listo para build_strategy.
    """

    strategy_type: str
    dims: tuple[ParamDim, ...]
    finalize: Callable[[dict], dict]

    @property
    def lows(self) -> np.ndarray:
        return np.array([d.low for d in self.dims], dtype=float)

    @property
    def highs(self) -> np.ndarray:
        return np.array([d.high for d in self.dims], dtype=float)

    @property
    def dim(self) -> int:
        return len(self.dims)

    def midpoint(self) -> np.ndarray:
        return (self.lows + self.highs) / 2.0

    def to_params(self, vector: np.ndarray) -> dict:
        raw: dict[str, float | int] = {}
        for d, value in zip(self.dims, vector):
            clipped = float(np.clip(value, d.low, d.high))
            raw[d.name] = int(round(clipped)) if d.is_int else clipped
        return self.finalize(raw)


# --- finalizadores: restricciones de coherencia que el vector plano no captura -----


def _finalize_momentum(raw: dict) -> dict:
    fast = int(raw["fast_sma_window"])
    slow = int(raw["slow_sma_window"])
    # La media rapida debe ser mas corta que la lenta.
    if fast >= slow:
        fast = max(5, slow - 5)
    breakout = int(raw["breakout_lookback"])
    # min_bars debe cubrir la ventana mas larga (validacion de la config lo exige).
    min_bars = slow + max(breakout, 14) + 1
    return {
        "fast_sma_window": fast,
        "slow_sma_window": slow,
        "atr_window": 14,
        "breakout_lookback": breakout,
        "min_atr_pct": raw["min_atr_pct"],
        "risk_atr_multiple": raw["risk_atr_multiple"],
        "reward_atr_multiple": raw["reward_atr_multiple"],
        "require_breakout": True,
        "min_bars": min_bars,
        "min_breadth": raw["min_breadth"],
        "min_relative_strength": raw["min_relative_strength"],
    }


def _finalize_mean_reversion(raw: dict) -> dict:
    lookback = int(raw["lookback"])
    entry_z = float(raw["entry_z"])
    exit_z = float(raw["exit_z"])
    # El objetivo de salida debe quedar por debajo del umbral de entrada.
    if exit_z >= entry_z:
        exit_z = max(0.0, entry_z - 0.1)
    return {
        "lookback": lookback,
        "entry_z": entry_z,
        "exit_z": exit_z,
        "stop_atr_mult": raw["stop_atr_mult"],
        "atr_window": 14,
        "min_std_pct": raw["min_std_pct"],
        "min_bars": lookback + 10,
        "min_breadth": raw["min_breadth"],
        "max_relative_strength": raw["max_relative_strength"],
    }


def _momentum_space() -> ParamSpace:
    return ParamSpace(
        strategy_type="crypto_momentum",
        dims=(
            ParamDim("fast_sma_window", 5, 30, is_int=True),
            ParamDim("slow_sma_window", 20, 80, is_int=True),
            ParamDim("breakout_lookback", 2, 20, is_int=True),
            ParamDim("min_atr_pct", 0.05, 2.0),
            ParamDim("risk_atr_multiple", 1.0, 4.0),
            ParamDim("reward_atr_multiple", 1.5, 6.0),
            ParamDim("min_breadth", 0.0, 0.8),
            ParamDim("min_relative_strength", -1.0, 0.5),
        ),
        finalize=_finalize_momentum,
    )


def _mean_reversion_space() -> ParamSpace:
    return ParamSpace(
        strategy_type="mean_reversion",
        dims=(
            ParamDim("lookback", 10, 40, is_int=True),
            ParamDim("entry_z", 1.0, 3.5),
            ParamDim("exit_z", 0.0, 1.0),
            ParamDim("stop_atr_mult", 1.0, 4.0),
            ParamDim("min_std_pct", 0.05, 1.0),
            ParamDim("min_breadth", 0.0, 0.8),
            ParamDim("max_relative_strength", -0.2, 1.0),
        ),
        finalize=_finalize_mean_reversion,
    )


SPACES: dict[str, ParamSpace] = {
    "crypto_momentum": _momentum_space(),
    "mean_reversion": _mean_reversion_space(),
}


def get_space(strategy_type: str) -> ParamSpace:
    space = SPACES.get(strategy_type)
    if space is None:
        known = ", ".join(sorted(SPACES))
        raise ValueError(f"No search space for '{strategy_type}'. Known: {known}")
    return space
