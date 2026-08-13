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


# --- las seis primitivas TEMATICAS ---------------------------------------------------
#
# NINGUNA dimension toca un umbral de senal, y no es un olvido: la cobertura de un tema crece
# de forma monotona con el calendario de captura (dieciseis de las treinta fuentes empezaron a
# existir el dia que arranco la captura), asi que "cobertura del tema" esta correlacionada
# casi uno a uno con la FECHA. Un piso de tono sorteable permitiria al optimizador elegir
# implicitamente EN QUE TRAMO DE HISTORIA se le deja operar a la estrategia, y rankearia por
# disponibilidad de datos en vez de por criterio. Es el mismo fallo que el docstring de
# `observation/signal_radar.py` describe para `MIN_SIGNAL_COVERAGE`, y aqui seria peor porque
# la senal es la primitiva y no un anadido.
#
# Consecuencia aceptada y declarada: la capa de senal de estas seis SE AFIRMA, no se optimiza.
# Si esta mal, el backtest dira "esta familia no anade nada sobre su nucleo", que es
# exactamente la medicion que se busca. La via para encenderla es la que ya existe:
# `scoring/signal_study.py` inyecta el umbral desde FUERA del espacio de busqueda.


def _finalize_liquidation_cascade(raw: dict) -> dict:
    mean_window = int(raw["mean_window"])
    atr_window = int(raw["atr_window"])
    return {
        "mean_window": mean_window,
        "atr_window": atr_window,
        "entry_stretch_atr": raw["entry_stretch_atr"],
        "entry_range_atr": raw["entry_range_atr"],
        "close_location_max": raw["close_location_max"],
        "min_atr_pct": raw["min_atr_pct"],
        "stop_atr_mult": raw["stop_atr_mult"],
        "target_atr_mult": raw["target_atr_mult"],
        "min_bars": mean_window + atr_window + 1,
        "allow_short": True,
    }


def _finalize_vol_term_structure(raw: dict) -> dict:
    fast = int(raw["rv_fast_window"])
    slow = int(raw["rv_slow_window"])
    # La config RECHAZA fast >= slow, asi que el vector plano tiene que repararse aqui o el
    # optimizador produciria candidatos invalidos en la esquina del hipercubo.
    if fast >= slow:
        fast = max(5, slow // 2)
    breakout = int(raw["breakout_lookback"])
    return {
        "rv_fast_window": fast,
        "rv_slow_window": slow,
        "max_compression": raw["max_compression"],
        "breakout_lookback": breakout,
        "atr_window": int(raw["atr_window"]),
        "min_atr_pct": raw["min_atr_pct"],
        "stop_atr_mult": raw["stop_atr_mult"],
        "reward_atr_mult": raw["reward_atr_mult"],
        "min_bars": slow + breakout + 1,
        "allow_short": True,
    }


def _finalize_event_calendar_drift(raw: dict) -> dict:
    drift = int(raw["drift_window"])
    confirm = min(int(raw["confirm_window"]), drift - 1)
    atr_window = int(raw["atr_window"])
    min_drift = float(raw["min_drift_pct"])
    return {
        "drift_window": drift,
        "confirm_window": max(1, confirm),
        "min_drift_pct": min_drift,
        # El techo tiene que quedar por encima del piso con holgura o la banda es vacia.
        "max_drift_pct": max(float(raw["max_drift_pct"]), min_drift * 1.5),
        "atr_window": atr_window,
        "min_atr_pct": raw["min_atr_pct"],
        "stop_atr_mult": raw["stop_atr_mult"],
        "reward_atr_mult": raw["reward_atr_mult"],
        "min_bars": drift + atr_window + 10,
        "allow_short": True,
    }


def _finalize_attention_ignition(raw: dict) -> dict:
    trend_window = int(raw["trend_window"])
    volume_lookback = int(raw["volume_lookback"])
    atr_window = int(raw["atr_window"])
    return {
        "volume_lookback": volume_lookback,
        "volume_mult": raw["volume_mult"],
        "close_location_min": raw["close_location_min"],
        "trend_window": trend_window,
        "atr_window": atr_window,
        "min_atr_pct": raw["min_atr_pct"],
        "stop_atr_mult": raw["stop_atr_mult"],
        "reward_atr_mult": raw["reward_atr_mult"],
        "min_bars": max(trend_window, volume_lookback) + atr_window + 1,
        # La tesis, no un default: ver el docstring de attention_ignition.
        "allow_short": False,
    }


def _finalize_flow_persistence(raw: dict) -> dict:
    trend_window = int(raw["trend_window"])
    slope_window = min(int(raw["slope_window"]), trend_window - 1)
    persistence_window = int(raw["persistence_window"])
    atr_window = int(raw["atr_window"])
    return {
        "trend_window": trend_window,
        "slope_window": max(1, slope_window),
        "persistence_window": persistence_window,
        "min_persistence": raw["min_persistence"],
        "pullback_atr": raw["pullback_atr"],
        "atr_window": atr_window,
        "stop_atr_mult": raw["stop_atr_mult"],
        "reward_atr_mult": raw["reward_atr_mult"],
        "min_bars": trend_window + max(persistence_window, atr_window) + 1,
        "allow_short": True,
    }


def _finalize_signal_composite(raw: dict) -> dict:
    trigger = int(raw["trigger_window"])
    trend = int(raw["trend_window"])
    if trend <= trigger:
        trend = trigger + 10
    cross_lookback = int(raw["cross_lookback"])
    atr_window = int(raw["atr_window"])
    return {
        "trigger_window": trigger,
        "trend_window": trend,
        "cross_lookback": cross_lookback,
        "max_stretch_atr": raw["max_stretch_atr"],
        "atr_window": atr_window,
        "min_atr_pct": raw["min_atr_pct"],
        "stop_atr_mult": raw["stop_atr_mult"],
        "reward_atr_mult": raw["reward_atr_mult"],
        "min_bars": trend + cross_lookback + atr_window + 1,
        "allow_short": True,
    }


def _liquidation_cascade_space() -> ParamSpace:
    return ParamSpace(
        strategy_type="liquidation_cascade",
        dims=(
            ParamDim("mean_window", 10, 40, is_int=True),
            ParamDim("atr_window", 7, 28, is_int=True),
            ParamDim("entry_stretch_atr", 1.0, 4.0),
            ParamDim("entry_range_atr", 1.0, 3.0),
            ParamDim("close_location_max", 0.05, 0.45),
            ParamDim("min_atr_pct", 0.1, 2.0),
            ParamDim("stop_atr_mult", 1.0, 4.0),
            ParamDim("target_atr_mult", 1.0, 5.0),
        ),
        finalize=_finalize_liquidation_cascade,
    )


def _vol_term_structure_space() -> ParamSpace:
    return ParamSpace(
        strategy_type="vol_term_structure",
        dims=(
            ParamDim("rv_fast_window", 5, 20, is_int=True),
            ParamDim("rv_slow_window", 30, 120, is_int=True),
            ParamDim("max_compression", 0.4, 1.2),
            ParamDim("breakout_lookback", 5, 40, is_int=True),
            ParamDim("atr_window", 7, 28, is_int=True),
            ParamDim("min_atr_pct", 0.05, 2.0),
            ParamDim("stop_atr_mult", 1.0, 4.0),
            ParamDim("reward_atr_mult", 1.5, 6.0),
        ),
        finalize=_finalize_vol_term_structure,
    )


def _event_calendar_drift_space() -> ParamSpace:
    return ParamSpace(
        strategy_type="event_calendar_drift",
        dims=(
            ParamDim("drift_window", 2, 20, is_int=True),
            ParamDim("confirm_window", 1, 10, is_int=True),
            ParamDim("min_drift_pct", 0.5, 8.0),
            ParamDim("max_drift_pct", 5.0, 30.0),
            ParamDim("atr_window", 7, 28, is_int=True),
            ParamDim("min_atr_pct", 0.05, 2.0),
            ParamDim("stop_atr_mult", 1.0, 4.0),
            ParamDim("reward_atr_mult", 1.0, 5.0),
        ),
        finalize=_finalize_event_calendar_drift,
    )


def _attention_ignition_space() -> ParamSpace:
    return ParamSpace(
        strategy_type="attention_ignition",
        dims=(
            ParamDim("volume_lookback", 10, 60, is_int=True),
            ParamDim("volume_mult", 1.5, 5.0),
            ParamDim("close_location_min", 0.55, 0.95),
            ParamDim("trend_window", 20, 100, is_int=True),
            ParamDim("atr_window", 7, 28, is_int=True),
            ParamDim("min_atr_pct", 0.05, 2.0),
            ParamDim("stop_atr_mult", 1.0, 4.0),
            ParamDim("reward_atr_mult", 1.5, 6.0),
        ),
        finalize=_finalize_attention_ignition,
    )


def _flow_persistence_space() -> ParamSpace:
    return ParamSpace(
        strategy_type="flow_persistence",
        dims=(
            ParamDim("trend_window", 20, 120, is_int=True),
            ParamDim("slope_window", 3, 30, is_int=True),
            ParamDim("persistence_window", 10, 60, is_int=True),
            ParamDim("min_persistence", 0.50, 0.80),
            ParamDim("pullback_atr", 0.25, 3.0),
            ParamDim("atr_window", 7, 28, is_int=True),
            ParamDim("stop_atr_mult", 1.0, 4.0),
            ParamDim("reward_atr_mult", 1.5, 6.0),
        ),
        finalize=_finalize_flow_persistence,
    )


def _signal_composite_space() -> ParamSpace:
    return ParamSpace(
        strategy_type="signal_composite",
        dims=(
            ParamDim("trigger_window", 5, 40, is_int=True),
            ParamDim("trend_window", 30, 150, is_int=True),
            ParamDim("cross_lookback", 1, 10, is_int=True),
            ParamDim("max_stretch_atr", 0.5, 5.0),
            ParamDim("atr_window", 7, 28, is_int=True),
            ParamDim("min_atr_pct", 0.05, 2.0),
            ParamDim("stop_atr_mult", 1.0, 4.0),
            ParamDim("reward_atr_mult", 1.5, 6.0),
        ),
        finalize=_finalize_signal_composite,
    )


SPACES: dict[str, ParamSpace] = {
    "crypto_momentum": _momentum_space(),
    "mean_reversion": _mean_reversion_space(),
    "liquidation_cascade": _liquidation_cascade_space(),
    "vol_term_structure": _vol_term_structure_space(),
    "event_calendar_drift": _event_calendar_drift_space(),
    "attention_ignition": _attention_ignition_space(),
    "flow_persistence": _flow_persistence_space(),
    "signal_composite": _signal_composite_space(),
}


def get_space(strategy_type: str) -> ParamSpace:
    space = SPACES.get(strategy_type)
    if space is None:
        known = ", ".join(sorted(SPACES))
        raise ValueError(f"No search space for '{strategy_type}'. Known: {known}")
    return space
