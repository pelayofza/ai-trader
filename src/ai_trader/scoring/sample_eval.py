from __future__ import annotations

import dataclasses
import logging
from datetime import datetime

import pandas as pd

from ai_trader.backtest.engine import DEFAULT_STARTING_EQUITY, BacktestEngine
from ai_trader.config import AppConfig, StrategySpec

logger = logging.getLogger(__name__)

# Penalizacion para una muestra cuyo backtest falla (config degenerada, ventana vacia).
# Un Calmar realista es del orden de unidades; un fallo debe puntuar claramente peor
# que cualquier resultado valido, sin ser un -inf que desestabilice la agregacion.
FAILURE_PENALTY = -5.0


def evaluate_sample(
    base_config: AppConfig,
    spec: StrategySpec,
    bars: dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
    *,
    split_ratio: float = 0.7,
    starting_equity: float = DEFAULT_STARTING_EQUITY,
) -> float:
    """
    Corre UNA muestra (un dict de barras = un escenario/path) por el backtest real y
    devuelve su headline score: Calmar OUT-OF-SAMPLE (ventana test).

    La estrategia candidata sustituye a la del config base; TODO lo demas (universo,
    riesgo, ejecucion, fees, slippage) es el del sistema en vivo. Lo que se puntua es
    lo que operaria de verdad. Un fallo de backtest puntua FAILURE_PENALTY, no rompe
    el barrido.

    Esta funcion es AGNOSTICA de la libreria sintetica: recibe barras ya cargadas y no
    conoce ningun 'ai_v1'/'ai_v2'. Quien elige el sustrato es quien la llama (por
    defecto, run_optimization con DEFAULT_LIBRARY_ID = 'ai_v2').
    """
    config = dataclasses.replace(base_config, strategies=[spec])
    try:
        result = BacktestEngine.from_bars(
            config, bars, starting_equity=starting_equity
        ).run(start, end, split_ratio=split_ratio)
    except Exception as exc:  # noqa: BLE001 - una muestra mala no debe tumbar el barrido
        logger.warning("Sample backtest failed (%s); scoring as penalty", exc)
        return FAILURE_PENALTY
    return result.headline_score
