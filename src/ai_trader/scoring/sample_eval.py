from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from ai_trader.backtest.engine import (
    DEFAULT_STARTING_EQUITY,
    BacktestEngine,
    resolve_split_cutoff,
)
from ai_trader.backtest.metrics import (
    DEFAULT_HEADLINE_WEIGHTS,
    HeadlineWeights,
    daily_returns,
    kurtosis,
    skewness,
)
from ai_trader.config import AppConfig, StrategySpec
from ai_trader.scoring.baselines import Baseline, compute_baselines

logger = logging.getLogger(__name__)

# Penalizacion para una muestra cuyo backtest falla (config degenerada, ventana vacia).
# El headline vive en unidades de Sharpe (tipicamente -2..2), asi que -5 es claramente
# peor que cualquier resultado valido sin ser un -inf que desestabilice la agregacion.
FAILURE_PENALTY = -5.0


@dataclass(slots=True, frozen=True)
class SampleEvaluation:
    """
    Resultado out-of-sample de UNA muestra. Ademas del score lleva las piezas que el
    descuento por multiples pruebas necesita (Sharpe crudo, nº de observaciones OOS y
    los momentos de los retornos), para que el DSR no tenga que reconstruirlas ni
    aproximarlas desde el score penalizado.
    """

    score: float
    sharpe: float
    turnover: float
    max_drawdown_pct: float
    num_trades: int
    oos_observations: int
    returns_skew: float
    returns_kurtosis: float
    failed: bool = False

    @classmethod
    def failure(cls) -> SampleEvaluation:
        return cls(
            score=FAILURE_PENALTY,
            sharpe=0.0,
            turnover=0.0,
            max_drawdown_pct=0.0,
            num_trades=0,
            oos_observations=0,
            returns_skew=0.0,
            returns_kurtosis=3.0,
            failed=True,
        )


def evaluate_sample_detailed(
    base_config: AppConfig,
    spec: StrategySpec,
    bars: dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
    *,
    split_ratio: float = 0.7,
    starting_equity: float = DEFAULT_STARTING_EQUITY,
    headline_weights: HeadlineWeights = DEFAULT_HEADLINE_WEIGHTS,
) -> SampleEvaluation:
    """
    Corre UNA muestra (un dict de barras = un escenario/path) por el backtest real y
    devuelve su evaluacion OUT-OF-SAMPLE (ventana test): headline score
    `Sharpe - lambda*turnover - kappa*maxDD` mas los estadisticos que lo componen.

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
            config, bars, starting_equity=starting_equity, headline_weights=headline_weights
        ).run(start, end, split_ratio=split_ratio)
    except Exception as exc:  # noqa: BLE001 - una muestra mala no debe tumbar el barrido
        logger.warning("Sample backtest failed (%s); scoring as penalty", exc)
        return SampleEvaluation.failure()

    metrics = result.test.metrics
    returns = daily_returns(result.test.equity_curve)
    return SampleEvaluation(
        score=result.headline_score,
        sharpe=metrics.sharpe,
        turnover=metrics.turnover,
        max_drawdown_pct=metrics.max_drawdown_pct,
        num_trades=metrics.num_trades,
        oos_observations=len(returns),
        returns_skew=skewness(returns),
        returns_kurtosis=kurtosis(returns),
    )


def evaluate_sample(
    base_config: AppConfig,
    spec: StrategySpec,
    bars: dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
    *,
    split_ratio: float = 0.7,
    starting_equity: float = DEFAULT_STARTING_EQUITY,
    headline_weights: HeadlineWeights = DEFAULT_HEADLINE_WEIGHTS,
) -> float:
    """Atajo escalar de `evaluate_sample_detailed`: solo el headline score OOS."""
    return evaluate_sample_detailed(
        base_config, spec, bars, start, end,
        split_ratio=split_ratio,
        starting_equity=starting_equity,
        headline_weights=headline_weights,
    ).score


def evaluate_baselines(
    base_config: AppConfig,
    bars: dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
    *,
    split_ratio: float = 0.7,
    starting_equity: float = DEFAULT_STARTING_EQUITY,
    headline_weights: HeadlineWeights = DEFAULT_HEADLINE_WEIGHTS,
) -> dict[str, Baseline]:
    """
    Puntua los baselines pasivos de UNA muestra en su ventana OUT-OF-SAMPLE.

    El corte train/test sale de `resolve_split_cutoff`, la misma funcion que usa el
    motor de backtest: los baselines se miden en la ventana exacta de la estrategia, y
    con los costes del config (fee_rate + slippage) para que nadie compare gratis.

    El slippage que pagan es el PLANO de referencia, no el modelo de microestructura que
    paga la estrategia: un baseline hace dos operaciones sobre los activos mas liquidos
    del universo, y darle el coste barato solo puede ENDURECER el liston que la
    estrategia tiene que superar. La direccion prudente del error.
    """
    cutoff = resolve_split_cutoff(start, end, split_ratio=split_ratio)
    return compute_baselines(
        bars,
        cutoff,
        end,
        starting_equity=starting_equity,
        fee_rate=base_config.execution.fee_rate,
        slippage_bps=base_config.execution.slippage_bps,
        weights=headline_weights,
    )
