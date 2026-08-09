from __future__ import annotations

from ai_trader.scoring.aggregate import DEFAULT_CVAR_ALPHA, RewardStats, aggregate_reward
from ai_trader.scoring.baselines import (
    Baseline,
    BaselineGate,
    compute_baselines,
    gate,
)
from ai_trader.scoring.cem import CEMConfig, CEMResult, maximize
from ai_trader.scoring.overfit import (
    DeflatedSharpe,
    PBOResult,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from ai_trader.scoring.sample_eval import (
    SampleEvaluation,
    evaluate_baselines,
    evaluate_sample,
    evaluate_sample_detailed,
)
from ai_trader.scoring.scenario_split import ScenarioSplit, split_scenarios
from ai_trader.scoring.search_space import SPACES, ParamSpace, get_space

__all__ = [
    "DEFAULT_CVAR_ALPHA",
    "RewardStats",
    "aggregate_reward",
    "Baseline",
    "BaselineGate",
    "compute_baselines",
    "gate",
    "CEMConfig",
    "CEMResult",
    "maximize",
    "DeflatedSharpe",
    "PBOResult",
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "ScenarioSplit",
    "split_scenarios",
    "SPACES",
    "ParamSpace",
    "get_space",
    "SampleEvaluation",
    "evaluate_baselines",
    "evaluate_sample",
    "evaluate_sample_detailed",
]
