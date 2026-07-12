from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_trader.app.runner import RunnerConfig
from ai_trader.execution.paper import PaperExecutionConfig
from ai_trader.risk.engine import RiskLimits

DEFAULT_CONFIG_PATH = Path("config") / "default.toml"


@dataclass(slots=True)
class StrategySpec:
    type: str
    id: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AppConfig:
    runner: RunnerConfig
    risk: RiskLimits
    execution: PaperExecutionConfig
    strategies: list[StrategySpec]


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))

    universe = raw.get("universe") or []
    if not universe:
        raise ValueError(f"'universe' is empty in {config_path}")

    strategies = [
        StrategySpec(
            type=item["type"],
            id=item.get("id"),
            params=dict(item.get("params", {})),
        )
        for item in raw.get("strategies", [])
    ]
    if not strategies:
        raise ValueError(f"No strategies configured in {config_path}")

    return AppConfig(
        runner=RunnerConfig(symbols=list(universe), **raw.get("runner", {})),
        risk=RiskLimits(**raw.get("risk", {})),
        execution=PaperExecutionConfig(**raw.get("execution", {})),
        strategies=strategies,
    )
