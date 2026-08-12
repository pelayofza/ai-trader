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
class SignalsConfig:
    """
    El radar de senales externas. APAGADO por defecto, y eso es una decision.

    Encenderlo lee el archivo crudo de `data/signals_raw/`, deriva las diecisiete fuentes
    y construye el bloque de observacion (`observation/signal_radar.py`). Con `enabled =
    false` no se lee nada, el radar sale vacio, la cobertura es 0 y todas las puertas de
    senales se saltan: el sistema opera EXACTAMENTE igual que antes de que existieran.

    LO QUE NO HAY AQUI, Y ES EL PUNTO: ningun umbral. Ni el minimo de cobertura, ni los
    pisos de tono, ni las escalas de las magnitudes de evento. Todo eso son constantes
    razonadas en codigo, y estan ahi justamente para que no se puedan ajustar —ni desde un
    fichero de configuracion ni desde un sorteo del optimizador— contra el resultado. Lo
    unico configurable de las senales es si se encienden y de donde se leen.
    """

    enabled: bool = False
    # Raiz del archivo crudo. Vacio = la de `signals/store.py`. Es una RUTA, no un umbral.
    raw_root: str = ""


@dataclass(slots=True)
class AppConfig:
    runner: RunnerConfig
    risk: RiskLimits
    execution: PaperExecutionConfig
    strategies: list[StrategySpec]
    # Con default: todo lo que ya construia un AppConfig (estudios, tests, el generador
    # sintetico) sigue construyendolo sin tocar una linea, y con el radar apagado.
    signals: SignalsConfig = field(default_factory=SignalsConfig)


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
        # El splat es ESTRICTO a proposito (una clave de mas revienta al cargar, en vez de
        # ignorarse en silencio), asi que una seccion nueva exige dataclass + campo + esta
        # linea. Sin `[signals]` en el fichero, el radar queda apagado.
        signals=SignalsConfig(**raw.get("signals", {})),
    )
