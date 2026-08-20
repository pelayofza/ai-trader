"""
Generador de datos sinteticos para el backtest.

Pieza independiente del resto del sistema: no consume salidas de otras piezas
(estrategias, riesgo, RL). Produce OHLCV multi-activo y correlacionado que el motor
de backtest lee tal cual via BarProvider / BacktestEngine.from_bars.

Arquitectura:
- La IA (ScenarioDesigner) disena escenarios macro diversos y, por escenario, define
  la "fisica": movimientos de factores + shocks + tilts por activo. No dibuja velas.
- Un motor numerico determinista (PathEngine) sintetiza las velas a partir de esa
  fisica mediante un modelo de factores, de modo que las correlaciones entre activos
  son siempre consistentes y la matriz siempre valida.
- Cada escenario genera un ensemble Monte Carlo de N paths (misma fisica, distinta
  semilla) para que la estrategia no se sobreajuste a una unica realizacion.
- Todo se persiste en disco (SyntheticStore) para no re-ejecutar la IA.
"""

from ai_trader.research.synthetic.scenarios import (
    FactorPhase,
    FactorShock,
    ScenarioSpec,
)
from ai_trader.research.synthetic.universe import (
    DEFAULT_FACTORS,
    DEFAULT_UNIVERSE,
    SyntheticAsset,
    SyntheticUniverse,
)

__all__ = [
    "DEFAULT_FACTORS",
    "DEFAULT_UNIVERSE",
    "FactorPhase",
    "FactorShock",
    "ScenarioSpec",
    "SyntheticAsset",
    "SyntheticUniverse",
]
