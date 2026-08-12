"""
Indicadores tecnicos compartidos por varias estrategias.

Existe porque `_atr` estaba DUPLICADO byte a byte en `strategies/momentum_crypto.py` y
`strategies/mean_reversion.py`. Dos copias del mismo indicador es una divergencia
esperando a pasar: corregir el ATR en una y no en la otra dejaria a dos estrategias que
dicen usar el mismo indicador usando indicadores distintos, y eso no lo detecta ningun
test, porque cada estrategia comprueba los suyos.

El cuerpo se movio TAL CUAL desde las estrategias, sin reescribir ni reordenar ninguna
operacion: el orden de las operaciones en coma flotante es el mismo, asi que los backtests
congelados en `tests/golden/` tienen que salir identicos byte a byte. Esa es la prueba de
que el cambio fue equivalente y no una reimplementacion con el mismo nombre.

Los dos modulos que lo usaban conservan el nombre `_atr` como alias que delega
(`from ... import atr as _atr`), de modo que ninguna llamada existente cambia.
"""
from __future__ import annotations

import pandas as pd

from ai_trader.shared import bars as bar_schema


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range suavizado con la media exponencial de Wilder.

    El rango verdadero de cada barra es el mayor de tres distancias —el recorrido del dia,
    y los dos huecos contra el cierre anterior—, de forma que un salto nocturno cuenta como
    movimiento aunque el rango intradia sea estrecho. El suavizado usa `alpha = 1/window`
    con `adjust=False`, que es la formulacion original de Wilder y no la media movil
    simple: pondera mas lo reciente sin descartar la historia de golpe.
    """
    high = bar_schema.series(df, bar_schema.HIGH)
    low = bar_schema.series(df, bar_schema.LOW)
    close = bar_schema.series(df, bar_schema.CLOSE)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.ewm(alpha=1 / window, adjust=False).mean()


__all__ = ["atr"]
