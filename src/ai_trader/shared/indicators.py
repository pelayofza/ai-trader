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

QUE DEVUELVEN Y POR QUE UNA SERIE
---------------------------------
Todo lo de aqui devuelve una SERIE alineada con el indice de las barras, no el ultimo
escalar. Los helpers de `observation/features.py` son privados y devuelven el valor de hoy
porque el vector de observacion solo necesita hoy; una estrategia, en cambio, compara el
valor de hoy con el de ayer, lo desplaza para no compararse consigo misma y lo mete en
`Signal.features`. Duplicar la version escalar aqui seria repetir el error que este modulo
existe para arreglar.

REGLA DEL DESPLAZAMIENTO. Los indicadores que responden "¿ha superado el precio de hoy
algo del pasado?" (`donchian_high`, `donchian_low`, `volume_ratio`) excluyen la barra de
hoy con `shift(1)` ANTES de la ventana. Sin eso el maximo de la ventana incluiria el
maximo de hoy y la comparacion `close > donchian_high` seria imposible de satisfacer casi
siempre; es la misma correccion que ya lleva el filtro de ruptura de `momentum_crypto`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ai_trader.shared import bars as bar_schema

# Observaciones por ano con las que se anualiza la volatilidad realizada. El default es el
# de cripto (24/7, una barra por dia natural). La AUTORIDAD sobre este numero es
# `backtest/metrics.py::periods_per_year_for_symbols`, que lo deduce de la clase de activo;
# aqui es un parametro con default y no una importacion porque `shared/` esta por debajo de
# `backtest/` en el grafo de dependencias y no puede importar de el.
DEFAULT_PERIODS_PER_YEAR = 365.0


def true_range(df: pd.DataFrame) -> pd.Series:
    """
    Rango verdadero de cada barra: el mayor de tres distancias —el recorrido del dia y los
    dos huecos contra el cierre anterior—, de forma que un salto nocturno cuenta como
    movimiento aunque el rango intradia sea estrecho.

    El cuerpo salio de `atr()` SIN reordenar ninguna operacion, por el mismo motivo por el
    que `atr()` salio de las estrategias: el orden en coma flotante es el mismo, asi que
    los backtests congelados en `tests/golden/` siguen saliendo identicos byte a byte.
    """
    high = bar_schema.series(df, bar_schema.HIGH)
    low = bar_schema.series(df, bar_schema.LOW)
    close = bar_schema.series(df, bar_schema.CLOSE)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range suavizado con la media exponencial de Wilder.

    El suavizado usa `alpha = 1/window` con `adjust=False`, que es la formulacion original
    de Wilder y no la media movil simple: pondera mas lo reciente sin descartar la historia
    de golpe.
    """
    return true_range(df).ewm(alpha=1 / window, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    """Media movil simple. `min_periods=window`: sin ventana completa no hay valor, NaN."""
    return series.rolling(window=window, min_periods=window).mean()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Desviacion MUESTRAL (ddof=1), coherente con `volatility_pct` del motor de metricas
    y con el z-score de `mean_reversion`."""
    return series.rolling(window=window, min_periods=window).std(ddof=1)


def log_returns(close: pd.Series) -> pd.Series:
    """Retornos logaritmicos. NaN donde el precio no es positivo: un precio <= 0 no es un
    dato malo que rellenar, es una barra que no se puede usar."""
    prices = pd.to_numeric(close, errors="coerce")
    valid = prices > 0
    logp = pd.Series(np.where(valid, np.log(prices.where(valid, 1.0)), np.nan), index=prices.index)
    return logp.diff()


def realized_vol(
    close: pd.Series,
    window: int,
    *,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> pd.Series:
    """Volatilidad realizada anualizada, en PORCENTAJE.

    Se publica anualizada y en % para que el numero sea legible en un log y en
    `Signal.features`; quien la usa como RATIO (vol corta / vol larga) obtiene lo mismo con
    o sin el factor, porque se cancela.
    """
    return rolling_std(log_returns(close), window) * np.sqrt(periods_per_year) * 100.0


def roc(close: pd.Series, horizon: int) -> pd.Series:
    """Tasa de cambio simple sobre `horizon` barras, en tanto por uno."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    prices = pd.to_numeric(close, errors="coerce")
    previous = prices.shift(horizon)
    return prices / previous.where(previous > 0) - 1.0


def donchian_high(df: pd.DataFrame, window: int) -> pd.Series:
    """Techo del canal de Donchian EXCLUYENDO la barra de hoy (ver la regla del modulo)."""
    high = bar_schema.series(df, bar_schema.HIGH)
    return high.shift(1).rolling(window=window, min_periods=window).max()


def donchian_low(df: pd.DataFrame, window: int) -> pd.Series:
    """Suelo del canal de Donchian EXCLUYENDO la barra de hoy."""
    low = bar_schema.series(df, bar_schema.LOW)
    return low.shift(1).rolling(window=window, min_periods=window).min()


def close_location(df: pd.DataFrame) -> pd.Series:
    """
    Donde cierra el precio DENTRO del rango del dia: 0 = en el minimo, 1 = en el maximo.

    Es la forma barata de distinguir "cayo un 8% y rebotó" de "cayo un 8% y cerro en
    minimos". Un rango nulo (barra plana) devuelve 0,5 y no NaN: la barra existe y su
    cierre esta, trivialmente, en el centro de un rango de anchura cero. Devolver NaN
    obligaria a cada llamante a decidir lo mismo, y alguno decidiria distinto.
    """
    high = bar_schema.series(df, bar_schema.HIGH)
    low = bar_schema.series(df, bar_schema.LOW)
    close = bar_schema.series(df, bar_schema.CLOSE)

    span = high - low
    location = (close - low) / span.where(span > 0)
    return location.where(span > 0, 0.5).where(span.notna() & close.notna())


def volume_ratio(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Volumen de hoy dividido por su MEDIANA movil (excluyendo hoy).

    Mediana y no media, y no es un detalle: estas series tienen colas gruesas, asi que un
    solo dia de volumen x20 subiria la media lo bastante como para que el propio pico
    dejara de parecer un pico. La mediana es insensible a eso, que es exactamente lo que se
    quiere de un denominador que representa "el volumen normal de este activo".
    """
    if bar_schema.VOLUME not in df.columns:
        return pd.Series(np.nan, index=df.index)
    volume = bar_schema.series(df, bar_schema.VOLUME)
    baseline = volume.shift(1).rolling(window=window, min_periods=window).median()
    return volume / baseline.where(baseline > 0)


def up_share(close: pd.Series, window: int) -> pd.Series:
    """
    Fraccion de barras con retorno POSITIVO en la ventana. 0,5 = tantas arriba como abajo.

    Mide persistencia, que es otra cosa que la pendiente: una subida de +20% concentrada en
    dos dias y otra de +20% repartida en treinta tienen la misma pendiente y muy distinta
    `up_share`. El primer retorno es NaN (no tiene barra previa) y se mantiene NaN en vez
    de contar como bajada, para que la primera ventana no salga sesgada a la baja.
    """
    changes = pd.to_numeric(close, errors="coerce").diff()
    positive = (changes > 0).astype(float).where(changes.notna())
    return positive.rolling(window=window, min_periods=window).mean()


__all__ = [
    "DEFAULT_PERIODS_PER_YEAR",
    "atr",
    "close_location",
    "donchian_high",
    "donchian_low",
    "log_returns",
    "realized_vol",
    "roc",
    "rolling_std",
    "sma",
    "true_range",
    "up_share",
    "volume_ratio",
]
