from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from ai_trader.shared import bars as bar_schema
from ai_trader.shared.instruments import AssetClass

# --- ventanas canonicas del espacio de observacion -----------------------------
# Son FIJAS y forman parte de la definicion del espacio de observacion (no son
# parametros de estrategia). Cambiarlas cambia la semantica del vector, asi que
# viven aqui como constantes documentadas.
RETURN_HORIZONS = (1, 5, 20, 60)
SMA_FAST = 20
SMA_SLOW = 50
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_WINDOW = 14
VOL_WINDOWS = (20, 60)
DONCHIAN_WINDOW = 20
DRAWDOWN_WINDOW = 60
VOLUME_WINDOW = 20

TRADING_DAYS_PER_YEAR = 365  # coherente con backtest/metrics.py

# Numero minimo de barras para que el horizonte mas largo este definido. Por debajo,
# las features que dependen de el degradan a 0.0 (documentado, no es un error).
MIN_OBS_BARS = max(SMA_SLOW, DRAWDOWN_WINDOW, max(RETURN_HORIZONS)) + 1

# Orden canonico del bloque de features del propio activo. El vector de observacion
# se construye SIEMPRE en este orden; es el contrato con cualquier politica/RL.
OWN_ASSET_FEATURES: tuple[str, ...] = (
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "sma_ratio",
    "price_sma_z",
    "rsi",
    "macd_norm",
    "atr_pct",
    "realized_vol_20",
    "realized_vol_60",
    "donchian_pos",
    "dist_to_high",
    "dist_to_low",
    "volume_ratio",  # OJO: el volumen sintetico es un proxy simplista.
    "drawdown_from_peak",
)


def _sanitize(value: float) -> float:
    """Cualquier NaN/inf (ventana insuficiente, division por cero) se colapsa a 0.0.
    Es la convencion del espacio de observacion: 'feature no disponible' = neutro."""
    if value is None or not math.isfinite(value):
        return 0.0
    return float(value)


def _log_return(close: pd.Series, horizon: int) -> float:
    if len(close) <= horizon:
        return 0.0
    now = close.iloc[-1]
    then = close.iloc[-1 - horizon]
    if now <= 0 or then <= 0:
        return 0.0
    return _sanitize(math.log(now / then))


def _rsi(close: pd.Series, window: int) -> float:
    if len(close) <= window:
        return 0.0
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return _sanitize(100.0 - (100.0 / (1.0 + rs)))


def _macd_norm(close: pd.Series) -> float:
    if len(close) < MACD_SLOW:
        return 0.0
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    price = close.iloc[-1]
    if price <= 0:
        return 0.0
    # Histograma MACD normalizado por precio: comparable entre activos de distinto nivel.
    return _sanitize((macd_line.iloc[-1] - signal.iloc[-1]) / price)


def _atr_pct(df: pd.DataFrame, close: pd.Series) -> float:
    if len(df) <= ATR_WINDOW:
        return 0.0
    high = bar_schema.series(df, bar_schema.HIGH)
    low = bar_schema.series(df, bar_schema.LOW)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / ATR_WINDOW, adjust=False).mean().iloc[-1]
    price = close.iloc[-1]
    if price <= 0:
        return 0.0
    return _sanitize(atr / price * 100.0)


def _realized_vol(close: pd.Series, window: int) -> float:
    if len(close) <= window:
        return 0.0
    log_ret = np.log(close / close.shift(1)).iloc[-window:]
    std = log_ret.std(ddof=1)
    return _sanitize(std * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)


def _donchian(df: pd.DataFrame, close: pd.Series) -> tuple[float, float, float]:
    """Posicion en el canal Donchian (0..1) y distancias % a maximo/minimo de N dias."""
    if len(df) < DONCHIAN_WINDOW:
        return 0.5, 0.0, 0.0
    high = bar_schema.series(df, bar_schema.HIGH).iloc[-DONCHIAN_WINDOW:]
    low = bar_schema.series(df, bar_schema.LOW).iloc[-DONCHIAN_WINDOW:]
    hi = float(high.max())
    lo = float(low.min())
    price = float(close.iloc[-1])
    rng = hi - lo
    pos = 0.5 if rng <= 0 else (price - lo) / rng
    dist_high = 0.0 if price <= 0 else (hi - price) / price * 100.0
    dist_low = 0.0 if price <= 0 else (price - lo) / price * 100.0
    return _sanitize(pos), _sanitize(dist_high), _sanitize(dist_low)


def _volume_ratio(df: pd.DataFrame) -> float:
    if bar_schema.VOLUME not in df.columns or len(df) < VOLUME_WINDOW:
        return 0.0
    volume = bar_schema.series(df, bar_schema.VOLUME)
    avg = volume.iloc[-VOLUME_WINDOW:].mean()
    if avg <= 0:
        return 0.0
    return _sanitize(volume.iloc[-1] / avg - 1.0)


def _drawdown_from_peak(close: pd.Series) -> float:
    window = close.iloc[-DRAWDOWN_WINDOW:]
    peak = float(window.max())
    price = float(close.iloc[-1])
    if peak <= 0:
        return 0.0
    return _sanitize((peak - price) / peak * 100.0)


def _sma_features(close: pd.Series) -> tuple[float, float]:
    if len(close) < SMA_SLOW:
        return 0.0, 0.0
    sma_fast = close.rolling(SMA_FAST, min_periods=SMA_FAST).mean().iloc[-1]
    sma_slow = close.rolling(SMA_SLOW, min_periods=SMA_SLOW).mean().iloc[-1]
    std_slow = close.rolling(SMA_SLOW, min_periods=SMA_SLOW).std(ddof=1).iloc[-1]
    price = float(close.iloc[-1])

    ratio = 0.0 if pd.isna(sma_fast) or pd.isna(sma_slow) or sma_slow <= 0 else sma_fast / sma_slow - 1.0
    z = 0.0 if pd.isna(sma_slow) or pd.isna(std_slow) or std_slow <= 0 else (price - sma_slow) / std_slow
    return _sanitize(ratio), _sanitize(z)


def build_own_asset_features(bars: pd.DataFrame) -> dict[str, float]:
    """
    Features del mercado del PROPIO activo, calculadas en el ultimo cierre de la
    ventana. La ventana ya viene recortada anti look-ahead por el reloj historico,
    asi que aqui no hay riesgo de futuro: solo se lee hasta la ultima barra.

    Devuelve SIEMPRE las mismas claves (OWN_ASSET_FEATURES), con 0.0 donde la ventana
    es demasiado corta para definir la feature.
    """
    if bars is None or bars.empty:
        return dict.fromkeys(OWN_ASSET_FEATURES, 0.0)

    close = bar_schema.series(bars, bar_schema.CLOSE).dropna()
    if close.empty:
        return dict.fromkeys(OWN_ASSET_FEATURES, 0.0)

    sma_ratio, price_sma_z = _sma_features(close)
    donchian_pos, dist_high, dist_low = _donchian(bars, close)

    return {
        "ret_1d": _log_return(close, 1),
        "ret_5d": _log_return(close, 5),
        "ret_20d": _log_return(close, 20),
        "ret_60d": _log_return(close, 60),
        "sma_ratio": sma_ratio,
        "price_sma_z": price_sma_z,
        "rsi": _rsi(close, RSI_WINDOW),
        "macd_norm": _macd_norm(close),
        "atr_pct": _atr_pct(bars, close),
        "realized_vol_20": _realized_vol(close, 20),
        "realized_vol_60": _realized_vol(close, 60),
        "donchian_pos": donchian_pos,
        "dist_to_high": dist_high,
        "dist_to_low": dist_low,
        "volume_ratio": _volume_ratio(bars),
        "drawdown_from_peak": _drawdown_from_peak(close),
    }


# One-hot de clase de activo. Orden fijo = parte del contrato del vector.
ASSET_CLASS_ONEHOT: tuple[str, ...] = ("is_crypto", "is_stock", "is_macro")


def asset_class_onehot(asset_class: AssetClass | None) -> dict[str, float]:
    return {
        "is_crypto": 1.0 if asset_class == AssetClass.CRYPTO else 0.0,
        "is_stock": 1.0 if asset_class == AssetClass.STOCK else 0.0,
        # 'macro' no es una AssetClass del sistema (solo CRYPTO/STOCK/PREDICTION);
        # se marca como macro todo lo que no sea ni cripto ni accion ni prediccion.
        "is_macro": 1.0
        if asset_class not in (AssetClass.CRYPTO, AssetClass.STOCK, AssetClass.PREDICTION)
        else 0.0,
    }


@dataclass(slots=True)
class Observation:
    """
    Espacio de observacion explicito de una decision: TODO lo que la politica ve en
    el momento de decidir sobre `symbol`, respetando el anti look-ahead (solo datos
    hasta el cierre de hoy).

    Se compone de tres bloques con orden canonico y estable:
      - `own_asset`   : mercado del propio activo (OWN_ASSET_FEATURES)
      - `regime`      : cross-sectional / regimen (REGIME_FEATURES); {} si no hay provider
      - `asset_class` : one-hot de clase de activo (ASSET_CLASS_ONEHOT)

    `to_vector()` aplana a un np.ndarray en ese orden; es el contrato con el RL.
    """

    symbol: str
    as_of: datetime
    own_asset: dict[str, float]
    asset_class: dict[str, float] = field(default_factory=dict)
    regime: dict[str, float] = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        return np.array(
            [self.own_asset.get(name, 0.0) for name in OWN_ASSET_FEATURES]
            + [self.regime.get(name, 0.0) for name in self.regime_feature_names]
            + [self.asset_class.get(name, 0.0) for name in ASSET_CLASS_ONEHOT],
            dtype=float,
        )

    @property
    def regime_feature_names(self) -> tuple[str, ...]:
        # Import diferido para no acoplar el modulo de features al de regime.
        from ai_trader.observation.regime import REGIME_FEATURES

        return REGIME_FEATURES

    def feature_names(self) -> tuple[str, ...]:
        return OWN_ASSET_FEATURES + self.regime_feature_names + ASSET_CLASS_ONEHOT

    def as_dict(self) -> dict[str, float]:
        merged: dict[str, float] = {}
        merged.update(self.own_asset)
        merged.update(self.regime)
        merged.update(self.asset_class)
        return merged
