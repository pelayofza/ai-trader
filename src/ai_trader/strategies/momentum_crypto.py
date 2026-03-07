from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
import logging

from ai_trader.shared.schemas import Side, Signal


logger = logging.getLogger(__name__)

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _get_price_column(df: pd.DataFrame, *candidates: str) -> pd.Series:
    for name in candidates:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    raise KeyError(f"Missing expected columns. Tried: {candidates}. Available: {list(df.columns)}")


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = _get_price_column(df, "high", "High")
    low = _get_price_column(df, "low", "Low")
    close = _get_price_column(df, "close", "Close")

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.ewm(alpha=1 / window, adjust=False).mean()


@dataclass(slots=True)
class CryptoMomentumConfig:
    timeframe: str = "1d"
    fast_sma_window: int = 10
    slow_sma_window: int = 20
    atr_window: int = 14
    breakout_lookback: int = 5
    min_atr_pct: float = 0.2
    risk_atr_multiple: float = 2.0
    reward_atr_multiple: float = 3.0
    min_bars: int = 30

    def __post_init__(self) -> None:
        if self.fast_sma_window <= 0:
            raise ValueError("fast_sma_window must be > 0")
        if self.slow_sma_window <= 0:
            raise ValueError("slow_sma_window must be > 0")
        if self.atr_window <= 0:
            raise ValueError("atr_window must be > 0")
        if self.breakout_lookback <= 1:
            raise ValueError("breakout_lookback must be > 1")
        if self.min_atr_pct <= 0:
            raise ValueError("min_atr_pct must be > 0")
        if self.risk_atr_multiple <= 0:
            raise ValueError("risk_atr_multiple must be > 0")
        if self.reward_atr_multiple <= 0:
            raise ValueError("reward_atr_multiple must be > 0")
        if self.min_bars < self.slow_sma_window:
            raise ValueError("min_bars must be >= slow_sma_window")


class CryptoMomentumStrategy:
    strategy_id = "crypto_momentum_v1"

    def __init__(self, config: CryptoMomentumConfig | None = None) -> None:
        self.config = config or CryptoMomentumConfig()

    def generate_signal(self, symbol: str, bars: pd.DataFrame) -> Signal | None:
        if bars is None or bars.empty:
            return None

        if len(bars) < self.config.min_bars:
            return None

        close = _get_price_column(bars, "close", "Close")
        high = _get_price_column(bars, "high", "High")

        fast_sma = _sma(close, self.config.fast_sma_window)
        slow_sma = _sma(close, self.config.slow_sma_window)
        atr = _atr(bars, self.config.atr_window)

        latest_close = float(close.iloc[-1])
        latest_fast_sma = fast_sma.iloc[-1]
        latest_slow_sma = slow_sma.iloc[-1]
        latest_atr = atr.iloc[-1]

        if pd.isna(latest_fast_sma) or pd.isna(latest_slow_sma) or pd.isna(latest_atr):
            return None

        if latest_close <= 0:
            return None

        atr_pct = float(latest_atr / latest_close * 100.0)

        recent_high = high.shift(1).rolling(
            window=self.config.breakout_lookback,
            min_periods=self.config.breakout_lookback,
        ).max()
        breakout_level = recent_high.iloc[-1]

        if pd.isna(breakout_level):
            return None

        trend_ok = bool(latest_close > latest_fast_sma and latest_fast_sma > latest_slow_sma)
        breakout_ok = True
        volatility_ok = bool(atr_pct >= self.config.min_atr_pct)
        
        logger.info(
            (
                "Momentum check | symbol=%s | close=%.6f | fast_sma=%.6f | slow_sma=%.6f | "
                "atr_pct=%.2f | trend_ok=%s | volatility_ok=%s"
            ),
            symbol,
            latest_close,
            float(latest_fast_sma),
            float(latest_slow_sma),
            atr_pct,
            trend_ok,
            volatility_ok,
        )

        if not (trend_ok and volatility_ok):
            logger.info("Strategy produced no signal for symbol=%s", symbol)
            return None

        stop_loss = float(latest_close - (latest_atr * self.config.risk_atr_multiple))
        take_profit = float(latest_close + (latest_atr * self.config.reward_atr_multiple))

        confidence = self._build_confidence(
            latest_close=latest_close,
            fast_sma=float(latest_fast_sma),
            slow_sma=float(latest_slow_sma),
            atr_pct=atr_pct,
            breakout_level=float(breakout_level),
        )

        logger.info("Strategy produced BUY signal for symbol=%s", symbol)

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            timeframe=self.config.timeframe,
            timestamp=utc_now(),
            side=Side.BUY,
            confidence=confidence,
            entry_price=latest_close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=(
                "Momentum breakout: close > SMA20 > SMA50, "
                f"ATR%={atr_pct:.2f}"
            ),
            features={
                "close": latest_close,
                "sma_fast": float(latest_fast_sma),
                "sma_slow": float(latest_slow_sma),
                "atr": float(latest_atr),
                "atr_pct": atr_pct,
                "breakout_level": float(breakout_level),
            },
            signal_id=f"{self.strategy_id}:{symbol}:{uuid4().hex[:12]}",
        )

    def _build_confidence(
        self,
        *,
        latest_close: float,
        fast_sma: float,
        slow_sma: float,
        atr_pct: float,
        breakout_level: float,
    ) -> float:
        trend_strength = min(max((fast_sma - slow_sma) / latest_close, 0.0), 0.05) / 0.05
        breakout_strength = min(max((latest_close - breakout_level) / latest_close, 0.0), 0.03) / 0.03
        volatility_strength = min(max(atr_pct / 4.0, 0.0), 1.0)

        raw_score = (
            0.45 * trend_strength
            + 0.35 * breakout_strength
            + 0.20 * volatility_strength
        )

        confidence = 0.55 + (raw_score * 0.35)
        return round(min(max(confidence, 0.55), 0.90), 2)