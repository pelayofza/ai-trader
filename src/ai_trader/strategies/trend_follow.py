import pandas as pd
from ai_trader.indicators.ta import sma, atr
from ai_trader.signals.models import TradeProposal

def generate_trend_follow_proposal(symbol: str, df: pd.DataFrame) -> TradeProposal:
    if df is None or df.empty or len(df) < 60:
        return TradeProposal(symbol, "NONE", None, None, None, None, "Not enough bars (need >= 60)")

    close = df["Close"]
    sma50 = sma(close, 50)
    atr14 = atr(df, 14)

    entry = float(close.iloc[-1])
    sma_last = sma50.iloc[-1]
    atr_last = atr14.iloc[-1]

    if pd.isna(sma_last) or pd.isna(atr_last) or atr_last <= 0:
        return TradeProposal(symbol, "NONE", None, None, None, None, "Indicators not ready (SMA50/ATR14)")

    if entry <= float(sma_last):
        return TradeProposal(symbol, "NONE", None, None, None, None, "Close is below SMA50 (no trend-long)")

    # Risk model
    stop = entry - 2.0 * float(atr_last)
    target = entry + 3.0 * float(atr_last)

    risk = entry - stop
    reward = target - entry
    rr = (reward / risk) if risk > 0 else None

    return TradeProposal(
        symbol=symbol,
        side="LONG",
        entry=entry,
        stop=stop,
        target=target,
        rr=rr,
        reason="Trend-follow: Close > SMA50"
    )