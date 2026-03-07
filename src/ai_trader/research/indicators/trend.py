import pandas as pd
from ai_trader.indicators.ta import sma, atr


def trend_snapshot(df: pd.DataFrame) -> dict:
    """
    Requires df columns: Open, High, Low, Close, Volume
    Returns a dict with trend/volatility metrics.
    """
    if df is None or df.empty or len(df) < 60:
        return {"ok": False, "reason": "Not enough bars (need >= 60)"}

    close = df["Close"]

    sma50 = sma(close, 50)
    atr14 = atr(df, 14)

    last_close = float(close.iloc[-1])
    last_sma50 = float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else None
    last_atr14 = float(atr14.iloc[-1]) if pd.notna(atr14.iloc[-1]) else None

    if last_sma50 is None:
        return {"ok": False, "reason": "SMA50 not available yet"}

    direction = "UP" if last_close >= last_sma50 else "DOWN"

    atr_pct = None
    if last_atr14 is not None and last_close:
        atr_pct = (last_atr14 / last_close) * 100.0

    return {
        "ok": True,
        "close": last_close,
        "sma50": last_sma50,
        "direction": direction,
        "atr14": last_atr14,
        "atr14_pct": atr_pct,
    }