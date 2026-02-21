import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df, window: int = 14) -> pd.Series:
    """
    df must contain: High, Low, Close
    Returns ATR series.
    """
    prev_close = df["Close"].shift(1)
    tr = true_range(df["High"], df["Low"], prev_close)
    return tr.rolling(window=window, min_periods=window).mean()