from __future__ import annotations

import pandas as pd

OPEN = "open"
HIGH = "high"
LOW = "low"
CLOSE = "close"
VOLUME = "volume"

OHLCV_COLUMNS = (OPEN, HIGH, LOW, CLOSE, VOLUME)


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forma canonica de un DataFrame de barras: columnas OHLCV en minuscula,
    indice datetime UTC ordenado y sin duplicados.

    Todo proveedor debe devolver barras que ya hayan pasado por aqui, de modo que
    ningun consumidor tenga que adivinar si la columna se llama "close" o "Close".
    """
    normalized = df.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]

    normalized.index = pd.to_datetime(normalized.index, utc=True, errors="coerce")
    normalized = normalized[~normalized.index.isna()]
    normalized = normalized.sort_index()
    normalized = normalized[~normalized.index.duplicated(keep="last")]

    return normalized


def series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise KeyError(f"Missing column '{column}'. Available: {list(df.columns)}")
    return pd.to_numeric(df[column], errors="coerce")


def last_close(df: pd.DataFrame) -> float | None:
    if df is None or df.empty or CLOSE not in df.columns:
        return None

    value = pd.to_numeric(df[CLOSE], errors="coerce").iloc[-1]
    if pd.isna(value):
        return None

    return float(value)
